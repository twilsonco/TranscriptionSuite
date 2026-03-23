"""
WebSocket endpoint for external audio streaming STT with real-time segment delivery
and optional Audio Notebook saving.

Compatible with the Omi External Custom STT protocol:
  https://docs.omi.me/developer/backend/custom-stt

Endpoint: /ws/omi  (URL kept for backward compatibility with Omi devices)

Query params:
  token             - API token for authentication (required unless local loopback)
  codec             - "opus" (default) or "pcm"
  sample_rate       - Input sample rate in Hz (default: 16000)
  language          - BCP-47 language code, e.g. "en" (optional; auto-detect if omitted)
  save_to_notebook  - "true"/"1" to save final high-quality transcript to Audio Notebook

Protocol:
  Client → Server: binary audio frames  OR  JSON {"type": "CloseStream"}
  Server → Client: {"segments": [...], "is_partial": bool}

  Segments are delivered progressively as speech pauses are detected.
  A final delivery with is_partial=false is sent on a graceful CloseStream close.

Sessions and cross-connection stitching:
  Connections sharing the same API token are grouped into a conversation.
  A new connection that arrives within inactivity_timeout_s of the previous frame
  continues the same conversation; a later connection starts fresh.
  All segment timestamps are absolute seconds from the conversation start.

Notebook saving:
  When save_to_notebook=true (first connection's value wins for the session),
  the complete conversation audio is re-transcribed at high quality (with full
  diarization) and saved to the Audio Notebook when the conversation ends.
  This save is a background asyncio.Task that does not block other connections.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from server.api.routes.utils import is_local_auth_bypass_host
from server.core.token_store import get_token_store

# Match local-mode trust logic used by the rest of the server (see main.py)
_TLS_MODE: bool = os.environ.get("TLS_ENABLED", "false").lower() == "true"

logger = logging.getLogger(__name__)

router = APIRouter()

# PCM constants matching Omi device protocol defaults
PCM_SAMPLE_RATE = 16000
PCM_DTYPE = np.int16

# Config defaults (overridable from the ws_stt config section)
_DEFAULT_INACTIVITY_S: float = 90.0
_DEFAULT_SEGMENT_SILENCE_S: float = 5.0
_DEFAULT_MIN_SEGMENT_S: float = 0.5
_DEFAULT_CONTEXT_WINDOW_S: float = 300.0
_DEFAULT_MAX_SEGMENT_S: float = 300.0
_DEFAULT_MAX_CONV_S: float = 7200.0


# ---------------------------------------------------------------------------
# Cross-connection session state
# ---------------------------------------------------------------------------


@dataclass
class WsAudioSession:
    """
    Per-token cross-connection audio conversation state.

    One session spans all WebSocket connections that share an API token
    and arrive within inactivity_s seconds of each other.
    """

    session_key: str  # sha256(token) truncated to 24 hex chars

    # All audio accumulated so far in this conversation (float32 @ sample_rate)
    conv_chunks: list[np.ndarray] = field(default_factory=list)

    # Flat list of all segments streamed to client(s) during this conversation
    streamed_segments: list[dict[str, Any]] = field(default_factory=list)

    # Monotonic timestamps for gap detection
    conv_start_monotonic: float = field(default_factory=time.monotonic)
    last_audio_monotonic: float = field(default_factory=time.monotonic)

    # Flags set by first-connection-wins
    save_to_notebook: bool = False
    language: str | None = None

    # Background notebook save task handle
    save_task: asyncio.Task | None = None  # type: ignore[type-arg]

    # Config values loaded when conversation starts
    inactivity_s: float = _DEFAULT_INACTIVITY_S
    segment_silence_s: float = _DEFAULT_SEGMENT_SILENCE_S
    min_segment_s: float = _DEFAULT_MIN_SEGMENT_S
    context_window_s: float = _DEFAULT_CONTEXT_WINDOW_S
    max_segment_s: float = _DEFAULT_MAX_SEGMENT_S
    max_conv_s: float = _DEFAULT_MAX_CONV_S

    @property
    def conv_duration_s(self) -> float:
        return sum(len(c) for c in self.conv_chunks) / PCM_SAMPLE_RATE

    def conv_audio(self) -> np.ndarray:
        if not self.conv_chunks:
            return np.array([], dtype=np.float32)
        return np.concatenate(self.conv_chunks)


# Module-level session store (keyed by session_key derived from the API token)
_ws_sessions: dict[str, WsAudioSession] = {}


def _session_key(token: str) -> str:
    """Derive a stable, safe session key from an API token."""
    return hashlib.sha256(token.encode()).hexdigest()[:24]


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------


def _load_ws_cfg() -> dict[str, Any]:
    """
    Return the ws_stt config section, falling back to omi_stt for backward compat.
    """
    try:
        from server.config import get_config

        cfg = get_config()
        ws_cfg: Any = cfg.get("ws_stt", default=None)
        if ws_cfg is None:
            ws_cfg = cfg.get("omi_stt", default={})
        return ws_cfg or {}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------


def _is_authorized(token: str | None, client_host: str | None) -> bool:
    """
    Return True if the request is authorized.

    In non-TLS mode loopback connections are trusted without a token,
    consistent with how other HTTP routes work (see main.py).
    In TLS mode a valid token is always required.
    """
    if not _TLS_MODE and is_local_auth_bypass_host(client_host):
        return True
    if not token:
        return False
    token = token.strip()
    if not token:
        return False
    return get_token_store().validate_token(token) is not None


# ---------------------------------------------------------------------------
# Audio decoding helpers
# ---------------------------------------------------------------------------


def _decode_pcm(data: bytes, sample_rate: int = PCM_SAMPLE_RATE) -> np.ndarray:
    """Interpret raw bytes as signed 16-bit little-endian PCM, return float32."""
    pcm = np.frombuffer(data, dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0


def _make_opus_decoder(sample_rate: int = PCM_SAMPLE_RATE) -> Any:
    """Create an opuslib Decoder or raise ImportError with an actionable message."""
    try:
        import opuslib  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "opuslib is not installed. "
            "Install it with: uv sync --extra omi\n"
            "You also need the system libopus library: brew install opus"
        ) from exc
    return opuslib.Decoder(sample_rate, channels=1)


def _decode_opus_frame(decoder: Any, data: bytes) -> np.ndarray:
    """Decode a single Opus packet and return float32 PCM."""
    # 5760 = max Opus frame (120 ms at 48 kHz) — large enough for any valid packet
    # regardless of the frame duration the sender chose (20 ms, 40 ms, 60 ms, etc.).
    try:
        pcm_bytes = decoder.decode(data, frame_size=5760)
    except Exception:
        logger.debug(
            "_decode_opus_frame: %d-byte frame rejected: first_bytes=%s",
            len(data),
            data[:16].hex(),
        )
        raise
    pcm = np.frombuffer(pcm_bytes, dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0


# ---------------------------------------------------------------------------
# Transcription helpers
# ---------------------------------------------------------------------------


async def _transcribe_audio(
    audio: np.ndarray,
    sample_rate: int,
    language: str | None,
    *,
    enable_diarization: bool = False,
) -> list[dict[str, Any]]:
    """
    Transcribe *audio* (float32 PCM) using the loaded model manager.

    When *enable_diarization* is True, attempts diarization in priority order:
      1. Native integrated diarization (WhisperX ``transcribe_with_diarization``)
      2. Pyannote parallel diarization fallback (reads HF_TOKEN from env/config)
      3. Plain transcription without speaker labels

    When False (default, for streaming chunks), uses only plain transcription
    to avoid blocking the event loop for the full diarization pipeline.

    Returns a list of segment dicts: {text, start, end, speaker?}
    """
    import os as _os

    from server.core.model_manager import get_model_manager

    model_manager = get_model_manager()
    engine = model_manager.transcription_engine

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        sf.write(str(tmp_path), audio, sample_rate)

    try:
        if enable_diarization:
            from server.config import get_config as _get_config

            _cfg = _get_config()
            hf_token: str | None = (
                _os.environ.get("HF_TOKEN", "").strip()
                or str(_cfg.get("diarization", "hf_token") or "").strip()
            ) or None

            # --- Attempt 1: Native integrated diarization (e.g. WhisperX) ---
            backend = getattr(engine, "_backend", None)

            if backend is not None and hf_token and hasattr(backend, "transcribe_with_diarization"):
                try:
                    diar_result = await asyncio.to_thread(
                        backend.transcribe_with_diarization,
                        audio,
                        audio_sample_rate=sample_rate,
                        language=language,
                        hf_token=hf_token,
                    )
                    segments: list[dict[str, Any]] = []
                    for seg in diar_result.segments:
                        entry: dict[str, Any] = {
                            "text": seg.get("text", "").strip(),
                            "start": round(float(seg.get("start", 0.0)), 3),
                            "end": round(float(seg.get("end", 0.0)), 3),
                        }
                        if seg.get("speaker"):
                            entry["speaker"] = seg["speaker"]
                        if entry["text"]:
                            segments.append(entry)
                    if segments:
                        logger.debug(
                            "_transcribe_audio: native diarization → %d segments", len(segments)
                        )
                        return segments
                except Exception as exc:
                    logger.debug("_transcribe_audio: native diarization failed: %s", exc)

            # --- Attempt 2: Pyannote parallel diarization fallback ---
            try:
                from server.core.parallel_diarize import (
                    transcribe_and_diarize,
                    transcribe_then_diarize,
                )
                from server.core.speaker_merge import build_speaker_segments

                use_parallel = _cfg.get("diarization", "parallel", default=True)
                diarize_fn = transcribe_and_diarize if use_parallel else transcribe_then_diarize

                result, diar_result = await asyncio.to_thread(
                    diarize_fn,
                    engine=engine,
                    model_manager=model_manager,
                    file_path=str(tmp_path),
                    language=language,
                    word_timestamps=True,
                )
                if diar_result is not None:
                    diar_dicts = [seg.to_dict() for seg in diar_result.segments]
                    merged_segments, _, _ = build_speaker_segments(result.words, diar_dicts)
                    if merged_segments:
                        logger.debug(
                            "_transcribe_audio: pyannote diarization → %d segments",
                            len(merged_segments),
                        )
                        return [
                            {
                                "text": s.get("text", "").strip(),
                                "start": round(float(s.get("start", 0.0)), 3),
                                "end": round(float(s.get("end", 0.0)), 3),
                                **({"speaker": s["speaker"]} if s.get("speaker") else {}),
                            }
                            for s in merged_segments
                            if s.get("text", "").strip()
                        ]
            except Exception as exc:
                logger.debug("_transcribe_audio: pyannote fallback failed: %s", exc)

        # --- Plain transcription (always attempted; only path when enable_diarization=False) ---
        result = await asyncio.to_thread(
            engine.transcribe_file,
            str(tmp_path),
            language=language,
            word_timestamps=False,
        )
        plain_segments = [
            {
                "text": seg.get("text", "").strip(),
                "start": round(float(seg.get("start", 0.0)), 3),
                "end": round(float(seg.get("end", 0.0)), 3),
            }
            for seg in result.segments
            if seg.get("text", "").strip()
        ]
        logger.debug(
            "_transcribe_audio: plain transcription → %d segments", len(plain_segments)
        )
        return plain_segments

    finally:
        try:
            tmp_path.unlink()
        except Exception:
            pass


async def _transcribe_with_context(
    session: WsAudioSession,
    pending: np.ndarray,
    sample_rate: int,
) -> list[dict[str, Any]]:
    """
    Transcribe *pending* audio using a rolling context window for quality.

    Prepends the last ``context_window_s`` seconds of accumulated conversation
    audio before *pending* so the model has conversational context.  Resulting
    segments that fall within the context region are discarded; only segments
    in the *pending* region are returned.

    Returned segment timestamps are **absolute seconds from the conversation
    start**, so they stitch correctly across multiple flush calls and
    reconnections.
    """
    conv_dur = session.conv_duration_s
    context_start = max(0.0, conv_dur - session.context_window_s)
    context_start_frame = int(context_start * sample_rate)

    if session.conv_chunks:
        full_conv = np.concatenate(session.conv_chunks)
        context_audio = full_conv[context_start_frame:]
    else:
        context_audio = np.array([], dtype=np.float32)

    context_dur = len(context_audio) / sample_rate  # conv_dur - context_start

    combined = np.concatenate([context_audio, pending]) if len(context_audio) > 0 else pending
    combined_dur = len(combined) / sample_rate

    logger.info(
        "transcribe_with_context[%s]: context=%.1fs, pending=%.1fs, combined=%.1fs, conv_total=%.1fs",
        session.session_key[:8],
        context_dur,
        len(pending) / sample_rate,
        combined_dur,
        conv_dur,
    )

    raw_segments = await _transcribe_audio(combined, sample_rate, session.language)

    # Filter and timestamp-adjust
    result: list[dict[str, Any]] = []
    for seg in raw_segments:
        t_start = float(seg.get("start", 0.0))
        t_end = float(seg.get("end", t_start))

        # Skip segments entirely within the context region
        if t_end <= context_dur:
            continue

        # Clip start to context boundary for partially-overlapping segments
        effective_start = max(t_start, context_dur)

        # Convert to conversation-absolute timestamps:
        #   absolute = context_start_in_conv + position_in_combined
        abs_start = context_start + effective_start
        abs_end = context_start + t_end

        entry: dict[str, Any] = {
            "text": seg.get("text", "").strip(),
            "start": round(abs_start, 3),
            "end": round(abs_end, 3),
        }
        if seg.get("speaker"):
            entry["speaker"] = seg["speaker"]
        if entry["text"]:
            result.append(entry)

    return result


# ---------------------------------------------------------------------------
# Notebook save (background)
# ---------------------------------------------------------------------------


async def _save_conversation_to_notebook(session: WsAudioSession, sample_rate: int) -> None:
    """
    Re-transcribe the full conversation at high quality and save to the Audio Notebook.

    Runs as an asyncio.Task so it does not block any connection.
    """
    if not session.conv_chunks:
        logger.info(
            "save_conversation[%s]: no audio accumulated, skipping", session.session_key[:8]
        )
        return

    conv_audio = session.conv_audio()
    conv_dur = len(conv_audio) / sample_rate

    logger.info(
        "save_conversation[%s]: reprocessing full conversation (%.1fs) for notebook save",
        session.session_key[:8],
        conv_dur,
    )

    try:
        segments = await _transcribe_audio(conv_audio, sample_rate, session.language, enable_diarization=True)

        if not segments:
            logger.warning(
                "save_conversation[%s]: transcription returned no segments, skipping save",
                session.session_key[:8],
            )
            return

        full_text = " ".join(s.get("text", "") for s in segments).strip()

        from datetime import datetime

        from server.api.routes.notebook import save_to_notebook_sync

        conv_wall = datetime.fromtimestamp(
            time.time() - (time.monotonic() - session.conv_start_monotonic)
        )

        recording_id = await asyncio.to_thread(
            save_to_notebook_sync,
            audio_source=conv_audio,
            sample_rate=sample_rate,
            filename=f"conversation_{conv_wall.strftime('%Y%m%d_%H%M%S')}.wav",
            transcription_text=full_text,
            duration_seconds=conv_dur,
            segments=segments,
            language=session.language,
            transcription_backend=None,
            title=None,
            recorded_at=conv_wall,
        )

        logger.info(
            "save_conversation[%s]: saved to notebook (recording_id=%s, dur=%.1fs)",
            session.session_key[:8],
            recording_id,
            conv_dur,
        )

    except Exception as exc:
        logger.error(
            "save_conversation[%s]: failed — %s",
            session.session_key[:8],
            exc,
            exc_info=True,
        )


# ---------------------------------------------------------------------------
# Segment flush helper
# ---------------------------------------------------------------------------


async def _flush_pending(
    session: WsAudioSession,
    pending_chunks: list[np.ndarray],
    sample_rate: int,
    websocket: WebSocket,
) -> None:
    """
    Transcribe *pending_chunks* using a context window, send segments to the
    client immediately, then append the pending audio to conv_chunks.

    On transcription failure the audio is still saved to conv_chunks so the
    full-conversation reprocess can recover it.
    """
    if not pending_chunks:
        return

    pending = np.concatenate(pending_chunks)
    pending_dur = len(pending) / sample_rate

    if pending_dur < session.min_segment_s:
        logger.debug(
            "flush_pending[%s]: pending too short (%.2fs < %.2fs), appending without transcription",
            session.session_key[:8],
            pending_dur,
            session.min_segment_s,
        )
        session.conv_chunks.extend(pending_chunks)
        return

    # Skip silent/noise-only audio — Whisper hallucinate-loops indefinitely on silence
    rms = float(np.sqrt(np.mean(pending ** 2)))
    _SILENCE_RMS = 0.005  # ~-46 dBFS; well below any speech
    if rms < _SILENCE_RMS:
        logger.debug(
            "flush_pending[%s]: skipping silent audio (rms=%.5f < %.5f), appending without transcription",
            session.session_key[:8],
            rms,
            _SILENCE_RMS,
        )
        session.conv_chunks.extend(pending_chunks)
        return

    logger.info(
        "flush_pending[%s]: flushing %.1fs pending (conv=%.1fs, rms=%.4f)",
        session.session_key[:8],
        pending_dur,
        session.conv_duration_s,
        rms,
    )

    try:
        segments = await _transcribe_with_context(session, pending, sample_rate)
    except Exception as exc:
        logger.error(
            "flush_pending[%s]: transcription error — %s",
            session.session_key[:8],
            exc,
            exc_info=True,
        )
        # Preserve audio for full-conversation reprocess
        session.conv_chunks.extend(pending_chunks)
        return

    # Append to conversation history AFTER transcription (so context is correct)
    session.conv_chunks.extend(pending_chunks)

    if segments:
        session.streamed_segments.extend(segments)
        _first_text = segments[0].get("text", "")[:60].strip()
        _last_text = segments[-1].get("text", "")[:60].strip() if len(segments) > 1 else ""
        logger.info(
            'flush_pending[%s]: sending %d segment(s)  %.1fs\u2013%.1fs  "%s"%s',
            session.session_key[:8],
            len(segments),
            segments[0].get("start", 0),
            segments[-1].get("end", 0),
            _first_text,
            f' \u2026 "{_last_text}"' if _last_text else "",
        )
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_json({"segments": segments, "is_partial": True})
    else:
        logger.debug(
            "flush_pending[%s]: no new segments in this flush", session.session_key[:8]
        )


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------


@router.websocket("/ws/stt")
@router.websocket("/ws/omi")  # backward-compat alias for Omi devices
async def ws_stt_endpoint(
    websocket: WebSocket,
    token: str = "",
    codec: str = "pcm",
    sample_rate: int = PCM_SAMPLE_RATE,
    language: str | None = None,
    save_to_notebook: bool = False,
) -> None:
    """
    Streaming STT WebSocket endpoint (compatible with Omi External Custom STT protocol).

    Accepts audio frames and delivers transcription segments progressively as
    speech pauses are detected.  Multiple connections within the same inactivity
    window that share the same API token are stitched into one conversation.

    When save_to_notebook=true, the complete conversation audio is
    re-transcribed at high quality and saved to the Audio Notebook when the
    conversation ends.

    Connect:
      ws://<host>:9786/ws/stt?token=<api-token>&codec=pcm&save_to_notebook=true
    """
    await websocket.accept()

    # --- Authentication ---
    client_host = websocket.client.host if websocket.client else None
    if not _is_authorized(token, client_host):
        await websocket.send_json(
            {"error": "Unauthorized", "message": "Invalid or missing API token"}
        )
        await websocket.close(code=4401)
        return

    # --- Load config and check enabled flag ---
    ws_cfg = _load_ws_cfg()
    if not ws_cfg.get("enabled", True):
        await websocket.send_json(
            {"error": "Service Unavailable", "message": "WebSocket STT endpoint is disabled"}
        )
        await websocket.close(code=4503)
        return

    # --- Verify a transcription model is loaded ---
    try:
        from server.core.model_manager import get_model_manager

        mm = get_model_manager()
        status = mm.get_status()
        if not status.get("transcription", {}).get("loaded", False):
            await websocket.send_json(
                {
                    "error": "Service Unavailable",
                    "message": "Transcription model is not loaded",
                }
            )
            await websocket.close(code=4503)
            return
    except Exception as exc:
        logger.error("ws_stt: failed to access model manager: %s", exc)
        await websocket.send_json({"error": "Internal Error", "message": str(exc)})
        await websocket.close(code=4500)
        return

    # --- Build audio decoder ---
    opus_decoder: Any | None = None
    codec_lower = codec.strip().lower()
    if codec_lower == "opus":
        try:
            opus_decoder = _make_opus_decoder(sample_rate)
        except ImportError as exc:
            await websocket.send_json({"error": "Configuration Error", "message": str(exc)})
            await websocket.close(code=4500)
            return
    elif codec_lower != "pcm":
        await websocket.send_json(
            {
                "error": "Bad Request",
                "message": f"Unsupported codec '{codec}'. Use 'opus' or 'pcm'.",
            }
        )
        await websocket.close(code=4400)
        return

    # --- Derive config values ---
    inactivity_s = float(ws_cfg.get("inactivity_timeout_s", _DEFAULT_INACTIVITY_S))
    segment_silence_s = float(ws_cfg.get("segment_silence_s", _DEFAULT_SEGMENT_SILENCE_S))
    min_segment_s = float(ws_cfg.get("min_segment_s", _DEFAULT_MIN_SEGMENT_S))
    context_window_s = float(ws_cfg.get("context_window_s", _DEFAULT_CONTEXT_WINDOW_S))
    max_segment_s = float(ws_cfg.get("max_segment_s", _DEFAULT_MAX_SEGMENT_S))
    max_conv_s = float(ws_cfg.get("max_conversation_s", _DEFAULT_MAX_CONV_S))

    # --- Derive or resume session ---
    effective_token = token.strip() if token.strip() else f"_local_{client_host}"
    skey = _session_key(effective_token)

    now_mono = time.monotonic()
    existing = _ws_sessions.get(skey)

    if (
        existing is not None
        and (now_mono - existing.last_audio_monotonic) < inactivity_s
    ):
        # Stitch: continue existing conversation
        session = existing
        logger.info(
            "ws_stt[%s]: stitching to existing conversation "
            "(gap=%.1fs, conv_dur=%.1fs, segments=%d)",
            skey[:8],
            now_mono - session.last_audio_monotonic,
            session.conv_duration_s,
            len(session.streamed_segments),
        )
        # Cancel any prematurely-scheduled save task
        if session.save_task and not session.save_task.done():
            session.save_task.cancel()
            session.save_task = None
            logger.debug("ws_stt[%s]: cancelled premature save task", skey[:8])
    else:
        # New conversation
        if existing is not None:
            logger.info(
                "ws_stt[%s]: starting new conversation "
                "(previous ended %.1fs ago, prev_dur=%.1fs)",
                skey[:8],
                now_mono - existing.last_audio_monotonic,
                existing.conv_duration_s,
            )
        session = WsAudioSession(session_key=skey)
        session.inactivity_s = inactivity_s
        session.segment_silence_s = segment_silence_s
        session.min_segment_s = min_segment_s
        session.context_window_s = context_window_s
        session.max_segment_s = max_segment_s
        session.max_conv_s = max_conv_s
        session.save_to_notebook = save_to_notebook
        session.language = language
        _ws_sessions[skey] = session
        logger.info(
            "ws_stt[%s]: new conversation started "
            "(save_to_notebook=%s, language=%s, inactivity=%.0fs, segment_silence=%.1fs)",
            skey[:8],
            save_to_notebook,
            language or "auto",
            inactivity_s,
            segment_silence_s,
        )

    # --- Main receive loop ---
    pending_chunks: list[np.ndarray] = []
    pending_frames: int = 0
    last_audio_time = time.monotonic()
    close_requested = False
    conversation_ended = False

    try:
        while True:
            # Hard conversation length cap
            current_conv_dur = session.conv_duration_s + pending_frames / sample_rate
            if current_conv_dur >= max_conv_s:
                logger.warning(
                    "ws_stt[%s]: max conversation duration reached (%.0fs), closing",
                    skey[:8],
                    max_conv_s,
                )
                break

            # Wait for next frame; timeout fires after segment_silence_s of silence
            try:
                message = await asyncio.wait_for(
                    websocket.receive(), timeout=segment_silence_s
                )
            except TimeoutError:
                silence_dur = time.monotonic() - last_audio_time
                logger.info(
                    "ws_stt[%s]: receive timeout (silence=%.1fs, pending=%.1fs)",
                    skey[:8],
                    silence_dur,
                    pending_frames / sample_rate,
                )

                # Flush pending segment (if any)
                if pending_chunks:
                    await _flush_pending(session, pending_chunks, sample_rate, websocket)
                    pending_chunks = []
                    pending_frames = 0

                # Inactivity timeout → end of conversation
                if silence_dur >= inactivity_s:
                    logger.info(
                        "ws_stt[%s]: inactivity timeout (%.0fs), conversation ended",
                        skey[:8],
                        inactivity_s,
                    )
                    conversation_ended = True
                    break

                continue

            # Disconnected
            if message.get("type") == "websocket.disconnect":
                logger.debug("ws_stt[%s]: client disconnected", skey[:8])
                break

            # JSON control message
            if "text" in message:
                try:
                    msg_data = json.loads(message["text"])
                except json.JSONDecodeError:
                    logger.warning("ws_stt[%s]: ignoring non-JSON text frame", skey[:8])
                    continue

                msg_type = msg_data.get("type", "")
                if msg_type == "CloseStream":
                    logger.info("ws_stt[%s]: received CloseStream", skey[:8])
                    close_requested = True
                    break
                else:
                    logger.debug(
                        "ws_stt[%s]: ignoring unknown JSON type '%s'", skey[:8], msg_type
                    )
                continue

            # Binary audio frame
            if "bytes" in message:
                raw: bytes = message["bytes"]
                if not raw:
                    continue

                try:
                    if codec_lower == "opus":
                        chunk = _decode_opus_frame(opus_decoder, raw)
                    else:
                        chunk = _decode_pcm(raw, sample_rate)
                except Exception as exc:
                    logger.warning("ws_stt[%s]: audio decode error: %s", skey[:8], exc)
                    continue

                pending_chunks.append(chunk)
                pending_frames += len(chunk)
                last_audio_time = time.monotonic()
                session.last_audio_monotonic = last_audio_time

                # Force-flush if pending buffer is full
                if pending_frames / sample_rate >= max_segment_s:
                    logger.info(
                        "ws_stt[%s]: pending buffer full (%.1fs), force flushing",
                        skey[:8],
                        pending_frames / sample_rate,
                    )
                    await _flush_pending(session, pending_chunks, sample_rate, websocket)
                    pending_chunks = []
                    pending_frames = 0

        # --- Post-loop: flush any remaining audio ---
        if pending_chunks:
            logger.info(
                "ws_stt[%s]: flushing final %.1fs of pending audio",
                skey[:8],
                pending_frames / sample_rate,
            )
            await _flush_pending(session, pending_chunks, sample_rate, websocket)

        # Send a final summary on graceful close
        if close_requested and websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_json(
                {"segments": session.streamed_segments, "is_partial": False}
            )

        # Schedule notebook save if the conversation is ending
        should_save = (
            session.save_to_notebook
            and session.conv_chunks
            and (conversation_ended or close_requested)
        )
        if should_save:
            logger.info(
                "ws_stt[%s]: scheduling conversation notebook save (%.1fs of audio)",
                skey[:8],
                session.conv_duration_s,
            )
            session.save_task = asyncio.create_task(
                _save_conversation_to_notebook(session, sample_rate),
                name=f"notebook_save_{skey[:8]}",
            )

    except WebSocketDisconnect:
        logger.debug("ws_stt[%s]: WebSocketDisconnect", skey[:8])
        # Preserve buffered audio for potential stitch on reconnect
        if pending_chunks:
            session.conv_chunks.extend(pending_chunks)
            session.last_audio_monotonic = time.monotonic()

    except Exception as exc:
        logger.error("ws_stt[%s]: unexpected error: %s", skey[:8], exc, exc_info=True)
        if websocket.client_state == WebSocketState.CONNECTED:
            try:
                await websocket.send_json({"error": "Internal Error", "message": str(exc)})
            except Exception:
                pass

    finally:
        if websocket.client_state == WebSocketState.CONNECTED:
            try:
                await websocket.close()
            except Exception:
                pass

        logger.info(
            "ws_stt[%s]: connection closed (conv_dur=%.1fs, total_segments=%d)",
            skey[:8],
            session.conv_duration_s,
            len(session.streamed_segments),
        )
