"""
WebSocket endpoint for Omi external custom STT integration.

Implements the Omi External Custom STT Service protocol:
https://docs.omi.me/developer/backend/custom-stt

Endpoint: /ws/omi
Query params:
  token  - API token for authentication (required)
  codec  - Audio codec: "opus" (default) or "pcm"
  sample_rate - Input sample rate in Hz (default: 16000)
  language    - BCP-47 language code, e.g. "en" (optional, auto-detect if omitted)

Protocol:
  Client → Server: binary audio frames  OR  JSON {"type": "CloseStream"}
  Server → Client: {"segments": [{"text", "speaker", "start", "end"}, ...]}

The endpoint buffers all inbound audio, transcribes the full recording on
CloseStream (or inactivity timeout), and responds with a single segments
payload before closing.

Diarization is performed automatically when the loaded backend supports it
(i.e. WhisperX with a valid HF_TOKEN).  If diarization is unavailable the
speaker field is simply omitted from each segment.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from server.core.token_store import get_token_store
from server.api.routes.utils import is_local_auth_bypass_host

# Match local-mode trust logic used by the rest of the server (see main.py)
_TLS_MODE: bool = os.environ.get("TLS_ENABLED", "false").lower() == "true"

logger = logging.getLogger(__name__)

router = APIRouter()

# Default inactivity timeout (matches Omi's 90-second idle close)
DEFAULT_TIMEOUT_S: float = 90.0
# PCM frame format sent over the wire by Omi
PCM_SAMPLE_RATE = 16000
PCM_DTYPE = np.int16


# ---------------------------------------------------------------------------
# Audio decoding helpers
# ---------------------------------------------------------------------------


def _decode_pcm(data: bytes, sample_rate: int = PCM_SAMPLE_RATE) -> np.ndarray:
    """Interpret raw bytes as signed 16-bit little-endian PCM and return float32."""
    pcm = np.frombuffer(data, dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0


def _make_opus_decoder(sample_rate: int = PCM_SAMPLE_RATE):
    """Create an opuslib Decoder, or raise ImportError with a clear message."""
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
    # opuslib.Decoder.decode returns signed 16-bit PCM bytes
    pcm_bytes = decoder.decode(data, frame_size=960)  # 60 ms at 16 kHz
    pcm = np.frombuffer(pcm_bytes, dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------


def _is_authorized(token: str | None, client_host: str | None) -> bool:
    """
    Return True if the request is authorized.

    In non-TLS (local) mode, loopback connections are trusted without a token,
    consistent with how ``require_admin`` works for HTTP routes.
    In TLS mode, a valid token is always required.
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
# Transcription helper
# ---------------------------------------------------------------------------


async def _transcribe_audio(
    audio: np.ndarray,
    sample_rate: int,
    language: str | None,
) -> list[dict[str, Any]]:
    """
    Transcribe *audio* using the loaded model manager engine.

    Tries diarization first (WhisperX path); falls back to plain transcription
    if diarization is unavailable or raises.

    Returns a list of OMI-format segment dicts.
    """
    from server.core.model_manager import get_model_manager

    model_manager = get_model_manager()
    engine = model_manager.transcription_engine

    # Write audio to a temporary WAV file (transcribe_file handles resampling)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        sf.write(str(tmp_path), audio, sample_rate)

    try:
        # Attempt diarization if backend supports it
        backend = getattr(engine, "_backend", None)
        diarization_result = None

        if backend is not None and hasattr(backend, "transcribe_with_diarization"):
            hf_token: str | None = None
            try:
                hf_token = model_manager.hf_token
            except AttributeError:
                pass

            if hf_token:
                try:
                    diarization_result = await asyncio.to_thread(
                        backend.transcribe_with_diarization,
                        audio,
                        audio_sample_rate=sample_rate,
                        language=language,
                        hf_token=hf_token,
                    )
                except Exception as e:
                    logger.warning(
                        "Diarization failed, falling back to plain transcription: %s", e
                    )
                    diarization_result = None

        if diarization_result is not None:
            # Map DiarizedTranscriptionResult → OMI segments
            segments: list[dict[str, Any]] = []
            for seg in diarization_result.segments:
                entry: dict[str, Any] = {
                    "text": seg.get("text", "").strip(),
                    "start": round(float(seg.get("start", 0.0)), 3),
                    "end": round(float(seg.get("end", 0.0)), 3),
                }
                speaker = seg.get("speaker")
                if speaker:
                    entry["speaker"] = speaker
                if entry["text"]:
                    segments.append(entry)
            return segments

        # Plain transcription fallback
        result = await asyncio.to_thread(
            engine.transcribe_file,
            str(tmp_path),
            language=language,
            word_timestamps=False,
        )

        segments = []
        for seg in result.segments:
            entry = {
                "text": seg.get("text", "").strip(),
                "start": round(float(seg.get("start", 0.0)), 3),
                "end": round(float(seg.get("end", 0.0)), 3),
            }
            if entry["text"]:
                segments.append(entry)
        return segments

    finally:
        try:
            tmp_path.unlink()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# WebSocket handler
# ---------------------------------------------------------------------------


@router.websocket("/ws/omi")
async def omi_stt_endpoint(
    websocket: WebSocket,
    token: str = "",
    codec: str = "opus",
    sample_rate: int = PCM_SAMPLE_RATE,
    language: str | None = None,
) -> None:
    """
    Omi External Custom STT WebSocket endpoint.

    Connect with:
      ws://<host>:9786/ws/omi?token=<api_token>&codec=opus
    or
      ws://<host>:9786/ws/omi?token=<api_token>&codec=pcm
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

    # Check omi_stt.enabled in config
    try:
        from server.config import get_config

        cfg = get_config()
        omi_cfg = cfg.get("omi_stt", default={})
        if not omi_cfg.get("enabled", True):
            await websocket.send_json(
                {"error": "Service Unavailable", "message": "OMI STT endpoint is disabled"}
            )
            await websocket.close(code=4503)
            return
    except Exception:
        pass  # If config is unreadable, default to enabled

    # Check that the model manager has a loaded model
    try:
        from server.core.model_manager import get_model_manager

        model_manager = get_model_manager()
        status = model_manager.get_status()
        if not status.get("transcription", {}).get("loaded", False):
            await websocket.send_json(
                {"error": "Service Unavailable", "message": "Transcription model is not loaded"}
            )
            await websocket.close(code=4503)
            return
    except Exception as e:
        logger.error("Failed to access model manager: %s", e)
        await websocket.send_json({"error": "Internal Error", "message": str(e)})
        await websocket.close(code=4500)
        return

    # Build audio decoder
    opus_decoder: Any | None = None
    codec_lower = codec.strip().lower()
    if codec_lower == "opus":
        try:
            opus_decoder = _make_opus_decoder(sample_rate)
        except ImportError as e:
            await websocket.send_json({"error": "Configuration Error", "message": str(e)})
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

    # Determine inactivity timeout from config
    timeout_s = DEFAULT_TIMEOUT_S
    try:
        from server.config import get_config

        cfg = get_config()
        omi_cfg = cfg.get("omi_stt", default={})
        timeout_s = float(omi_cfg.get("inactivity_timeout_s", DEFAULT_TIMEOUT_S))
    except Exception:
        pass

    logger.info(
        "OMI STT session started (codec=%s, sample_rate=%d, language=%s, timeout=%.0fs)",
        codec_lower,
        sample_rate,
        language or "auto",
        timeout_s,
    )

    audio_chunks: list[np.ndarray] = []
    total_frames: int = 0

    try:
        while True:
            # Wait for a message with inactivity timeout
            try:
                message = await asyncio.wait_for(websocket.receive(), timeout=timeout_s)
            except TimeoutError:
                logger.info("OMI STT: inactivity timeout (%.0fs), closing session", timeout_s)
                break

            if message.get("type") == "websocket.disconnect":
                logger.debug("OMI STT: client disconnected")
                break

            if "text" in message:
                # JSON control message
                try:
                    msg_data = json.loads(message["text"])
                except json.JSONDecodeError:
                    logger.warning("OMI STT: ignoring invalid JSON frame")
                    continue

                msg_type = msg_data.get("type", "")
                if msg_type == "CloseStream":
                    logger.debug("OMI STT: received CloseStream")
                    break
                else:
                    logger.debug("OMI STT: ignoring unknown JSON message type '%s'", msg_type)

            elif "bytes" in message:
                raw: bytes = message["bytes"]
                if not raw:
                    continue

                try:
                    if codec_lower == "opus":
                        chunk = _decode_opus_frame(opus_decoder, raw)
                    else:
                        chunk = _decode_pcm(raw, sample_rate)
                    audio_chunks.append(chunk)
                    total_frames += len(chunk)
                except Exception as e:
                    logger.warning("OMI STT: audio decode error: %s", e)
                    continue

        # --- Transcribe ---
        if not audio_chunks:
            logger.info("OMI STT: no audio received, sending empty segments")
            await websocket.send_json({"segments": []})
        else:
            duration_s = total_frames / sample_rate
            logger.info(
                "OMI STT: transcribing %.1fs of audio (%d frames)", duration_s, total_frames
            )
            combined = np.concatenate(audio_chunks)
            try:
                segments = await _transcribe_audio(combined, sample_rate, language)
                await websocket.send_json({"segments": segments})
                logger.info("OMI STT: sent %d segment(s)", len(segments))
            except Exception as e:
                logger.error("OMI STT: transcription error: %s", e, exc_info=True)
                await websocket.send_json({"error": "Transcription failed", "message": str(e)})

    except WebSocketDisconnect:
        logger.debug("OMI STT: WebSocketDisconnect")

    except Exception as e:
        logger.error("OMI STT: unexpected error: %s", e, exc_info=True)
        if websocket.client_state == WebSocketState.CONNECTED:
            try:
                await websocket.send_json({"error": "Internal Error", "message": str(e)})
            except Exception:
                pass

    finally:
        if websocket.client_state == WebSocketState.CONNECTED:
            try:
                await websocket.close()
            except Exception:
                pass
        logger.info("OMI STT session ended")
