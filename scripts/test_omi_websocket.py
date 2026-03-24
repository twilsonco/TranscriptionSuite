#!/usr/bin/env python3
"""
Test script for the /ws/omi WebSocket endpoint.

Reads WAV files, sends PCM audio to the OMI STT endpoint, and saves the
JSON segment output to samples/output/<stem>_<timestamp>.json.

Usage:
    # Process all WAV/FLAC files in samples/input/
    python scripts/test_omi_websocket.py

    # Process specific files
    python scripts/test_omi_websocket.py samples/input/1min_test.wav

    # Different server or language
    python scripts/test_omi_websocket.py --server ws://10.0.1.90:9786
    python scripts/test_omi_websocket.py --language en

Requirements (all in server/backend/.venv):
    soundfile, numpy, scipy, websockets

The admin API token is read automatically from the token store file.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
import websockets
from scipy.signal import resample as sp_resample

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).parent.parent
SAMPLES_INPUT = REPO_ROOT / "samples" / "input"
SAMPLES_OUTPUT = REPO_ROOT / "samples" / "output"

TOKEN_FILE = (
    Path.home()
    / "Library"
    / "Application Support"
    / "TranscriptionSuite"
    / "data"
    / "tokens"
    / "tokens.json"
)

TARGET_SAMPLE_RATE = 16_000
# 960 samples = 60 ms at 16 kHz (standard Opus frame size, fine for PCM too)
CHUNK_FRAMES = 960
SUPPORTED_EXTENSIONS = {".wav", ".flac", ".ogg"}

# ---------------------------------------------------------------------------
# Token
# ---------------------------------------------------------------------------


def get_admin_token() -> str:
    """Read the first non-revoked admin token from the local token store."""
    if not TOKEN_FILE.exists():
        raise FileNotFoundError(
            f"Token store not found: {TOKEN_FILE}\n"
            "Is the server running and the DATA_DIR set correctly?"
        )
    data = json.loads(TOKEN_FILE.read_text())
    for tok in data.get("tokens", []):
        if tok.get("is_admin") and not tok.get("is_revoked"):
            return tok["token"]
    raise RuntimeError(f"No active admin token found in {TOKEN_FILE}")


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------


def load_as_pcm_int16(path: Path) -> tuple[bytes, float]:
    """
    Load an audio file, resample to 16 kHz mono, and encode as Int16 LE PCM.

    Returns (pcm_bytes, duration_seconds).
    """
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)

    # Mix to mono
    audio = audio.mean(axis=1)

    # Resample to 16 kHz if needed
    if sr != TARGET_SAMPLE_RATE:
        target_len = int(round(len(audio) * TARGET_SAMPLE_RATE / sr))
        audio = sp_resample(audio, target_len).astype(np.float32)
        sr = TARGET_SAMPLE_RATE

    duration = len(audio) / sr

    # Clip and convert to Int16
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767).astype(np.int16)
    return pcm.tobytes(), duration


# ---------------------------------------------------------------------------
# WebSocket send / receive
# ---------------------------------------------------------------------------


async def process_file(
    path: Path,
    token: str,
    server_url: str,
    language: str | None,
    receive_timeout: float,
    speaker_labels: bool = False,
) -> list[dict]:
    """
    Stream audio to /ws/omi as PCM, collect all response messages, and return them.

    The endpoint delivers partial segments as speech pauses are detected, then a
    final message with is_partial=False after CloseStream.  Both the sender and
    receiver run concurrently so partial messages are not silently dropped.
    """
    print(f"    Loading audio …")
    pcm_bytes, duration = load_as_pcm_int16(path)
    print(f"    Duration    : {duration:.1f}s")
    print(f"    PCM payload : {len(pcm_bytes):,} bytes  ({len(pcm_bytes) // 2:,} samples)")

    uri = f"{server_url}/ws/stt?token={token}&codec=pcm&sample_rate={TARGET_SAMPLE_RATE}"
    if language:
        uri += f"&language={language}"
    if speaker_labels:
        uri += "&speaker_labels=true"

    print(f"    Connecting  : {server_url}/ws/stt")

    responses: list[dict] = []
    chunk_bytes = CHUNK_FRAMES * 2  # 2 bytes per Int16 sample

    async with websockets.connect(
        uri, max_size=2**24, open_timeout=15, ping_interval=None
    ) as ws:
        # Sender: stream all chunks then signal end
        async def _send() -> None:
            n_chunks = 0
            for offset in range(0, len(pcm_bytes), chunk_bytes):
                await ws.send(pcm_bytes[offset : offset + chunk_bytes])
                n_chunks += 1
                await asyncio.sleep(0.001)  # yield to event loop
            print(f"    Audio sent  : {n_chunks} chunk(s)  → sending CloseStream …")
            await ws.send(json.dumps({"type": "CloseStream"}))

        # Receiver: collect all messages until server closes the connection
        async def _recv() -> None:
            print(f"    Waiting for transcription (timeout={receive_timeout:.0f}s) …")
            while True:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=receive_timeout)
                    payload = json.loads(raw)
                    responses.append(payload)
                    segs = payload.get("segments", [])
                    is_partial = payload.get("is_partial", True)
                    tag = "partial" if is_partial else "final"
                    print(
                        f"    ← [{tag}] {len(segs)} segment(s)"
                        + (f': "{segs[0].get("text", "")[:60]}"' if segs else "")
                    )
                except websockets.exceptions.ConnectionClosedOK:
                    break
                except websockets.exceptions.ConnectionClosedError as exc:
                    print(f"    ← connection closed with error: {exc}")
                    break
                except asyncio.TimeoutError:
                    print(f"    ← receive timeout after {receive_timeout:.0f}s")
                    break

        await asyncio.gather(_send(), _recv())

    return responses


async def run(files: list[Path], server_url: str, language: str | None, speaker_labels: bool = False) -> None:
    SAMPLES_OUTPUT.mkdir(parents=True, exist_ok=True)

    token = get_admin_token()
    print(f"Token : {token[:8]}…{token[-6:]}")
    print(f"Server: {server_url}")
    print(f"Files : {[p.name for p in files]}")
    if speaker_labels:
        print("Speaker labels: enabled (speaker_labels=true — diarization runs after CloseStream)")
    print()

    passed = 0
    failed = 0

    for path in files:
        if not path.exists():
            print(f"[SKIP] Not found: {path}")
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            print(f"[SKIP] Unsupported extension '{path.suffix}' — use WAV/FLAC: {path.name}")
            continue

        print(f"[{path.name}]")

        # Dynamic receive timeout: 10× real-time, minimum 120 s, max 900 s
        # When speaker_labels=True, allow extra time for diarization (up to 30min per hour of audio)
        try:
            audio_info = sf.info(str(path))
            estimated_duration = audio_info.duration
        except Exception:
            estimated_duration = 600.0

        receive_timeout = max(120.0, min(estimated_duration * 10, 900.0))
        if speaker_labels:
            receive_timeout = max(receive_timeout, estimated_duration * 30, 600.0)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_name = f"{path.stem}_{ts}.json"
        out_path = SAMPLES_OUTPUT / out_name

        try:
            responses = await process_file(path, token, server_url, language, receive_timeout, speaker_labels=speaker_labels)
        except Exception as exc:
            print(f"    ✗  Error: {exc}\n")
            failed += 1
            continue

        if not responses:
            print("    ✗  No responses received\n")
            failed += 1
            continue

        # Prefer the final (is_partial=False) message; fall back to merging partials
        result: dict = {}
        for r in reversed(responses):
            if not r.get("is_partial", True):
                result = r
                break
        if not result:
            merged_segs: list[dict] = []
            for r in responses:
                merged_segs.extend(r.get("segments", []))
            result = {"segments": merged_segs, "is_partial": True}

        error = next((r.get("error") for r in responses if r.get("error")), None)
        if error:
            print(f"    ✗  Server error: {error}\n")
            failed += 1
            continue

        segments: list[dict] = result.get("segments", [])
        print(f"    ✓  {len(segments)} segment(s) [{len(responses)} message(s)]")
        if segments:
            first_text = segments[0].get("text", "").strip()
            print(f'    ↳ First : "{first_text[:90]}"')
            has_speaker = any("speaker" in s for s in segments)
            has_diarization_response = any(r.get("diarization_complete") for r in responses)
            if has_speaker:
                speaker_set = {s.get("speaker") for s in segments if s.get("speaker")}
                print(f"    ↳ Diarization: yes ✓ ({len(speaker_set)} speaker(s): {', '.join(sorted(speaker_set))})")
            elif has_diarization_response:
                print(f"    ↳ Diarization: ran but no speaker labels produced (check logs)")
            else:
                print(f"    ↳ Diarization: no (use --speaker-labels to request, or --save-to-notebook for async)")

        output = {
            "source_file": path.name,
            "processed_at": ts,
            "server": server_url,
            "language_hint": language,
            "messages_received": len(responses),
            "segment_count": len(segments),
            "responses": responses,
        }
        out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
        print(f"    → Saved: {out_path.relative_to(REPO_ROOT)}\n")
        passed += 1

    print(f"Done. {passed} passed, {failed} failed.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test the /ws/omi OMI External STT WebSocket endpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/test_omi_websocket.py\n"
            "  python scripts/test_omi_websocket.py samples/input/1min_test.wav\n"
            "  python scripts/test_omi_websocket.py --speaker-labels samples/input/1min_test.wav\n"
            "  python scripts/test_omi_websocket.py --dir /path/to/audio/files\n"
            "  python scripts/test_omi_websocket.py --server ws://10.0.1.90:9786 --language en\n"
        ),
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Audio file(s) to process. Default: all WAV/FLAC in --dir (or samples/input/)",
    )
    parser.add_argument(
        "--server",
        default="ws://localhost:9786",
        help="WebSocket server base URL (default: ws://localhost:9786)",
    )
    parser.add_argument(
        "--language",
        default=None,
        metavar="LANG",
        help="BCP-47 language code, e.g. 'en'. Default: auto-detect.",
    )
    parser.add_argument(
        "--speaker-labels",
        action="store_true",
        help="Request speaker diarization after CloseStream (speaker_labels=true). "
             "Diarization runs on full conversation audio and the server sends a "
             "second final message with speaker-labelled segments. Requires HF_TOKEN.",
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=None,
        metavar="DIRECTORY",
        help="Directory of audio files to process (default: samples/input/). "
             "Ignored if file paths are given as positional arguments.",
    )
    args = parser.parse_args()

    if args.files:
        paths = [Path(f) for f in args.files]
    else:
        input_dir = args.dir if args.dir else SAMPLES_INPUT
        paths = sorted(
            p for p in input_dir.iterdir()
            if p.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if not paths:
            print(f"No WAV/FLAC files found in {input_dir}")
            print("Copy sample files there first, or pass file paths as arguments.")
            sys.exit(1)

    asyncio.run(run(paths, args.server, args.language, speaker_labels=args.speaker_labels))


if __name__ == "__main__":
    main()
