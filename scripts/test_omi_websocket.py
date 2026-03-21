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
) -> dict:
    """
    Stream audio to /ws/omi as PCM and return the parsed JSON response.
    """
    print(f"    Loading audio …")
    pcm_bytes, duration = load_as_pcm_int16(path)
    print(f"    Duration    : {duration:.1f}s")
    print(f"    PCM payload : {len(pcm_bytes):,} bytes  ({len(pcm_bytes) // 2:,} samples)")

    uri = f"{server_url}/ws/omi?token={token}&codec=pcm&sample_rate={TARGET_SAMPLE_RATE}"
    if language:
        uri += f"&language={language}"

    print(f"    Connecting  : {server_url}/ws/omi")

    async with websockets.connect(uri, max_size=2**24, open_timeout=15) as ws:
        # Send PCM in fixed-size chunks
        chunk_bytes = CHUNK_FRAMES * 2  # 2 bytes per Int16 sample
        n_chunks = 0
        for offset in range(0, len(pcm_bytes), chunk_bytes):
            await ws.send(pcm_bytes[offset : offset + chunk_bytes])
            n_chunks += 1

        print(f"    Audio sent  : {n_chunks} chunk(s)  → sending CloseStream …")
        await ws.send(json.dumps({"type": "CloseStream"}))

        print(f"    Waiting for transcription (timeout={receive_timeout:.0f}s) …")
        raw = await asyncio.wait_for(ws.recv(), timeout=receive_timeout)

    result = json.loads(raw)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def run(files: list[Path], server_url: str, language: str | None) -> None:
    SAMPLES_OUTPUT.mkdir(parents=True, exist_ok=True)

    token = get_admin_token()
    print(f"Token : {token[:8]}…{token[-6:]}")
    print(f"Server: {server_url}")
    print(f"Files : {[p.name for p in files]}")
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
        try:
            audio_info = sf.info(str(path))
            estimated_duration = audio_info.duration
        except Exception:
            estimated_duration = 600.0

        receive_timeout = max(120.0, min(estimated_duration * 10, 900.0))

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_name = f"{path.stem}_{ts}.json"
        out_path = SAMPLES_OUTPUT / out_name

        try:
            result = await process_file(path, token, server_url, language, receive_timeout)
        except Exception as exc:
            print(f"    ✗  Error: {exc}\n")
            failed += 1
            continue

        segments: list[dict] = result.get("segments", [])
        error = result.get("error")

        if error:
            print(f"    ✗  Server error: {error} — {result.get('message', '')}\n")
            failed += 1
            continue

        print(f"    ✓  {len(segments)} segment(s) returned")
        if segments:
            first_text = segments[0].get("text", "").strip()
            print(f'    ↳ First : "{first_text[:90]}"')
            has_speaker = any("speaker" in s for s in segments)
            print(f"    ↳ Diarization: {'yes' if has_speaker else 'no (token missing or unsupported backend)'}")

        output = {
            "source_file": path.name,
            "processed_at": ts,
            "server": server_url,
            "language_hint": language,
            "segment_count": len(segments),
            "result": result,
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
            "  python scripts/test_omi_websocket.py --server ws://10.0.1.90:9786 --language en\n"
        ),
    )
    parser.add_argument(
        "files",
        nargs="*",
        help=f"Audio file(s) to process. Default: all WAV/FLAC in {SAMPLES_INPUT}",
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
    args = parser.parse_args()

    if args.files:
        paths = [Path(f) for f in args.files]
    else:
        paths = sorted(
            p for p in SAMPLES_INPUT.iterdir()
            if p.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if not paths:
            print(f"No WAV/FLAC files found in {SAMPLES_INPUT}")
            print("Copy sample files there first, or pass file paths as arguments.")
            sys.exit(1)

    asyncio.run(run(paths, args.server, args.language))


if __name__ == "__main__":
    main()
