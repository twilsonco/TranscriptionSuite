#!/usr/bin/env python3
"""
Test script for the /ws/stt WebSocket STT endpoint (canonical path).
Also works with the /ws/omi backward-compat alias.

Streams a WAV/FLAC file as PCM audio and collects all progressive segment
responses, then prints the assembled transcript.  The endpoint delivers
partial results as speech pauses are detected, so multiple JSON messages
will be received before the connection closes.

Usage:
    # Process all WAV/FLAC files in samples/input/
    python scripts/test_ws_stt.py

    # Process a specific file
    python scripts/test_ws_stt.py samples/input/1min_test.wav

    # Different server or language
    python scripts/test_ws_stt.py --server ws://10.0.1.90:9786 --language en

    # Save final transcript to Audio Notebook
    python scripts/test_ws_stt.py --save-to-notebook samples/input/1min_test.wav

    # Stitch test: send the file as two sequential connections and verify
    # they are merged into one conversation
    python scripts/test_ws_stt.py --stitch-test samples/input/1min_test.wav

    # Control inter-chunk delay (default: 0.001s = fast sequential, like Omi stored audio)
    python scripts/test_ws_stt.py --chunk-delay 0.06 samples/input/1min_test.wav  # real-time
    python scripts/test_ws_stt.py --chunk-delay 0.0  samples/input/1min_test.wav  # max speed

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
CHUNK_FRAMES = 960  # 60 ms at 16 kHz
SUPPORTED_EXTENSIONS = {".wav", ".flac", ".ogg"}


# ---------------------------------------------------------------------------
# Token
# ---------------------------------------------------------------------------


def get_admin_token() -> str:
    """Read the first non-revoked admin token from the local token store."""
    if not TOKEN_FILE.exists():
        raise FileNotFoundError(
            f"Token store not found: {TOKEN_FILE}\n"
            "Is the server running and DATA_DIR set correctly?"
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
    Load *path*, resample to 16 kHz mono, encode as little-endian Int16 PCM.

    Returns (pcm_bytes, duration_seconds).
    """
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)

    if sr != TARGET_SAMPLE_RATE:
        target_len = int(round(len(audio) * TARGET_SAMPLE_RATE / sr))
        audio = sp_resample(audio, target_len).astype(np.float32)

    duration = len(audio) / TARGET_SAMPLE_RATE
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767).astype(np.int16)
    return pcm.tobytes(), duration


# ---------------------------------------------------------------------------
# WebSocket send / receive helper
# ---------------------------------------------------------------------------


async def stream_audio(
    pcm_bytes: bytes,
    token: str,
    server_url: str,
    language: str | None,
    save_to_notebook: bool,
    chunk_delay: float,
    graceful_close: bool = True,
) -> list[dict]:
    """
    Open a single WebSocket connection, stream *pcm_bytes* in chunks, and
    collect all response messages until the server closes the connection.

    Returns the list of all received JSON payloads.
    """
    uri = f"{server_url}/ws/stt?token={token}&codec=pcm&sample_rate={TARGET_SAMPLE_RATE}"
    if language:
        uri += f"&language={language}"
    if save_to_notebook:
        uri += "&save_to_notebook=true"

    print(f"    Connecting  : {server_url}/ws/stt")

    responses: list[dict] = []
    chunk_bytes = CHUNK_FRAMES * 2  # 2 bytes per Int16 sample

    async with websockets.connect(
        uri, max_size=2**24, open_timeout=15, ping_interval=None
    ) as ws:
        # Sender coroutine — streams all chunks then sends CloseStream
        async def _send() -> None:
            n_sent = 0
            for offset in range(0, len(pcm_bytes), chunk_bytes):
                await ws.send(pcm_bytes[offset : offset + chunk_bytes])
                n_sent += 1
                # Always yield to the event loop so _recv() can interleave;
                # chunk_delay=0.0 uses sleep(0) which is a bare yield.
                await asyncio.sleep(chunk_delay)
            print(f"    Audio sent  : {n_sent} chunk(s)")
            if graceful_close:
                await ws.send(json.dumps({"type": "CloseStream"}))
                print("    CloseStream sent")

        # Receiver coroutine — collects until connection closes
        async def _recv() -> None:
            while True:
                try:
                    raw = await ws.recv()
                    payload = json.loads(raw)
                    responses.append(payload)
                    segs = payload.get("segments", [])
                    is_partial = payload.get("is_partial", True)
                    tag = "partial" if is_partial else "final"
                    print(
                        f"    ← [{tag}] {len(segs)} segment(s)"
                        + (
                            f': "{segs[0].get("text", "")[:60]}"'
                            if segs
                            else ""
                        )
                    )
                except websockets.exceptions.ConnectionClosedOK:
                    break
                except websockets.exceptions.ConnectionClosedError as exc:
                    print(f"    ← connection closed with error: {exc}")
                    break

        await asyncio.gather(_send(), _recv())

    return responses


# ---------------------------------------------------------------------------
# Per-file test runner
# ---------------------------------------------------------------------------


def _all_segments(responses: list[dict]) -> list[dict]:
    """Return the de-duplicated final segment list from a response list.

    Prefers the final (is_partial=False) blob if present; otherwise merges
    all partial segments.
    """
    for r in reversed(responses):
        if not r.get("is_partial", True):
            return r.get("segments", [])
    merged: list[dict] = []
    for r in responses:
        merged.extend(r.get("segments", []))
    return merged


async def test_file(
    path: Path,
    token: str,
    server_url: str,
    language: str | None,
    save_to_notebook: bool,
    chunk_delay: float,
    stitch_test: bool,
) -> bool:
    """Run the test for a single audio file.  Returns True on pass."""
    if not path.exists():
        print(f"[SKIP] Not found: {path}")
        return True
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        print(f"[SKIP] Unsupported extension '{path.suffix}': {path.name}")
        return True

    print(f"[{path.name}]")
    pcm_bytes, duration = load_as_pcm_int16(path)
    print(f"    Duration    : {duration:.1f}s")
    print(f"    PCM payload : {len(pcm_bytes):,} bytes")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    SAMPLES_OUTPUT.mkdir(parents=True, exist_ok=True)

    try:
        if stitch_test:
            return await _run_stitch_test(
                path, pcm_bytes, duration, token, server_url,
                language, save_to_notebook, chunk_delay, ts,
            )
        else:
            return await _run_single_test(
                path, pcm_bytes, token, server_url,
                language, save_to_notebook, chunk_delay, ts,
            )
    except Exception as exc:
        print(f"    ✗  Error: {exc}\n")
        return False


async def _run_single_test(
    path: Path,
    pcm_bytes: bytes,
    token: str,
    server_url: str,
    language: str | None,
    save_to_notebook: bool,
    chunk_delay: float,
    ts: str,
) -> bool:
    responses = await stream_audio(
        pcm_bytes, token, server_url, language, save_to_notebook, chunk_delay,
    )
    if not responses:
        print("    ✗  No responses received\n")
        return False

    segments = _all_segments(responses)
    error = next((r.get("error") for r in responses if r.get("error")), None)
    if error:
        print(f"    ✗  Server error: {error}\n")
        return False

    print(f"    ✓  {len(segments)} total segment(s) [{len(responses)} message(s)]")
    if segments:
        print(f'    ↳ First: "{segments[0].get("text", "")[:90]}"')
        print(f'    ↳ Last : "{segments[-1].get("text", "")[:90]}"')
        has_speaker = any("speaker" in s for s in segments)
        print(f"    ↳ Diarization: {'yes ✓' if has_speaker else 'no (HF_TOKEN missing or unsupported backend)'}")
    if save_to_notebook:
        queued = any(r.get("notebook_save_queued") for r in responses)
        print(f"    ↳ Notebook save: {'queued (check logs)' if queued else 'will run after inactivity timeout'}")

    out_path = SAMPLES_OUTPUT / f"{path.stem}_{ts}.json"
    out_path.write_text(
        json.dumps(
            {
                "source_file": path.name,
                "processed_at": ts,
                "server": server_url,
                "language_hint": language,
                "messages_received": len(responses),
                "segment_count": len(segments),
                "responses": responses,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    print(f"    → Saved: {out_path.relative_to(REPO_ROOT)}\n")
    return True


async def _run_stitch_test(
    path: Path,
    pcm_bytes: bytes,
    duration: float,
    token: str,
    server_url: str,
    language: str | None,
    save_to_notebook: bool,
    chunk_delay: float,
    ts: str,
) -> bool:
    """
    Stitch test: send the first half as one connection, reconnect immediately
    and send the second half.  Verifies that timestamps are contiguous and that
    the conversation is treated as one session.
    """
    print("    [stitch-test] Sending first half …")
    half = len(pcm_bytes) // 2
    # Align to 2-byte (Int16) boundary
    half = (half // 2) * 2

    # First connection — no CloseStream (simulate abrupt disconnect)
    responses_a = await stream_audio(
        pcm_bytes[:half],
        token, server_url, language, save_to_notebook, chunk_delay,
        graceful_close=False,
    )
    print("    [stitch-test] First half done.  Reconnecting immediately …")

    # Second connection — should stitch if gap < inactivity_timeout_s
    responses_b = await stream_audio(
        pcm_bytes[half:],
        token, server_url, language, save_to_notebook, chunk_delay,
        graceful_close=True,
    )
    print("    [stitch-test] Second half done.")

    segs_a = _all_segments(responses_a)
    segs_b = _all_segments(responses_b)

    combined = segs_a + segs_b
    print(f"    ✓  Stitch test: {len(segs_a)} + {len(segs_b)} = {len(combined)} segments")

    if segs_a and segs_b:
        last_a_end = segs_a[-1].get("end", 0.0)
        first_b_start = segs_b[0].get("start", 0.0) if segs_b else None
        if first_b_start is not None:
            gap = first_b_start - last_a_end
            if gap < 0:
                print(
                    f"    ↳ Timestamps overlap? last_a.end={last_a_end:.1f}s, first_b.start={first_b_start:.1f}s"
                )
            else:
                print(
                    f"    ↳ Timestamp gap between halves: {gap:.1f}s "
                    f"(last_a.end={last_a_end:.1f}s → first_b.start={first_b_start:.1f}s)"
                )
        half_dur = (half // 2) / TARGET_SAMPLE_RATE
        print(f"    ↳ Expected break point ~{half_dur:.1f}s into {duration:.1f}s audio")

    out_path = SAMPLES_OUTPUT / f"{path.stem}_stitch_{ts}.json"
    out_path.write_text(
        json.dumps(
            {
                "source_file": path.name,
                "test": "stitch",
                "processed_at": ts,
                "server": server_url,
                "first_half_segments": segs_a,
                "second_half_segments": segs_b,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    print(f"    → Saved: {out_path.relative_to(REPO_ROOT)}\n")
    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def run(
    files: list[Path],
    server_url: str,
    language: str | None,
    save_to_notebook: bool,
    chunk_delay: float,
    stitch_test: bool,
) -> None:
    token = get_admin_token()
    print(f"Token : {token[:8]}…{token[-6:]}")
    print(f"Server: {server_url}")
    print(f"Files : {[p.name for p in files]}")
    if save_to_notebook:
        print("Notebook save: enabled (save_to_notebook=true)")
    if stitch_test:
        print("Stitch test: enabled (two connections per file)")
    print()

    passed = 0
    failed = 0

    for path in files:
        ok = await test_file(
            path, token, server_url, language, save_to_notebook, chunk_delay, stitch_test,
        )
        if ok:
            passed += 1
        else:
            failed += 1

    print(f"Done.  {passed} passed, {failed} failed.")
    if failed:
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test the /ws/stt WebSocket STT endpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/test_ws_stt.py\n"
            "  python scripts/test_ws_stt.py samples/input/1min_test.wav\n"
            "  python scripts/test_ws_stt.py --save-to-notebook samples/input/1min_test.wav\n"
            "  python scripts/test_ws_stt.py --stitch-test samples/input/1min_test.wav\n"
            "  python scripts/test_ws_stt.py --server ws://10.0.1.90:9786 --language en\n"
        ),
    )
    parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        help="Audio files to test (default: all files in samples/input/)",
    )
    parser.add_argument(
        "--server",
        default="ws://127.0.0.1:9786",
        help="WebSocket server base URL (default: ws://127.0.0.1:9786)",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="BCP-47 language code, e.g. 'en' (default: auto-detect)",
    )
    parser.add_argument(
        "--save-to-notebook",
        action="store_true",
        help="Set save_to_notebook=true on the connection (saves final transcript to notebook)",
    )
    parser.add_argument(
        "--chunk-delay",
        type=float,
        default=0.001,
        metavar="SECONDS",
        help="Delay between audio chunks in seconds (default: 0.001 — fast sequential, "
             "like Omi sending stored audio). Use 0.06 for real-time pacing, 0.0 for "
             "absolute max speed.",
    )
    parser.add_argument(
        "--stitch-test",
        action="store_true",
        help="Send file as two sequential connections to verify conversation stitching.",
    )

    args = parser.parse_args()

    if args.files:
        files = [Path(f) for f in args.files]
    else:
        files = sorted(
            p for p in SAMPLES_INPUT.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if not files:
            print(f"No audio files found in {SAMPLES_INPUT}  (WAV/FLAC/OGG)")
            sys.exit(1)

    asyncio.run(
        run(
            files=files,
            server_url=args.server,
            language=args.language,
            save_to_notebook=args.save_to_notebook,
            chunk_delay=args.chunk_delay,
            stitch_test=args.stitch_test,
        )
    )


if __name__ == "__main__":
    main()
