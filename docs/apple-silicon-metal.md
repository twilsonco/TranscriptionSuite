# Apple Silicon / Metal Backend — Setup & Usage Guide

Features added in commits `948da33`–`233e85a` (2025-03-21 to 2025-03-22).

---

## Overview of Changes

| Commit | What was added |
|--------|---------------|
| `948da33` | `mlx-whisper` dependency, `pyproject.toml` updates |
| `e83270e` | MLX Whisper backend (`mlx_whisper_backend.py`), Omi WebSocket endpoint (`/ws/omi`), MLX model registry entries in the dashboard |
| `c998fd6` | **Metal** runtime profile option in the dashboard UI (Server view + Settings) |
| `233e85a` | `LOG_LEVEL` / `LOG_DIR` env var overrides, MLX beam-search fix (greedy fallback) |

---

## 1. Server Startup — Bare-Metal (Metal/MLX)

The server runs directly on macOS without Docker when you use the `.venv` uvicorn binary.  
All path arguments must be fully expanded (no `$HOME` in quoted env vars on macOS).

```bash
cd /path/to/TranscriptionSuite

DATA_DIR="/Users/<you>/Library/Application Support/TranscriptionSuite/data" \
HF_HOME="/Users/<you>/Library/Application Support/TranscriptionSuite/models" \
HF_TOKEN="hf_..." \
MAIN_TRANSCRIBER_MODEL="mlx-community/whisper-large-v3-mlx" \
server/backend/.venv/bin/uvicorn server.api.main:app \
  --host 0.0.0.0 --port 9786
```

> **Tip — debug logging:** Add `LOG_LEVEL=DEBUG LOG_DIR="$DATA_DIR/logs"` to the env block above to get a structured JSON log at `$LOG_DIR/server.log`.

The server is ready when:

```bash
curl -s http://localhost:9786/ready | python3 -m json.tool
```

returns `"loaded": true` with `"backend": "mlx_whisper"` and `"features.mlx.available": true`.

---

## 2. Metal Runtime Profile — Dashboard

In the dashboard the Metal profile can be selected from two places:

**Settings → Server Profile**
1. Open the dashboard (e.g. `npm run dev:electron` from `dashboard/`).
2. Click the gear ⚙ icon → **Settings**.
3. Under *Runtime Profile*, select **Metal (Apple Silicon)**.
4. The model selector will show only `mlx-community/*` models.
5. Click **Save**. The dashboard stores `runtimeProfile: "metal"` in its config.

**Server View (quick toggle)**
- The Server panel also exposes the profile dropdown so you can switch without opening Settings.

When `metal` is selected, the dashboard starts/restarts the server using the MLX model and bare-metal uvicorn path rather than Docker.

---

## 3. Transcribing a File

### 3a. Terminal (curl)

Basic transcription, no diarization:

```bash
curl -s -X POST http://localhost:9786/api/transcribe/file \
  -F "file=@/path/to/audio.wav" \
  -w "\nHTTP_STATUS: %{http_code}\n"
```

With speaker diarization (requires `HF_TOKEN` and pyannote model access):

```bash
curl -s -X POST http://localhost:9786/api/transcribe/file \
  -F "file=@/path/to/audio.wav" \
  -F "diarization=true" \
  -w "\nHTTP_STATUS: %{http_code}\n"
```

The response is a JSON object:

```jsonc
{
  "text": "...",                  // full transcript
  "language": "en",
  "language_probability": 1.0,
  "duration": 60.0,
  "num_speakers": 2,              // present when diarization=true
  "segments": [
    {
      "text": "Hello world.",
      "start": 0.0,
      "end": 2.5,
      "speaker": "SPEAKER_00",   // present when diarization=true
      "words": [...]              // per-word timestamps + speaker
    }
  ],
  "words": [...]                  // flat word list with speaker labels
}
```

Useful optional form fields:

| Field | Default | Description |
|-------|---------|-------------|
| `language` | auto-detect | BCP-47 code, e.g. `"en"` |
| `diarization` | `false` | Enable speaker diarization |
| `min_speakers` | auto | Hint minimum speaker count |
| `max_speakers` | auto | Hint maximum speaker count |
| `initial_prompt` | none | Context string to guide transcription |

### 3b. Dashboard — File Transcription

1. Open the dashboard.
2. Navigate to the **Transcribe** view.
3. Drag-and-drop or browse to an audio/video file.
4. Enable *Speaker Diarization* if desired.
5. Click **Transcribe**.

Results are shown inline and can be exported as `.srt` or `.txt`.

### 3c. Dashboard — Audio Notebook

Files can also be sent to the Audio Notebook for storage and later review:
- Enable *Add to Notebook* in the transcription panel, or
- Set `auto_add_to_audio_notebook: true` in `server/config.yaml`.

---

## 4. MLX Backend Notes

- **Model selection**: Any `mlx-community/*` Whisper model works (e.g. `mlx-community/whisper-large-v3-mlx`, `mlx-community/whisper-large-v3-turbo`).  The backend is auto-selected when the model name matches `mlx-community/*`.
- **Beam search**: MLX Whisper only supports greedy decoding.  If `beam_size > 1` is configured (the default is 5), the backend silently falls back to greedy.  This has no user-visible impact.
- **Diarization**: Pyannote diarization works with the MLX backend exactly as with other backends — transcription runs on Metal, diarization runs on CPU (or the device specified in `config.yaml`).
- **Performance**: ~3 s per minute of audio on an M-series chip with `whisper-large-v3-mlx`.

---

## 5. Logging Configuration

Two new env vars override the logging section of `server/config.yaml`:

| Variable | Config key | Example |
|----------|-----------|---------|
| `LOG_LEVEL` | `logging.level` | `DEBUG`, `INFO`, `WARNING` |
| `LOG_DIR` | `logging.directory` | `/path/to/logs` |

The log file is written to `$LOG_DIR/server.log` in structured JSON (one object per line, via structlog).

Example tail / pretty-print:

```bash
tail -f "/Users/<you>/Library/Application Support/TranscriptionSuite/data/logs/server.log" \
  | python3 -c "import sys,json; [print(json.dumps(json.loads(l),indent=2)) for l in sys.stdin]"
```

---

## 6. Omi Wearable Device — Custom STT via WebSocket

The `/ws/omi` WebSocket endpoint lets an [Omi](https://www.omi.me/) wearable stream audio to this server for transcription.

See **[`docs/omi_external_custom_STT_service_docs.md`](omi_external_custom_STT_service_docs.md)** for the full protocol spec.

### Quick setup

1. Ensure the server is running and reachable from the Omi device (e.g. on the same Wi-Fi, or via Tailscale).
2. In the Omi app, go to **Settings → Developer → Custom STT** and enter:
   - **WebSocket URL**: `ws://<server-ip>:9786/ws/omi?token=<your-api-token>`
   - **Codec**: `opus` (default) or `pcm`
3. The server buffers inbound audio, transcribes on `CloseStream` (or 90 s timeout), and returns a `segments` JSON payload.
4. Diarization is included automatically when the loaded backend supports it.

**Configuration** (`server/config.yaml`):

```yaml
omi_stt:
  enabled: true
  inactivity_timeout_s: 90  # match Omi's server-side idle timeout
```

**Authentication**: the `token` query parameter is validated against the server's API token store (same tokens used by the remote web client).  On localhost the token check is bypassed.

**Opus dependency** (only needed for Omi's default `opus` codec):

```bash
brew install opus
cd server/backend && uv sync --extra omi
```

### Test script

```bash
python3 scripts/test_omi_websocket.py \
  --url ws://localhost:9786/ws/omi \
  --file samples/input/1min_test.wav \
  --codec pcm
```

Run `python3 scripts/test_omi_websocket.py --help` for all options.

---

## 7. Config Reference — New Sections

Sections added/expanded in `server/config.yaml` by these commits:

```yaml
# Logging (env vars LOG_LEVEL / LOG_DIR also override these)
logging:
  level: "INFO"          # or DEBUG / WARNING / ERROR
  directory: "/data/logs"
  console_output: true
  file_name: "server.log"
  max_size_mb: 10
  backup_count: 5

# Omi External Custom STT
omi_stt:
  enabled: true
  inactivity_timeout_s: 90
```

The `main_transcriber.model` value drives backend selection automatically:

| Model pattern | Backend |
|--------------|---------|
| `mlx-community/*` | `mlx_whisper` (Metal, Apple Silicon) |
| `nvidia/*` | NeMo / Parakeet |
| `*/VibeVoice-ASR*` | VibeVoice-ASR |
| anything else | `faster_whisper` (WhisperX) |
