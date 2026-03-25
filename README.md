# Audio LLM Demo

Real-time audio transcription demo powered by Amphion (vLLM) with TEN VAD speech segmentation.

## Prerequisites

- Python 3.10+
- A running vLLM server with Amphion (OpenAI-compatible API)
- OpenSSL (for self-signed certificate generation)

## Quick Start

```bash
# Install dependencies
pip install -r backend/requirements.txt

# Set vLLM endpoint (default: http://localhost:8000)
export VLLM_BASE_URL="http://localhost:8000"
export VLLM_MODEL_NAME="Amphion/Amphion-3B"

# Start the server
bash start.sh
```

Open `https://<your-server-ip>:8443` in your browser.

> On first visit, the browser will warn about the self-signed certificate.
> Click **Advanced** -> **Proceed** to continue.

## Configuration

| Environment Variable | Default | Description |
|---|---|---|
| `VLLM_BASE_URL` | `http://localhost:8000` | vLLM server address |
| `VLLM_MODEL_NAME` | `Amphion/Amphion-3B` | Model name |
| `VAD_THRESHOLD` | `0.5` | VAD speech probability threshold |
| `VAD_SMOOTHING_ALPHA` | `0.35` | EMA smoothing for VAD probability (larger = smoother) |
| `VAD_START_FRAMES` | `3` | Consecutive speech frames required to start a segment |
| `VAD_END_FRAMES` | `SILENCE_DURATION_MS/10` | Consecutive non-speech frames required to end a segment |
| `SILENCE_DURATION_MS` | `600` | Silence duration (ms) to end a speech segment |
| `PORT` | `8443` | HTTPS server port |

## Architecture

```
Browser (Mic) --WSS--> FastAPI --HTTP--> vLLM (Amphion)
                          |
                       TEN VAD
                    (speech detection)
```

- **Frontend**: Web Audio API AudioWorklet captures 16kHz PCM, sends via WebSocket
- **Backend**: FastAPI with two concurrent async tasks per connection:
  - VAD Task: processes audio frames, detects speech segments (non-blocking)
  - LLM Task: consumes segments from asyncio.Queue, calls vLLM API (independent)
- **Hotwords**: Managed in the browser UI, synced to backend via WebSocket in real-time
