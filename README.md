# Visionnaire

Real-time computer-vision accessibility assistant for the OptiVision hackathon.

Laptop webcam frames are streamed over a binary WebSocket to an inference server
(local or NVIDIA Brev GPU). The server runs YOLOv8n and returns a structured JSON
decision. The laptop speaks the decision with pyttsx3.

## Current MVP

```
Webcam → JPEG → binary WebSocket → YOLOv8n → Decision → JSON WebSocket → pyttsx3
```

## Future

```
YOLO + MiDaS + Face + OCR → Fusion Engine → Decision
```

Those extra models are **not** implemented yet. Empty stubs under
`server/models/` and `server/fusion/` are reserved for later.

## Architecture

```
LAPTOP / CLIENT                         NVIDIA BREV GPU / SERVER
─────────────────                       ──────────────────────────
OpenCV webcam                           WebSocket server (0.0.0.0)
JPEG encode                             YOLOv8n inference
WebSocket client   ── binary frames ──► pipeline (YOLO only for MVP)
pyttsx3 audio      ◄── JSON decision ── structured Decision
```

### Responsibilities

| Side   | Owns                                              | Must not own                          |
|--------|---------------------------------------------------|---------------------------------------|
| Client | camera, JPEG, WebSocket client, TTS               | YOLO / any heavy AI inference         |
| Server | WebSocket server, YOLO, pipeline, decision logic  | camera or speaker implementations     |
| Shared | binary frame protocol (`shared/protocol.py`)      | —                                     |

Client and server are strictly separated. Transport (`websocket.py`) is separate
from inference (`pipeline.py` / `models/yolo.py`).

### Package layout

```
src/optivisionhackathon/
├── client/
│   ├── main.py
│   ├── camera.py
│   ├── websocket.py
│   └── speaker.py
├── server/
│   ├── main.py
│   ├── websocket.py
│   ├── pipeline.py
│   ├── schemas.py
│   ├── models/
│   │   ├── yolo.py          ← implemented
│   │   ├── depth.py         ← future
│   │   ├── face.py          ← future
│   │   └── ocr.py           ← future
│   └── fusion/
│       ├── engine.py        ← future
│       └── cooldown.py      ← future
├── shared/
│   └── protocol.py
├── data/faces/
├── scripts/
└── tests/
requirements.txt
.env.example
```

## Binary WebSocket protocol

Defined in `shared/protocol.py`:

```
┌──────────────┬──────────────┬────────────────────┐
│ Magic        │ Payload Len  │ JPEG Payload       │
│ 4 bytes      │ 4 bytes BE   │ N bytes            │
└──────────────┴──────────────┴────────────────────┘
```

- Magic: `b"VISH"`
- Length: unsigned 32-bit big-endian
- Frames travel as **binary** WebSocket messages (never base64 / never JSON images)
- Server responses are **JSON only**

```python
encode_frame(jpeg_bytes) -> bytes
decode_frame(packet) -> bytes
```

## Environment variables

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

| Variable                   | Default                 | Used by |
|----------------------------|-------------------------|---------|
| `VISIONNAIRE_SERVER_URL`   | `ws://localhost:8765`   | Client  |
| `VISIONNAIRE_HOST`         | `0.0.0.0`               | Server  |
| `VISIONNAIRE_PORT`         | `8765`                  | Server  |
| `VISIONNAIRE_CAMERA_INDEX` | `0`                     | Client  |
| `VISIONNAIRE_JPEG_QUALITY` | `80`                    | Client  |
| `VISIONNAIRE_TARGET_FPS`   | `5`                     | Client  |

Do **not** hardcode a Brev IP. For Brev, set:

```bash
VISIONNAIRE_SERVER_URL=ws://<BREV_ADDRESS>:8765
```

## Local development

Requires Python 3.10 or 3.11.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
cp .env.example .env
```

### Run the server (terminal 1)

```bash
python -m optivisionhackathon.server.main
```

At startup you should see CUDA / device logs, for example:

```
CUDA available: True
GPU: NVIDIA L4
Inference device: cuda
Loading YOLOv8n...
```

On a machine without CUDA it falls back to CPU automatically.

### Run the client (terminal 2)

```bash
python -m optivisionhackathon.client.main
```

The client captures at ~5 FPS, waits for each decision (no unbounded frame queue),
and speaks non-empty messages via pyttsx3.

## NVIDIA Brev deployment

1. Start a GPU instance on NVIDIA Brev.
2. Clone this repo on the instance and install dependencies (with CUDA-enabled torch).
3. Run the **same** server code — only env config changes:

```bash
export VISIONNAIRE_HOST=0.0.0.0
export VISIONNAIRE_PORT=8765
python -m optivisionhackathon.server.main
```

4. On the laptop, point the client at the instance:

```bash
export VISIONNAIRE_SERVER_URL=ws://<BREV_ADDRESS>:8765
python -m optivisionhackathon.client.main
```

No code changes are required between local and Brev — only environment variables.

Ensure the Brev firewall / port mapping exposes `VISIONNAIRE_PORT` (default 8765).

## Tests

```bash
pip install pytest
pytest src/optivisionhackathon/tests/test_protocol.py -v
```

## Current MVP limitations

- Only YOLOv8n is used (no depth, face ID, or OCR)
- Distance is a coarse heuristic from bounding-box height, not real meters
- Only a curated set of COCO obstacle classes is treated as relevant
- Simple per-key speech cooldown; full fusion engine is not implemented
- Single-client request/response loop (hackathon-friendly, not production-scaled)

## Planned future models

| Module | Role                                      |
|--------|-------------------------------------------|
| MiDaS  | Monocular depth estimation                |
| Face   | Known-face recognition                    |
| OCR    | Scene text (signs, labels)                |
| Fusion | Combine signals into a single Decision    |
