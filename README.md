# Visionnaire

Real-time multi-modal computer-vision accessibility assistant for the OptiVision hackathon.

Laptop webcam frames are streamed over a binary WebSocket to an inference server
(local or NVIDIA Brev GPU). The server runs YOLOv8n obstacle detection, monocular
depth and metric distance estimation, deep face recognition, and scene text / OCR,
fused together into a coherent, prioritized decision. The laptop displays annotated
live visual overlays and speaks spoken assistance in real time via non-blocking TTS.

## Full Multi-Modal Pipeline

```
Webcam → JPEG → binary WebSocket → YOLO + Depth + Face + OCR → Fusion Engine → Decision → JSON WebSocket → pyttsx3 / eSpeak-ng
```

## Architecture

```
LAPTOP / CLIENT                         NVIDIA GPU / BREV SERVER
─────────────────                       ──────────────────────────
OpenCV webcam                           WebSocket server (0.0.0.0)
JPEG encode                             YOLOv8n obstacle detection
HUD & multi-box overlays                Pinhole depth / distance estimation
WebSocket client   ── binary frames ──► Deep face recognition (MobileNet embeddings)
eSpeak-ng audio    ◄── JSON decision ── Scene text OCR (signage & labels)
                                        Fusion engine + intelligent cooldown
```

### Responsibilities

| Side   | Owns                                              | Must not own                          |
|--------|---------------------------------------------------|---------------------------------------|
| Client | Camera, JPEG, WebSocket client, TTS, live HUD     | YOLO / deep AI inference              |
| Server | WebSocket server, YOLO, Depth, Face, OCR, Fusion  | Camera or speaker implementations     |
| Shared | Binary frame protocol (`shared/protocol.py`)      | —                                     |

Client and server are strictly separated. Transport (`websocket.py`) is separate
from inference (`pipeline.py`).

### Package Layout

```
src/optivisionhackathon/
├── client/
│   ├── main.py              # Laptop client loop (camera, HUD preview, TTS)
│   ├── camera.py            # OpenCV camera capture & JPEG encoding
│   ├── display.py           # Multi-modal visual overlays (obstacles, faces, signs)
│   ├── result.py            # Structured decision JSON parser
│   ├── speaker.py           # Non-blocking, non-overlapping TTS (espeak-ng / pyttsx3)
│   └── websocket.py         # Async WebSocket client
├── server/
│   ├── main.py              # Server entrypoint with CUDA & multi-modal init
│   ├── websocket.py         # WebSocket server transport
│   ├── pipeline.py          # Full multi-modal inference pipeline
│   ├── schemas.py           # Typed dataclasses (Detection, FaceDetection, TextDetection, Decision)
│   ├── models/
│   │   ├── yolo.py          # YOLOv8n obstacle detector (CUDA/CPU)
│   │   ├── depth.py         # Pinhole depth & metric distance estimation
│   │   ├── face.py          # Deep face recognition with MobileNet embeddings
│   │   └── ocr.py           # Scene text & signage recognition
│   └── fusion/
│       ├── engine.py        # Multi-modal priority fusion & utterance generator
│       └── cooldown.py      # Intelligent cooldown with priority escalation
├── shared/
│   └── protocol.py          # Binary frame packet protocol
├── data/
│   └── faces/
│       └── embeddings.pkl   # Registered face embeddings database
├── scripts/
│   ├── register_face.py     # CLI face registration tool
│   └── verify_flow.py       # End-to-end verification script
└── tests/
    ├── test_protocol.py     # Binary protocol tests
    ├── test_pipeline.py     # Pipeline & speech formatting tests
    ├── test_fusion.py       # Multi-modal fusion, depth, face, and OCR tests
    └── test_end_to_end.py   # Full integration flow tests
```

## Binary WebSocket Protocol

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

## Environment Variables

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

| Variable                        | Default                 | Used by | Description                                     |
|---------------------------------|-------------------------|---------|-------------------------------------------------|
| `VISIONNAIRE_SERVER_URL`        | `ws://localhost:8765`   | Client  | WebSocket URL to connect to                     |
| `VISIONNAIRE_HOST`              | `0.0.0.0`               | Server  | WebSocket bind address                          |
| `VISIONNAIRE_PORT`              | `8765`                  | Server  | WebSocket port                                  |
| `VISIONNAIRE_CAMERA_INDEX`      | `0`                     | Client  | Webcam device index                             |
| `VISIONNAIRE_JPEG_QUALITY`      | `80`                    | Client  | JPEG compression quality (1-100)                |
| `VISIONNAIRE_TARGET_FPS`        | `5`                     | Client  | Capture and send framerate                      |
| `VISIONNAIRE_ENABLE_DEPTH`      | `true`                  | Server  | Enable monocular depth & metric distance        |
| `VISIONNAIRE_ENABLE_FACE`       | `true`                  | Server  | Enable face recognition                         |
| `VISIONNAIRE_ENABLE_OCR`        | `true`                  | Server  | Enable scene text recognition                   |
| `VISIONNAIRE_CRITICAL_DIST`     | `1.5`                   | Server  | Critical proximity distance in meters           |
| `VISIONNAIRE_WARNING_DIST`      | `3.0`                   | Server  | Warning proximity distance in meters            |
| `VISIONNAIRE_FACE_THRESHOLD`    | `0.65`                  | Server  | Face cosine similarity threshold                |
| `VISIONNAIRE_OCR_CONF`          | `0.40`                  | Server  | OCR minimum confidence                          |
| `VISIONNAIRE_DECISION_COOLDOWN` | `2.5`                   | Server  | Repetition speech cooldown (seconds)            |
| `VISIONNAIRE_CRITICAL_COOLDOWN` | `1.0`                   | Server  | Critical alerts cooldown (seconds)              |

## Registering Known Faces

Register faces directly using the webcam or an image file:

```bash
# Register from webcam:
uv run python -m optivisionhackathon.scripts.register_face --name "Souhail" --webcam

# Register from an image:
uv run python -m optivisionhackathon.scripts.register_face --name "Souhail" --image path/to/photo.jpg

# List all registered faces:
uv run python -m optivisionhackathon.scripts.register_face --list

# Delete a face:
uv run python -m optivisionhackathon.scripts.register_face --delete "Souhail"
```

## Running the Application

### 1. Run the Server

```bash
uv run python -m optivisionhackathon.server.main
```

At startup, the server automatically detects CUDA (NVIDIA GPU) or falls back to CPU,
loads YOLOv8n, initializes Depth, loads Face embeddings, and starts listening.

### 2. Run the Client

```bash
uv run python -m optivisionhackathon.client.main
```

The client streams webcam frames at ~5 FPS, renders live OpenCV preview with bounding
boxes (color-coded by distance severity), face tags, and signs, and speaks audio
alerts via non-blocking TTS. Press `q` to exit.

## Running Tests

Run the complete test suite with `uv run pytest`:

```bash
uv run pytest -v
```
