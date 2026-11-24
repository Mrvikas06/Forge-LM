# ForgeLM Backend Service

High-performance inference and streaming API built with **FastAPI**, **Uvicorn**, and **Pydantic**.

---

## Features

- **REST & SSE Streaming**: Full support for standard synchronous responses and Server-Sent Events (`text/event-stream`) for live token streaming.
- **KV Cache Accelerated**: Decodes autoregressively using $O(N)$ cached key-value states.
- **Multi-Turn Chat Formatting**: Formats conversations automatically using `<|user|>` and `<|assistant|>` chat templates.
- **Hardware Telemetry**: Real-time VRAM allocation, CUDA device tracking, and health metrics.
- **Graceful Lifecycle Management**: Loads checkpoints dynamically with automatic fallback to initialized weights for testing.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Service root and metadata |
| `GET` | `/health` | Hardware telemetry, CUDA availability, VRAM allocation |
| `GET` | `/model/info` | Architecture metadata (parameter count, layers, heads) |
| `POST` | `/generate` | Autoregressive text completion |
| `POST` | `/generate/stream` | Token-by-token SSE streaming generation |
| `POST` | `/chat` | Multi-turn conversational instruction completion |
| `POST` | `/chat/stream` | Token-by-token SSE streaming chat |

---

## Launching the Server

### Using the CLI Script
```powershell
# Default (0.0.0.0:8000)
python scripts/serve.py

# Custom host and port
python scripts/serve.py --host 127.0.0.1 --port 8080

# Serve with specific checkpoint
python scripts/serve.py --checkpoint checkpoints/best_model.pt
```

### Direct Entrypoint
```powershell
python backend/main.py
```

---

## Interactive Documentation

Once the server is running, navigate to:
- Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
- ReDoc: [http://localhost:8000/redoc](http://localhost:8000/redoc)
