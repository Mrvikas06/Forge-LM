# ForgeLM Web Frontend

Modern dark-mode interactive web application for conversational chat, prompt completion, and real-time generation analysis with **ForgeLM**.

---

## Features

- **Dual Interaction Modes**:
  - **Chat Mode**: Conversational interface with bubble streams formatted via `<|user|>` and `<|assistant|>` templates.
  - **Playground Mode**: Raw prompt continuation with curated preset stories, explanations, and adventures.
- **Dynamic Hyperparameter Sliders**:
  - Real-time adjustment of Temperature (0.0 – 1.5), Top-k (0 – 100), Top-p nucleus (0.1 – 1.0), and Repetition Penalty (1.0 – 2.0).
  - Maximum output sequence length slider (16 – 512 tokens).
  - Toggle switch for $O(N)$ KV Cache acceleration.
- **Server-Sent Events (SSE) Streaming**:
  - Token-by-token real-time streaming with blinking terminal cursor.
  - Live token count and generation throughput benchmark (`tok/s`).
- **Live Hardware Telemetry**:
  - Continuous status polling of GPU / CPU device name and active VRAM allocation.
  - Model architecture metadata display (layers, attention heads, parameter count).
- **Zero-Build Architecture**:
  - Runs natively via modern HTML5, CSS3, and ES6 JavaScript without requiring bundlers or Node.js build steps.
  - Mounted directly on the backend server at `/app`.

---

## Launching the Web Interface

### 1. Launch via CLI (Recommended)
```powershell
python scripts/web.py
```
This boots the FastAPI backend and automatically opens [http://localhost:8000/app](http://localhost:8000/app) in your default browser.

### 2. Standalone Frontend Preview
You can also preview the UI using Python's built-in HTTP server or npm:
```powershell
# Using Python
cd frontend
python -m http.server 3000

# Using npm
cd frontend
npm start
```
