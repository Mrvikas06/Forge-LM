"""FastAPI application for ForgeLM inference and text generation services."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse

from forgelm.backend.schemas import (
    ChatRequest,
    ChatResponse,
    DiagnosticsResponse,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    ModelInfoResponse,
)
from forgelm.backend.service import model_service
from forgelm.utils.logging import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan context manager for startup and shutdown hooks."""
    logger.info("Initializing ForgeLM backend service...")
    if not model_service.is_loaded:
        model_service.load()
    yield
    logger.info("ForgeLM backend service shutting down.")


app = FastAPI(
    title="ForgeLM Inference API",
    description="REST & SSE Streaming API for ForgeLM causal language model.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for web frontends (e.g. React / Vite)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent.parent / "frontend"


@app.get("/", tags=["Info"])
def root() -> dict[str, str]:
    """Root metadata endpoint."""
    return {
        "service": "ForgeLM Inference API",
        "version": "1.0.0",
        "docs_url": "/docs",
        "health_url": "/health",
        "ui_url": "/app",
    }


@app.get("/app", response_class=HTMLResponse, tags=["UI"])
def serve_frontend_app() -> HTMLResponse:
    """Serve interactive web frontend."""
    index_file = FRONTEND_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Frontend assets not found")
    content = index_file.read_text(encoding="utf-8")
    content = content.replace('href="style.css"', 'href="/app/style.css"')
    content = content.replace('src="app.js"', 'src="/app/app.js"')
    return HTMLResponse(content=content)


@app.get("/app/style.css", tags=["UI"])
def serve_frontend_css() -> Response:
    """Serve frontend stylesheet."""
    css_file = FRONTEND_DIR / "style.css"
    if not css_file.exists():
        raise HTTPException(status_code=404, detail="CSS stylesheet not found")
    return Response(content=css_file.read_text(encoding="utf-8"), media_type="text/css")


@app.get("/app/app.js", tags=["UI"])
def serve_frontend_js() -> Response:
    """Serve frontend JavaScript application."""
    js_file = FRONTEND_DIR / "app.js"
    if not js_file.exists():
        raise HTTPException(status_code=404, detail="JavaScript asset not found")
    return Response(content=js_file.read_text(encoding="utf-8"), media_type="application/javascript")


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def get_health() -> HealthResponse:
    """Telemetry and GPU status check."""
    return model_service.get_health()


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Monitoring"])
def get_model_info() -> ModelInfoResponse:
    """Retrieve model architecture specifications."""
    try:
        return model_service.get_model_info()
    except RuntimeError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))


@app.get("/model/diagnostics", response_model=DiagnosticsResponse, tags=["Monitoring"])
def get_model_diagnostics() -> DiagnosticsResponse:
    """Retrieve unified diagnostics across tokenizer, model, and experiments."""
    try:
        return model_service.get_diagnostics()
    except Exception as e:
        logger.error(f"Diagnostics error: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))



@app.post("/generate", response_model=GenerateResponse, tags=["Inference"])
def generate_text(req: GenerateRequest) -> GenerateResponse:
    """Generate text completion from a prompt."""
    try:
        return model_service.generate(req)
    except Exception as e:
        logger.error(f"Generation error: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/generate/stream", tags=["Inference"])
def stream_generate_text(req: GenerateRequest) -> StreamingResponse:
    """Stream generated tokens in Server-Sent Events (SSE) format."""

    def event_stream():
        try:
            for token in model_service.stream_generate(req):
                payload = json.dumps({"token": token, "done": False})
                yield f"data: {payload}\n\n"
            # Signal completion
            yield f"data: {json.dumps({'token': '', 'done': True})}\n\n"
        except Exception as e:
            logger.error(f"Stream error: {e}", exc_info=True)
            yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/chat", response_model=ChatResponse, tags=["Inference"])
def chat(req: ChatRequest) -> ChatResponse:
    """Generate assistant response from conversation messages."""
    try:
        return model_service.chat(req)
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/chat/stream", tags=["Inference"])
def stream_chat(req: ChatRequest) -> StreamingResponse:
    """Stream assistant response tokens in Server-Sent Events (SSE) format."""

    def event_stream():
        try:
            for token in model_service.stream_chat(req):
                payload = json.dumps({"token": token, "done": False})
                yield f"data: {payload}\n\n"
            # Signal completion
            yield f"data: {json.dumps({'token': '', 'done': True})}\n\n"
        except Exception as e:
            logger.error(f"Chat stream error: {e}", exc_info=True)
            yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
