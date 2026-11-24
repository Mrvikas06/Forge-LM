"""Unit and integration tests for ForgeLM FastAPI backend."""

import pytest
from fastapi.testclient import TestClient

from forgelm.backend.app import app
from forgelm.backend.service import model_service


@pytest.fixture(scope="module")
def client():
    """Create a test client with loaded model service."""
    # Ensure model is initialized
    if not model_service.is_loaded:
        model_service.load(device_str="cpu")
    with TestClient(app) as test_client:
        yield test_client


def test_root_endpoint(client):
    """Test root service information endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "ForgeLM" in data["service"]
    assert "version" in data
    assert "ui_url" in data


def test_frontend_serving(client):
    """Test integrated frontend HTML, CSS, and JS serving."""
    # Test HTML app endpoint
    html_res = client.get("/app")
    assert html_res.status_code == 200
    assert "text/html" in html_res.headers["content-type"]
    assert "ForgeLM" in html_res.text

    # Test CSS endpoint
    css_res = client.get("/app/style.css")
    assert css_res.status_code == 200
    assert "text/css" in css_res.headers["content-type"]

    # Test JS endpoint
    js_res = client.get("/app/app.js")
    assert js_res.status_code == 200
    assert "application/javascript" in js_res.headers["content-type"]


def test_health_endpoint(client):
    """Test health telemetry endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "device" in data
    assert "gpu_available" in data


def test_model_info_endpoint(client):
    """Test model architecture info endpoint."""
    response = client.get("/model/info")
    assert response.status_code == 200
    data = response.json()
    assert data["vocab_size"] == 4096
    assert data["num_parameters"] > 0
    assert data["d_model"] == 384
    assert data["n_layers"] == 6


def test_generate_endpoint(client):
    """Test standard non-streaming text generation."""
    payload = {
        "prompt": "Once upon a time",
        "max_tokens": 15,
        "temperature": 0.8,
        "top_k": 20,
        "top_p": 0.9,
    }
    response = client.post("/generate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prompt"] == "Once upon a time"
    assert "completion" in data
    assert len(data["text"]) >= len("Once upon a time")
    assert data["tokens_generated"] > 0
    assert data["duration_seconds"] >= 0.0


def test_generate_stream_endpoint(client):
    """Test SSE streaming text generation."""
    payload = {
        "prompt": "The little bear",
        "max_tokens": 10,
        "temperature": 0.5,
    }
    response = client.post("/generate/stream", json=payload)
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    content = response.text
    assert "data:" in content
    assert '"done": true' in content or '"done":true' in content


def test_chat_endpoint(client):
    """Test multi-turn instruction chat endpoint."""
    payload = {
        "messages": [
            {"role": "user", "content": "Hello, tell me a quick fact."}
        ],
        "max_tokens": 15,
        "temperature": 0.7,
    }
    response = client.post("/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["role"] == "assistant"
    assert isinstance(data["content"], str)
    assert data["tokens_generated"] > 0


def test_chat_stream_endpoint(client):
    """Test SSE streaming chat endpoint."""
    payload = {
        "messages": [
            {"role": "user", "content": "Sing a cheerful song."}
        ],
        "max_tokens": 10,
    }
    response = client.post("/chat/stream", json=payload)
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    assert "data:" in response.text


def test_invalid_generate_request(client):
    """Test request validation and error handling."""
    # Empty prompt violates min_length=1
    response = client.post("/generate", json={"prompt": ""})
    assert response.status_code == 422

    # Negative max_tokens violates ge=1
    response = client.post("/generate", json={"prompt": "Valid", "max_tokens": -5})
    assert response.status_code == 422
