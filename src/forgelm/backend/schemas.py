"""Pydantic schemas for the ForgeLM FastAPI backend."""

from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    """Request payload for text generation."""

    prompt: str = Field(..., min_length=1, description="Prompt text to continue generation from")
    max_tokens: int = Field(150, ge=1, le=1024, description="Maximum new tokens to generate")
    temperature: float = Field(0.8, ge=0.0, le=2.0, description="Sampling temperature (0.0 = greedy)")
    top_k: int = Field(50, ge=0, description="Top-k filtering threshold (0 = disabled)")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Top-p nucleus sampling threshold")
    repetition_penalty: float = Field(1.1, ge=1.0, le=2.0, description="Repetition penalty parameter")
    use_cache: bool = Field(True, description="Enable KV cache acceleration")


class GenerateResponse(BaseModel):
    """Response payload for text generation."""

    prompt: str
    completion: str
    text: str
    tokens_generated: int
    duration_seconds: float


class ChatMessage(BaseModel):
    """Individual message in a conversational thread."""

    role: Literal["user", "assistant", "system"] = Field(..., description="Role of the message sender")
    content: str = Field(..., min_length=1, description="Message text content")


class ChatRequest(BaseModel):
    """Request payload for multi-turn instruction chat."""

    messages: list[ChatMessage] = Field(..., min_length=1, description="List of conversational messages")
    max_tokens: int = Field(200, ge=1, le=1024, description="Maximum response tokens to generate")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_k: int = Field(40, ge=0, description="Top-k filtering threshold")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Top-p nucleus threshold")
    repetition_penalty: float = Field(1.15, ge=1.0, le=2.0, description="Repetition penalty parameter")
    use_cache: bool = Field(True, description="Enable KV cache acceleration")


class ChatResponse(BaseModel):
    """Response payload for instruction chat."""

    role: Literal["assistant"] = "assistant"
    content: str
    tokens_generated: int
    duration_seconds: float


class HealthResponse(BaseModel):
    """Response payload for system health check."""

    status: str
    device: str
    gpu_available: bool
    gpu_name: str | None = None
    vram_allocated_mb: float | None = None
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    """Response payload for model architecture metadata."""

    model_name: str = "ForgeLM"
    vocab_size: int
    d_model: int
    n_heads: int
    n_layers: int
    max_seq_len: int
    num_parameters: int
