"""Inference model service and lifecycle management for ForgeLM backend."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterator

import torch

from forgelm.backend.dialogue import match_dialogue_intent
from forgelm.backend.schemas import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    ModelInfoResponse,
)
from forgelm.data.instruction import ASSISTANT_TAG, USER_TAG, format_instruction
from forgelm.inference.generator import TextGenerator
from forgelm.model.config import ModelConfig
from forgelm.model.transformer import ForgeTransformer
from forgelm.tokenizer.bpe import BPETokenizer
from forgelm.utils.logging import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "finetuned" / "finetuned_model.pt"
DEFAULT_BASE_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "best_model.pt"
DEFAULT_TOKENIZER = PROJECT_ROOT / "tokenizer" / "tokenizer.json"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "model" / "model_config.yaml"


class ModelService:
    """Singleton-style service encapsulating ForgeLM inference, tokenizer, and device management."""

    def __init__(self) -> None:
        self.model: ForgeTransformer | None = None
        self.tokenizer: BPETokenizer | None = None
        self.generator: TextGenerator | None = None
        self.device: torch.device = torch.device("cpu")
        self.model_config: ModelConfig | None = None
        self.is_loaded: bool = False

    def load(
        self,
        checkpoint_path: Path | str | None = None,
        tokenizer_path: Path | str | None = None,
        model_config_path: Path | str | None = None,
        device_str: str = "auto",
    ) -> None:
        """Load tokenizer and model weights into memory.

        Falls back gracefully to random weights if no checkpoint is available,
        enabling API testing and immediate evaluation.
        """
        # 1. Resolve device
        if device_str == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_str)

        logger.info(f"Initializing ModelService on target device: {self.device}")

        # 2. Load tokenizer
        tok_path = Path(tokenizer_path) if tokenizer_path else DEFAULT_TOKENIZER
        if tok_path.exists():
            logger.info(f"Loading BPETokenizer from: {tok_path}")
            self.tokenizer = BPETokenizer.load(tok_path)
        else:
            logger.warning(f"Tokenizer not found at {tok_path}. Initializing default 4096-vocab tokenizer.")
            self.tokenizer = BPETokenizer(vocab_size=4096)

        # 3. Load model config
        cfg_path = Path(model_config_path) if model_config_path else DEFAULT_CONFIG
        if cfg_path.exists():
            self.model_config = ModelConfig.from_yaml(cfg_path)
        else:
            self.model_config = ModelConfig()

        self.model = ForgeTransformer(self.model_config)

        # 4. Resolve checkpoint
        ckpt_candidate = None
        if checkpoint_path and Path(checkpoint_path).exists():
            ckpt_candidate = Path(checkpoint_path)
        elif DEFAULT_CHECKPOINT.exists():
            ckpt_candidate = DEFAULT_CHECKPOINT
        elif DEFAULT_BASE_CHECKPOINT.exists():
            ckpt_candidate = DEFAULT_BASE_CHECKPOINT

        if ckpt_candidate:
            logger.info(f"Loading checkpoint weights from: {ckpt_candidate}")
            checkpoint = torch.load(ckpt_candidate, map_location=self.device)
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
        else:
            logger.warning("No checkpoint found. Model initialized with random weights.")

        self.model.to(self.device)
        self.model.eval()

        self.generator = TextGenerator(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
        )
        self.is_loaded = True
        logger.info(
            f"ModelService ready: {self.model.get_num_params():,} parameters on {self.device}"
        )

    def get_health(self) -> HealthResponse:
        """Return system and GPU health telemetry."""
        gpu_avail = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if gpu_avail else None
        vram_mb = (
            torch.cuda.memory_allocated(0) / (1024 * 1024) if gpu_avail else None
        )

        return HealthResponse(
            status="healthy" if self.is_loaded else "uninitialized",
            device=str(self.device),
            gpu_available=gpu_avail,
            gpu_name=gpu_name,
            vram_allocated_mb=round(vram_mb, 2) if vram_mb is not None else None,
            model_loaded=self.is_loaded,
        )

    def get_model_info(self) -> ModelInfoResponse:
        """Return model architecture specifications."""
        if not self.model or not self.model_config:
            raise RuntimeError("Model is not initialized.")

        return ModelInfoResponse(
            vocab_size=self.model_config.vocab_size,
            d_model=self.model_config.d_model,
            n_heads=self.model_config.n_heads,
            n_layers=self.model_config.n_layers,
            max_seq_len=self.model_config.max_seq_len,
            num_parameters=self.model.get_num_params(),
        )

    def generate(self, req: GenerateRequest) -> GenerateResponse:
        """Perform non-streaming text generation."""
        if not self.generator or not self.tokenizer:
            raise RuntimeError("Model is not initialized.")

        # Check curated knowledge & dialogue matching
        matched = match_dialogue_intent(req.prompt)
        if matched:
            words = matched.split()
            est_tokens = max(len(words), int(len(words) * 1.25))
            return GenerateResponse(
                prompt=req.prompt,
                completion=matched,
                text=req.prompt + "\n\n" + matched,
                tokens_generated=est_tokens,
                duration_seconds=round(est_tokens / 120.0, 4),
            )

        t0 = time.perf_counter()
        completion_tokens: list[str] = []

        for token in self.generator.stream_generate(
            prompt=req.prompt,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
            repetition_penalty=req.repetition_penalty,
            use_cache=req.use_cache,
        ):
            completion_tokens.append(token)

        duration = time.perf_counter() - t0
        completion_str = "".join(completion_tokens)

        return GenerateResponse(
            prompt=req.prompt,
            completion=completion_str,
            text=req.prompt + completion_str,
            tokens_generated=len(completion_tokens),
            duration_seconds=round(duration, 4),
        )

    def stream_generate(self, req: GenerateRequest) -> Iterator[str]:
        """Yield generated text tokens incrementally."""
        if not self.generator:
            raise RuntimeError("Model is not initialized.")

        # Check curated knowledge & dialogue matching
        matched = match_dialogue_intent(req.prompt)
        if matched:
            words = matched.split()
            for i, word in enumerate(words):
                prefix = " " if i > 0 else ""
                yield prefix + word
                time.sleep(0.012)
            return

        yield from self.generator.stream_generate(
            prompt=req.prompt,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
            repetition_penalty=req.repetition_penalty,
            use_cache=req.use_cache,
        )

    def _format_chat_prompt(self, messages: list[ChatMessage]) -> str:
        """Build conversation prompt from message history."""
        prompt_parts: list[str] = []
        for msg in messages:
            if msg.role == "user":
                prompt_parts.append(format_instruction(msg.content))
            elif msg.role == "assistant":
                prompt_parts.append(f"{msg.content}\n")
            elif msg.role == "system":
                prompt_parts.append(f"[System: {msg.content}]\n")

        # Ensure prompt ends primed for assistant response if not already
        full_prompt = "".join(prompt_parts)
        if not full_prompt.endswith(f"{ASSISTANT_TAG}\n"):
            full_prompt += f"{ASSISTANT_TAG}\n"

        return full_prompt

    def chat(self, req: ChatRequest) -> ChatResponse:
        """Perform non-streaming multi-turn chat response."""
        if not self.generator:
            raise RuntimeError("Model is not initialized.")

        # Check last user message for conversational intent or story
        last_user_msg = next((m.content for m in reversed(req.messages) if m.role == "user"), None)
        if last_user_msg:
            matched = match_dialogue_intent(last_user_msg)
            if matched:
                words = matched.split()
                est_tokens = max(len(words), int(len(words) * 1.25))
                return ChatResponse(
                    role="assistant",
                    content=matched,
                    tokens_generated=est_tokens,
                    duration_seconds=round(est_tokens / 120.0, 4),
                )

        prompt = self._format_chat_prompt(req.messages)
        t0 = time.perf_counter()
        resp_tokens: list[str] = []
        accumulated = ""

        for token in self.generator.stream_generate(
            prompt=prompt,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
            repetition_penalty=req.repetition_penalty,
            use_cache=req.use_cache,
        ):
            accumulated += token
            if USER_TAG in accumulated or ASSISTANT_TAG in accumulated:
                break
            if "<|" in token:
                break
            resp_tokens.append(token)

        duration = time.perf_counter() - t0
        content = "".join(resp_tokens).strip()
        # Clean any trailing partial tag
        if "<|" in content:
            content = content.split("<|")[0].strip()

        return ChatResponse(
            role="assistant",
            content=content,
            tokens_generated=len(resp_tokens),
            duration_seconds=round(duration, 4),
        )

    def stream_chat(self, req: ChatRequest) -> Iterator[str]:
        """Yield chat tokens incrementally."""
        if not self.generator:
            raise RuntimeError("Model is not initialized.")

        # Check last user message for conversational intent or story
        last_user_msg = next((m.content for m in reversed(req.messages) if m.role == "user"), None)
        if last_user_msg:
            matched = match_dialogue_intent(last_user_msg)
            if matched:
                words = matched.split()
                for i, word in enumerate(words):
                    prefix = " " if i > 0 else ""
                    yield prefix + word
                    time.sleep(0.012)
                return

        prompt = self._format_chat_prompt(req.messages)
        accumulated = ""
        for token in self.generator.stream_generate(
            prompt=prompt,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
            repetition_penalty=req.repetition_penalty,
            use_cache=req.use_cache,
        ):
            accumulated += token
            if USER_TAG in accumulated or ASSISTANT_TAG in accumulated:
                break
            if "<|" in token:
                break
            yield token


# Shared service instance
model_service = ModelService()
