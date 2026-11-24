"""Autoregressive text generation engine for ForgeLM."""

from __future__ import annotations

from typing import Iterator, Sequence

import torch
import torch.nn as nn

from forgelm.inference.sampling import sample_next_token
from forgelm.tokenizer.bpe import BPETokenizer
from forgelm.utils.logging import get_logger

logger = get_logger(__name__)


class TextGenerator:
    """Autoregressive text generator with sampling and streaming support.

    Handles prompt encoding, context window management, token sampling,
    and decoding back to human-readable text.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: BPETokenizer,
        device: torch.device | str | None = None,
    ) -> None:
        """Initialize TextGenerator.

        Args:
            model: Trained ForgeTransformer language model.
            tokenizer: Trained BPETokenizer instance.
            device: Target execution device ('cuda', 'cpu', or None for auto).
        """
        self.model = model
        self.tokenizer = tokenizer

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        self.model.to(self.device)
        self.model.eval()

        self.eos_id = tokenizer.eos_id
        self.max_context = getattr(model, "config", None)
        self.max_seq_len = self.max_context.max_seq_len if self.max_context else 512

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        stop_tokens: Sequence[int] | None = None,
    ) -> str:
        """Generate text completion from a prompt string.

        Args:
            prompt: Input text prompt.
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature (0.0 = greedy argmax).
            top_k: Top-k filtering threshold.
            top_p: Nucleus cumulative probability threshold.
            repetition_penalty: Penalty for repeating prior tokens.
            stop_tokens: Optional token IDs that trigger early stopping.

        Returns:
            Generated text string including the prompt.
        """
        chunks: list[str] = [prompt]
        for token_text in self.stream_generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            stop_tokens=stop_tokens,
        ):
            chunks.append(token_text)

        return "".join(chunks)

    @torch.no_grad()
    def stream_generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        stop_tokens: Sequence[int] | None = None,
    ) -> Iterator[str]:
        """Yield generated text tokens one by one as they are produced.

        Args:
            prompt: Input text prompt.
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature.
            top_k: Top-k filtering.
            top_p: Nucleus sampling.
            repetition_penalty: Penalty on previously generated tokens.
            stop_tokens: Extra stop token IDs.

        Yields:
            Decoded text strings for each new token.
        """
        # Encode initial prompt tokens
        input_ids = self.tokenizer.encode(prompt, add_bos=True, add_eos=False)
        tokens_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        stops = {self.eos_id}
        if stop_tokens:
            stops.update(stop_tokens)

        generated_ids: list[int] = []

        for _ in range(max_new_tokens):
            # Crop to context window if input length exceeds max_seq_len
            if tokens_tensor.size(1) > self.max_seq_len:
                tokens_cond = tokens_tensor[:, -self.max_seq_len :]
            else:
                tokens_cond = tokens_tensor

            # Forward pass: get logits for last position
            logits, _ = self.model(tokens_cond)
            next_token_logits = logits[:, -1, :]  # (1, vocab_size)

            # Sample next token
            next_token = sample_next_token(
                logits=next_token_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                generated_tokens=input_ids + generated_ids,
            )

            token_id = next_token.item()

            # Check for EOS or custom stop token
            if token_id in stops:
                break

            generated_ids.append(token_id)
            tokens_tensor = torch.cat([tokens_tensor, next_token], dim=1)

            # Decode single token text and yield
            decoded_chunk = self.tokenizer.decode([token_id], skip_special_tokens=True)
            yield decoded_chunk
