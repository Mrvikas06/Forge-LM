"""Instruction tuning dataset and template formatting utilities for Supervised Fine-Tuning (SFT)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import Dataset

from forgelm.tokenizer.bpe import BPETokenizer
from forgelm.utils.logging import get_logger

logger = get_logger(__name__)

USER_TAG = "<|user|>"
ASSISTANT_TAG = "<|assistant|>"


def format_instruction(instruction: str, response: str | None = None) -> str:
    """Format an instruction and optional response into a standardized chat template.

    Template:
        <|user|>
        {instruction}
        <|assistant|>
        {response}

    Args:
        instruction: The user prompt or instruction string.
        response: Optional model response string.

    Returns:
        Formatted text string.
    """
    cleaned_instruction = instruction.strip()
    formatted = f"{USER_TAG}\n{cleaned_instruction}\n{ASSISTANT_TAG}\n"

    if response is not None:
        formatted += f"{response.strip()}"

    return formatted


class InstructionDataset(Dataset):
    """Dataset for Supervised Fine-Tuning (SFT) with response-only loss masking.

    Ensures that cross-entropy loss is computed exclusively on the assistant's
    response tokens, masking out the user instruction prompt.
    """

    def __init__(
        self,
        data: Sequence[dict[str, str]] | str | Path,
        tokenizer: BPETokenizer,
        max_seq_len: int = 512,
        pad_token_id: int = 0,
    ) -> None:
        """Initialize InstructionDataset.

        Args:
            data: List of dicts with 'instruction' and 'response' keys, or path to JSON file.
            tokenizer: Trained BPETokenizer instance.
            max_seq_len: Maximum sequence length.
            pad_token_id: Special token ID used for padding and loss masking.
        """
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id

        if isinstance(data, (str, Path)):
            path = Path(data)
            if not path.exists():
                raise FileNotFoundError(f"Instruction dataset file not found: {path}")
            with open(path, "r", encoding="utf-8") as f:
                self.records = json.load(f)
        else:
            self.records = list(data)

        self.samples = self._prepare_samples()

    def _prepare_samples(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Pre-tokenize all instruction-response pairs with prompt masking."""
        samples: list[tuple[torch.Tensor, torch.Tensor]] = []

        for record in self.records:
            instruction = record.get("instruction", "")
            response = record.get("response", "")

            if not instruction or not response:
                continue

            # Format prompt and response parts
            prompt_str = format_instruction(instruction)
            prompt_ids = self.tokenizer.encode(prompt_str, add_bos=True, add_eos=False)

            # Response ends with EOS token
            resp_ids = self.tokenizer.encode(response, add_bos=False, add_eos=True)

            # Concatenate full sequence: [BOS, prompt..., response..., EOS]
            full_ids = prompt_ids + resp_ids

            if len(full_ids) < 2:
                continue

            # Crop if exceeding max_seq_len + 1
            if len(full_ids) > self.max_seq_len + 1:
                full_ids = full_ids[: self.max_seq_len + 1]

            # Input x is all tokens except last; Target y is shifted by 1
            input_ids = full_ids[:-1]
            target_ids = full_ids[1:]

            # Mask out prompt positions in targets so loss is only calculated on response
            prompt_len = len(prompt_ids)
            masked_targets = list(target_ids)

            # Target position i corresponds to input token i predicting full_ids[i+1]
            # Prompt tokens extend from index 0 up to prompt_len - 1
            for i in range(min(prompt_len - 1, len(masked_targets))):
                masked_targets[i] = self.pad_token_id

            # Pad sequences to max_seq_len
            pad_length = self.max_seq_len - len(input_ids)
            if pad_length > 0:
                input_ids = input_ids + [self.pad_token_id] * pad_length
                masked_targets = masked_targets + [self.pad_token_id] * pad_length
            else:
                input_ids = input_ids[: self.max_seq_len]
                masked_targets = masked_targets[: self.max_seq_len]

            samples.append((
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(masked_targets, dtype=torch.long),
            ))

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx]
