# Model Architecture

Decoder-only causal Transformer implemented from scratch in PyTorch.

## Architecture Highlights

- **Model Type**: Causal Transformer (GPT-style decoder-only)
- **Parameters**: ~12.5M parameters (designed to fit within 2 GB VRAM during training)
- **Block Layout**: Pre-LayerNorm residual architecture (`x + SubLayer(LayerNorm(x))`)
- **Attention**: Multi-head causal self-attention with lower-triangular causal masking
- **MLP**: Feed-forward network with Gaussian Error Linear Unit (GELU) activation
- **Weight Tying**: Shares token embedding matrix `wte` with language modeling head `lm_head`
- **Initialization**: Truncated normal distribution ($\mathcal{N}(0, 0.02)$), scaled by $1 / \sqrt{2 \times n\_layers}$ on residual projections

## Hyperparameters (model_config.yaml)

| Parameter | Value | Description |
|---|---|---|
| `vocab_size` | 4,096 | Vocabulary size from trained BPE tokenizer |
| `d_model` | 384 | Embedding / hidden dimension |
| `n_heads` | 6 | Number of attention heads ($d_{head} = 64$) |
| `n_layers` | 6 | Number of Transformer decoder blocks |
| `d_ff` | 1,536 | Feed-forward dimension ($4 \times d_{model}$) |
| `max_seq_len` | 512 | Context window sequence length |
| `dropout` | 0.1 | Regularization dropout rate |
| `bias` | False | No bias in linear projections |

## Python Usage

```python
import torch
from forgelm.model import ForgeTransformer, ModelConfig

# Instantiate from project config
config = ModelConfig.from_yaml("configs/model/model_config.yaml")
model = ForgeTransformer(config)

print(f"Parameters: {model.get_num_params():,}")

# Forward pass
input_ids = torch.randint(0, config.vocab_size, (2, 64))
targets = torch.randint(0, config.vocab_size, (2, 64))

logits, loss = model(input_ids, targets=targets)
print(f"Logits shape: {logits.shape}")  # torch.Size([2, 64, 4096])
print(f"Loss: {loss.item():.4f}")
```
