# Pretraining Engine

Pretraining pipeline, optimization loop, and memory-mapped binary dataset for ForgeLM.

## Architecture & Features

- **Memory-Mapped Binary Datasets**: `PretrainingDataset` streams `uint16` token arrays from `.bin` files via `numpy.memmap`, keeping RAM usage constant and minimal.
- **Selective Weight Decay**: Decoupled AdamW applies weight decay (0.1) exclusively to 2D weight matrices (linear layers and embeddings) while keeping 1D vectors (biases, LayerNorm scales) unpenalized.
- **Learning Rate Scheduling**: Linear warmup followed by cosine annealing with a minimum residual learning rate ratio (0.1).
- **Gradient Accumulation**: Enables effective batch sizes ($16 \times 4 = 64$) matching larger GPU architectures while staying strictly within consumer VRAM limits (RTX 3050 6 GB).
- **Automatic Mixed Precision (AMP)**: Uses PyTorch's native `torch.amp.autocast` and `GradScaler` for accelerated training and 50% activation memory savings.
- **Full Checkpointing**: Saves model weights, optimizer state, scaler, scheduler, step count, and loss metrics. Automatically maintains `best_model.pt` based on validation loss.

## Hyperparameters (training_config.yaml)

| Parameter | Value | Description |
|---|---|---|
| `batch_size` | 16 | Micro-batch size per forward pass |
| `gradient_accumulation_steps` | 4 | Effective batch size = $16 \times 4 = 64$ |
| `learning_rate` | 5.0e-4 | Peak learning rate |
| `min_lr_ratio` | 0.1 | Minimum learning rate (5.0e-5) |
| `weight_decay` | 0.1 | AdamW weight decay on 2D parameters |
| `grad_clip` | 1.0 | Gradient clipping maximum $L_2$ norm |
| `max_steps` | 10,000 | Total training steps |
| `warmup_steps` | 500 | Linear warmup steps |
| `eval_interval` | 250 | Steps between validation evaluations |
| `checkpoint_interval` | 1,000 | Steps between checkpoint saves |
| `mixed_precision` | True | Automatic Mixed Precision (AMP fp16) |

## Usage

### 1. Tokenize Dataset to Binary

```powershell
python scripts/tokenize_dataset.py
```

Generates:
- `data/tokenized/train.bin`
- `data/tokenized/val.bin`

### 2. Launch Pretraining

```powershell
# Standard training
python scripts/train.py

# Custom parameters
python scripts/train.py --batch-size 16 --max-steps 5000 --learning-rate 3e-4

# Resume from checkpoint
python scripts/train.py --resume checkpoints/checkpoint_step_1000.pt
```
