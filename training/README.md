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

---

## Supervised Fine-Tuning (SFT)

Instruction tuning adapts a pretrained ForgeLM checkpoint into a conversational model using prompt-response pairs formatted with standardized role tags (`<|user|>` and `<|assistant|>`).

### Loss Masking
Cross-entropy loss is computed **exclusively** on the assistant's response tokens. The user instruction tokens and padding are masked using `ignore_index=0`, preventing the model from wasting gradient capacity memorizing user prompts.

### Fine-Tuning Hyperparameters (`finetune_config.yaml`)

| Parameter | Value | Description |
|---|---|---|
| `learning_rate` | 3.0e-5 | Reduced learning rate to prevent catastrophic forgetting |
| `batch_size` | 8 | Micro-batch size |
| `gradient_accumulation_steps` | 4 | Effective batch size = 32 |
| `max_steps` | 1,000 | Total SFT optimization steps |
| `warmup_steps` | 50 | Short warmup phase |
| `weight_decay` | 0.01 | Moderate weight decay |

### SFT CLI Usage

```powershell
# Launch fine-tuning with base model and instructions
python scripts/finetune.py --base-checkpoint checkpoints/best_model.pt --instruction-data data/instructions.json

# Interactive chat with fine-tuned model
python scripts/chat.py --checkpoint checkpoints/finetuned/finetuned_model.pt
```
