# Training Subsystem Architecture

The `src/forgelm/training/` package, paired with `configs/training/`, implements the full pretraining and Supervised Fine-Tuning (SFT) engine for ForgeLM.

---

## 1. Role & Purpose of the Training Subsystem

Training a causal language model transforms randomly initialized weight matrices into a coherent probability distribution over sequences via next-token prediction:

$$\mathcal{L}_{\text{NLL}}(\theta) = -\frac{1}{T} \sum_{t=1}^T \log P(x_t \mid x_{<t}; \theta)$$

The training subsystem coordinates:
1. **Optimization**: AdamW optimizer with selective weight decay and numerical stabilization.
2. **Learning Rate Dynamics**: Warmup followed by cosine decay down to a defined minimum learning rate.
3. **Gradient Management**: Micro-batch gradient accumulation and global gradient norm clipping.
4. **Precision**: Automatic Mixed Precision (AMP `fp16`) for GPU acceleration with GradScaler.
5. **State Persistence**: Atomic checkpointing preserving model weights, optimizer state, step counter, and metadata.

---

## 2. Mathematical Components

### Cosine Annealing with Warmup

To prevent early divergence while weights are randomly oriented, the learning rate increases linearly from $0$ to $\eta_{\text{peak}}$ during $T_{\text{warmup}}$ steps, followed by a cosine decay:

$$\eta_t = \begin{cases} 
\eta_{\text{peak}} \cdot \frac{t}{T_{\text{warmup}}} & \text{if } t < T_{\text{warmup}} \\
\eta_{\text{min}} + \frac{1}{2}(\eta_{\text{peak}} - \eta_{\text{min}}) \left(1 + \cos\left(\frac{t - T_{\text{warmup}}}{T_{\text{max}} - T_{\text{warmup}}} \pi\right)\right) & \text{if } t \ge T_{\text{warmup}}
\end{cases}$$

### Selective Weight Decay ($L_2$ Regularization)

Applying weight decay to 1D vectors (biases and layer normalization scales) distorts network scale and harms convergence. ForgeLM explicitly splits parameters into two groups:
- **Decay Group** ($\lambda = 0.01$): All 2D weight matrices (attention projections, MLP dense layers, token embeddings).
- **No-Decay Group** ($\lambda = 0.0$): 1D parameters (RMSNorm scales, affine biases).

### Effective Batch Size

Gradient accumulation simulates large batch training without exceeding GPU VRAM limits:

$$B_{\text{effective}} = B_{\text{micro}} \times G_{\text{accum}} \times S_{\text{seq}}$$

With $B_{\text{micro}} = 32$, $G_{\text{accum}} = 4$, and $S_{\text{seq}} = 512$, each optimizer step processes:
$$32 \times 4 \times 512 = 65,536 \text{ tokens/step}$$

---

## 3. Package File Structure

- `src/forgelm/training/config.py`: `TrainingConfig` dataclass with YAML loading and validation.
- `src/forgelm/training/optim.py`: Parameter group builder and cosine learning rate scheduler.
- `src/forgelm/training/trainer.py`: Main `Trainer` loop managing micro-steps, AMP scaling, and validation.
- `configs/training/train_config.yaml`: Default hyperparameters for base pretraining.
- `configs/training/finetune_config.yaml`: Hyperparameters for instruction tuning.

---

## 4. Usage Commands

```powershell
# Pretrain base model from scratch
python scripts/train.py --config configs/training/train_config.yaml --max-steps 1000

# Supervised fine-tuning with prompt masking
python scripts/finetune.py --base-checkpoint checkpoints/best_model.pt --max-steps 200
```
