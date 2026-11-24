# Evaluation System

Quantitative evaluation engine and validation benchmark suite for ForgeLM.

## Metrics

The evaluation module tracks standard causal language modeling performance metrics:

1. **Cross-Entropy Loss**:
   $$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \log P(y_i \mid x_{<i})$$
2. **Perplexity (PPL)**: Exponential of cross-entropy loss measuring how well the model predicts the token distribution:
   $$\text{PPL} = \exp(\mathcal{L})$$
3. **Top-1 Accuracy**: Fraction of tokens where the highest-probability logit equals the ground-truth target token.
4. **Top-5 Accuracy**: Fraction of tokens where the ground-truth target appears within the 5 highest-probability logits.
5. **Bits-per-token (BPC / BPB)**: Information-theoretic compression metric in base 2:
   $$\text{BPC} = \frac{\mathcal{L}}{\ln(2)}$$

## Python API Usage

```python
import torch
from forgelm.evaluation import Evaluator
from forgelm.model import ForgeTransformer, ModelConfig
from forgelm.data import PretrainingDataset

# 1. Load model
config = ModelConfig.from_yaml("configs/model/model_config.yaml")
model = ForgeTransformer(config)

# 2. Load validation dataset
dataset = PretrainingDataset("data/tokenized/val.bin", seq_len=config.max_seq_len)

# 3. Evaluate
evaluator = Evaluator(model=model)
metrics = evaluator.evaluate(dataset, batch_size=16)

print(f"Validation Loss : {metrics['loss']:.4f}")
print(f"Perplexity      : {metrics['perplexity']:.2f}")
print(f"Top-1 Accuracy  : {metrics['top1_accuracy'] * 100:.2f}%")
print(f"Top-5 Accuracy  : {metrics['top5_accuracy'] * 100:.2f}%")
```

## CLI Checkpoint Evaluation

Evaluate any saved `.pt` checkpoint against the tokenized validation set:

```powershell
# Evaluate best checkpoint
python scripts/evaluate.py --checkpoint checkpoints/best_model.pt

# Evaluate specific checkpoint with batch limit for quick check
python scripts/evaluate.py --checkpoint checkpoints/checkpoint_step_1000.pt --max-batches 50
```
