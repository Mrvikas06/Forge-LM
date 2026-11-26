# Evaluation Subsystem Architecture

The `src/forgelm/evaluation/` package provides evaluation protocols and statistical metrics for assessing ForgeLM checkpoints without data leakage.

---

## 1. Role & Purpose of the Evaluation Subsystem

Training loss only reflects how well a model memorizes the training data. The evaluation subsystem assesses generalization on unseen validation splits (`data/tokenized/val.bin` or validation instruction pairs) using standardized statistical metrics.

All evaluation routines run with:
- `torch.no_grad()`: Disables autograd graph construction, saving ~60% VRAM.
- `model.eval()`: Deactivates dropout layers for deterministic evaluation.

---

## 2. Mathematical Metric Definitions

### Cross-Entropy Validation Loss ($\mathcal{L}$)
Mean negative log-likelihood across all $N$ validation tokens:

$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^N \log P(y_i \mid x_{<i})$$

### Perplexity ($\text{PPL}$)
Perplexity represents the effective branching factor of the language model—the number of equally probable choices the model is choosing between at each step:

$$\text{PPL} = \exp(\mathcal{L}) = e^{\mathcal{L}}$$

- A lower perplexity indicates higher certainty and better linguistic modeling.
- A random model on a 4,096-token vocabulary has $\text{PPL} \approx 4,096$.
- A well-trained language model on TinyStories achieves $\text{PPL} < 25$.

### Bits-Per-Character ($\text{BPC}$)
Because tokenizers have different vocabulary sizes, Perplexity cannot be compared across models with different tokenizers. **Bits-Per-Character** normalizes the cross-entropy loss to the underlying UTF-8 text length:

$$\text{BPC} = \frac{\mathcal{L}}{\ln(2)} \times \frac{N_{\text{tokens}}}{N_{\text{characters}}}$$

### Top-1 & Top-k Next-Token Accuracy
- **Top-1 Accuracy**: Fraction of tokens where the model's top prediction ($\arg\max$) matches the ground truth.
- **Top-5 Accuracy**: Fraction of tokens where the ground truth is present within the model's top 5 logit candidates.

### Generation Diversity Metrics (Distinct-N)
Measures vocabulary richness and repetition in generated sequences:

$$\text{Distinct-N} = \frac{|\text{Unique } n\text{-grams}|}{|\text{Total } n\text{-grams}|}$$

---

## 3. Package File Structure

- `src/forgelm/evaluation/metrics.py`: Mathematical implementations of $\text{PPL}$, $\text{BPC}$, Accuracy, and Distinct-N.
- `src/forgelm/evaluation/evaluator.py`: `Evaluator` class managing validation batches, loss accumulation, and metrics aggregation.
- `scripts/evaluate_checkpoint.py`: Standalone CLI utility for evaluating checkpoints.

---

## 4. Usage Commands

```powershell
# Evaluate best checkpoint on validation dataset
python scripts/evaluate_checkpoint.py --checkpoint checkpoints/best_model.pt --max-batches 50
```
