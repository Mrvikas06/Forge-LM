# ForgeLM Unified System Architecture

This document provides the end-to-end architectural blueprint of ForgeLM, detailing how the `tokenizer/`, `training/`, `evaluation/`, and `experiments/` subsystems interconnect to form a production-grade, educational causal language model pipeline.

---

## 1. Subsystem Interaction Map

```
┌────────────────────────────────────────────────────────────────────────┐
│                          RAW DATASET CORPUS                            │
│                  data/raw/ & data/instructions.json                    │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 1. TOKENIZER SUBSYSTEM (tokenizer/ & src/forgelm/tokenizer/)           │
│    • Vocabulary: 4,096 tokens (4 Special + 256 Bytes + 3,836 Merges)  │
│    • Zero Out-of-Vocabulary (OOV) byte-level fallback guarantee        │
│    • Artifact: tokenizer/tokenizer.json                                │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Tokenized Stream
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 2. TRAINING SUBSYSTEM (src/forgelm/training/ & configs/training/)      │
│    • Architecture: 12.5M Parameter Decoder (L=6, H=6, D=384, V=4096)  │
│    • Optimizer: AdamW with 2D/1D selective weight decay splitting     │
│    • Schedule: Linear Warmup + Cosine Annealing decay                 │
│    • Features: AMP fp16, Gradient Accumulation, Early Stopping        │
│    • Output: checkpoints/final_model.pt & checkpoints/best_model.pt    │
└──────────────┬──────────────────────────────────────────┬──────────────┘
               │ Per-Step Metrics                         │ Model Weights
               ▼                                          ▼
┌──────────────────────────────────────┐  ┌──────────────────────────────┐
│ 3. EXPERIMENT TRACKING               │  │ 4. EVALUATION SUBSYSTEM      │
│    (experiments/ & src/.../tracker/) │  │    (src/forgelm/evaluation/) │
│    • Streaming JSONL metrics logs    │  │    • Perplexity (PPL)        │
│    • Isolated timestamped run dirs   │  │    • Bits-Per-Character (BPC)│
│    • Summary & Matplotlib curves     │  │    • Top-1 & Top-5 Accuracies│
│    • Multi-run comparison CLI        │  │    • Distinct-N Diversity    │
└──────────────────────────────────────┘  └──────────────┬───────────────┘
                                                         │
                                                         ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 5. INFERENCE & STUDIO UI (src/forgelm/backend/ & frontend/)            │
│    • REST endpoints: /generate, /chat, /health, /model/info, /diagnostics│
│    • Streaming endpoints: /generate/stream, /chat/stream (SSE)         │
│    • O(N) incremental decoding with KV Cache                           │
│    • Curated Dialogue Engine for instant conversational replies        │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Directory Matrix & Responsibilities

| Subsystem Folder | Primary Modules | Key Artifacts | Role in Pipeline |
| :--- | :--- | :--- | :--- |
| **`tokenizer/`** | `bpe.py`, `special_tokens.py` | `tokenizer.json` | Converts text to/from subword IDs with zero OOV loss. |
| **`training/`** | `trainer.py`, `optim.py`, `config.py` | `train_config.yaml`, `finetune_config.yaml` | Executes optimization loops, cosine schedules, and checkpointing. |
| **`evaluation/`** | `evaluator.py`, `metrics.py` | `evaluate_checkpoint.py` | Unbiased statistical validation ($\text{PPL}$, $\text{BPC}$, Accuracy). |
| **`experiments/`** | `tracker.py`, `visualizer.py` | `metrics.jsonl`, `summary.json`, `loss_curves.png` | Run logging, metric streaming, and comparative visualization. |

---

## 3. Subsystem Deep Dive

### I. Tokenizer Subsystem
- **Why It Exists**: Language models require numerical sequences. Character-level tokenization produces overly long contexts, while word-level tokenization creates massive vocabularies with frequent out-of-vocabulary errors. BPE balances context efficiency with manageable matrix sizes ($V=4096$).
- **Byte Fallback**: Indices 4..259 represent raw byte values `0x00` through `0xFF`. Unseen characters decompose into raw UTF-8 bytes, guaranteeing that no input ever causes an error or yields `<unk>`.

### II. Training Subsystem
- **Why It Exists**: Pretraining trains the model to minimize cross-entropy loss over billions of tokens, instilling grammar, facts, and reasoning. SFT aligns the pretrained model to instruction prompts using prompt-loss masking.
- **Selective Weight Decay**: 2D weight matrices (linear layers, projections, embeddings) receive $L_2$ decay ($\lambda = 0.01$), while 1D tensors (RMSNorm gains, biases) receive $\lambda = 0.0$ to prevent scale collapse.

### III. Evaluation Subsystem
- **Why It Exists**: Evaluates models on held-out data with no gradient tracking (`torch.no_grad()`).
- **Key Metrics**:
  - **Perplexity ($\text{PPL}$)**: Measures model uncertainty per token ($e^{\mathcal{L}}$).
  - **Bits-Per-Character ($\text{BPC}$)**: Normalizes loss across different tokenizers ($\frac{\mathcal{L}}{\ln 2}$).
  - **Top-1 / Top-5 Accuracy**: Measures exact prediction rate.
  - **Distinct-N**: Assesses generation lexical richness and detects repetitive loops.

### IV. Experiments Subsystem
- **Why It Exists**: Machine learning is deeply empirical. The `experiments/runs/` registry records every hyperparameter configuration, streaming loss, throughput, and final validation loss for every training attempt.
- **Crash Recovery**: Streaming `.jsonl` files guarantee that if a machine reboots or runs out of memory, all steps up to the interruption are preserved.
