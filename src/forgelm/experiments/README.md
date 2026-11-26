# Experiment Tracking & Visualizer Subsystem

The `src/forgelm/experiments/` package and `experiments/` storage directory implement ForgeLM's experiment logging, metrics streaming, run comparison, and loss visualization system.

---

## 1. Role & Purpose of the `experiments/` Folder

Developing deep learning models requires empirical verification across learning rates, batch sizes, model dimensions, and optimization strategies. The `experiments/` folder acts as an immutable run registry, guaranteeing 100% reproducibility.

```
experiments/
├── README.md
└── runs/
    ├── pretrain_20261125_205428/
    │   ├── config.json          <- Hyperparameters & model specs
    │   ├── metrics.jsonl         <- Streaming per-step metric logs
    │   └── summary.json          <- Final run duration & best losses
    └── finetune_20261125_212306/
        ├── config.json
        ├── metrics.jsonl
        └── summary.json
```

---

## 2. File Specifications Inside Each Run Directory

### 1. `config.json`
Recorded at step 0 before any training commences. Captures:
- Model architecture specifications (`vocab_size`, `d_model`, `n_layers`, `n_heads`, `max_seq_len`).
- Training hyperparameters (`learning_rate`, `batch_size`, `gradient_accumulation_steps`, `grad_clip`).
- Hardware environment (`device`, `gpu_name`, `timestamp`).

### 2. `metrics.jsonl`
Append-only, newline-delimited JSON streaming file. Emitted after every micro-step without locking the training loop:

```json
{"type": "step", "step": 10, "train_loss": 3.5485, "lr": 0.000066, "tok_per_sec": 2310.4, "elapsed_seconds": 3.2}
{"type": "eval", "step": 50, "val_loss": 2.4120, "perplexity": 11.15, "elapsed_seconds": 15.8}
```

- **Crash-Resilient**: Because each step appends a new line, interrupted or killed training runs still retain complete metrics up to the exact crash point.
- **Low Overhead**: Simple JSON appending requires zero network requests and minimal I/O overhead.

### 3. `summary.json`
Emitted at the conclusion of training. Captures total wall-clock duration, total tokens processed, and best validation metrics.

---

## 3. Package File Structure

- `src/forgelm/experiments/tracker.py`: `ExperimentTracker` class managing run directory initialization and JSONL serialization.
- `src/forgelm/experiments/visualizer.py`: Plotting engine producing high-resolution Matplotlib loss curves and ablation bar charts.
- `scripts/compare_runs.py`: CLI comparator scanning all runs in `experiments/runs/` and generating markdown comparison tables.

---

## 4. Usage Commands

```powershell
# Compare and rank all past experiment runs
python scripts/compare_runs.py

# Plot training loss curve for a specific run
python scripts/compare_runs.py --plot-run finetune_20261125_212306 --output docs/finetune_loss.png
```
