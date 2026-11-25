# ForgeLM Experiment Tracking & Analysis

Structured experiment management, loss curve plotting, and generation ablation benchmarks for **ForgeLM**.

---

## Architecture & Features

- **JSONL Streaming Telemetry**: `ExperimentTracker` streams per-step metrics (`train_loss`, `lr`, `tok_per_sec`, `vram_mb`) and evaluation metrics (`val_loss`, `perplexity`) to newline-delimited JSON without holding metrics in RAM.
- **Optional TensorBoard Synchronization**: Automatically detects and synchronizes scalar curves to `SummaryWriter` if installed.
- **Publication-Grade Visualizations**: Built-in `plot_loss_curves` generates dual-axis charts depicting raw loss, simple moving average (SMA) trends, validation cross-entropy, and perplexity dynamics using `matplotlib`.
- **Ablation & Benchmarking Suite**: Systematic performance profiling comparing decoding latency with and without KV caching, across temperature schedules, and sequence lengths.

---

## Directory Structure

```
experiments/
├── README.md
├── runs/
│   └── <run_name>_<timestamp>/
│       ├── config.json         # Run metadata, hardware specs, hyperparameters
│       ├── metrics.jsonl       # Streaming step and evaluation records
│       ├── summary.json        # Final metrics, best val loss, duration
│       └── loss_curves.png     # Rendered loss and perplexity curves
└── reports/
    ├── generation_benchmark.json
    ├── ablation_report.md
    └── kv_cache_speedup.png
```

---

## Usage

### 1. Training with Experiment Tracking

Pretraining and fine-tuning automatically log experiments to `experiments/runs/`:

```powershell
# Pretraining with custom experiment name
python scripts/train.py --experiment-name pretrain_tinystories

# Fine-tuning with tracking
python scripts/finetune.py --experiment-name sft_dialogue
```

### 2. Plotting Loss Curves

Render publication-ready loss and perplexity plots from logged metrics:

```powershell
# Plot latest run
python scripts/plot_metrics.py

# Plot specific run directory
python scripts/plot_metrics.py --run-dir experiments/runs/pretrain_tinystories_20261125_120000
```

### 3. Generation & KV Cache Ablation Benchmark

Benchmark decoding throughput ($O(N)$ vs $O(N^2)$) and temperature sampling:

```powershell
python scripts/benchmark_generation.py
```

Outputs:
- Markdown performance table at `experiments/reports/ablation_report.md`
- Throughput speedup bar chart at `experiments/reports/kv_cache_speedup.png`
- Detailed metrics JSON at `experiments/reports/generation_benchmark.json`
