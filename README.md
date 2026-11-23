# ForgeLM

An educational GPT-style language model built entirely from random initialization. The goal is to implement and understand every stage of the modern language model pipeline from first principles without relying on pretrained weights.

---

## Overview

ForgeLM implements a complete causal language model pipeline from scratch, prioritizing correctness, reproducibility, modular design, and execution within consumer GPU memory limits.

## Project Goals

- Build a custom tokenizer from scratch (Byte-Pair Encoding).
- Implement decoder-only Transformer components (causal self-attention, feed-forward layers, normalization, positional embeddings).
- Train a language model from random initialization on a curated corpus.
- Implement training optimization (learning rate scheduling, gradient accumulation, mixed precision).
- Evaluate model quality using cross-entropy loss and perplexity.
- Implement autoregressive text generation with temperature, top-k, and top-p sampling.
- Implement KV caching for accelerated generation.
- Explore instruction tuning and alignment.
- Serve inference via a lightweight FastAPI backend and interactive frontend.

## Pipeline Architecture

```
Raw Dataset
  -> Data Processing
    -> Tokenizer
      -> Tokenized Dataset
        -> Transformer
          -> Pretraining
            -> Evaluation
              -> Generation
                -> Instruction Tuning
                  -> Inference
```

## Hardware and Environment

- Operating System: Windows 11
- GPU: NVIDIA GeForce RTX 3050 (6 GB VRAM)
- RAM: 32 GB
- Python: 3.11+
- Framework: PyTorch

All architecture and training hyperparameters are tuned for execution within a 6 GB VRAM envelope using mixed precision and gradient accumulation.

## Project Status

- **Phase 1: Project Foundation and Environment Setup** - Complete
- **Phase 1.5: Dataset Collection and Preprocessing** - Complete (TinyStories corpus integrated)
- **Phase 2: Custom BPE Tokenizer** - Complete (4,096 vocab Byte-level BPE)
- **Phase 2.5: Transformer Architecture** - Next

## Quickstart

### 1. Clone and Create Virtual Environment

```powershell
git clone https://github.com/Mrvikas06/Forge-LM.git
cd ForgeLM
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 2. Install PyTorch with CUDA Support

Install the official CUDA-enabled PyTorch wheel matching your local driver (example for CUDA 12.1):

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install Dependencies and ForgeLM Package

```powershell
pip install -r requirements.txt
pip install -e .
```

### 4. Verify GPU and Environment

```powershell
python test_gpu.py
```

### 5. Run Test Suite

```powershell
pytest
```

## Dataset Pipeline

ForgeLM uses the **TinyStories** corpus (Microsoft Research), which provides a constrained English vocabulary suited for sub-30M parameter models trained on consumer hardware.

### Prepare Processed Data

```powershell
# Process full dataset into clean text splits
python scripts/prepare_data.py

# Or process a small subset for rapid debugging
python scripts/prepare_data.py --max-samples 10000
```

Raw data and processed text outputs are stored under `data/` and excluded from Git.

## Project Structure

```
ForgeLM/
├── configs/              # Declarative YAML configurations
│   ├── data/             #   Dataset and preprocessing configuration
│   ├── model/            #   Transformer architecture parameters
│   └── training/         #   Optimization and training parameters
├── data/                 # Data directory (ignored by git)
│   ├── raw/              #   Raw downloaded files
│   ├── processed/        #   Cleaned plain-text splits
│   └── tokenized/        #   Pre-tokenized binary/tensor arrays
├── tokenizer/            # Tokenizer training and serialization
├── model/                # Transformer architecture implementation
├── training/             # Pretraining engine and optimization loop
├── evaluation/           # Evaluation metrics and perplexity benchmarks
├── inference/            # Text generation engine and KV cache
├── backend/              # FastAPI model server
├── frontend/             # Interactive user interface
├── experiments/          # Run logs and hyperparameter records
├── checkpoints/          # Model weights (ignored by git)
├── logs/                 # Training logs (ignored by git)
├── scripts/              # Data processing and maintenance scripts
├── tests/                # Automated pytest suite
├── docs/                 # Project documentation
├── notebooks/            # Exploratory analysis notebooks
├── src/forgelm/          # Core Python package
│   ├── config/           #   Configuration loader
│   ├── data/             #   Text cleaning and data processing
│   └── utils/            #   Device detection, seed, and logging
├── test_gpu.py           # Hardware verification script
├── pyproject.toml        # Package metadata and build definition
├── requirements.txt      # Project dependencies
├── .gitignore
├── .env.example
├── LICENSE
└── README.md
```

## Development Roadmap

| Phase | Description | Status |
|---|---|---|
| 1 | Project foundation, packaging, and testing setup | Complete |
| 1.5 | Dataset collection and text preprocessing | Complete |
| 2 | Byte-Pair Encoding (BPE) tokenizer from scratch | Complete |
| 2.5 | Decoder-only Transformer architecture | Next |
| 3 | Pretraining engine with mixed precision | Planned |
| 4 | Evaluation metrics and validation benchmarks | Planned |
| 5 | Autoregressive generation and sampling strategies | Planned |
| 6 | KV cache inference optimization | Planned |
| 7 | Instruction tuning and fine-tuning pipeline | Planned |
| 8 | FastAPI backend service | Planned |
| 9 | Interactive web frontend | Planned |

## License

MIT License. See [LICENSE](LICENSE) for details.
