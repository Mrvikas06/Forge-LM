# ForgeLM

An educational GPT-style language model built entirely from random initialization. The goal is to understand and implement every component of the modern LLM pipeline -- from raw data to interactive text generation.

---

## Overview

ForgeLM is a hands-on learning project that walks through building a small language model from scratch. No pretrained weights are used. Every piece of the pipeline is implemented, trained, and evaluated on consumer hardware.

This project is not about achieving state-of-the-art results. It is about deeply understanding how language models work by building one yourself.

## Project Goals

- Build a custom tokenizer from scratch (BPE or similar).
- Implement Transformer components: attention, feed-forward layers, embeddings, normalization.
- Train a language model from random initialization on a real dataset.
- Understand training optimization: learning rate scheduling, gradient clipping, mixed precision.
- Evaluate model performance with perplexity and loss tracking.
- Implement autoregressive text generation with sampling strategies.
- Explore instruction tuning and alignment techniques.
- Develop a lightweight inference system with KV cache.
- Serve the model through a FastAPI backend and interactive frontend.

## Pipeline Architecture

```
Raw Dataset
  -> Data Processing
    -> Tokenizer
      -> Tokenized Dataset
        -> Transformer Model
          -> Pretraining
            -> Evaluation
              -> Text Generation
                -> Instruction Tuning
                  -> Inference Engine
```

## Hardware

This project is developed and tested on:

| Component | Specification |
|---|---|
| OS | Windows 11 |
| GPU | NVIDIA RTX 3050 (6 GB VRAM) |
| RAM | 32 GB |
| Python | 3.11 |
| Framework | PyTorch |

All design decisions account for limited GPU memory. Mixed precision training, gradient accumulation, and efficient batching will be used to stay within the 6 GB VRAM budget.

## Project Status

**Current phase: Foundation and Environment Setup**

The project structure, development tooling, configuration system, and test infrastructure are in place. No model, tokenizer, or training code has been implemented yet.

## Installation

### 1. Clone the repository

```powershell
git clone https://github.com/Mrvikas06/ForgeLM.git
cd ForgeLM
```

### 2. Create a virtual environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install PyTorch with CUDA support

Check your CUDA version and install the matching PyTorch build. For CUDA 12.1:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

For other CUDA versions, see: https://pytorch.org/get-started/locally/

### 4. Install remaining dependencies

```powershell
pip install -r requirements.txt
```

### 5. Install ForgeLM as a local package

```powershell
pip install -e .
```

### 6. Verify installation

```powershell
python -c "import forgelm; print(forgelm.__version__)"
```

## GPU Verification

Run the GPU test script to confirm your CUDA setup:

```powershell
python test_gpu.py
```

This will report your GPU name, available memory, and run a basic compute test. If CUDA is not available, it will print instructions for installing a CUDA-compatible PyTorch build.

## Running Tests

```powershell
pytest
```

## Project Structure

```
ForgeLM/
├── configs/              # YAML configuration files
│   ├── model/            #   Model architecture config
│   ├── training/         #   Training hyperparameter config
│   └── data/             #   Dataset config
├── data/                 # Dataset storage (excluded from Git)
│   ├── raw/
│   ├── processed/
│   └── tokenized/
├── tokenizer/            # Custom tokenizer (Phase 2)
├── model/                # Transformer implementation (Phase 2)
├── training/             # Training engine (Phase 3)
├── evaluation/           # Evaluation and metrics (Phase 4)
├── inference/            # Generation and inference (Phase 5+)
├── backend/              # FastAPI server (Phase 8)
├── frontend/             # Web interface (Phase 9)
├── experiments/          # Experiment tracking
├── checkpoints/          # Saved model weights (excluded from Git)
├── logs/                 # Training logs (excluded from Git)
├── scripts/              # Utility scripts
├── tests/                # Pytest test suite
├── docs/                 # Documentation
├── notebooks/            # Jupyter notebooks
├── src/forgelm/          # Python package
│   ├── config/           #   Configuration loading
│   └── utils/            #   Device, seed, logging utilities
├── test_gpu.py           # GPU verification script
├── pyproject.toml        # Python packaging
├── requirements.txt      # Dependencies
├── .gitignore
├── .env.example
├── LICENSE
└── README.md
```

## Roadmap

| Phase | Description | Status |
|---|---|---|
| 1 | Project foundation and environment setup | Complete |
| 1.5 | Dataset collection and preprocessing | Planned |
| 2 | Tokenizer and model architecture | Planned |
| 3 | Pretraining engine | Planned |
| 4 | Evaluation system | Planned |
| 5 | Text generation | Planned |
| 6 | Instruction tuning | Planned |
| 7 | Inference engine with KV cache | Planned |
| 8 | FastAPI backend | Planned |
| 9 | Interactive frontend | Planned |
| 10 | Experiment tracking and analysis | Planned |

## License

MIT License. See [LICENSE](LICENSE).
