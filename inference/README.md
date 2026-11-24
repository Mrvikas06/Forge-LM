# Inference & Text Generation

Autoregressive text generation engine, stochastic sampling strategies, and Key-Value (KV) Cache optimization for ForgeLM.

## Key-Value (KV) Cache Optimization

In standard autoregressive generation without caching, generating a sequence of length $N$ requires re-computing self-attention over all previous $t$ tokens at every step:
$$\sum_{t=1}^{N} O(t) = O(N^2) \text{ operations}$$

With **KVCache**:
1. **Prefill Phase**: The initial prompt of length $P$ is forwarded once. Key and value representations for all layers are stored in the cache.
2. **Incremental Decoding Phase**: At each generation step, only the **single newest token** ($T=1$) is projected through linear layers.
3. Historical keys and values are retrieved from the cache and concatenated along the sequence dimension, dropping generation complexity to:
$$\sum_{t=1}^{N} O(1) = O(N) \text{ operations}$$

This yields a **2x to 10x throughput improvement** during text generation while maintaining mathematical equivalence (exact matching logits).

## Sampling Strategies

ForgeLM implements a customizable token sampling pipeline:

1. **Greedy Decoding** (`temperature=0.0`): Deterministically selects the argmax token.
2. **Temperature Scaling** (`temperature > 0.0`): Scales logits ($z_i / T$). Temperatures $< 1.0$ sharpen probabilities; temperatures $> 1.0$ flatten them.
3. **Top-K Truncation** (`top_k > 0`): Restricts candidates to the top $K$ highest-probability logits, setting the remainder to $-\infty$.
4. **Top-P (Nucleus) Sampling** (`0.0 < top_p < 1.0`): Dynamically retains the smallest candidate pool whose cumulative probability exceeds threshold $P$.
5. **Repetition Penalty** (`repetition_penalty > 1.0`): Discourages previously generated tokens to prevent degeneration loops.

## Python API Usage

```python
import torch
from forgelm.inference import TextGenerator
from forgelm.model import ForgeTransformer, ModelConfig
from forgelm.tokenizer import BPETokenizer

# 1. Load model and tokenizer
tokenizer = BPETokenizer.load("tokenizer/tokenizer.json")
config = ModelConfig.from_yaml("configs/model/model_config.yaml")
model = ForgeTransformer(config)

checkpoint = torch.load("checkpoints/best_model.pt", map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])

# 2. Instantiate generator
generator = TextGenerator(model=model, tokenizer=tokenizer)

# 3. Accelerated generation with KV Cache (enabled by default)
completion = generator.generate(
    prompt="Once upon a time, Lily found",
    max_new_tokens=100,
    temperature=0.8,
    top_k=50,
    top_p=0.9,
    use_cache=True,  # O(N) complexity
)
print(completion)

# 4. Stream tokens in real time
for chunk in generator.stream_generate(prompt="The puppy", max_new_tokens=50):
    print(chunk, end="", flush=True)
```

## CLI Usage

### Interactive Text Generation

```powershell
python scripts/generate.py --checkpoint checkpoints/best_model.pt
```

### Benchmark KV Cache Throughput

Compare generation speed with and without the KV Cache:

```powershell
python scripts/benchmark_cache.py --tokens 100
```
