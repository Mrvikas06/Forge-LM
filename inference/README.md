# Inference & Text Generation

Autoregressive text generation engine and stochastic sampling strategies for ForgeLM.

## Sampling Strategies

ForgeLM implements a customizable token sampling pipeline:

1. **Greedy Decoding** (`temperature=0.0`):
   Deterministically selects the argmax token with highest probability.
2. **Temperature Scaling** (`temperature > 0.0`):
   Scales logits before softmax ($z_i / T$). Temperatures $< 1.0$ concentrate probability on high-confidence tokens; temperatures $> 1.0$ encourage diversity.
3. **Top-K Truncation** (`top_k > 0`):
   Restricts candidate tokens to the $K$ highest probability values, masking all other logits to $-\infty$.
4. **Top-P (Nucleus) Sampling** (`0.0 < top_p < 1.0`):
   Filters tokens to the smallest set whose cumulative probability reaches threshold $P$, adapting candidate pool size based on model confidence.
5. **Repetition Penalty** (`repetition_penalty > 1.0`):
   Penalizes logits of tokens already generated in the context window, eliminating looping and degradation artifacts common in small language models.

## Python API Usage

```python
import torch
from forgelm.inference import TextGenerator
from forgelm.model import ForgeTransformer, ModelConfig
from forgelm.tokenizer import BPETokenizer

# 1. Load tokenizer and model
tokenizer = BPETokenizer.load("tokenizer/tokenizer.json")
config = ModelConfig.from_yaml("configs/model/model_config.yaml")
model = ForgeTransformer(config)

checkpoint = torch.load("checkpoints/best_model.pt", map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])

# 2. Instantiate generator
generator = TextGenerator(model=model, tokenizer=tokenizer)

# 3. Generate text
completion = generator.generate(
    prompt="Once upon a time, Lily found",
    max_new_tokens=100,
    temperature=0.8,
    top_k=50,
    top_p=0.9,
    repetition_penalty=1.1,
)
print(completion)

# 4. Stream tokens as they are generated
for token_chunk in generator.stream_generate(prompt="The puppy", max_new_tokens=50):
    print(token_chunk, end="", flush=True)
```

## CLI Usage

### One-Shot Generation

```powershell
python scripts/generate.py --checkpoint checkpoints/best_model.pt --prompt "Once upon a time" --temperature 0.7
```

### Interactive Streaming Mode

Run without `--prompt` to enter an interactive session:

```powershell
python scripts/generate.py --checkpoint checkpoints/best_model.pt
```
