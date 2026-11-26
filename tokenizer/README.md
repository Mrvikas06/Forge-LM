# Tokenizer Subsystem & Artifact Storage

The `tokenizer/` directory serves as the centralized, persistent artifact store for ForgeLM's custom Byte-Pair Encoding (BPE) subword tokenizer. It decouples the learned tokenization vocabulary from the model architecture, ensuring identical token boundaries across data preparation, model pretraining, fine-tuning, evaluation, and production inference.

---

## 1. Role & Purpose of the `tokenizer/` Folder

In transformer language models, neural attention and embedding layers operate on discrete integer indices ($0 \le i < V$), not raw text strings. The tokenizer subsystem is responsible for translating human-readable strings into numerical token vectors and reconstructing text from model output logits.

```
                    ┌─────────────────────────┐
                    │    Raw UTF-8 String     │
                    └────────────┬────────────┘
                                 │ encode()
                                 ▼
                    ┌─────────────────────────┐
                    │ Integer Token Sequence  │
                    │   e.g. [1, 260, 412, 2] │
                    └────────────┬────────────┘
                                 │ Embedding Lookup
                                 ▼
                    ┌─────────────────────────┐
                    │  Transformer Input d_m  │
                    └─────────────────────────┘
```

The `tokenizer/` folder stores the compiled subword rules in a versioned format (`tokenizer.json`), which is read by:
1. **Data Preprocessing**: `scripts/tokenize_dataset.py` to compile raw text corpora into binary memory-mapped files (`data/tokenized/train.bin`).
2. **Pretraining Engine**: `scripts/train.py` to configure the embedding matrix dimensions ($V = 4,096$).
3. **Instruction Tuning**: `scripts/finetune.py` to format `<|user|>` and `<|assistant|>` chat templates and mask prompt tokens.
4. **Evaluation**: `scripts/evaluate.py` to calculate accurate Perplexity ($\text{PPL}$) and Bits-Per-Character ($\text{BPC}$).
5. **Inference & Web API**: `src/forgelm/backend/service.py` and `scripts/generate.py` for real-time autoregressive text decoding.

---

## 2. Tokenizer Architecture & Vocabulary Structure

ForgeLM employs a custom **Byte-Pair Encoding (BPE)** algorithm with a byte-level fallback guarantee, implemented in [src/forgelm/tokenizer/bpe.py](file:///c:/Users/vikas/Documents/learn/FORGE/src/forgelm/tokenizer/bpe.py).

### Vocabulary Partitioning ($V = 4,096$)

| Range | Token Count | Description | Purpose |
| :--- | :---: | :--- | :--- |
| `0` | 1 | `<pad>` | Padding token for batch alignment |
| `1` | 1 | `<bos>` | Beginning-of-sequence delimiter |
| `2` | 1 | `<eos>` | End-of-sequence delimiter (triggers generation stop) |
| `3` | 1 | `<unk>` | Unknown token fallback (virtually unused due to byte fallback) |
| `4 – 259` | 256 | Base UTF-8 Bytes (`0x00 – 0xFF`) | Direct representation of every possible byte |
| `260 – 4095` | 3,836 | Learned BPE Merges | Frequent English words, morphemes, prefixes, and suffixes |

### Zero Out-of-Vocabulary (OOV) Guarantee
Because all 256 individual byte values are explicitly present in the base vocabulary (indices 4 through 259), the tokenizer can represent **any** arbitrary Unicode string (emojis, mathematical symbols, foreign scripts, code) without ever falling back to `<unk>` or throwing an error. Unseen characters decompose into their constituent UTF-8 bytes.

---

## 3. Storage Format: `tokenizer.json`

The file `tokenizer/tokenizer.json` (~192 KB) stores the complete state in JSON:

```json
{
  "vocab_size": 4096,
  "special_tokens": {
    "<pad>": 0,
    "<bos>": 1,
    "<eos>": 2,
    "<unk>": 3
  },
  "merges": [
    [108, 105, 260],
    [36, 120, 261],
    ...
  ]
}
```

- **`vocab_size`**: Integer defining the exact vocabulary upper bound.
- **`special_tokens`**: Dictionary mapping reserved symbol strings to fixed integer IDs.
- **`merges`**: Ordered list of triples `[left_id, right_id, new_id]` recording each iterative pair merge learned from the training corpus.

---

## 4. CLI Inspection & Verification

To inspect the tokenizer vocabulary, verify compression ratios, and test string segmentation:

```powershell
# Run the tokenizer inspection CLI
python scripts/inspect_tokenizer.py --text "Once upon a time in a sunny forest"

# Run roundtrip integrity tests
pytest tests/test_tokenizer.py -v
```
