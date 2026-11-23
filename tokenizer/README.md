# Tokenizer

Byte-Pair Encoding (BPE) tokenizer implemented from scratch with byte-level fallback.

## Vocabulary Details

- **Vocabulary Size**: 4,096 tokens
- **Special Tokens**:
  - `<pad>`: ID 0
  - `<bos>`: ID 1
  - `<eos>`: ID 2
  - `<unk>`: ID 3
- **Base Bytes**: IDs 4 – 259 (all 256 individual UTF-8 bytes)
- **Learned Merges**: IDs 260 – 4095 (3,836 BPE merge rules)
- **Zero OOV Guarantee**: Byte fallback ensures any arbitrary UTF-8 string can be encoded and decoded without `<unk>` loss.

## Artifacts

- `tokenizer/tokenizer.json`: Serialized vocabulary and merge rules (~192 KB).

## Usage

### Training

```powershell
# Train BPE tokenizer on processed corpus
python scripts/train_tokenizer.py --vocab-size 4096 --max-lines 50000
```

### Python API

```python
from forgelm.tokenizer import BPETokenizer

# Load trained tokenizer
tokenizer = BPETokenizer.load("tokenizer/tokenizer.json")

# Encode text with BOS/EOS
tokens = tokenizer.encode("Once upon a time, Lily found a bird.", add_bos=True, add_eos=True)
print(tokens)

# Decode tokens back to text
text = tokenizer.decode(tokens, skip_special_tokens=True)
print(text)
```
