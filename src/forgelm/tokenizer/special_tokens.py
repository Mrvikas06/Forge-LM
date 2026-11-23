"""Special token constants and configuration."""

PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"

DEFAULT_SPECIAL_TOKENS = {
    PAD_TOKEN: 0,
    BOS_TOKEN: 1,
    EOS_TOKEN: 2,
    UNK_TOKEN: 3,
}

# Number of special tokens reserved at the beginning of the vocabulary
NUM_SPECIAL_TOKENS = len(DEFAULT_SPECIAL_TOKENS)

# Number of standard byte tokens (0..255)
NUM_BYTE_TOKENS = 256

# Base vocabulary size (special tokens + raw bytes)
BASE_VOCAB_SIZE = NUM_SPECIAL_TOKENS + NUM_BYTE_TOKENS
