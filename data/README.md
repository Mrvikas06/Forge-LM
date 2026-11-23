# Data Directory

Stores datasets at every stage of the pipeline.

| Subdirectory | Purpose |
|---|---|
| `raw/` | Original downloaded datasets (unmodified). |
| `processed/` | Cleaned and preprocessed text data. |
| `tokenized/` | Tokenized and chunked data ready for training. |

Populated during **Phase 1.5: Dataset Collection and Processing**.

Data files are excluded from Git via `.gitignore`.
