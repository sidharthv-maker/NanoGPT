# NanoGPT

A minimal GPT implementation in PyTorch trained on Shakespeare's works to generate Shakespearean-style text.

## Architecture

- **Multi-head causal self-attention** with a lower-triangular mask
- **4 Transformer blocks**, each with attention + feed-forward (4× expansion) + LayerNorm
- **Token + positional embeddings** (learned)
- Character-level vocabulary (no tokenizer needed)

Default hyperparameters: `emb_dim=256`, `heads=4`, `seq_len=256`, 150 training epochs.

## Requirements

```
torch
```

## Usage

```bash
python model.py
```

The script will:
1. Load `data/shkp.txt` (the Shakespeare corpus)
2. Train for 150 epochs, printing loss each epoch
3. Prompt you for a seed string, then generate 500 characters of Shakespearean text

**Note:** The `device` variable on line 115 is currently commented out. Uncomment line 5 and the `.to(device)` / `x_batch.to(device)` calls to enable GPU training.

## Data

`data/shkp.txt` — the complete works of Shakespeare (included in the repo).
