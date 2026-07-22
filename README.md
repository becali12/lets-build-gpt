# Let's Build GPT

A from-scratch character-level Transformer language model trained on the works of Fyodor Dostoevsky.

This project follows [Andrej Karpathy's "Let's build GPT from scratch"](https://www.youtube.com/watch?v=kCc8FmEb1nY) video tutorial. The code and architecture are based on his teaching.

## What's here

- `bigram.py` — a complete GPT-style Transformer: multi-head causal self-attention, feed-forward layers, residual connections, LayerNorm, and dropout. Includes a training loop and text generation that writes output to a file.
- `my-gpt.ipynb` — the exploratory notebook where the initial components were developed step by step.
- `Brothers_Karamazov.txt`, `Crime_and_Punishment.txt` — the raw training text.

## Architecture

The model (`BigramLanguageModel`) is a decoder-only Transformer with:

- **6 layers** of Transformer blocks
- **6 attention heads** per block (head size = 64)
- **384-dimensional** token and position embeddings
- Feed-forward layers with 4x inner expansion and ReLU activation
- Pre-norm LayerNorm and 20% dropout for regularization
- Character-level tokenization (vocab ≈ 83 unique characters)

## How it works

1. **Tokenize.** The two novels are concatenated and turned into a character-level vocabulary — every unique character becomes a token.
2. **Split.** 90% of the text is used for training, 10% held out for validation.
3. **Train.** Random blocks of text (context length 128, batch size 64) are fed to the Transformer, which learns to predict the next character. Optimized with AdamW for 5,000 steps.
4. **Generate.** After training, the model generates 10,000 characters of Dostoevsky-flavored text and saves it to `generated_text.txt`.

## Running it

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python bigram.py
```

Training runs on CUDA or Apple Silicon GPU (MPS) if available, otherwise falls back to CPU. On a base M4 Mac Mini, expect training to complete in roughly 30 minutes.

## Credits

Based on [Andrej Karpathy's "Let's build GPT"](https://www.youtube.com/watch?v=kCc8FmEb1nY) tutorial and his [nanoGPT](https://github.com/karpathy/nanoGPT) repository. All credit for the architecture and teaching goes to him.
