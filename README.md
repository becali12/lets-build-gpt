# build_your_gpt

A from-scratch language model trained on the works of Fyodor Dostoevsky.

This is a learning project that follows the "build a GPT from scratch" path: start
with a simple character-level bigram model and grow it toward a full transformer.
The training corpus is two of Dostoevsky's novels - *The Brothers Karamazov* and
*Crime and Punishment* - so the model learns to babble in his (translated) voice.

## What's here

- `bigram.py` - a self-contained baseline: a character-level bigram language model
  built with PyTorch, plus its training loop and a text sampler.
- `my-gpt.ipynb` - the exploratory notebook where the pieces are developed step by step.
- `Brothers_Karamazov.txt`, `Crime_and_Punishment.txt` - the raw training text
  (Constance Garnett translations).

## How it works

1. **Tokenize.** The two novels are concatenated and turned into a character-level
   vocabulary - every unique character becomes a token.
2. **Split.** 90% of the text is used for training, 10% held out for validation.
3. **Train.** Random blocks of text (context length 8, batch size 32) are fed to a
   `BigramLanguageModel`, which learns to predict the next character. Optimized with
   AdamW for 10k steps.
4. **Generate.** Starting from a seed token, the model samples one character at a time
   to produce Dostoevsky-flavored text.

## Running it

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python bigram.py
```

This trains the bigram model and prints a 300-character sample of generated text.

## Roadmap

The bigram model is intentionally the simplest possible starting point. The plan is to
build up from here toward a full GPT - self-attention, multiple heads, feed-forward
layers, and positional embeddings - one component at a time.
