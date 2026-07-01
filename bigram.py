import torch
import torch.nn as nn
from torch.nn import functional as F

# ---- hyperparameters ----
batch_size = 32          # how many independent sequences we process in parallel
block_size = 8           # maximum context length for predictions
max_iters = 10000        # number of training steps
learning_rate = 1e-3
# -------------------------

torch.manual_seed(1337)  # reproducible runs

# ---- data ----
with open('Brothers_Karamazov.txt', 'r', encoding='utf-8') as f:
    karamazov_bros = f.read()

with open('Crime_and_Punishment.txt', 'r', encoding='utf-8') as f:
    raskolnikov = f.read()

text = '\n'.join([karamazov_bros, raskolnikov])

# all the unique characters that occur in the text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# character-level tokenizer
strtoint = {c: i for i, c in enumerate(chars)}
inttostr = {i: c for i, c in enumerate(chars)}
encode = lambda s: [strtoint[c] for c in s]          # string -> list of ints
decode = lambda li: ''.join(inttostr[i] for i in li)  # list of ints -> string

# train / validation split
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    """Grab a random batch of inputs (x) and targets (y)."""
    data = train_data if split == 'train' else val_data
    random_positions = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in random_positions])
    y = torch.stack([data[i + 1:i + 1 + block_size] for i in random_positions])
    return x, y


# ---- model ----
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)  # (B, T, C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :]                        # last time step -> (B, C)
            probs = F.softmax(logits, dim=-1)                # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)          # (B, T+1)
        return idx


def generate_answer(model, max_new_tokens=100):
    context = torch.zeros((1, 1), dtype=torch.long)
    print(decode(model.generate(context, max_new_tokens)[0].tolist()))


# ---- train ----
def main():
    model = BigramLanguageModel(vocab_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for step in range(max_iters):
        xb, yb = get_batch('train')

        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()   # figure out which direction each weight should move
        optimizer.step()  # nudge the weights that way

    print(f"final loss: {loss.item():.4f}")

    # sample some text from the trained model
    generate_answer(model, 300)


if __name__ == '__main__':
    main()
