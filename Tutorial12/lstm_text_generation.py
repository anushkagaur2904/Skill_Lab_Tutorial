from pathlib import Path
import random
import string

import torch
from torch import nn


DATA_FILE = Path(__file__).with_name("data.txt")


def load_text():
    text = DATA_FILE.read_text(encoding="utf-8")
    text = " ".join(text.split())
    if len(text) < 500:
        raise ValueError("data.txt should contain at least 500 characters.")
    return text


def make_sequences(encoded, seq_length):
    inputs, targets = [], []
    for i in range(len(encoded) - seq_length):
        inputs.append(encoded[i : i + seq_length])
        targets.append(encoded[i + seq_length])
    return torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long)


class CharLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, hidden_size=128, layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_size, num_layers=layers, batch_first=True
        )
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        output, hidden = self.lstm(x, hidden)
        return self.fc(output[:, -1, :]), hidden


def generate(model, seed, char_to_idx, idx_to_char, length=300, temperature=0.8):
    model.eval()
    chars = list(seed)
    hidden = None

    with torch.no_grad():
        # Feed the complete seed through the LSTM first so the hidden state
        # represents the whole starting phrase.
        seed_ids = torch.tensor(
            [[char_to_idx.get(ch, 0) for ch in seed]], dtype=torch.long
        )
        logits, hidden = model(seed_ids, hidden)

        for _ in range(length):
            input_ids = torch.tensor(
                [[char_to_idx.get(chars[-1], 0)]], dtype=torch.long
            )
            logits, hidden = model(input_ids, hidden)
            probabilities = torch.softmax(logits / temperature, dim=-1).squeeze()
            next_id = torch.multinomial(probabilities, 1).item()
            chars.append(idx_to_char[next_id])

    return "".join(chars)


def main():
    random.seed(42)
    torch.manual_seed(42)

    text = load_text()
    chars = sorted(set(text + string.whitespace))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for ch, i in char_to_idx.items()}
    encoded = [char_to_idx[ch] for ch in text]

    # Hyper-parameter experiment: sequence length and LSTM units.
    sequence_length = 60
    hidden_size = 128
    epochs = 20
    learning_rate = 0.002

    X, y = make_sequences(encoded, sequence_length)
    model = CharLSTM(len(chars), hidden_size=hidden_size)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print(f"Characters: {len(text)}")
    print(f"Vocabulary size: {len(chars)}")
    print(f"Sequences: {len(X)}")
    print(
        f"Hyper-parameters: sequence_length={sequence_length}, "
        f"hidden_size={hidden_size}, epochs={epochs}"
    )

    batch_size = 128
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(X))
        total_loss = 0.0

        for start in range(0, len(X), batch_size):
            idx = order[start : start + batch_size]
            batch_x, batch_y = X[idx], y[idx]

            optimizer.zero_grad()
            logits, _ = model(batch_x)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_x)

        average_loss = total_loss / len(X)
        if epoch == 1 or epoch % 2 == 0:
            print(f"Epoch {epoch:02d}/{epochs} - Loss: {average_loss:.4f}")

    seed = "Generative AI"
    generated = generate(
        model, seed, char_to_idx, idx_to_char, length=300, temperature=0.8
    )

    print("\nGenerated text:\n")
    print(generated)

    Path("generated_text.txt").write_text(generated, encoding="utf-8")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "char_to_idx": char_to_idx,
            "idx_to_char": idx_to_char,
            "sequence_length": sequence_length,
        },
        "lstm_text_generator.pt",
    )
    print("\nSaved: generated_text.txt and lstm_text_generator.pt")


if __name__ == "__main__":
    main()
