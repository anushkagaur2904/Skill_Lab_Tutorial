import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn


def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i : i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


class RNNForecaster(nn.Module):
    def __init__(self, hidden_size=32):
        super().__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        output, _ = self.rnn(x)
        return self.fc(output[:, -1, :])


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    # 1. Generate synthetic sine-wave data with noise.
    t = np.linspace(0, 40 * np.pi, 1200)
    data = np.sin(t) + 0.08 * np.random.randn(len(t))

    # 2. Create fixed-length input-output sequences.
    sequence_length = 30
    X, y = create_sequences(data, sequence_length)
    split = int(0.8 * len(X))

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    X_train = torch.from_numpy(X_train).unsqueeze(-1)
    y_train = torch.from_numpy(y_train).unsqueeze(-1)
    X_test = torch.from_numpy(X_test).unsqueeze(-1)
    y_test = torch.from_numpy(y_test).unsqueeze(-1)

    # 3. Define the RNN model.
    model = RNNForecaster(hidden_size=32)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 4. Train using MSE loss and Adam.
    epochs = 30
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        prediction = model(X_train)
        loss = criterion(prediction, y_train)
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % 5 == 0:
            print(f"Epoch {epoch:02d}/{epochs} - Loss: {loss.item():.6f}")

    # 5. Predict on the test set.
    model.eval()
    with torch.no_grad():
        test_prediction = model(X_test)
        test_loss = criterion(test_prediction, y_test).item()

    print(f"Test MSE: {test_loss:.6f}")

    # 6. Visualize original data and predictions.
    predicted = test_prediction.squeeze().numpy()
    test_start = sequence_length + split

    plt.figure(figsize=(12, 5))
    plt.plot(data, label="Original time series")
    plt.plot(
        np.arange(test_start, test_start + len(predicted)),
        predicted,
        label="RNN prediction",
    )
    plt.axvline(test_start, linestyle="--", label="Test start")
    plt.title("PyTorch RNN Time-Series Forecasting")
    plt.xlabel("Time step")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig("rnn_forecast.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
