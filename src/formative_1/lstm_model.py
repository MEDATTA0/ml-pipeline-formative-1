"""Two-head LSTM: one shared encoder predicts the next-day return and volatility."""

import numpy as np
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class TwoHeadLSTM(torch.nn.Module):
    """Last-step LSTM state (plus an optional ticker embedding) feeds a 2-output linear head."""

    def __init__(
        self, n_features, hidden=64, layers=2, dropout=0.2, n_tickers=0, emb_dim=8
    ):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            n_features, hidden, layers, batch_first=True, dropout=dropout
        )
        self.embed = torch.nn.Embedding(n_tickers, emb_dim) if n_tickers else None
        self.head = torch.nn.Linear(hidden + (emb_dim if n_tickers else 0), 2)

    def forward(self, x, ticker=None):
        out, _ = self.lstm(x)
        h = out[:, -1]
        if self.embed is not None:
            h = torch.cat([h, self.embed(ticker)], dim=1)
        return self.head(h)


def predict(model, X, ticker=None, batch_size=4096):
    """Scaled predictions, shape (n, 2): [return, volatility]."""
    model.eval()
    outputs = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch_ticker = (
                ticker[i : i + batch_size].to(DEVICE) if ticker is not None else None
            )
            outputs.append(model(X[i : i + batch_size].to(DEVICE), batch_ticker).cpu())
    return torch.cat(outputs).numpy()


def _train_epoch(model, optimizer, X, Y, ticker, indices, batch_size):
    """One pass over the `indices` samples; returns the mean training loss."""
    model.train()
    losses = []
    for start in range(0, len(indices), batch_size):
        batch = indices[start : start + batch_size]
        batch_ticker = ticker[batch].to(DEVICE) if ticker is not None else None

        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(
            model(X[batch].to(DEVICE), batch_ticker), Y[batch].to(DEVICE)
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses))


def fit(
    model,
    X,
    Y,
    train,
    val,
    ticker=None,
    epochs=15,
    patience=3,
    batch_size=512,
    lr=1e-3,
    seed=42,
    verbose=True,
):
    """Train on the `train` mask and keep the weights with the best validation loss.

    `X` and `Y` are scaled tensors; `Y` has columns [return, volatility]. The validation
    loss is the mean MSE across both heads, and training stops after `patience` epochs
    without improvement. Returns the model and a list with one dict of losses per epoch.
    """
    model.to(DEVICE)
    rng = np.random.RandomState(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_idx = np.flatnonzero(train)
    val_idx = np.flatnonzero(val)
    val_ticker = ticker[val_idx] if ticker is not None else None
    y_val = Y[val_idx].numpy()

    best_loss, best_state, epochs_without_gain = np.inf, None, 0
    history = []
    for epoch in range(1, epochs + 1):
        rng.shuffle(train_idx)
        train_loss = _train_epoch(model, optimizer, X, Y, ticker, train_idx, batch_size)

        val_by_head = ((predict(model, X[val_idx], val_ticker) - y_val) ** 2).mean(
            axis=0
        )
        val_loss = float(val_by_head.mean())
        history.append(
            {
                "epoch": epoch,
                "train": train_loss,
                "val": val_loss,
                "val_return": float(val_by_head[0]),
                "val_vol": float(val_by_head[1]),
            }
        )
        if verbose:
            print(
                f"epoch {epoch:2d}  train {train_loss:.4f}  val {val_loss:.4f}  "
                f"(return {val_by_head[0]:.4f}, vol {val_by_head[1]:.4f})"
            )

        if val_loss < best_loss:
            best_loss, epochs_without_gain = val_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            epochs_without_gain += 1
            if epochs_without_gain >= patience:
                break

    model.load_state_dict(best_state)
    return model, history
