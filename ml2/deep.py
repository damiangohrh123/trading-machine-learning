"""GPU neural nets: an MLP on the features, optionally plus a GRU over the last days of price action."""
import copy

import numpy as np
import torch
import torch.nn as nn

from ml2 import settings as S

DEV = "cuda"
CHUNK = 65536  # prediction batch size


class Net(nn.Module):
    def __init__(self, n_feat: int, hidden: int, dropout: float, use_seq: bool, seq_ch: int = 5, gru_hidden: int = 64):
        super().__init__()
        self.gru = nn.GRU(seq_ch, gru_hidden, batch_first=True) if use_seq else None
        width = n_feat + (gru_hidden if use_seq else 0)
        self.mlp = nn.Sequential(nn.Linear(width, hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden // 2, 1))

    def forward(self, x, seq=None):
        if self.gru is not None:
            x = torch.cat([x, self.gru(seq)[1][-1]], dim=1)  # last hidden state summarises the window
        return self.mlp(x).squeeze(-1)


def _window(seq_all: torch.Tensor, t: torch.Tensor, s: torch.Tensor, length: int) -> torch.Tensor:
    """Gather the `length` days ending at t for each (t, s) pair -> (batch, length, channels)."""
    steps = torch.arange(1 - length, 1, device=DEV)
    return seq_all[(t[:, None] + steps).clamp(min=0), s[:, None]]


def fit_predict(params: dict, Xtr, ytr, Xte, ctx: dict, use_seq: bool) -> np.ndarray:
    torch.manual_seed(S.SEED)
    ds, classify = ctx["ds"], ctx["target"] == "class_top"
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    prep = lambda X: torch.tensor(np.clip((X - mu) / sd, -5, 5), dtype=torch.float32, device=DEV)  # noqa: E731
    xtr, xte = prep(Xtr), prep(Xte)
    ytr_t = torch.tensor(ytr, dtype=torch.float32, device=DEV)
    t_all = torch.tensor(ds.t, dtype=torch.long, device=DEV)
    s_all = torch.tensor(ds.s, dtype=torch.long, device=DEV)
    seq_all = torch.tensor(ds.seq, device=DEV) if use_seq else None
    tr, te = torch.tensor(ctx["tr"], device=DEV), torch.tensor(ctx["te"], device=DEV)
    length = params.get("seq_len", 30)

    net = Net(xtr.shape[1], params["hidden"], params["dropout"], use_seq).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    loss_fn = nn.BCEWithLogitsLoss() if classify else nn.MSELoss()

    def forward(x, rows, t_idx, s_idx):
        seq = _window(seq_all, t_idx[rows], s_idx[rows], length) if use_seq else None
        return net(x, seq)

    # rows are date-sorted, so the last 10% of the training rows is a time-ordered validation set
    n_val = len(xtr) // 10
    n_fit = len(xtr) - n_val
    best, best_state = float("inf"), None
    for _ in range(params["epochs"]):
        net.train()
        for idx in torch.randperm(n_fit, device=DEV).split(params["batch"]):
            opt.zero_grad()
            loss = loss_fn(forward(xtr[idx], tr[idx], t_all, s_all), ytr_t[idx])
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            val = np.mean([loss_fn(forward(xtr[i], tr[i], t_all, s_all), ytr_t[i]).item()
                           for i in torch.arange(n_fit, len(xtr), device=DEV).split(CHUNK)])
        if val < best:  # keep the epoch with the lowest validation loss
            best, best_state = val, copy.deepcopy(net.state_dict())
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        out = [forward(xte[i], te[i], t_all, s_all).float().cpu().numpy()
               for i in torch.arange(len(xte), device=DEV).split(CHUNK)]
    return np.concatenate(out)
