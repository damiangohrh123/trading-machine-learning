"""GPU neural nets: an MLP on the features, optionally with a sequence encoder over recent price action."""
import copy
import time

import numpy as np
import torch
import torch.nn as nn

from ml2 import settings as S

DEV = "cuda"
CHUNK = 65536  # prediction batch size


class TrialTimeout(Exception):
    """Raised when a training run passes its deadline, so heavy configs cannot stall a search."""


class Encoder(nn.Module):
    """Summarise a (batch, length, channels) window into one vector."""

    def __init__(self, arch: str, channels: int, hidden: int, length: int):
        super().__init__()
        self.arch = arch
        if arch == "gru":
            self.net = nn.GRU(channels, hidden, batch_first=True)
        elif arch == "tcn":  # dilated convolutions widen the receptive field cheaply
            layers, c = [], channels
            for d in (1, 2, 4, 8):
                layers += [nn.Conv1d(c, hidden, 3, padding=d, dilation=d), nn.GELU()]
                c = hidden
            self.net = nn.Sequential(*layers)
        elif arch == "transformer":
            self.proj = nn.Linear(channels, hidden)
            self.pos = nn.Parameter(torch.zeros(length, hidden))  # learned position embedding
            layer = nn.TransformerEncoderLayer(hidden, nhead=4, dim_feedforward=2 * hidden, dropout=0.1,
                                               batch_first=True)
            self.net = nn.TransformerEncoder(layer, num_layers=2)
        else:
            raise ValueError(arch)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        if self.arch == "gru":
            return self.net(seq)[1][-1]                  # last hidden state
        if self.arch == "tcn":
            return self.net(seq.transpose(1, 2))[:, :, -1]  # features at the most recent day
        return self.net(self.proj(seq) + self.pos).mean(dim=1)


class Net(nn.Module):
    def __init__(self, n_feat: int, hidden: int, dropout: float, arch: str, seq_hidden: int, length: int,
                 channels: int = 5):
        super().__init__()
        self.enc = Encoder(arch, channels, seq_hidden, length) if arch != "none" else None
        width = n_feat + (seq_hidden if self.enc else 0)
        self.mlp = nn.Sequential(nn.Linear(width, hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden // 2, 1))

    def forward(self, x, seq=None):
        if self.enc is not None:
            x = torch.cat([x, self.enc(seq)], dim=1)
        return self.mlp(x).squeeze(-1)


def _ic_loss(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Negative correlation with the target: trains the ranking quality directly."""
    p, t = pred - pred.mean(), y - y.mean()
    return -(p * t).sum() / (p.norm() * t.norm() + 1e-8)


def _window(seq_all: torch.Tensor, t: torch.Tensor, s: torch.Tensor, length: int) -> torch.Tensor:
    """Gather the `length` days ending at t for each (t, s) pair -> (batch, length, channels)."""
    steps = torch.arange(1 - length, 1, device=DEV)
    return seq_all[(t[:, None] + steps).clamp(min=0), s[:, None]]


def fit_predict(params: dict, Xtr, ytr, Xte, ctx: dict, arch: str) -> np.ndarray:
    """arch is 'none' (plain MLP), 'gru', 'tcn' or 'transformer'."""
    torch.manual_seed(S.SEED)
    ds, classify = ctx["ds"], ctx["target"] == "class_top"
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    prep = lambda X: torch.tensor(np.clip((X - mu) / sd, -5, 5), dtype=torch.float32, device=DEV)  # noqa: E731
    xtr, xte = prep(Xtr), prep(Xte)
    ytr_t = torch.tensor(ytr, dtype=torch.float32, device=DEV)
    t_all = torch.tensor(ds.t, dtype=torch.long, device=DEV)
    s_all = torch.tensor(ds.s, dtype=torch.long, device=DEV)
    use_seq = arch != "none"
    seq_all = torch.tensor(ds.seq, device=DEV) if use_seq else None
    tr, te = torch.tensor(ctx["tr"], device=DEV), torch.tensor(ctx["te"], device=DEV)
    length = params.get("seq_len", 30)

    net = Net(xtr.shape[1], params["hidden"], params["dropout"], arch, params.get("seq_hidden", 64), length).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    if classify:
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        loss_fn = _ic_loss if params.get("loss") == "ic" else nn.MSELoss()

    def forward(x, rows):
        seq = _window(seq_all, t_all[rows], s_all[rows], length) if use_seq else None
        return net(x, seq)

    # rows are date-sorted, so the last 10% of the training rows is a time-ordered validation set
    n_val = len(xtr) // 10
    n_fit = len(xtr) - n_val
    best, best_state = float("inf"), None
    for _ in range(params["epochs"]):
        net.train()
        for idx in torch.randperm(n_fit, device=DEV).split(params["batch"]):
            if ctx["deadline"] and time.time() > ctx["deadline"]:
                raise TrialTimeout
            opt.zero_grad()
            loss_fn(forward(xtr[idx], tr[idx]), ytr_t[idx]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            val = np.mean([loss_fn(forward(xtr[i], tr[i]), ytr_t[i]).item()
                           for i in torch.arange(n_fit, len(xtr), device=DEV).split(CHUNK)])
        if val < best:  # keep the epoch with the lowest validation loss
            best, best_state = val, copy.deepcopy(net.state_dict())
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        out = [forward(xte[i], te[i]).float().cpu().numpy() for i in torch.arange(len(xte), device=DEV).split(CHUNK)]
    return np.concatenate(out)
