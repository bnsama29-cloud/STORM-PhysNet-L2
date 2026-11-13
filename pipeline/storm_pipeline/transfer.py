"""Shared pieces of the cross-satellite transfer experiments (GOES-16/18 and, for new runs, GRASP).

Recipe (as in grasp_transfer.py, which the AGERS 2026 paper used): pretrained block-A GOES-15 checkpoints;
variants zeroshot (no training), heads (encoder frozen), full (all weights) and scratch (random init); loss is the
horizon-weighted MSE; Adam with gradient clipping 1.0; best-validation checkpoint with early stopping."""
import copy
from pathlib import Path

import numpy as np, torch, torch.nn as nn, yaml
from torch.utils.data import DataLoader

from .data import SW_COLS, HSTEPS
from .families import configure, build as build_family

H_WEIGHT = {1: 1.0, 6: 0.7, 12: 0.5, 24: 0.4}           # same per-horizon weights as train_block.py


def horizon_weights(device):
    assert all(h in H_WEIGHT for h in HSTEPS), f"no loss weight for horizons {HSTEPS}"
    return torch.tensor([H_WEIGHT[h] for h in HSTEPS], device=device)


class Models:
    """Builds and loads the pretrained families with the horizon set of storm_pipeline.data.HSTEPS."""

    def __init__(self, repo, ckpt_dir, scratch_dir, device):
        from src.training.trainer import Trainer
        self.Trainer, self.dev, self.ckpt, self.scratch = Trainer, device, Path(ckpt_dir), Path(scratch_dir)
        self.cfg = yaml.safe_load(open(Path(repo) / "configs/config.yaml"))
        self.cfg["data"]["forecast_horizons"] = [float(h) for h in HSTEPS]

    def build(self, fam):
        c = configure(copy.deepcopy(self.cfg), fam)
        c["training"]["checkpoint_dir"] = str(self.scratch / "ckpt"); c["training"]["log_dir"] = str(self.scratch / "tb")
        tr = self.Trainer(c); m = build_family(fam, tr, len(SW_COLS), self.dev); tr.writer.close(); return m.to(self.dev)

    def pretrained(self, fam, seed):
        m = self.build(fam)
        m.load_state_dict(torch.load(self.ckpt / f"{fam}_s{seed}_best.pt", map_location=self.dev, weights_only=True))
        return m


def freeze_heads_only(m, fam):
    """Leave only the output heads trainable (STORM-PhysNet: everything after the encoder; Transformers: the final fc)."""
    if hasattr(m, "freeze_encoder"):                      # STORM-PhysNet families
        m.freeze_encoder()
        if hasattr(m, "pos_emb") and isinstance(m.pos_emb, torch.nn.Parameter): m.pos_emb.requires_grad = False
    else:
        for n, p in m.named_parameters(): p.requires_grad = n.startswith("fc.")


def run_fwd(m, bt):
    return m(x_sw=bt["x_sw"], x_flux=bt["x_flux"], y_persist=bt["y_persist"])["flux_pred"]


def wmse(p, y, hw):
    return (((p - y) ** 2).mean(0) * hw).sum()


@torch.no_grad()
def predict(m, ds, dev):
    m.eval(); P = []
    for bt in DataLoader(ds, batch_size=256):
        bt = {k: v.to(dev) for k, v in bt.items()}; P.append(run_fwd(m, bt).cpu().numpy())
    return np.concatenate(P).astype(np.float32)


def train(m, sets, lr, seed, params, dev, epochs=30, patience=8):
    """Returns (model at best validation loss, epochs run, best validation loss)."""
    hw = horizon_weights(dev); torch.manual_seed(seed)
    opt = torch.optim.Adam(params, lr=lr)
    g = torch.Generator(); g.manual_seed(seed)
    tl = DataLoader(sets["train"], batch_size=64, shuffle=True, drop_last=True, generator=g)
    best, bad, best_state, ep = float("inf"), 0, None, 0
    for ep in range(epochs):
        m.train()
        for bt in tl:
            bt = {k: v.to(dev) for k, v in bt.items()}
            opt.zero_grad(); loss = wmse(run_fwd(m, bt), bt["y_flux"], hw); loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        m.eval(); v, n = 0.0, 0
        with torch.no_grad():
            for bt in DataLoader(sets["val"], batch_size=256):
                bt = {k: x.to(dev) for k, x in bt.items()}; v += wmse(run_fwd(m, bt), bt["y_flux"], hw).item() * len(bt["x_sw"]); n += len(bt["x_sw"])
        v /= max(n, 1)
        if v < best - 1e-6: best, bad, best_state = v, 0, copy.deepcopy(m.state_dict())
        else:
            bad += 1
            if bad >= patience: break
    if best_state is not None: m.load_state_dict(best_state)
    return m, ep + 1, best
