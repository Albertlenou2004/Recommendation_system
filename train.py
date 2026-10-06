"""
train.py - STEP 3.3 Training loop + hyper-parameter tuning (PyTorch)

* Loss: MSE on explicit ratings        * Optimiser: AdamW (weight decay = L2 regularisation)
* ReduceLROnPlateau scheduler          * Early stopping on VALIDATION RMSE (test set never touched)
"""
import itertools
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import DEFAULT_HP, SEED, TUNING_EPOCHS, TUNING_GRID
from model import HybridNCF


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_loader(df, batch_size, shuffle, device):
    u = torch.as_tensor(df["user_idx"].to_numpy(copy=True), dtype=torch.long)
    i = torch.as_tensor(df["item_idx"].to_numpy(copy=True), dtype=torch.long)
    r = torch.as_tensor(df["rating"].to_numpy(copy=True), dtype=torch.float32)
    return DataLoader(TensorDataset(u, i, r), batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def rmse_on(model, loader, device):
    model.eval()
    se, n = 0.0, 0
    for u, i, r in loader:
        u, i, r = u.to(device), i.to(device), r.to(device)
        pred = model(u, i).clamp(1, 5)
        se += ((pred - r) ** 2).sum().item()
        n += len(r)
    return (se / n) ** 0.5


def train_model(ds, fb, hp, device, verbose=True):
    """Train one model with hyper-parameters `hp`; returns (best_model, history, best_val_rmse)."""
    set_seed()
    model = HybridNCF(fb.user_features, fb.item_features, hp["emb_dim"], hp["hidden"],
                      hp["dropout"], ds.train["rating"].mean()).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=1)
    loss_fn = nn.MSELoss()
    train_loader = make_loader(ds.train, hp["batch_size"], True, device)
    val_loader = make_loader(ds.val, 4096, False, device)

    best_val, best_state, wait, history = float("inf"), None, 0, []
    for epoch in range(1, hp["epochs"] + 1):
        model.train()
        total = 0.0
        for u, i, r in train_loader:
            u, i, r = u.to(device), i.to(device), r.to(device)
            loss = loss_fn(model(u, i), r)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)          # stabilise training
            opt.step()
            total += loss.item() * len(r)
        train_rmse = (total / len(ds.train)) ** 0.5
        val_rmse = rmse_on(model, val_loader, device)
        sched.step(val_rmse)
        history.append({"epoch": epoch, "train_rmse": train_rmse, "val_rmse": val_rmse})
        if verbose:
            print(f"    epoch {epoch:02d}  train_rmse={train_rmse:.4f}  val_rmse={val_rmse:.4f}")
        if val_rmse < best_val - 1e-4:
            best_val, wait = val_rmse, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= hp["patience"]:
                if verbose:
                    print(f"    early stopping at epoch {epoch}")
                break
    model.load_state_dict(best_state)
    return model, history, best_val


def tune_hyperparameters(ds, fb, device):
    """Grid search over embedding size, learning rate and dropout; selection on validation RMSE."""
    keys = list(TUNING_GRID)
    rows = []
    for combo in itertools.product(*[TUNING_GRID[k] for k in keys]):
        hp = {**DEFAULT_HP, **dict(zip(keys, combo)), "epochs": TUNING_EPOCHS}
        t0 = time.time()
        _, _, val_rmse = train_model(ds, fb, hp, device, verbose=False)
        rows.append({**dict(zip(keys, combo)), "val_rmse": val_rmse, "seconds": round(time.time() - t0, 1)})
        print(f"[tune] {dict(zip(keys, combo))} -> val_rmse={val_rmse:.4f}")
    results = pd.DataFrame(rows).sort_values("val_rmse").reset_index(drop=True)
    best = {**DEFAULT_HP, **{k: results[k].iloc[0].item() for k in keys}}
    return best, results


@torch.no_grad()
def predict_matrix(model, n_users, n_items, device, user_chunk=100):
    """Predicted rating for EVERY (user, item) pair -> array (U, I) clipped to [1, 5]."""
    model.eval()
    out = np.zeros((n_users, n_items), dtype=np.float32)
    items = torch.arange(n_items, device=device)
    for start in range(0, n_users, user_chunk):
        us = torch.arange(start, min(start + user_chunk, n_users), device=device)
        u = us.repeat_interleave(n_items)
        i = items.repeat(len(us))
        out[start:start + len(us)] = model(u, i).clamp(1, 5).view(len(us), n_items).cpu().numpy()
    return out
