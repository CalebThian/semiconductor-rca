"""Sequential trainer for MultiTaskNeuralGranger (V1).

Trains all 3N predictors sequentially: all Task 1 targets, then all
Task 2A targets, then all Task 2B targets. Each predictor is trained
independently with prediction loss + group-lasso sparsity penalty.

Time-series-aware train/val split: first (1 - val_frac) fraction is
training, the remainder is validation (no shuffling across the split
boundary to avoid leaking future into past).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.models.mt_neural_granger import MultiTaskNeuralGranger, TASKS


@dataclass
class TrainConfig:
    lookback: int = 5
    hidden_dim: int = 64
    use_hybrid: bool = False
    gate_init: float = 0.0
    ridge_lambda: float = 1e-3
    lambda_sparse: float = 0.01
    lr: float = 1e-3
    batch_size: int = 64
    max_epochs: int = 200
    patience: int = 20
    val_frac: float = 0.2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    verbose: bool = True


@dataclass
class TrainLog:
    """Records per-(task, target) training outcome."""
    final_train_loss: dict = field(default_factory=dict)
    final_val_loss: dict = field(default_factory=dict)
    epochs_trained: dict = field(default_factory=dict)


def _time_series_split(n_samples: int, val_frac: float):
    """Return (train_indices, val_indices) split by time order."""
    n_val = int(n_samples * val_frac)
    n_train = n_samples - n_val
    train_idx = list(range(n_train))
    val_idx = list(range(n_train, n_samples))
    return train_idx, val_idx


def _train_one_predictor(
    mlp,
    dataset,
    config: TrainConfig,
) -> tuple[float, float, int]:
    """Train a single PerTargetMLP. Returns (train_loss, val_loss, epochs)."""
    device = config.device
    mlp = mlp.to(device)

    train_idx, val_idx = _time_series_split(len(dataset), config.val_frac)

    # 一次性把樣本堆成張量並搬上 GPU —— 取代 DataLoader。
    # 原本每個 epoch 都會對全部樣本呼叫一次 __getitem__(7299 次 × 100 epoch
    # × 75 個預測器 ≈ 5500 萬次),且每個 batch 都做一次 host→device 傳輸。
    # 資料量僅約 3.6 MB,整份常駐 GPU 即可。
    def _stack(idx):
        xs = torch.as_tensor(
            np.stack([np.asarray(dataset[k][0]) for k in idx]), dtype=torch.float32)
        ys = torch.as_tensor(
            np.stack([np.asarray(dataset[k][1]) for k in idx]), dtype=torch.float32)
        return xs.to(device), ys.to(device)

    Xtr, Ytr = _stack(train_idx)
    Xva, Yva = _stack(val_idx) 
    if hasattr(mlp, "fit_linear_closed_form"):
        mlp.fit_linear_closed_form(Xtr, Ytr)

    optimizer = torch.optim.Adam( [p for p in mlp.parameters() if p.requires_grad], lr=config.lr)
    n_tr, bs = len(Xtr), config.batch_size

    best_val = float("inf")
    best_state = None
    patience_counter = 0
    epochs_run = 0
    avg_train = float("nan")

    for epoch in range(config.max_epochs):
        # --- train ---
        mlp.train()
        perm = torch.randperm(n_tr, device=device)
        train_loss_sum = torch.zeros((), device=device)   # 留在 GPU,避免每批同步
        n_train_batches = 0
        for s in range(0, n_tr, bs):
            b = perm[s:s + bs]
            x, y = Xtr[b], Ytr[b]
            pred = mlp(x)
            mse = ((pred - y) ** 2).mean()
            loss = mse + config.lambda_sparse * mlp.group_lasso_loss()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            train_loss_sum += mse.detach()
            n_train_batches += 1

        # --- validate(資料不大,一次算完)---
        mlp.eval()
        with torch.no_grad():
            avg_val = float(((mlp(Xva) - Yva) ** 2).mean())

        avg_train = float(train_loss_sum) / max(n_train_batches, 1)
        epochs_run = epoch + 1

        # --- early stopping ---
        if avg_val < best_val - 1e-5:
            best_val = avg_val
            best_state = {k: v.detach().clone() for k, v in mlp.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                break

    if best_state is not None:
        mlp.load_state_dict(best_state)

    return avg_train, best_val, epochs_run


def train_mt_neural_granger(
    data: np.ndarray,
    config: TrainConfig | None = None,
    segment_lengths: list[int] | None = None,
) -> tuple[MultiTaskNeuralGranger, TrainLog]:
    """Train the full MT-NGC model on a time series.

    Args:
        data: (T, N) array
        config: training configuration

    Returns:
        (trained_model, train_log)
    """
    if config is None:
        config = TrainConfig()

    T, N = data.shape
    # Z-score normalize each variable independently. This is essential:
    # without it, variables with large scale dominate the prediction loss
    # and the group-lasso sparsity behaves inconsistently across variables.
    # (Discovered via dry-run: V1 collapsed to AUROC 0.5 on un-normalized
    #  wild-scale data while Granger, which normalizes internally, was fine.)
    data = data.astype(np.float32)
    data_mean = data.mean(axis=0, keepdims=True)
    data_std = data.std(axis=0, keepdims=True) + 1e-8
    data = (data - data_mean) / data_std
    model = MultiTaskNeuralGranger(
        n_variables=N,
        lookback=config.lookback,
        hidden_dim=config.hidden_dim,
        use_hybrid=config.use_hybrid,
        gate_init=config.gate_init,
        ridge_lambda=config.ridge_lambda,
    )
    log = TrainLog()

    for task in TASKS:
        if config.verbose:
            print(f"\n{'='*55}\nTraining {task}\n{'='*55}")
        t0 = time.time()

        for i in range(N):
            mlp = model.predictors[task][i]
            dataset = model.make_dataset(data, task, target_var=i,
                                         segment_lengths=segment_lengths)
            tr, vl, ep = _train_one_predictor(mlp, dataset, config)

            key = f"{task}__target_{i}"
            log.final_train_loss[key] = tr
            log.final_val_loss[key] = vl
            log.epochs_trained[key] = ep

        if config.verbose:
            dt = time.time() - t0
            mean_val = np.mean([
                log.final_val_loss[f"{task}__target_{i}"] for i in range(N)
            ])
            print(f"  {task} done in {dt:.1f}s, mean val MSE: {mean_val:.4f}")

    return model, log