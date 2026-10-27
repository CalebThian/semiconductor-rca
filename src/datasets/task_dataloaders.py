"""Data preparation for the 3 prediction tasks of V1 method.

Each task takes a multivariate time series (T, N) and produces
(input, target) pairs suitable for training a PerTargetMLP.

Convention: for each task and each target variable i, the dataset
yields (input_vector, scalar_target) pairs where:
- input_vector shape depends on the task (see method_design_v1.md §3)
- scalar_target is the value to predict (x_i at some time t)
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

def build_valid_targets(
    segment_lengths: list[int] | None,
    total_T: int,
    lookback: int,
) -> list[int]:
    """Compute valid target time-indices that don't cross segment boundaries.

    A target at global time t is valid only if its lookback window
    [t-lookback, t) lies entirely within the same segment as t.

    Args:
        segment_lengths: list of segment lengths. If None, treat the whole
            series as one segment (backward-compatible with continuous data).
        total_T: total number of time steps (for the None case / validation).
        lookback: L. Use 0 for tasks without history (Task2A).

    Returns:
        Sorted list of valid global time-indices t.
    """
    if segment_lengths is None:
        segment_lengths = [total_T]
    assert sum(segment_lengths) == total_T, (
        f"segment_lengths sum {sum(segment_lengths)} != total_T {total_T}"
    )

    valid = []
    offset = 0
    for seg_len in segment_lengths:
        # Within this segment, local target index must be >= lookback
        # (so the lookback window stays inside the segment).
        for local_t in range(lookback, seg_len):
            valid.append(offset + local_t)
        offset += seg_len
    return valid


class Task1TemporalDataset(Dataset):
    """Task 1: temporal forecasting.

    For each (t, i) where t ≥ L:
        input  = [x_0(t-L), ..., x_0(t-1), x_1(t-L), ..., x_{N-1}(t-1)]
        target = x_i(t)

    Args:
        data: (T, N) array
        target_var: index i of the target variable
        lookback: L
    """
    def __init__(self, data: np.ndarray, target_var: int, lookback: int,segment_lengths: list[int] | None = None,):
        assert data.ndim == 2, f"Expected 2D data, got shape {data.shape}"
        self.data = torch.tensor(data, dtype=torch.float32)
        self.target_var = target_var
        self.L = lookback
        self.T, self.N = data.shape
        # Valid target indices that don't cross segment boundaries
        self.valid_t = build_valid_targets(segment_lengths, self.T, lookback)
        self.n_samples = len(self.valid_t)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        t = self.valid_t[idx]  # actual time index (segment-safe)
        history = self.data[t - self.L : t, :]  # shape (L, N)
        history = history.T  # shape (N, L)
        input_vec = history.flatten()  # shape (N*L,)
        target = self.data[t, self.target_var]
        return input_vec, target

    @staticmethod
    def feature_to_variable(n_vars: int, lookback: int) -> list[int]:
        """Return the feature_to_variable mapping for this task."""
        mapping = []
        for j in range(n_vars):
            mapping.extend([j] * lookback)
        return mapping


class Task2AInstantaneousDataset(Dataset):
    """Task 2A: pure instantaneous imputation (no history).

    For each (t, i):
        input  = [x_j(t) for j in range(N) if j != i]
        target = x_i(t)

    Args:
        data: (T, N) array
        target_var: index i
    """

    def __init__(
        self, data: np.ndarray, target_var: int,
        segment_lengths: list[int] | None = None,
    ):
        assert data.ndim == 2
        self.data = torch.tensor(data, dtype=torch.float32)
        self.target_var = target_var
        self.T, self.N = data.shape
        self.other_vars = [j for j in range(self.N) if j != target_var]
        self.other_vars_t = torch.tensor(self.other_vars, dtype=torch.long)
        # lookback=0: every time-step is a valid target (no history needed)
        self.valid_t = build_valid_targets(segment_lengths, self.T, 0)

    def __len__(self) -> int:
        return len(self.valid_t)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        t = self.valid_t[idx]
        input_vec = self.data[t, self.other_vars_t]  # shape (N-1,)
        target = self.data[t, self.target_var]
        return input_vec, target

    def feature_to_variable(self) -> list[int]:
        """Return mapping (per-target since 'other_vars' depends on target)."""
        return self.other_vars.copy()


class Task2BMixedDataset(Dataset):
    """Task 2B: mixed imputation (history + same-time-step others).

    For each (t, i) where t ≥ L:
        input = [history (N*L)] ++ [x_j(t) for j != i] (N-1)
        target = x_i(t)

    Args:
        data: (T, N) array
        target_var: index i
        lookback: L
    """

    def __init__(
        self, data: np.ndarray, target_var: int, lookback: int,
        segment_lengths: list[int] | None = None,
    ):
        assert data.ndim == 2
        self.data = torch.tensor(data, dtype=torch.float32)
        self.target_var = target_var
        self.L = lookback
        self.T, self.N = data.shape
        self.other_vars = [j for j in range(self.N) if j != target_var]
        self.other_vars_t = torch.tensor(self.other_vars, dtype=torch.long)
        self.valid_t = build_valid_targets(segment_lengths, self.T, lookback)
        self.n_samples = len(self.valid_t)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        t = self.valid_t[idx]
        history = self.data[t - self.L : t, :].T.flatten()
        same_time = self.data[t, self.other_vars_t]
        input_vec = torch.cat([history, same_time], dim=0)
        target = self.data[t, self.target_var]
        return input_vec, target

    def feature_to_variable(self, n_vars: int, lookback: int) -> list[int]:
        """Return mapping for Task 2B input layout."""
        mapping = []
        # History part: each variable has L features
        for j in range(n_vars):
            mapping.extend([j] * lookback)
        # Same-time-step part: other variables (without target_var)
        mapping.extend(self.other_vars)
        return mapping


# Convenience function to compute input_dim for each task
def get_input_dim(task: str, n_vars: int, lookback: int) -> int:
    """Compute the input dimension for a given task.

    Args:
        task: "task1" | "task2a" | "task2b"
        n_vars: N
        lookback: L
    """
    if task == "task1":
        return n_vars * lookback
    elif task == "task2a":
        return n_vars - 1
    elif task == "task2b":
        return n_vars * lookback + (n_vars - 1)
    else:
        raise ValueError(f"Unknown task: {task}")