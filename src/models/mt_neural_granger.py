"""Multi-Task Neural Granger Causality (MT-NGC) — V1 main model.

V1 architecture (fully independent predictors, no backbone sharing):
  - For each of 3 tasks (temporal, instantaneous, mixed) and each of N
    target variables, instantiate one independent PerTargetMLP.
  - Total: 3N independent small networks.
  - Train sequentially (all Task 1, then all Task 2A, then all Task 2B).
  - Extract a (N, N) edge-score matrix per task from learned weights.
  - Combine via geometric mean for a single consensus graph.

This is the V1 baseline; backbone sharing is a future ablation.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.datasets.task_dataloaders import (
    Task1TemporalDataset,
    Task2AInstantaneousDataset,
    Task2BMixedDataset,
    get_input_dim,
)
from src.models.per_target_mlp import PerTargetMLP
from src.models.hybrid_per_target_mlp import HybridPerTargetMLP


TASKS = ["task1", "task2a", "task2b"]


class MultiTaskNeuralGranger(nn.Module):
    """V1 method: 3N independent per-target predictors.

    Args:
        n_variables: N
        lookback: L (used by task1 and task2b)
        hidden_dim: hidden width of each PerTargetMLP
    """

    def __init__(
        self,
        n_variables: int,
        lookback: int = 5,
        hidden_dim: int = 64,
        use_hybrid: bool = False,
        gate_init: float = 0.0,
        ridge_lambda: float = 1e-3,
    ):
        super().__init__()
        self.N = n_variables
        self.L = lookback
        self.hidden_dim = hidden_dim
        self.use_hybrid = use_hybrid
        self.gate_init = gate_init
        self.ridge_lambda = ridge_lambda

        # Build 3N predictors, organized as ModuleDict per task.
        # predictors[task][i] = PerTargetMLP for target i under that task.
        self.predictors = nn.ModuleDict()
                        
        for task in TASKS:
            task_modules = nn.ModuleList()
            input_dim = get_input_dim(task, n_variables, lookback)

            for i in range(n_variables):
                # feature_to_variable depends on task and (for 2a/2b) target
                ftv = self._feature_to_variable(task, i)
                if use_hybrid:
                    mlp = HybridPerTargetMLP(
                        input_dim=input_dim,
                        feature_to_variable=ftv,
                        n_variables=n_variables,
                        hidden_dim=hidden_dim,
                        gate_init=gate_init,
                        ridge_lambda=ridge_lambda,
                    )
                else:
                    mlp = PerTargetMLP(
                        input_dim=input_dim,
                        feature_to_variable=ftv,
                        n_variables=n_variables,
                        hidden_dim=hidden_dim,
                    )
                task_modules.append(mlp)

            self.predictors[task] = task_modules

    def _feature_to_variable(self, task: str, target_var: int) -> list[int]:
        """Compute the feature_to_variable mapping for a (task, target)."""
        if task == "task1":
            return Task1TemporalDataset.feature_to_variable(self.N, self.L)
        elif task == "task2a":
            # other variables (without target)
            return [j for j in range(self.N) if j != target_var]
        elif task == "task2b":
            mapping = []
            for j in range(self.N):
                mapping.extend([j] * self.L)
            mapping.extend([j for j in range(self.N) if j != target_var])
            return mapping
        else:
            raise ValueError(f"Unknown task: {task}")

    def make_dataset(self, data: np.ndarray, task: str, target_var: int,
                     segment_lengths: list[int] | None = None):
        """Build the appropriate Dataset for (task, target_var).

        segment_lengths: optional trajectory segment lengths. When provided,
        lookback windows won't cross segment boundaries. None = whole series
        is one continuous segment (backward-compatible).
        """
        if task == "task1":
            return Task1TemporalDataset(data, target_var, self.L,
                                        segment_lengths=segment_lengths)
        elif task == "task2a":
            return Task2AInstantaneousDataset(data, target_var,
                                              segment_lengths=segment_lengths)
        elif task == "task2b":
            return Task2BMixedDataset(data, target_var, self.L,
                                      segment_lengths=segment_lengths)
        else:
            raise ValueError(f"Unknown task: {task}")

    def get_graph(self, task: str) -> np.ndarray:
        """Extract the (N, N) edge-score matrix for a task.

        A[i, j] = importance of source variable j when predicting target i.
        Diagonal is zeroed (no self-loops in output).
        """
        A = np.zeros((self.N, self.N), dtype=np.float32)
        for i in range(self.N):
            mlp = self.predictors[task][i]
            importance = mlp.get_input_importance().detach().cpu().numpy()
            A[i, :] = importance
        np.fill_diagonal(A, 0.0)
        return A

    def get_all_graphs(self) -> dict[str, np.ndarray]:
        """Return all 3 task graphs + geometric-mean consensus."""
        graphs = {task: self.get_graph(task) for task in TASKS}

        # Geometric mean consensus.
        # Normalize each graph to [0, 1] first so the geometric mean is
        # meaningful (raw importance scales differ across tasks).
        normed = []
        for task in TASKS:
            g = graphs[task].copy()
            gmax = g.max()
            if gmax > 0:
                g = g / gmax
            normed.append(g)

        # Geometric mean: (g1 * g2 * g3)^(1/3)
        # Add small epsilon to avoid issues; consensus is high only if all agree.
        stacked = np.stack(normed, axis=0)  # (3, N, N)
        consensus = np.prod(stacked, axis=0) ** (1.0 / 3.0)
        graphs["consensus"] = consensus.astype(np.float32)

        return graphs