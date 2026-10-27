"""Wrapper exposing MultiTaskNeuralGranger via the CausalDiscoveryMethod API.

This lets the V1 method plug into the existing experiment_runner and reuse
all Phase 1 evaluation metrics.

The `output_graph` parameter selects which learned graph to return as the
primary result ("task1", "task2a", "task2b", or "consensus"). All four
graphs are always stored in result.extra for analysis.
"""
from __future__ import annotations

import time

import numpy as np

from src.models.base import CausalDiscoveryMethod, CausalDiscoveryResult
from src.trainers.mt_trainer import TrainConfig, train_mt_neural_granger


class MTNeuralGrangerMethod(CausalDiscoveryMethod):
    """V1: Multi-Task Neural Granger Causality."""

    name = "mt_neural_granger"

    def __init__(
        self,
        lookback: int = 5,
        hidden_dim: int = 64,
        use_hybrid: bool = False,
        gate_init: float = 0.0,
        ridge_lambda: float = 1e-3,
        lambda_sparse: float = 0.01,
        lr: float = 1e-3,
        batch_size: int = 64,
        max_epochs: int = 200,
        patience: int = 20,
        output_graph: str = "consensus",
        verbose: bool = False,
        tasks: tuple | None = None,
    ):
        assert output_graph in {"task1", "task2a", "task2b", "consensus"}, (
            f"Invalid output_graph: {output_graph}"
        )
        # Restricting `tasks` skips training the graphs you never read.
        # consensus is the geometric mean of all three, so it cannot be
        # produced from a subset — fail here rather than silently returning
        # a consensus built from randomly initialized predictors.
        if tasks is not None:
            tasks = tuple(tasks)
            if output_graph == "consensus" and set(tasks) != {
                "task1", "task2a", "task2b"
            }:
                raise ValueError(
                    "output_graph='consensus' needs all three tasks, but "
                    f"tasks={tasks}. Either train all three or pick a single "
                    "task as output_graph."
                )
            if output_graph not in tasks:
                raise ValueError(
                    f"output_graph={output_graph!r} is not in tasks={tasks}."
                )
        self.output_graph = output_graph
        self.config = TrainConfig(
            lookback=lookback,
            hidden_dim=hidden_dim,
            use_hybrid=use_hybrid,
            gate_init=gate_init,
            ridge_lambda=ridge_lambda,
            lambda_sparse=lambda_sparse,
            lr=lr,
            batch_size=batch_size,
            max_epochs=max_epochs,
            patience=patience,
            verbose=verbose,
            tasks=tasks,
        )

    def fit(self, data: np.ndarray, **kwargs) -> CausalDiscoveryResult:
        t0 = time.time()
        segment_lengths = kwargs.get("segment_lengths", None)
        model, log = train_mt_neural_granger(
            data, self.config, segment_lengths=segment_lengths
        )
        self.model = model
        graphs = model.get_all_graphs()
        runtime = time.time() - t0

        primary = graphs[self.output_graph]

        return CausalDiscoveryResult(
            graph=primary,
            runtime_sec=runtime,
            extra={
                "all_graphs": graphs,           # task1/2a/2b/consensus
                "output_graph_used": self.output_graph,
                "train_log": {
                    "final_val_loss": log.final_val_loss,
                    "epochs_trained": log.epochs_trained,
                },
            },
        )