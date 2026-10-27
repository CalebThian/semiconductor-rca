"""Abstract base class for all causal discovery methods."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class CausalDiscoveryResult:
    """Unified output format for causal discovery methods.

    Convention: graph[i, j] represents the strength/probability of edge j -> i.
    (rows = effect, cols = cause)
    """
    graph: np.ndarray                   # (N, N) float, soft edge scores
    graph_binary: Optional[np.ndarray] = None   # (N, N) int, if method outputs binary
    runtime_sec: float = 0.0
    extra: dict = field(default_factory=dict)   # method-specific outputs


class CausalDiscoveryMethod(ABC):
    """Abstract interface. All methods (baselines + our methods) implement this."""

    name: str = "base"

    @abstractmethod
    def fit(self, data: np.ndarray, **kwargs) -> CausalDiscoveryResult:
        """Fit on (T, N) array and return a graph.

        Args:
            data: (T, N) numpy array, normalized.
            **kwargs: method-specific hyperparameters.

        Returns:
            CausalDiscoveryResult with graph of shape (N, N).
        """
        raise NotImplementedError
