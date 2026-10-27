"""Per-target MLP module: the core building block for our V1 method.

For a single target variable i, this module:
  1. Takes a flattened input vector (representing other variables' values
     across possibly multiple time lags).
  2. Predicts the target value via a small MLP.
  3. Exposes "input importance" per source variable, derived from the
     L2 norms of column-groups in the first-layer weights.
  4. Provides a group-lasso loss to encourage sparsity over source variables.

The mapping from input features to source variables is provided externally
via `feature_to_variable` (a list of length input_dim, where each element
is the source variable index that input feature belongs to).

For N variables, after training on N target predictors, the importance
vectors stacked together form an (N, N) edge-score matrix.
Convention: A[i, j] = importance of source variable j when predicting
target variable i. (rows = effect, cols = cause; consistent with the rest
of the codebase.)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PerTargetMLP(nn.Module):
    """A small MLP that predicts a single target variable.

    Args:
        input_dim: total dimension of flattened input (e.g., N*L for Task 1).
        feature_to_variable: list of length input_dim, mapping each input
            feature to its source variable index. Used for group-lasso
            grouping and for computing per-variable importance.
        n_variables: total number of variables N (used as output dimension
            of importance vector).
        hidden_dim: hidden layer width.
    """

    def __init__(
        self,
        input_dim: int,
        feature_to_variable: list[int],
        n_variables: int,
        hidden_dim: int = 64,
    ):
        super().__init__()
        assert len(feature_to_variable) == input_dim, (
            f"feature_to_variable length {len(feature_to_variable)} "
            f"doesn't match input_dim {input_dim}"
        )
        self.input_dim = input_dim
        self.n_variables = n_variables
        self.hidden_dim = hidden_dim

        # First layer: this is where causal signal lives.
        # W1 shape: (hidden_dim, input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

        # Pre-compute groupings: for each variable j, which input features?
        # We register as buffer so it moves with .to(device) but isn't trained.
        groups = [[] for _ in range(n_variables)]
        for feat_idx, var_idx in enumerate(feature_to_variable):
            groups[var_idx].append(feat_idx)

        # Store as a list of LongTensors (one per variable).
        # Some variables may have no features (e.g., the target itself in
        # Task 2A); their group is empty.
        self.groups = []
        for j, feats in enumerate(groups):
            self.register_buffer(
                f"group_{j}",
                torch.tensor(feats, dtype=torch.long),
            )
            self.groups.append(f"group_{j}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict the target value(s).

        Args:
            x: shape (batch_size, input_dim) or (input_dim,) for single sample.
        Returns:
            shape (batch_size,) or scalar predicted value.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze_out = True
        else:
            squeeze_out = False

        h = F.relu(self.fc1(x))
        out = self.fc2(h).squeeze(-1)

        if squeeze_out:
            out = out.squeeze(0)
        return out

    def get_input_importance(self) -> torch.Tensor:
        """Compute per-variable importance from first-layer weights.

        For each source variable j, sum the L2 norms of the columns of W1
        corresponding to features of variable j.

        Returns:
            Tensor of shape (n_variables,) with non-negative importance scores.
        """
        # W1 shape: (hidden_dim, input_dim)
        W1 = self.fc1.weight
        importance = torch.zeros(self.n_variables, device=W1.device)

        for j in range(self.n_variables):
            group_idx = getattr(self, f"group_{j}")
            if group_idx.numel() == 0:
                # No features belong to this variable (e.g., self in Task 2A)
                continue
            # Select columns corresponding to variable j
            W_j = W1[:, group_idx]  # shape (hidden_dim, n_features_for_j)
            # L2 norm across all entries (matrix Frobenius norm)
            importance[j] = torch.linalg.norm(W_j, ord="fro")

        return importance

    def group_lasso_loss(self) -> torch.Tensor:
        """Compute group-lasso penalty on first-layer weights.

        The penalty is the sum, over source variables j, of the L2 norm of
        the column block corresponding to variable j. This encourages entire
        variable-groups to be zeroed out.

        Returns:
            Scalar tensor (the penalty value).
        """
        # Same as sum of importance values (by construction)
        return self.get_input_importance().sum()