"""Continuous-feature policy (MLP): for compressed messages, not scalar IDs.

This module is the right head when the Actor input is a *continuous* feature
(e.g. Amanda's imperfect-state message ``m = f_phi(s)``). It is *not* the
policy for exact integer state IDs: probe scale tests show MLP-on-scalar
collapses as K grows (K=8 → ~0.33, K=16 → ~0.10), and stacking an MLP on top
of an embedding is strictly worse than embedding + linear. For scalar IDs use
``EmbeddingPolicy`` with ``num_hidden_layers=0``.
"""

from __future__ import annotations

import torch

from actor_observer_rl.agents.actor import ActionResult


class NeuralPolicy(torch.nn.Module):
    """One-hidden-layer MLP for continuous inputs (message / feature vectors)."""

    def __init__(self, input_dim: int, hidden_dim: int, num_actions: int) -> None:
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if num_actions <= 1:
            raise ValueError("num_actions must be greater than one")
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.network(input_tensor.float())

    def action_distribution(self, input_tensor: torch.Tensor) -> torch.distributions.Categorical:
        logits = self(input_tensor)
        if not torch.isfinite(logits).all():
            raise FloatingPointError("policy logits contain non-finite values")
        return torch.distributions.Categorical(logits=logits)

    def act(self, input_tensor: torch.Tensor, sample: bool = True) -> ActionResult:
        distribution = self.action_distribution(input_tensor)
        action = distribution.sample() if sample else distribution.probs.argmax(dim=-1)
        log_prob = distribution.log_prob(action)
        return ActionResult(action=action, log_prob=log_prob, probs=distribution.probs)
