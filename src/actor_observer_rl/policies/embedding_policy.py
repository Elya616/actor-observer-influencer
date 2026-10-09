"""Embedding-based policy for discrete scalar state IDs.

Professor Marbach asked us to try the Gemini suggestion: instead of feeding a
raw scalar ID into an MLP (where nearby IDs are treated as similar), look the
ID up in a learnable embedding table -- the same idea as word embeddings in
NLP. Each state ID becomes an independent vector; nearby IDs are *not* forced
to share structure unless training learns that.

This policy is for exact integer state IDs (perfect-state / probe setting). It
is *not* a drop-in for a continuous compressed message ``m = f_phi(s)``; that
case still needs a continuous feature map.
"""

from __future__ import annotations

import torch

from actor_observer_rl.agents.actor import ActionResult


class EmbeddingPolicy(torch.nn.Module):
    """``Embedding(num_states, d)`` then optional ReLU MLP, then action logits.

    Default (and recommended for scalar IDs): ``num_hidden_layers=0`` —
    embedding lookup then a single linear map to action logits. Scale / lr
    stress tests: pure embed beats embed+MLP1 (K=16: 0.96 vs 0.42) and beats
    MLP-on-scalar entirely. Keep hidden layers at 0 unless you have a reason.
    """

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        embedding_dim: int = 16,
        hidden_dim: int = 64,
        num_hidden_layers: int = 0,
    ) -> None:
        if num_states <= 0:
            raise ValueError("num_states must be positive")
        if num_actions <= 1:
            raise ValueError("num_actions must be greater than one")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if num_hidden_layers < 0:
            raise ValueError("num_hidden_layers must be non-negative")
        if num_hidden_layers > 0 and hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive when using hidden layers")
        super().__init__()
        self.num_states = num_states
        self.num_actions = num_actions
        self.embedding_dim = embedding_dim
        self.embedding = torch.nn.Embedding(num_states, embedding_dim)

        layers: list[torch.nn.Module] = []
        in_dim = embedding_dim
        for _ in range(num_hidden_layers):
            layers.append(torch.nn.Linear(in_dim, hidden_dim))
            layers.append(torch.nn.ReLU())
            in_dim = hidden_dim
        layers.append(torch.nn.Linear(in_dim, num_actions))
        self.head = torch.nn.Sequential(*layers)

    def _ids(self, input_tensor: torch.Tensor) -> torch.Tensor:
        # Scalar-ID observations arrive as floats; embedding needs Long indices.
        ids = input_tensor.float().reshape(-1).round().long()
        if torch.any(ids < 0) or torch.any(ids >= self.num_states):
            raise ValueError(
                f"state id out of range for Embedding(num_states={self.num_states})"
            )
        return ids

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        features = self.embedding(self._ids(input_tensor))
        logits = self.head(features)
        return logits.squeeze(0) if logits.shape[0] == 1 else logits

    def action_distribution(
        self, input_tensor: torch.Tensor
    ) -> torch.distributions.Categorical:
        logits = self(input_tensor)
        if not torch.isfinite(logits).all():
            raise FloatingPointError("policy logits contain non-finite values")
        return torch.distributions.Categorical(logits=logits)

    def act(self, input_tensor: torch.Tensor, sample: bool = True) -> ActionResult:
        distribution = self.action_distribution(input_tensor)
        action = distribution.sample() if sample else distribution.probs.argmax(dim=-1)
        log_prob = distribution.log_prob(action)
        return ActionResult(action=action, log_prob=log_prob, probs=distribution.probs)
