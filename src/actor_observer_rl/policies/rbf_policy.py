"""Radial-basis-function policy over a scalar state message.

In the informational-action and combined settings, the policy conditions on a
compressed scalar message ``m = f_phi(s)`` rather than on a one-hot state vector.
A plain linear head on that scalar is brittle as the number of states grows.

``RBFPolicy`` keeps the scalar interface but lifts the scalar into a bank of
Gaussian bump features, one centered at each integer state ID, before a linear
softmax head. A message near ID ``k`` activates bump ``k``. This gives the
linear head one-hot-like separability while still consuming a single scalar.

The features are differentiable in ``m``, so gradients can still flow back to a
learned compression function ``phi``.
"""

from __future__ import annotations

import torch

from actor_observer_rl.agents.actor import ActionResult


class RBFPolicy(torch.nn.Module):
    """Gaussian radial-basis feature map on a scalar ID, then linear softmax.

    The constructor accepts both call styles used in the current branches:

    ``RBFPolicy(num_states=K, num_actions=K)``
        Explicit feature count and action count.

    ``RBFPolicy(num_actions=K)``
        Convenience form for the controlled identity-action cases where the
        number of RBF centers equals the number of actions.
    """

    def __init__(
        self,
        num_states: int | None = None,
        num_actions: int | None = None,
        sigma: float = 0.5,
    ) -> None:
        if num_actions is None:
            if num_states is None:
                raise ValueError("num_actions is required")
            num_actions = int(num_states)
        if num_states is None:
            num_states = int(num_actions)
        if num_states <= 0:
            raise ValueError("num_states must be positive")
        if num_actions <= 1:
            raise ValueError("num_actions must be greater than one")
        if sigma <= 0.0:
            raise ValueError("sigma must be positive")
        super().__init__()
        self.num_states = int(num_states)
        self.num_actions = int(num_actions)
        self.sigma = float(sigma)
        self.register_buffer(
            "centers",
            torch.arange(self.num_states, dtype=torch.float32),
        )
        self.head = torch.nn.Linear(self.num_states, self.num_actions)

    def features(self, input_tensor: torch.Tensor) -> torch.Tensor:
        message = input_tensor.float()
        if message.ndim == 0:
            message = message.reshape(1)
        if message.shape[-1] != 1:
            message = message.reshape(-1, 1)
        else:
            message = message.reshape(-1, 1)
        features = torch.exp(-((message - self.centers) ** 2) / (2.0 * self.sigma**2))
        return features.squeeze(0) if features.shape[0] == 1 else features

    def _features(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.features(input_tensor)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        logits = self.head(self.features(input_tensor))
        return logits.squeeze(0) if logits.shape[0] == 1 else logits

    def action_distribution(
        self,
        input_tensor: torch.Tensor,
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
