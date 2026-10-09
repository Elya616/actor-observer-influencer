"""Softmax policy skeleton."""

from __future__ import annotations

import torch

from actor_observer_rl.agents.actor import ActionResult


class SoftmaxPolicy(torch.nn.Module):
    """Linear logits followed by a categorical action distribution."""

    def __init__(self, input_dim: int, num_actions: int) -> None:
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if num_actions <= 1:
            raise ValueError("num_actions must be greater than one")
        super().__init__()
        self.input_dim = input_dim
        self.num_actions = num_actions
        self.linear = torch.nn.Linear(input_dim, num_actions)

    def init_as_sum_decoder(
        self,
        sharpness: float = 4.0,
        *,
        message_scale: float = 1.0,
        message_shift: float = 0.0,
    ) -> None:
        """Initialize scalar logits as distance scores to action centers."""
        if self.input_dim != 1:
            raise ValueError("sum-style initialization requires input_dim=1")
        if sharpness <= 0.0:
            raise ValueError("sharpness must be positive")
        if message_scale <= 0.0:
            raise ValueError("message_scale must be positive")
        actions = torch.arange(self.num_actions, dtype=self.linear.weight.dtype)
        with torch.no_grad():
            self.linear.weight.copy_(
                (2.0 * sharpness * actions * float(message_scale)).reshape(-1, 1)
            )
            self.linear.bias.copy_(
                2.0 * sharpness * actions * float(message_shift)
                - sharpness * actions.pow(2)
            )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.linear(input_tensor.float())

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
