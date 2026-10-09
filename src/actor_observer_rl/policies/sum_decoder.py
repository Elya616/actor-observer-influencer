"""Fixed and learnable scalar-message policies for sum-message experiments."""

from __future__ import annotations

import torch

from actor_observer_rl.agents.actor import ActionResult


class SumDecoderPolicy(torch.nn.Module):
    """Map a scalar message near 0, 1, or 2 to the corresponding action.

    This policy is intentionally fixed. It provides a differentiable decoder
    for staged imperfect-state experiments where the expected useful message is
    known in advance, e.g. ``m = x_1 + x_2``.
    """

    def __init__(self, num_actions: int = 3, sharpness: float = 4.0) -> None:
        if num_actions <= 1:
            raise ValueError("num_actions must be greater than one")
        if sharpness <= 0.0:
            raise ValueError("sharpness must be positive")
        super().__init__()
        self.num_actions = int(num_actions)
        self.sharpness = float(sharpness)
        self.register_buffer(
            "action_values",
            torch.arange(self.num_actions, dtype=torch.float32),
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        message = input_tensor.float()
        if message.ndim == 0:
            message = message.reshape(1)
        if message.shape[-1] != 1:
            raise ValueError("SumDecoderPolicy expects a scalar message input")
        return -self.sharpness * (message - self.action_values).pow(2)

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


class LearnableSumDecoderPolicy(torch.nn.Module):
    """Quadratic scalar-message policy with learnable scale, shift, and sharpness."""

    def __init__(
        self,
        num_actions: int,
        *,
        init_sharpness: float = 4.0,
        init_scale: float = 1.0,
        init_shift: float = 0.0,
    ) -> None:
        if num_actions <= 1:
            raise ValueError("num_actions must be greater than one")
        if init_sharpness <= 0.0:
            raise ValueError("init_sharpness must be positive")
        super().__init__()
        self.num_actions = int(num_actions)
        self.register_buffer(
            "action_values",
            torch.arange(self.num_actions, dtype=torch.float32),
        )
        init_raw = float(torch.log(torch.expm1(torch.tensor(init_sharpness))))
        self.raw_beta = torch.nn.Parameter(torch.tensor(init_raw))
        self.scale = torch.nn.Parameter(torch.tensor(float(init_scale)))
        self.shift = torch.nn.Parameter(torch.tensor(float(init_shift)))

    @property
    def sharpness(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.raw_beta) + 1e-4

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        message = input_tensor.float()
        if message.ndim == 0:
            message = message.reshape(1)
        if message.shape[-1] != 1:
            raise ValueError("LearnableSumDecoderPolicy expects a scalar message")
        mapped = self.scale * message + self.shift
        square_distance = (mapped - self.action_values).pow(2).clamp(max=1.0e4)
        beta = self.sharpness.clamp(max=50.0)
        return -beta * square_distance

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
