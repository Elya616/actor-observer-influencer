"""Actor wrapper skeleton."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import torch


@dataclass(frozen=True)
class ActionResult:
    """Action sample and distribution data needed for policy-gradient updates."""

    action: torch.Tensor
    log_prob: torch.Tensor
    probs: torch.Tensor


class Actor:
    """Thin wrapper around a policy `pi_theta`."""

    def __init__(
        self,
        policy: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> None:
        self.policy = policy
        self.optimizer = optimizer

    def act(self, input_tensor: torch.Tensor, sample: bool = True) -> ActionResult:
        if not hasattr(self.policy, "act"):
            raise TypeError("policy must expose an act(input_tensor, sample) method")
        return self.policy.act(input_tensor, sample=sample)

    def update_from_loss(self, loss: torch.Tensor) -> None:
        if self.optimizer is None:
            raise RuntimeError("Actor.update_from_loss requires an optimizer")
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def parameters(self) -> Iterator[torch.nn.Parameter]:
        return self.policy.parameters()
