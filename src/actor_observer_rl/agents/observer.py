"""Observer wrapper.

The Observer is the explicit reward channel between the environment and the
Actor. The project PDF defines the Observer as the agent that "receives r_t
and reports it to the Actor". Case 1 uses an identity reward channel; later
cases may add noise, scaling, message production, or learnable compression
`f_phi`.
"""

from __future__ import annotations

from typing import Iterator

import torch


class ObserverMessageModule(torch.nn.Module):
    """One-layer implementation of the Observer message function `f_phi`.

    In Case 3, the Observer receives the full state `s_t` and emits a message
    `m_t = f_phi(s_t)`. The first foundation uses a single linear layer so the
    learnable parameters `phi` are easy to inspect and gradients remain clear.
    """

    def __init__(self, input_dim: int, message_dim: int) -> None:
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if message_dim <= 0:
            raise ValueError("message_dim must be positive")
        super().__init__()
        self.input_dim = input_dim
        self.message_dim = message_dim
        self.linear = torch.nn.Linear(input_dim, message_dim)

    def forward(self, full_state: torch.Tensor) -> torch.Tensor:
        return self.linear(full_state.float())


class MaskedObserverMessageModule(torch.nn.Module):
    """One-layer message function with a fixed input mask.

    This is useful for controlled imperfect-state experiments where Observers
    differ in the raw state information available to their compression
    function. A zero mask gives an Observer no state information; a partial
    mask exposes only selected features; an all-ones mask recovers the standard
    one-layer message module.
    """

    def __init__(
        self,
        input_dim: int,
        message_dim: int,
        input_mask: torch.Tensor | list[float],
    ) -> None:
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if message_dim <= 0:
            raise ValueError("message_dim must be positive")
        super().__init__()
        mask = torch.as_tensor(input_mask, dtype=torch.float32)
        if mask.shape != torch.Size([input_dim]):
            raise ValueError("input_mask length must match input_dim")
        self.input_dim = input_dim
        self.message_dim = message_dim
        self.register_buffer("input_mask", mask)
        self.linear = torch.nn.Linear(input_dim, message_dim)

    def forward(self, full_state: torch.Tensor) -> torch.Tensor:
        state = full_state.float()
        if state.shape[-1] != self.input_dim:
            raise ValueError("full_state length must match input_dim")
        return self.linear(state * self.input_mask)


class FixedTableMessageModule(torch.nn.Module):
    """Fixed message table for controlled Observer-selection experiments.

    The module expects a one-hot full-state vector and returns the corresponding
    row of a fixed message table. For example, a scalar table
    ``[0, 1, 1, 2]`` implements the known compression ``x_1 + x_2`` for the
    binary states ``(0,0), (1,0), (0,1), (1,1)``.
    """

    def __init__(self, messages_by_state: torch.Tensor | list[float] | list[list[float]]) -> None:
        super().__init__()
        table = torch.as_tensor(messages_by_state, dtype=torch.float32)
        if table.ndim == 1:
            table = table.unsqueeze(-1)
        if table.ndim != 2:
            raise ValueError("messages_by_state must be a 1-D or 2-D table")
        if table.shape[0] <= 0:
            raise ValueError("messages_by_state must contain at least one state")
        if table.shape[1] <= 0:
            raise ValueError("messages_by_state must contain at least one message dimension")
        self.input_dim = int(table.shape[0])
        self.message_dim = int(table.shape[1])
        self.register_buffer("message_table", table)

    def forward(self, full_state: torch.Tensor) -> torch.Tensor:
        state = full_state.float()
        if state.shape[-1] != self.input_dim:
            raise ValueError("full_state length must match message table state count")
        return state @ self.message_table


class Observer:
    """Observer that forwards rewards and optionally produces messages.

    For Case 1 the Observer is the conduit through which the Actor receives
    `r_t`; the trainer routes every environment reward through
    `report_reward` before it reaches the policy-gradient update. Case 3
    additionally uses `message` to emit `m_t = f_phi(s_t)` while preserving
    the autograd path into observer parameters.
    """

    def __init__(self, message_module: torch.nn.Module | None = None) -> None:
        self.message_module = message_module

    def message(self, full_state: torch.Tensor) -> torch.Tensor:
        if self.message_module is None:
            raise RuntimeError("Observer.message requires a message_module")
        return self.message_module(full_state)

    def report_reward(self, reward: float | torch.Tensor) -> float:
        """Forward the env reward to the Actor.

        Case 1: identity transformation. Subclasses may override to add
        noise, delay, or aggregation across multiple observers.
        """
        if isinstance(reward, torch.Tensor):
            return float(reward.detach().cpu().item())
        return float(reward)

    def parameters(self) -> Iterator[torch.nn.Parameter]:
        if self.message_module is None:
            return iter(())
        return self.message_module.parameters()
