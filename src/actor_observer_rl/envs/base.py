"""Base environment contracts for Actor-Observer experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class StepResult:
    """Result of one environment interaction.

    `state` is the full environment state `s_t`; `observation` is the
    actor-facing observation `o_t`. They are deliberately separate so partial
    observability can be introduced without changing the trainer contract.
    """

    state: Any
    observation: Any
    reward: float
    done: bool = False
    info: dict[str, Any] = field(default_factory=dict)


class BaseEnvironment(Protocol):
    """Minimal environment protocol inspired by Gymnasium, kept custom."""

    def reset(self, seed: int | None = None) -> tuple[Any, dict[str, Any]]:
        """Reset environment state and return actor observation plus metadata."""
        ...

    def step(self, action: int) -> StepResult:
        """Apply a discrete action and return state, observation, reward, done."""
        ...

    def get_full_state(self) -> Any:
        """Return the full state `s_t` when available."""
        ...

    def get_actor_observation(self) -> Any:
        """Return the actor-facing observation `o_t`."""
        ...

    def get_reward(self, state: Any, action: int) -> float:
        """Return scalar reward `r_t` for a state-action pair."""
        ...
