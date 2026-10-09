"""Case 1 single-state environment.

This environment implements the PDF's bandit-like setting where `|S| = 1`.
The single full state and the actor observation are both the static dummy
vector `s_static`; rewards come from the explicit table `r(a)`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import torch

from actor_observer_rl.envs.base import StepResult


@dataclass
class SingleStateBanditEnv:
    """Single static state with reward table `r(a)`.

    The environment has no transition dynamics. Each call to `step(action)`
    returns the same full state and observation, plus `reward_table[action]`.
    """

    reward_table: Sequence[float]
    state: Sequence[float] = (1.0,)
    _state_tensor: torch.Tensor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        rewards = [float(reward) for reward in self.reward_table]
        if not rewards:
            raise ValueError("reward_table must contain at least one action reward")

        state_values = [float(value) for value in self.state]
        if not state_values:
            raise ValueError("state must contain at least one value")

        self.reward_table = rewards
        self._state_tensor = torch.tensor(state_values, dtype=torch.float32)

    @property
    def num_actions(self) -> int:
        return len(self.reward_table)

    @property
    def state_dim(self) -> int:
        return int(self._state_tensor.numel())

    def reset(self, seed: int | None = None) -> tuple[torch.Tensor, dict[str, int | None]]:
        return self.get_actor_observation(), {"seed": seed, "state_index": 0}

    def step(self, action: int) -> StepResult:
        reward = self.get_reward(self.get_full_state(), action)
        return StepResult(
            state=self.get_full_state(),
            observation=self.get_actor_observation(),
            reward=reward,
            done=False,
            info={"action": int(action), "state_index": 0},
        )

    def get_full_state(self) -> torch.Tensor:
        return self._state_tensor.clone()

    def get_actor_observation(self) -> torch.Tensor:
        return self._state_tensor.clone()

    def get_reward(self, state: object, action: int) -> float:
        del state
        action_index = int(action)
        if action_index < 0 or action_index >= self.num_actions:
            raise IndexError(
                f"action {action_index} is outside valid range [0, {self.num_actions})"
            )
        return self.reward_table[action_index]
