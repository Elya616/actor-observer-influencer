"""Single-state noisy bandit for the Phase 2 (scale-the-actors) experiment.

Phase 2 of the scaling plan keeps the number of states small and instead grows
the number of Actors in the one-Observer / many-Actor setting of paper Section 4.
There a single ``BudgetedObserver`` holds a following-rate vector ``mu`` on the
capped simplex ``{mu >= 0, sum(mu) <= B}`` and must concentrate its *fixed*
budget on the high-utility Actors.

To isolate that attention mechanism we use the simplest possible task: one
state and one action per Actor. Actor ``j`` is a fixed-policy actor that always
plays action ``j``, so action ``j`` is a stand-in for "follow Actor ``j``".
Action ``a`` pays a mean quality ``q_a`` plus Gaussian noise:

    r = q_a + sigma * N(0, 1).

The noise is the whole point. With a deterministic table the Observer would
estimate every Actor's utility exactly from a single sample, so finding the good
Actors would be trivial no matter how many there are. With noise the Observer
needs several samples per Actor to separate good from bad, and under an equal
total-step budget each Actor is sampled fewer times as the crowd grows. That is
the scaling tension we want to measure: does the budgeted-following mechanism
still find the fixed set of good Actors as ``num_actions`` (the crowd) grows?

The interface mirrors ``IIDStateBanditEnv`` (single state, one-hot observation of
dimension 1) so the existing ``MultiActorTrainer`` drives it unchanged. The
generator is named ``_generator`` so the trainer can reseed it for reproducible
sub-streams.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import torch

from actor_observer_rl.envs.base import StepResult


@dataclass
class NoisyActorBanditEnv:
    """One-state bandit with per-action mean quality and Gaussian reward noise."""

    qualities: Sequence[float]
    noise_std: float = 1.0
    _qualities: torch.Tensor = field(init=False, repr=False)
    _generator: torch.Generator = field(init=False, repr=False)
    _is_reset: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        qualities = [float(value) for value in self.qualities]
        if len(qualities) < 2:
            raise ValueError("qualities must contain at least two actions")
        if self.noise_std < 0.0:
            raise ValueError("noise_std must be non-negative")
        self.qualities = qualities
        self._qualities = torch.tensor(qualities, dtype=torch.float32)
        self.noise_std = float(self.noise_std)
        self._generator = torch.Generator()

    @property
    def num_states(self) -> int:
        return 1

    @property
    def num_actions(self) -> int:
        return self._qualities.numel()

    @property
    def state_dim(self) -> int:
        return 1

    def reset(self, seed: int | None = None) -> tuple[torch.Tensor, dict[str, object]]:
        if seed is not None:
            self._generator.manual_seed(seed)
        self._is_reset = True
        return self.get_actor_observation(), {"seed": seed, "state_index": 0}

    def step(self, action: int) -> StepResult:
        if not self._is_reset:
            raise RuntimeError("reset must be called before step")
        reward = self.get_reward(0, action)
        return StepResult(
            state=self.get_full_state(),
            observation=self.get_actor_observation(),
            reward=reward,
            done=True,
            info={"action": int(action), "state_index": 0},
        )

    def get_full_state(self) -> torch.Tensor:
        return torch.ones(1, dtype=torch.float32)

    def get_actor_observation(self) -> torch.Tensor:
        return torch.ones(1, dtype=torch.float32)

    def mean_reward(self, action: int) -> float:
        """Noise-free expected reward of an action (the Actor's true utility)."""

        return float(self._qualities[self._check_action(action)].item())

    def get_reward(self, state: object, action: int) -> float:
        action_index = self._check_action(action)
        mean = float(self._qualities[action_index].item())
        if self.noise_std == 0.0:
            return mean
        noise = torch.randn((), generator=self._generator).item()
        return mean + self.noise_std * float(noise)

    def _check_action(self, action: int) -> int:
        action_index = int(action)
        if action_index < 0 or action_index >= self.num_actions:
            raise IndexError(
                f"action {action_index} is outside valid range [0, {self.num_actions})"
            )
        return action_index
