"""Case 2 i.i.d. state environment.

This environment implements the PDF's contextual-bandit setting:

`s_t ~ D`, `a_t ~ pi_theta(. | s_t)`, and `r_t = r(s_t, a_t)`.

Each sampled state is treated as a one-step episode. After `step(action)`,
`done=True`, so the trainer resets the environment and samples the next i.i.d.
state before the following action.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import torch

from actor_observer_rl.envs.base import StepResult


@dataclass
class IIDStateBanditEnv:
    """Contextual bandit with i.i.d. sampled states and reward table `r(s, a)`.

    ``observation_mode`` controls how the (perfect) state is exposed to the
    Actor policy:

    * ``"scalar_id"`` (default): a length-1 vector holding the raw state index
        ``[0.0], [1.0], ...``. This is the scalar-state representation Professor
        Marbach asked for, matching the imperfect-state/combined track where the
        policy input is a single scalar. A purely linear policy must now express
        the state->action map as one affine function of that scalar, which is
        the setup we want to stress-test as ``num_states`` grows.
    * ``"one_hot"``: a length-``num_states`` one-hot vector. Every state has its
        own coordinate, so a linear policy has an independent set of action
        logits per state. This is the earlier representation, kept for the
        historical Section 4 / Section 6 one-hot baselines.
    """

    reward_table: Sequence[Sequence[float]]
    state_distribution: Sequence[float]
    observation_mode: str = "scalar_id"
    _reward_table: list[list[float]] = field(init=False, repr=False)
    _state_distribution_tensor: torch.Tensor = field(init=False, repr=False)
    _generator: torch.Generator = field(init=False, repr=False)
    _current_state_index: int | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        rewards = [[float(reward) for reward in row] for row in self.reward_table]
        if not rewards:
            raise ValueError("reward_table must contain at least one state")

        num_actions = len(rewards[0])
        if num_actions <= 1:
            raise ValueError("reward_table must contain at least two actions")
        if any(len(row) != num_actions for row in rewards):
            raise ValueError("reward_table rows must all have the same action count")

        probabilities = [float(probability) for probability in self.state_distribution]
        if len(probabilities) != len(rewards):
            raise ValueError("state_distribution length must match reward_table rows")
        if any(probability < 0.0 for probability in probabilities):
            raise ValueError("state_distribution cannot contain negative probabilities")
        if not math.isclose(sum(probabilities), 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("state_distribution must sum to 1")

        if self.observation_mode not in ("one_hot", "scalar_id"):
            raise ValueError("observation_mode must be 'one_hot' or 'scalar_id'")

        self.reward_table = rewards
        self.state_distribution = probabilities
        self._reward_table = rewards
        self._state_distribution_tensor = torch.tensor(probabilities, dtype=torch.float32)
        self._generator = torch.Generator()

    @property
    def num_states(self) -> int:
        return len(self._reward_table)

    @property
    def num_actions(self) -> int:
        return len(self._reward_table[0])

    @property
    def state_dim(self) -> int:
        return self.num_states

    @property
    def observation_dim(self) -> int:
        """Dimension of the Actor observation vector under the current mode."""
        return 1 if self.observation_mode == "scalar_id" else self.num_states

    def reset(self, seed: int | None = None) -> tuple[torch.Tensor, dict[str, int | None]]:
        if seed is not None:
            self._generator.manual_seed(seed)
        self._current_state_index = self._sample_state_index()
        return self.get_actor_observation(), {
            "seed": seed,
            "state_index": self._current_state_index,
        }

    def step(self, action: int) -> StepResult:
        if self._current_state_index is None:
            raise RuntimeError("reset must be called before step")

        state_index = self._current_state_index
        state = self.get_full_state()
        observation = self.get_actor_observation()
        reward = self.get_reward(state_index, action)
        return StepResult(
            state=state,
            observation=observation,
            reward=reward,
            done=True,
            info={"action": int(action), "state_index": state_index},
        )

    def get_full_state(self) -> torch.Tensor:
        return self._one_hot(self._require_current_state_index())

    def get_actor_observation(self) -> torch.Tensor:
        return self._observe(self._require_current_state_index())

    def observation_for_index(self, state_index: int) -> torch.Tensor:
        """Return the Actor observation for a given state (mode-aware).

        Useful for deterministic evaluation sweeps over every state.
        """
        return self._observe(self._state_to_index(state_index))

    def _observe(self, state_index: int) -> torch.Tensor:
        if self.observation_mode == "scalar_id":
            return torch.tensor([float(state_index)], dtype=torch.float32)
        return self._one_hot(state_index)

    def get_reward(self, state: object, action: int) -> float:
        state_index = self._state_to_index(state)
        action_index = int(action)
        if action_index < 0 or action_index >= self.num_actions:
            raise IndexError(
                f"action {action_index} is outside valid range [0, {self.num_actions})"
            )
        return self._reward_table[state_index][action_index]

    def _sample_state_index(self) -> int:
        sample = torch.multinomial(
            self._state_distribution_tensor,
            num_samples=1,
            replacement=True,
            generator=self._generator,
        )
        return int(sample.item())

    def _one_hot(self, state_index: int) -> torch.Tensor:
        return torch.nn.functional.one_hot(
            torch.tensor(state_index),
            num_classes=self.num_states,
        ).to(dtype=torch.float32)

    def _require_current_state_index(self) -> int:
        if self._current_state_index is None:
            raise RuntimeError("reset must be called before reading state")
        return self._current_state_index

    def _state_to_index(self, state: object) -> int:
        if isinstance(state, int):
            state_index = state
        elif isinstance(state, torch.Tensor):
            if state.ndim == 0:
                state_index = int(state.item())
            else:
                if state.numel() != self.num_states:
                    raise ValueError("state tensor length must match num_states")
                state_index = int(state.detach().cpu().argmax().item())
        else:
            raise TypeError("state must be an int state index or one-hot tensor")

        if state_index < 0 or state_index >= self.num_states:
            raise IndexError(
                f"state {state_index} is outside valid range [0, {self.num_states})"
            )
        return state_index
