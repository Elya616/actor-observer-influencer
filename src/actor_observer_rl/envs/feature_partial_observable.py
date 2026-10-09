"""Partial-observation environment with explicit full-state features."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import torch

from actor_observer_rl.envs.base import StepResult


@dataclass
class FeaturePartialObservableStateEnv:
    """One-step environment where the Observer sees configured state features.

    `PartialObservableStateEnv` represents the full state as a one-hot state
    index. This variant keeps the same trainer contract but lets experiments
    expose explicit feature vectors such as `(x_1, x_2)` to the Observer. The
    Actor observation remains an uninformative zero vector.
    """

    state_features: Sequence[Sequence[float]]
    reward_table: Sequence[Sequence[float]]
    state_distribution: Sequence[float]
    observation_dim: int
    _state_features: list[list[float]] = field(init=False, repr=False)
    _reward_table: list[list[float]] = field(init=False, repr=False)
    _state_distribution_tensor: torch.Tensor = field(init=False, repr=False)
    _generator: torch.Generator = field(init=False, repr=False)
    _current_state_index: int | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        features = [[float(value) for value in row] for row in self.state_features]
        if not features:
            raise ValueError("state_features must contain at least one state")
        feature_dim = len(features[0])
        if feature_dim <= 0:
            raise ValueError("state_features must contain at least one feature")
        if any(len(row) != feature_dim for row in features):
            raise ValueError("state_features rows must all have the same length")

        rewards = [[float(reward) for reward in row] for row in self.reward_table]
        if len(rewards) != len(features):
            raise ValueError("reward_table row count must match state_features")
        num_actions = len(rewards[0])
        if num_actions <= 1:
            raise ValueError("reward_table must contain at least two actions")
        if any(len(row) != num_actions for row in rewards):
            raise ValueError("reward_table rows must all have the same action count")

        probabilities = [float(probability) for probability in self.state_distribution]
        if len(probabilities) != len(features):
            raise ValueError("state_distribution length must match state_features")
        if any(probability < 0.0 for probability in probabilities):
            raise ValueError("state_distribution cannot contain negative probabilities")
        if not math.isclose(sum(probabilities), 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("state_distribution must sum to 1")
        if self.observation_dim <= 0:
            raise ValueError("observation_dim must be positive")

        self.state_features = features
        self.reward_table = rewards
        self.state_distribution = probabilities
        self._state_features = features
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
        return len(self._state_features[0])

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
        return self.full_state_for_index(self._require_current_state_index())

    def get_actor_observation(self) -> torch.Tensor:
        self._require_current_state_index()
        return torch.zeros(self.observation_dim, dtype=torch.float32)

    def get_reward(self, state: object, action: int) -> float:
        state_index = self._state_to_index(state)
        action_index = int(action)
        if action_index < 0 or action_index >= self.num_actions:
            raise IndexError(
                f"action {action_index} is outside valid range [0, {self.num_actions})"
            )
        return self._reward_table[state_index][action_index]

    def full_state_for_index(self, state_index: int) -> torch.Tensor:
        return torch.tensor(
            self._state_features[self._validate_state_index(state_index)],
            dtype=torch.float32,
        )

    def actor_observation_for_index(self, state_index: int) -> torch.Tensor:
        self._validate_state_index(state_index)
        return torch.zeros(self.observation_dim, dtype=torch.float32)

    def state_probability(self, state_index: int) -> float:
        return self.state_distribution[self._validate_state_index(state_index)]

    def _sample_state_index(self) -> int:
        sample = torch.multinomial(
            self._state_distribution_tensor,
            num_samples=1,
            replacement=True,
            generator=self._generator,
        )
        return int(sample.item())

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
                state_vector = state.detach().cpu().float()
                for index, features in enumerate(self._state_features):
                    if torch.allclose(
                        state_vector,
                        torch.tensor(features, dtype=torch.float32),
                    ):
                        return index
                raise ValueError("state tensor does not match a configured state")
        else:
            raise TypeError("state must be an int state index or feature tensor")
        return self._validate_state_index(state_index)

    def _validate_state_index(self, state_index: int) -> int:
        if state_index < 0 or state_index >= self.num_states:
            raise IndexError(
                f"state {state_index} is outside valid range [0, {self.num_states})"
            )
        return int(state_index)

