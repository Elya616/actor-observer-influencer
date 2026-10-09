"""Tests for Case 3 partial-observability environment behavior."""

from __future__ import annotations

import pytest
import torch

from actor_observer_rl.envs import PartialObservableStateEnv


def test_reset_returns_zero_actor_observation_and_state_index() -> None:
    env = PartialObservableStateEnv(
        reward_table=[[1.0, 0.0], [0.0, 1.0]],
        state_distribution=[0.0, 1.0],
        observation_dim=1,
    )

    observation, info = env.reset(seed=123)

    assert torch.equal(observation, torch.tensor([0.0]))
    assert info == {"seed": 123, "state_index": 1}
    assert torch.equal(env.get_full_state(), torch.tensor([0.0, 1.0]))
    assert env.num_states == 2
    assert env.num_actions == 2
    assert env.state_dim == 2


def test_step_returns_full_state_zero_observation_and_reward() -> None:
    env = PartialObservableStateEnv(
        reward_table=[[1.0, 0.0], [0.25, 2.0]],
        state_distribution=[0.0, 1.0],
        observation_dim=2,
    )
    env.reset(seed=0)

    result = env.step(1)

    assert torch.equal(result.state, torch.tensor([0.0, 1.0]))
    assert torch.equal(result.observation, torch.tensor([0.0, 0.0]))
    assert result.reward == 2.0
    assert result.done is True
    assert result.info == {"action": 1, "state_index": 1}


def test_actor_observation_is_same_for_different_states() -> None:
    env = PartialObservableStateEnv(
        reward_table=[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
        state_distribution=[1.0, 0.0, 0.0],
        observation_dim=3,
    )

    observations = [env.actor_observation_for_index(index) for index in range(3)]

    assert all(torch.equal(observation, torch.zeros(3)) for observation in observations)


def test_state_sampling_is_reproducible_after_seeded_reset() -> None:
    def sample_sequence() -> list[int]:
        env = PartialObservableStateEnv(
            reward_table=[[1.0, 0.0], [0.0, 1.0], [0.2, 0.3]],
            state_distribution=[0.2, 0.3, 0.5],
            observation_dim=1,
        )
        indices: list[int] = []
        for step in range(10):
            _, info = env.reset(seed=7 if step == 0 else None)
            indices.append(int(info["state_index"]))
        return indices

    assert sample_sequence() == sample_sequence()


def test_get_reward_accepts_one_hot_state_tensor() -> None:
    env = PartialObservableStateEnv(
        reward_table=[[1.0, 0.0], [0.25, 2.0]],
        state_distribution=[0.5, 0.5],
        observation_dim=1,
    )

    reward = env.get_reward(torch.tensor([0.0, 1.0]), action=1)

    assert reward == 2.0


def test_step_before_reset_is_rejected() -> None:
    env = PartialObservableStateEnv(
        reward_table=[[1.0, 0.0], [0.0, 1.0]],
        state_distribution=[0.5, 0.5],
        observation_dim=1,
    )

    with pytest.raises(RuntimeError, match="reset must be called"):
        env.step(0)


def test_invalid_observation_dim_is_rejected() -> None:
    with pytest.raises(ValueError, match="observation_dim"):
        PartialObservableStateEnv(
            reward_table=[[1.0, 0.0], [0.0, 1.0]],
            state_distribution=[0.5, 0.5],
            observation_dim=0,
        )


def test_invalid_distribution_is_rejected() -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        PartialObservableStateEnv(
            reward_table=[[1.0, 0.0], [0.0, 1.0]],
            state_distribution=[0.7, 0.7],
            observation_dim=1,
        )
