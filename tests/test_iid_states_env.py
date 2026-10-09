"""Tests for Case 2 i.i.d. state environment behavior."""

from __future__ import annotations

import pytest
import torch

from actor_observer_rl.envs import IIDStateBanditEnv


def test_reset_samples_one_hot_state_observation() -> None:
    env = IIDStateBanditEnv(
        reward_table=[[1.0, 0.0], [0.0, 1.0]],
        state_distribution=[0.0, 1.0],
        observation_mode="one_hot",
    )

    observation, info = env.reset(seed=123)

    assert torch.equal(observation, torch.tensor([0.0, 1.0]))
    assert info == {"seed": 123, "state_index": 1}
    assert env.num_states == 2
    assert env.num_actions == 2
    assert env.state_dim == 2
    assert env.observation_dim == 2


def test_reset_samples_scalar_state_observation_by_default() -> None:
    env = IIDStateBanditEnv(
        reward_table=[[1.0, 0.0], [0.0, 1.0]],
        state_distribution=[0.0, 1.0],
    )

    observation, info = env.reset(seed=123)

    assert torch.equal(observation, torch.tensor([1.0]))
    assert info == {"seed": 123, "state_index": 1}
    assert env.observation_dim == 1
    assert torch.equal(env.observation_for_index(0), torch.tensor([0.0]))


def test_step_returns_state_dependent_reward_and_done() -> None:
    env = IIDStateBanditEnv(
        reward_table=[[1.0, 0.0], [0.25, 2.0]],
        state_distribution=[0.0, 1.0],
        observation_mode="one_hot",
    )
    env.reset(seed=0)

    result = env.step(1)

    assert torch.equal(result.state, torch.tensor([0.0, 1.0]))
    assert torch.equal(result.observation, torch.tensor([0.0, 1.0]))
    assert result.reward == 2.0
    assert result.done is True
    assert result.info == {"action": 1, "state_index": 1}


def test_state_sampling_is_reproducible_after_seeded_reset() -> None:
    def sample_sequence() -> list[int]:
        env = IIDStateBanditEnv(
            reward_table=[[1.0, 0.0], [0.0, 1.0], [0.2, 0.3]],
            state_distribution=[0.2, 0.3, 0.5],
        )
        indices: list[int] = []
        for step in range(10):
            _, info = env.reset(seed=7 if step == 0 else None)
            indices.append(int(info["state_index"]))
        return indices

    assert sample_sequence() == sample_sequence()


def test_get_reward_accepts_one_hot_state_tensor() -> None:
    env = IIDStateBanditEnv(
        reward_table=[[1.0, 0.0], [0.25, 2.0]],
        state_distribution=[0.5, 0.5],
    )

    reward = env.get_reward(torch.tensor([0.0, 1.0]), action=1)

    assert reward == 2.0


def test_step_before_reset_is_rejected() -> None:
    env = IIDStateBanditEnv(
        reward_table=[[1.0, 0.0], [0.0, 1.0]],
        state_distribution=[0.5, 0.5],
    )

    with pytest.raises(RuntimeError, match="reset must be called"):
        env.step(0)


def test_invalid_distribution_is_rejected() -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        IIDStateBanditEnv(
            reward_table=[[1.0, 0.0], [0.0, 1.0]],
            state_distribution=[0.7, 0.7],
        )


def test_non_rectangular_reward_table_is_rejected() -> None:
    with pytest.raises(ValueError, match="same action count"):
        IIDStateBanditEnv(
            reward_table=[[1.0, 0.0], [1.0]],
            state_distribution=[0.5, 0.5],
        )
