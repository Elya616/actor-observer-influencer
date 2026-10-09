"""Tests for Case 1 single-state environment behavior."""

from __future__ import annotations

import pytest
import torch

from actor_observer_rl.envs.single_state import SingleStateBanditEnv


def test_reset_returns_static_actor_observation() -> None:
    env = SingleStateBanditEnv(reward_table=[0.0, 1.0, 0.2])

    observation, info = env.reset(seed=123)

    assert torch.equal(observation, torch.tensor([1.0]))
    assert info == {"seed": 123, "state_index": 0}
    assert env.num_actions == 3
    assert env.state_dim == 1


def test_step_returns_reward_for_selected_action() -> None:
    env = SingleStateBanditEnv(reward_table=[0.0, 1.0, 0.2])

    result = env.step(1)

    assert torch.equal(result.state, torch.tensor([1.0]))
    assert torch.equal(result.observation, torch.tensor([1.0]))
    assert result.reward == 1.0
    assert result.done is False
    assert result.info["action"] == 1


def test_invalid_action_raises_clear_error() -> None:
    env = SingleStateBanditEnv(reward_table=[0.0, 1.0, 0.2])

    with pytest.raises(IndexError, match="outside valid range"):
        env.step(3)


def test_empty_reward_table_is_rejected() -> None:
    with pytest.raises(ValueError, match="reward_table"):
        SingleStateBanditEnv(reward_table=[])
