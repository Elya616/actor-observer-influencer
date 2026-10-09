"""Tests for the structured grid-world scaling environment."""

from __future__ import annotations

import math

import pytest
import torch

from actor_observer_rl.agents import Actor, Observer
from actor_observer_rl.envs import GridWorldBanditEnv
from actor_observer_rl.policies import NeuralPolicy
from actor_observer_rl.training import Trainer


def test_num_states_and_feature_range() -> None:
    env = GridWorldBanditEnv(grid_size=8)
    assert env.num_states == 64
    assert env.state_dim == 2
    for index in range(env.num_states):
        feature = env.feature(index)
        assert feature.shape == torch.Size([2])
        assert 0.0 < float(feature[0]) < 1.0
        assert 0.0 < float(feature[1]) < 1.0


def test_quadrant_optimal_actions_for_2x2() -> None:
    # 2x2 grid cell centers coincide with the four region centers, so each cell
    # has a distinct best action covering all four actions exactly once.
    env = GridWorldBanditEnv(grid_size=2, num_actions=4)
    optimal = env.optimal_actions()
    assert sorted(optimal) == [0, 1, 2, 3]


def test_correct_action_gives_highest_reward() -> None:
    env = GridWorldBanditEnv(grid_size=6, num_actions=4)
    for index in range(env.num_states):
        best = env.optimal_action(index)
        rewards = [env.get_reward(index, a) for a in range(env.num_actions)]
        assert max(range(env.num_actions), key=rewards.__getitem__) == best
        assert rewards[best] > 0.0


def test_reward_is_smooth_across_neighbors() -> None:
    # Neighboring cells should have similar reward for a fixed action (no
    # discontinuous high-variance jumps within a region).
    env = GridWorldBanditEnv(grid_size=16, num_actions=4)
    action = 0
    max_jump = 0.0
    for row in range(env.grid_size):
        for col in range(env.grid_size - 1):
            left = row * env.grid_size + col
            right = row * env.grid_size + col + 1
            jump = abs(env.get_reward(left, action) - env.get_reward(right, action))
            max_jump = max(max_jump, jump)
    # Adjacent cells differ by ~1/grid_size in coordinates; the reward gradient
    # is bounded, so the per-step jump stays small.
    assert max_jump < 0.2


def test_representative_states_cover_all_actions() -> None:
    env = GridWorldBanditEnv(grid_size=8, num_actions=4)
    representatives = env.representative_states()
    assert len(representatives) == 4
    optimal = [env.optimal_action(index) for index in representatives]
    assert sorted(optimal) == [0, 1, 2, 3]


def test_reset_is_reproducible_with_seed() -> None:
    env_a = GridWorldBanditEnv(grid_size=8)
    env_b = GridWorldBanditEnv(grid_size=8)
    _, info_a = env_a.reset(seed=123)
    _, info_b = env_b.reset(seed=123)
    assert info_a["state_index"] == info_b["state_index"]


def test_step_requires_reset() -> None:
    env = GridWorldBanditEnv(grid_size=4)
    with pytest.raises(RuntimeError):
        env.step(0)


def test_policy_learns_correct_actions_on_small_grid() -> None:
    grid_size = 4
    env = GridWorldBanditEnv(grid_size=grid_size, num_actions=4)
    torch.manual_seed(0)
    policy = NeuralPolicy(input_dim=2, hidden_dim=32, num_actions=4)
    actor = Actor(policy=policy)
    optimizer = torch.optim.Adam(actor.parameters(), lr=0.01)
    trainer = Trainer(
        env=env,
        actor=actor,
        actor_optimizer=optimizer,
        observer=Observer(),
        optimal_actions=env.optimal_actions(),
        reward_baseline="running_mean",
    )
    trainer.run(num_steps=15000, seed=0)

    optimal = env.optimal_actions()
    correct = 0
    with torch.no_grad():
        for index in range(env.num_states):
            distribution = actor.policy.action_distribution(env.feature(index))
            if int(distribution.probs.argmax().item()) == optimal[index]:
                correct += 1
    assert correct / env.num_states >= 0.9


def test_region_centers_count_matches_actions() -> None:
    env = GridWorldBanditEnv(grid_size=4, num_actions=9)
    assert env._region_centers.shape == torch.Size([9, 2])
    # every action index should be the best action for at least its own center
    for action in range(9):
        center = env._region_centers[action]
        distances = torch.linalg.vector_norm(env._region_centers - center, dim=1)
        assert int(torch.argmin(distances).item()) == action


def test_invalid_configuration_raises() -> None:
    with pytest.raises(ValueError):
        GridWorldBanditEnv(grid_size=0)
    with pytest.raises(ValueError):
        GridWorldBanditEnv(grid_size=4, num_actions=1)
    with pytest.raises(ValueError):
        GridWorldBanditEnv(grid_size=4, reward_sigma=0.0)


def test_reward_stays_in_unit_interval() -> None:
    env = GridWorldBanditEnv(grid_size=10, num_actions=4)
    assert math.isclose(env.reward_sigma, 0.25, rel_tol=1e-6)
    for index in range(env.num_states):
        for action in range(env.num_actions):
            reward = env.get_reward(index, action)
            assert 0.0 <= reward <= 1.0
