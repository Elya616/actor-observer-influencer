"""Tests for the Phase 2 noisy single-state actor bandit."""

from __future__ import annotations

import pytest
import torch

from actor_observer_rl.agents import BudgetedObserver
from actor_observer_rl.envs import NoisyActorBanditEnv
from actor_observer_rl.training import MultiActorTrainer, make_fixed_softmax_actor


def test_basic_shape_and_properties() -> None:
    env = NoisyActorBanditEnv(qualities=[1.0, 0.0, 0.0, 0.0])
    assert env.num_states == 1
    assert env.num_actions == 4
    assert env.state_dim == 1
    observation, info = env.reset(seed=0)
    assert observation.shape == torch.Size([1])
    assert info["state_index"] == 0


def test_mean_reward_matches_qualities() -> None:
    env = NoisyActorBanditEnv(qualities=[1.0, 0.5, 0.0])
    assert env.mean_reward(0) == pytest.approx(1.0)
    assert env.mean_reward(1) == pytest.approx(0.5)
    assert env.mean_reward(2) == pytest.approx(0.0)


def test_zero_noise_is_deterministic() -> None:
    env = NoisyActorBanditEnv(qualities=[1.0, 0.25], noise_std=0.0)
    env.reset(seed=0)
    for action in range(env.num_actions):
        assert env.step(action).reward == pytest.approx(env.mean_reward(action))


def test_noise_sample_mean_is_close_to_quality() -> None:
    env = NoisyActorBanditEnv(qualities=[2.0, 0.0], noise_std=1.0)
    env.reset(seed=123)
    samples = [env.step(0).reward for _ in range(5000)]
    assert abs(sum(samples) / len(samples) - 2.0) < 0.1


def test_step_requires_reset() -> None:
    env = NoisyActorBanditEnv(qualities=[1.0, 0.0])
    with pytest.raises(RuntimeError):
        env.step(0)


def test_invalid_configuration_raises() -> None:
    with pytest.raises(ValueError):
        NoisyActorBanditEnv(qualities=[1.0])
    with pytest.raises(ValueError):
        NoisyActorBanditEnv(qualities=[1.0, 0.0], noise_std=-0.1)


def test_invalid_action_raises() -> None:
    env = NoisyActorBanditEnv(qualities=[1.0, 0.0])
    env.reset(seed=0)
    with pytest.raises(IndexError):
        env.step(5)


def test_reset_reseeds_generator_reproducibly() -> None:
    env_a = NoisyActorBanditEnv(qualities=[1.0, 0.0], noise_std=1.0)
    env_b = NoisyActorBanditEnv(qualities=[1.0, 0.0], noise_std=1.0)
    env_a.reset(seed=7)
    env_b.reset(seed=7)
    assert env_a.step(0).reward == pytest.approx(env_b.step(0).reward)


def test_observer_concentrates_budget_on_good_actor_small_crowd() -> None:
    # Smoke test: with a small crowd and clear gap the budgeted observer should
    # put most of its budget on the single good actor.
    num_actors = 4
    env = NoisyActorBanditEnv(qualities=[1.0, 0.0, 0.0, 0.0], noise_std=1.0)
    actors = [
        make_fixed_softmax_actor(num_states=1, num_actions=num_actors, target_action=j)
        for j in range(num_actors)
    ]
    observer = BudgetedObserver(
        num_actors=num_actors,
        budget=1.0,
        utility_learning_rate=0.05,
        following_learning_rate=0.05,
        c=1.0,
        seed=0,
    )
    trainer = MultiActorTrainer(
        env=env,
        actors=actors,
        observer=observer,
        actor_selection_probs=[1.0 / num_actors] * num_actors,
        actor_optimizers=None,
    )
    trace = trainer.run(num_steps=6000, seed=0)
    final_mu = trace.final_following_rates
    assert final_mu is not None
    assert final_mu[0] / sum(final_mu) > 0.8
