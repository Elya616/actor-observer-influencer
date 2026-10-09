"""Tests for reward-weighted actor policy-gradient updates."""

from __future__ import annotations

import pytest
import torch

from actor_observer_rl.agents import Actor, Observer
from actor_observer_rl.envs import IIDStateBanditEnv, SingleStateBanditEnv
from actor_observer_rl.policies import SoftmaxPolicy
from actor_observer_rl.training import Trainer, reinforce_loss


def _zero_initialized_actor(input_dim: int = 1, num_actions: int = 3) -> Actor:
    policy = SoftmaxPolicy(input_dim=input_dim, num_actions=num_actions)
    with torch.no_grad():
        policy.linear.weight.zero_()
        policy.linear.bias.zero_()
    return Actor(policy)


def test_reinforce_loss_has_policy_gradient_sign() -> None:
    log_prob = torch.tensor(-0.5, requires_grad=True)

    loss = reinforce_loss(reward=2.0, log_prob=log_prob)
    loss.backward()

    assert loss.item() == pytest.approx(1.0)
    assert log_prob.grad is not None
    assert log_prob.grad.item() == pytest.approx(-2.0)


def test_case1_training_increases_best_action_probability() -> None:
    env = SingleStateBanditEnv(reward_table=[0.0, 1.0, 0.0])
    actor = _zero_initialized_actor()
    optimizer = torch.optim.SGD(actor.parameters(), lr=0.1)
    trainer = Trainer(env, actor, optimizer, optimal_action=1)

    initial_prob = actor.policy.action_distribution(env.get_actor_observation()).probs[1].item()
    trace = trainer.run(num_steps=800, seed=7)

    assert trace.final_optimal_action_prob is not None
    assert trace.final_optimal_action_prob > initial_prob
    assert trace.final_optimal_action_prob > 0.9
    assert all(torch.isfinite(torch.tensor(trace.losses)))


def test_case1_training_is_reproducible_with_fixed_seed() -> None:
    def run_once() -> tuple[list[int], list[float], list[float]]:
        env = SingleStateBanditEnv(reward_table=[0.0, 1.0, 0.0])
        actor = _zero_initialized_actor()
        optimizer = torch.optim.SGD(actor.parameters(), lr=0.1)
        trainer = Trainer(env, actor, optimizer, optimal_action=1)
        trace = trainer.run(num_steps=50, seed=11)
        return trace.actions, trace.rewards, trace.optimal_action_probs

    actions_a, rewards_a, probs_a = run_once()
    actions_b, rewards_b, probs_b = run_once()

    assert actions_a == actions_b
    assert rewards_a == rewards_b
    assert probs_a == pytest.approx(probs_b)


def test_trainer_routes_reward_through_observer() -> None:
    """Reward delivered to the actor must come from observer.report_reward."""

    seen: list[float] = []

    class RecordingObserver(Observer):
        def report_reward(self, reward):
            forwarded = super().report_reward(reward)
            seen.append(forwarded)
            return forwarded

    env = SingleStateBanditEnv(reward_table=[0.0, 1.0])
    actor = _zero_initialized_actor(num_actions=2)
    optimizer = torch.optim.SGD(actor.parameters(), lr=0.1)
    observer = RecordingObserver()
    trainer = Trainer(
        env,
        actor,
        actor_optimizer=optimizer,
        observer=observer,
        optimal_action=1,
    )

    trace = trainer.run(num_steps=20, seed=3)

    assert len(seen) == len(trace.rewards) == 20
    assert seen == trace.rewards


def test_observer_can_transform_reward_before_actor_sees_it() -> None:
    """A subclass scaling the reward should change what the actor learns from."""

    class ScalingObserver(Observer):
        def __init__(self, scale: float) -> None:
            super().__init__()
            self.scale = scale

        def report_reward(self, reward):
            forwarded = super().report_reward(reward)
            return self.scale * forwarded

    env = SingleStateBanditEnv(reward_table=[0.0, 1.0])
    actor = _zero_initialized_actor(num_actions=2)
    optimizer = torch.optim.SGD(actor.parameters(), lr=0.1)
    trainer = Trainer(
        env,
        actor,
        actor_optimizer=optimizer,
        observer=ScalingObserver(scale=0.0),
        optimal_action=1,
    )

    initial_prob = (
        actor.policy.action_distribution(env.get_actor_observation()).probs[1].item()
    )
    trace = trainer.run(num_steps=200, seed=5)

    assert trace.final_optimal_action_prob == pytest.approx(initial_prob, abs=1e-6)
    assert all(reward == 0.0 for reward in trace.rewards)


def test_case2_training_learns_different_best_actions_by_state() -> None:
    env = IIDStateBanditEnv(
        reward_table=[
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        state_distribution=[1 / 3, 1 / 3, 1 / 3],
        observation_mode="one_hot",
    )
    actor = _zero_initialized_actor(input_dim=3, num_actions=3)
    optimizer = torch.optim.SGD(actor.parameters(), lr=0.1)
    trainer = Trainer(
        env,
        actor,
        actor_optimizer=optimizer,
        observer=Observer(),
        optimal_actions=[0, 1, 2],
    )

    trace = trainer.run(num_steps=3000, seed=13)

    assert len(trace.state_indices) == 3000
    assert trace.final_optimal_action_prob is not None
    for state_index, optimal_action in enumerate([0, 1, 2]):
        observation = torch.nn.functional.one_hot(
            torch.tensor(state_index),
            num_classes=3,
        ).float()
        probs = actor.policy.action_distribution(observation).probs
        assert probs[optimal_action].item() > 0.9
    assert all(torch.isfinite(torch.tensor(trace.losses)))


def test_case2_training_is_reproducible_with_fixed_seed() -> None:
    def run_once() -> tuple[list[int], list[int], list[float], list[float]]:
        env = IIDStateBanditEnv(
            reward_table=[[1.0, 0.0], [0.0, 1.0]],
            state_distribution=[0.5, 0.5],
            observation_mode="one_hot",
        )
        actor = _zero_initialized_actor(input_dim=2, num_actions=2)
        optimizer = torch.optim.SGD(actor.parameters(), lr=0.1)
        trainer = Trainer(
            env,
            actor,
            actor_optimizer=optimizer,
            observer=Observer(),
            optimal_actions=[0, 1],
        )
        trace = trainer.run(num_steps=80, seed=23)
        return trace.state_indices, trace.actions, trace.rewards, trace.optimal_action_probs

    states_a, actions_a, rewards_a, probs_a = run_once()
    states_b, actions_b, rewards_b, probs_b = run_once()

    assert states_a == states_b
    assert actions_a == actions_b
    assert rewards_a == rewards_b
    assert probs_a == pytest.approx(probs_b)


def test_case2_routes_state_dependent_reward_through_observer() -> None:
    seen: list[float] = []

    class RecordingObserver(Observer):
        def report_reward(self, reward):
            forwarded = super().report_reward(reward)
            seen.append(forwarded)
            return forwarded

    env = IIDStateBanditEnv(
        reward_table=[[1.0, 1.0], [2.0, 2.0]],
        state_distribution=[0.0, 1.0],
        observation_mode="one_hot",
    )
    actor = _zero_initialized_actor(input_dim=2, num_actions=2)
    optimizer = torch.optim.SGD(actor.parameters(), lr=0.1)
    trainer = Trainer(
        env,
        actor,
        actor_optimizer=optimizer,
        observer=RecordingObserver(),
        optimal_actions=[0, 0],
    )

    trace = trainer.run(num_steps=10, seed=3)

    assert seen == trace.rewards
    assert seen == [2.0] * 10
