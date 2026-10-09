"""Tests for observer messages and Case 3 gradient flow."""

from __future__ import annotations

import torch

from actor_observer_rl.agents import Actor, Observer, ObserverMessageModule
from actor_observer_rl.envs import PartialObservableStateEnv
from actor_observer_rl.policies import SoftmaxPolicy
from actor_observer_rl.training import PartialObservableTrainer, build_actor_message_input
from actor_observer_rl.utils.seeding import set_global_seed


def test_observer_message_module_returns_expected_shape() -> None:
    message_module = ObserverMessageModule(input_dim=3, message_dim=2)
    full_state = torch.tensor([0.0, 1.0, 0.0])

    message = message_module(full_state)

    assert message.shape == torch.Size([2])


def test_actor_message_input_concatenates_observation_and_message() -> None:
    observation = torch.tensor([0.0])
    message = torch.tensor([0.25, -0.5])

    actor_input = build_actor_message_input(observation, message)

    assert torch.equal(actor_input, torch.tensor([0.0, 0.25, -0.5]))


def test_case3_observer_parameters_receive_gradients_through_message() -> None:
    set_global_seed(0)
    env = PartialObservableStateEnv(
        reward_table=[[1.0, 1.0], [1.0, 1.0]],
        state_distribution=[1.0, 0.0],
        observation_dim=1,
    )
    message_module = ObserverMessageModule(input_dim=env.state_dim, message_dim=2)
    observer = Observer(message_module=message_module)
    policy = SoftmaxPolicy(input_dim=3, num_actions=2)
    with torch.no_grad():
        policy.linear.weight.copy_(
            torch.tensor(
                [
                    [0.0, 1.0, -1.0],
                    [0.0, -1.0, 1.0],
                ]
            )
        )
        policy.linear.bias.zero_()
    actor = Actor(policy)
    actor_optimizer = torch.optim.SGD(actor.parameters(), lr=0.1)
    observer_optimizer = torch.optim.SGD(observer.parameters(), lr=0.1)
    trainer = PartialObservableTrainer(
        env=env,
        actor=actor,
        observer=observer,
        actor_optimizer=actor_optimizer,
        observer_optimizer=observer_optimizer,
        optimal_actions=[0, 0],
    )

    before = [parameter.detach().clone() for parameter in observer.parameters()]
    trace = trainer.run(num_steps=1, seed=5)
    after = [parameter.detach().clone() for parameter in observer.parameters()]

    assert trace.observer_grad_norms[0] > 0.0
    assert any(not torch.allclose(old, new) for old, new in zip(before, after))


def test_case3_reward_still_routes_through_observer_report_reward() -> None:
    class ZeroRewardObserver(Observer):
        def report_reward(self, reward):
            super().report_reward(reward)
            return 0.0

    set_global_seed(0)
    env = PartialObservableStateEnv(
        reward_table=[[1.0, 1.0], [1.0, 1.0]],
        state_distribution=[1.0, 0.0],
        observation_dim=1,
    )
    observer = ZeroRewardObserver(
        message_module=ObserverMessageModule(input_dim=env.state_dim, message_dim=2)
    )
    actor = Actor(SoftmaxPolicy(input_dim=3, num_actions=2))
    trainer = PartialObservableTrainer(
        env=env,
        actor=actor,
        observer=observer,
        actor_optimizer=torch.optim.SGD(actor.parameters(), lr=0.1),
        observer_optimizer=torch.optim.SGD(observer.parameters(), lr=0.1),
        optimal_actions=[0, 0],
    )

    trace = trainer.run(num_steps=3, seed=2)

    assert trace.rewards == [0.0, 0.0, 0.0]
    assert trace.actor_grad_norms == [0.0, 0.0, 0.0]
    assert trace.observer_grad_norms == [0.0, 0.0, 0.0]
