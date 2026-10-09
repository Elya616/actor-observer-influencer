"""Training tests for Case 4 multi-observer experiments."""

from __future__ import annotations

import pytest
import torch

from actor_observer_rl.agents import (
    Actor,
    FollowingRates,
    MultiObserver,
    Observer,
    ObserverMessageModule,
)
from actor_observer_rl.envs import PartialObservableStateEnv
from actor_observer_rl.policies import SoftmaxPolicy
from actor_observer_rl.training import MultiObserverTrainer
from actor_observer_rl.utils.seeding import set_global_seed


def _make_case4_trainer(
    seed: int = 0,
    num_observers: int = 2,
    message_dim: int = 2,
    learning_rate: float = 0.1,
) -> MultiObserverTrainer:
    set_global_seed(seed)
    env = PartialObservableStateEnv(
        reward_table=[[1.0, 0.0], [0.0, 1.0]],
        state_distribution=[0.5, 0.5],
        observation_dim=1,
    )
    observers = [
        Observer(
            message_module=ObserverMessageModule(
                input_dim=env.state_dim,
                message_dim=message_dim,
            )
        )
        for _ in range(num_observers)
    ]
    multi_observer = MultiObserver(
        observers=observers,
        following_rates=FollowingRates(num_observers=num_observers),
    )
    actor = Actor(SoftmaxPolicy(input_dim=1 + message_dim, num_actions=2))
    return MultiObserverTrainer(
        env=env,
        actor=actor,
        multi_observer=multi_observer,
        actor_optimizer=torch.optim.SGD(actor.parameters(), lr=learning_rate),
        observer_optimizer=torch.optim.SGD(
            list(multi_observer.observer_parameters()), lr=learning_rate
        ),
        following_optimizer=torch.optim.SGD(
            list(multi_observer.following_parameters()), lr=learning_rate
        ),
        optimal_actions=[0, 1],
    )


def test_case4_training_improves_expected_reward_with_zero_actor_observation() -> None:
    trainer = _make_case4_trainer(seed=0)
    initial_expected_reward, initial_mean_prob = trainer.evaluate_policy()

    trace = trainer.run(num_steps=3000, seed=11, eval_interval=100)

    assert trace.final_expected_reward is not None
    assert trace.final_mean_optimal_action_prob is not None
    assert initial_mean_prob is not None
    assert trace.final_expected_reward > initial_expected_reward
    assert trace.final_expected_reward > 0.8
    assert trace.final_mean_optimal_action_prob > initial_mean_prob
    assert trace.final_mean_optimal_action_prob > 0.8
    assert all(torch.isfinite(torch.tensor(trace.losses)))


def test_case4_training_records_following_weights_each_step() -> None:
    trainer = _make_case4_trainer(seed=2, num_observers=3)

    trace = trainer.run(num_steps=30, seed=4, eval_interval=10)

    assert len(trace.following_weights) == 30
    for weights in trace.following_weights:
        assert len(weights) == 3
        assert sum(weights) == pytest.approx(1.0, abs=1e-5)


def test_case4_training_records_selected_observer_indices() -> None:
    trainer = _make_case4_trainer(seed=7, num_observers=3)

    trace = trainer.run(num_steps=40, seed=13, eval_interval=20)

    assert len(trace.selected_observer_indices) == 40
    assert all(0 <= index < 3 for index in trace.selected_observer_indices)
    assert len(trace.selection_log_probs) == 40
    for index, log_prob in zip(
        trace.selected_observer_indices,
        trace.selection_log_probs,
        strict=True,
    ):
        assert log_prob <= 0.0
        assert log_prob > -10.0
        assert isinstance(index, int)


def test_case4_training_only_selected_observer_updates_each_step() -> None:
    set_global_seed(1)
    env = PartialObservableStateEnv(
        reward_table=[[1.0, 0.7], [0.7, 1.0]],
        state_distribution=[0.5, 0.5],
        observation_dim=1,
    )
    observers = [
        Observer(
            message_module=ObserverMessageModule(input_dim=env.state_dim, message_dim=2)
        )
        for _ in range(3)
    ]
    multi_observer = MultiObserver(
        observers=observers,
        following_rates=FollowingRates(num_observers=3),
    )
    actor = Actor(SoftmaxPolicy(input_dim=3, num_actions=2))
    trainer = MultiObserverTrainer(
        env=env,
        actor=actor,
        multi_observer=multi_observer,
        actor_optimizer=torch.optim.SGD(actor.parameters(), lr=0.1),
        observer_optimizer=torch.optim.SGD(
            list(multi_observer.observer_parameters()), lr=0.1
        ),
        following_optimizer=torch.optim.SGD(
            list(multi_observer.following_parameters()), lr=0.1
        ),
        optimal_actions=[0, 1],
    )

    before = [
        [parameter.detach().clone() for parameter in observer.parameters()]
        for observer in observers
    ]

    trace = trainer.run(num_steps=1, seed=8, eval_interval=1)
    selected = trace.selected_observer_indices[0]
    assert trace.rewards[0] > 0.0

    after = [
        [parameter.detach().clone() for parameter in observer.parameters()]
        for observer in observers
    ]
    for observer_index, (old_params, new_params) in enumerate(
        zip(before, after, strict=True)
    ):
        moved = any(
            not torch.allclose(old, new)
            for old, new in zip(old_params, new_params, strict=True)
        )
        if observer_index == selected:
            assert moved, f"observer {observer_index} was selected but did not update"
        else:
            assert not moved, (
                f"observer {observer_index} was not selected but parameters changed"
            )


def test_case4_training_records_per_observer_grad_norms() -> None:
    trainer = _make_case4_trainer(seed=1, num_observers=3)

    trace = trainer.run(num_steps=15, seed=8, eval_interval=5)

    assert len(trace.observer_grad_norms) == 15
    assert len(trace.following_grad_norms) == 15
    for per_observer_norms in trace.observer_grad_norms:
        assert len(per_observer_norms) == 3
    assert any(any(norm > 0.0 for norm in row) for row in trace.observer_grad_norms)
    assert any(norm > 0.0 for norm in trace.following_grad_norms)


def test_case4_grad_norm_is_zero_for_unselected_observers_each_step() -> None:
    trainer = _make_case4_trainer(seed=2, num_observers=3)

    trace = trainer.run(num_steps=12, seed=10, eval_interval=12)

    for selected, per_observer_norms in zip(
        trace.selected_observer_indices,
        trace.observer_grad_norms,
        strict=True,
    ):
        for observer_index, norm in enumerate(per_observer_norms):
            if observer_index != selected:
                assert norm == 0.0, (
                    f"observer {observer_index} got gradient on step where "
                    f"observer {selected} was selected"
                )


def test_case4_observer_and_following_parameters_receive_gradients() -> None:
    trainer = _make_case4_trainer(seed=3)

    before_observers = [
        parameter.detach().clone()
        for parameter in trainer.multi_observer.observer_parameters()
    ]
    before_following = [
        parameter.detach().clone()
        for parameter in trainer.multi_observer.following_parameters()
    ]

    trainer.run(num_steps=30, seed=5, eval_interval=10)

    after_observers = [
        parameter.detach().clone()
        for parameter in trainer.multi_observer.observer_parameters()
    ]
    after_following = [
        parameter.detach().clone()
        for parameter in trainer.multi_observer.following_parameters()
    ]

    assert any(
        not torch.allclose(old, new)
        for old, new in zip(before_observers, after_observers)
    )
    assert any(
        not torch.allclose(old, new)
        for old, new in zip(before_following, after_following)
    )


def test_case4_training_is_reproducible_with_fixed_seed() -> None:
    def run_once() -> tuple[
        list[int], list[int], list[int], list[float], list[list[float]]
    ]:
        trainer = _make_case4_trainer(seed=3)
        trace = trainer.run(num_steps=40, seed=17, eval_interval=10)
        return (
            trace.state_indices,
            trace.actions,
            trace.selected_observer_indices,
            trace.rewards,
            trace.following_weights,
        )

    states_a, actions_a, selected_a, rewards_a, weights_a = run_once()
    states_b, actions_b, selected_b, rewards_b, weights_b = run_once()

    assert states_a == states_b
    assert actions_a == actions_b
    assert selected_a == selected_b
    assert rewards_a == rewards_b
    for row_a, row_b in zip(weights_a, weights_b):
        assert row_a == pytest.approx(row_b)


def test_case4_reward_still_routes_through_primary_observer() -> None:
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
    primary = ZeroRewardObserver(
        message_module=ObserverMessageModule(input_dim=env.state_dim, message_dim=2)
    )
    secondary = Observer(
        message_module=ObserverMessageModule(input_dim=env.state_dim, message_dim=2)
    )
    multi_observer = MultiObserver(
        observers=[primary, secondary],
        following_rates=FollowingRates(num_observers=2),
    )
    actor = Actor(SoftmaxPolicy(input_dim=3, num_actions=2))
    trainer = MultiObserverTrainer(
        env=env,
        actor=actor,
        multi_observer=multi_observer,
        actor_optimizer=torch.optim.SGD(actor.parameters(), lr=0.1),
        observer_optimizer=torch.optim.SGD(
            list(multi_observer.observer_parameters()), lr=0.1
        ),
        following_optimizer=torch.optim.SGD(
            list(multi_observer.following_parameters()), lr=0.1
        ),
        optimal_actions=[0, 0],
    )

    trace = trainer.run(num_steps=3, seed=2, eval_interval=3)

    assert trace.rewards == [0.0, 0.0, 0.0]
    assert trace.actor_grad_norms == [0.0, 0.0, 0.0]
    assert trace.following_grad_norms == [0.0, 0.0, 0.0]
    for per_observer_norms in trace.observer_grad_norms:
        assert per_observer_norms == [0.0, 0.0]


def test_case4_training_raises_on_non_finite_loss() -> None:
    class NanRewardObserver(Observer):
        def report_reward(self, reward):
            super().report_reward(reward)
            return float("nan")

    set_global_seed(0)
    env = PartialObservableStateEnv(
        reward_table=[[1.0, 0.0], [0.0, 1.0]],
        state_distribution=[0.5, 0.5],
        observation_dim=1,
    )
    primary = NanRewardObserver(
        message_module=ObserverMessageModule(input_dim=env.state_dim, message_dim=2)
    )
    secondary = Observer(
        message_module=ObserverMessageModule(input_dim=env.state_dim, message_dim=2)
    )
    multi_observer = MultiObserver(
        observers=[primary, secondary],
        following_rates=FollowingRates(num_observers=2),
    )
    actor = Actor(SoftmaxPolicy(input_dim=3, num_actions=2))
    trainer = MultiObserverTrainer(
        env=env,
        actor=actor,
        multi_observer=multi_observer,
        actor_optimizer=torch.optim.SGD(actor.parameters(), lr=0.1),
        observer_optimizer=torch.optim.SGD(
            list(multi_observer.observer_parameters()), lr=0.1
        ),
        following_optimizer=torch.optim.SGD(
            list(multi_observer.following_parameters()), lr=0.1
        ),
        optimal_actions=[0, 1],
    )

    with pytest.raises(FloatingPointError, match="non-finite training loss"):
        trainer.run(num_steps=5, seed=2, eval_interval=5)


def test_case4_trainer_rejects_optimal_actions_length_mismatch() -> None:
    env = PartialObservableStateEnv(
        reward_table=[[1.0, 0.0], [0.0, 1.0]],
        state_distribution=[0.5, 0.5],
        observation_dim=1,
    )
    observers = [
        Observer(
            message_module=ObserverMessageModule(input_dim=env.state_dim, message_dim=2)
        )
        for _ in range(2)
    ]
    multi_observer = MultiObserver(
        observers=observers,
        following_rates=FollowingRates(num_observers=2),
    )
    actor = Actor(SoftmaxPolicy(input_dim=3, num_actions=2))

    with pytest.raises(ValueError, match="optimal_actions length"):
        MultiObserverTrainer(
            env=env,
            actor=actor,
            multi_observer=multi_observer,
            actor_optimizer=torch.optim.SGD(actor.parameters(), lr=0.1),
            observer_optimizer=torch.optim.SGD(
                list(multi_observer.observer_parameters()), lr=0.1
            ),
            following_optimizer=torch.optim.SGD(
                list(multi_observer.following_parameters()), lr=0.1
            ),
            optimal_actions=[0],
        )
