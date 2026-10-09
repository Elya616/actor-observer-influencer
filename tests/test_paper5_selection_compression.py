"""Tests for paper Section 5.2 selection plus compression learning."""

from __future__ import annotations

from statistics import fmean

import pytest
import torch

from actor_observer_rl.agents import (
    Actor,
    FollowingRates,
    MaskedObserverMessageModule,
    MultiObserver,
    Observer,
)
from actor_observer_rl.envs import FeaturePartialObservableStateEnv
from actor_observer_rl.policies import SumDecoderPolicy
from actor_observer_rl.training import StagedMultiObserverTrainer
from actor_observer_rl.utils.seeding import set_global_seed


STATE_FEATURES = [
    [0.0, 0.0],
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
]
STATE_SUMS = [0.0, 1.0, 1.0, 2.0]
SUM_REWARD_TABLE = [
    [1.0, -0.1, -0.1],
    [-0.1, 1.0, -0.1],
    [-0.1, 1.0, -0.1],
    [-0.1, -0.1, 1.0],
]
OBSERVER_MASKS = [
    [0.0, 0.0],
    [1.0, 0.0],
    [1.0, 1.0],
]


def _make_paper52_trainer(
    seed: int = 0,
    observer_learning_rate: float = 0.02,
    following_learning_rate: float = 0.005,
    init_logits: list[float] | None = None,
) -> StagedMultiObserverTrainer:
    set_global_seed(seed)
    env = FeaturePartialObservableStateEnv(
        state_features=STATE_FEATURES,
        reward_table=SUM_REWARD_TABLE,
        state_distribution=[0.25, 0.25, 0.25, 0.25],
        observation_dim=1,
    )
    observers = []
    for mask in OBSERVER_MASKS:
        message_module = MaskedObserverMessageModule(
            input_dim=2,
            message_dim=1,
            input_mask=mask,
        )
        with torch.no_grad():
            message_module.linear.weight.zero_()
            message_module.linear.bias.zero_()
        observers.append(Observer(message_module=message_module))
    multi_observer = MultiObserver(
        observers=observers,
        following_rates=FollowingRates(
            num_observers=len(observers),
            init_logits=init_logits,
        ),
    )
    actor = Actor(SumDecoderPolicy(num_actions=3, sharpness=4.0))
    return StagedMultiObserverTrainer(
        env=env,
        actor=actor,
        multi_observer=multi_observer,
        following_optimizer=torch.optim.SGD(
            list(multi_observer.following_parameters()),
            lr=following_learning_rate,
        ),
        observer_optimizer=torch.optim.SGD(
            list(multi_observer.observer_parameters()),
            lr=observer_learning_rate,
        ),
        optimal_actions=[0, 1, 1, 2],
    )


def test_feature_partial_observable_env_returns_configured_state_features() -> None:
    env = FeaturePartialObservableStateEnv(
        state_features=STATE_FEATURES,
        reward_table=SUM_REWARD_TABLE,
        state_distribution=[0.0, 0.0, 0.0, 1.0],
        observation_dim=1,
    )

    observation, info = env.reset(seed=3)

    assert info["state_index"] == 3
    assert torch.equal(observation, torch.tensor([0.0]))
    assert torch.equal(env.get_full_state(), torch.tensor([1.0, 1.0]))
    assert env.state_dim == 2
    assert env.get_reward(torch.tensor([1.0, 1.0]), action=2) == 1.0


def test_masked_observer_message_module_blocks_unavailable_features() -> None:
    module = MaskedObserverMessageModule(
        input_dim=2,
        message_dim=1,
        input_mask=[1.0, 0.0],
    )
    with torch.no_grad():
        module.linear.weight.fill_(1.0)
        module.linear.bias.zero_()

    assert module(torch.tensor([0.0, 1.0])).item() == pytest.approx(0.0)
    assert module(torch.tensor([1.0, 1.0])).item() == pytest.approx(1.0)


def test_paper52_selection_and_compression_learn_full_sum_observer() -> None:
    trainer = _make_paper52_trainer(seed=2)

    trace = trainer.run(num_steps=8000, seed=2, eval_interval=400)

    final_weights = trace.final_following_weights
    final_messages = trace.final_observer_messages_by_state
    assert final_weights is not None
    assert final_messages is not None
    full_sum_messages = [message[0] for message in final_messages[2]]
    assert final_weights[2] > 0.85
    assert trace.final_expected_reward is not None
    assert trace.final_expected_reward > 0.90
    assert trace.final_mean_optimal_action_prob is not None
    assert trace.final_mean_optimal_action_prob > 0.90
    assert _r_squared(full_sum_messages, STATE_SUMS) > 0.98


def test_paper52_only_selected_observer_updates_on_one_step() -> None:
    trainer = _make_paper52_trainer(seed=4)
    before = [
        [parameter.detach().clone() for parameter in observer.parameters()]
        for observer in trainer.multi_observer.observers
    ]

    trace = trainer.run(num_steps=1, seed=4, eval_interval=1)

    selected = trace.selected_observer_indices[0]
    after = [
        [parameter.detach().clone() for parameter in observer.parameters()]
        for observer in trainer.multi_observer.observers
    ]
    for observer_index, (old_params, new_params) in enumerate(
        zip(before, after, strict=True)
    ):
        moved = any(
            not torch.allclose(old, new)
            for old, new in zip(old_params, new_params, strict=True)
        )
        if observer_index == selected:
            assert moved
        else:
            assert not moved


def test_paper52_trace_records_observer_message_snapshots() -> None:
    trainer = _make_paper52_trainer(seed=5)

    trace = trainer.run(num_steps=20, seed=5, eval_interval=10)

    assert trace.evaluation_steps == [0, 10, 20]
    assert len(trace.observer_messages_by_state) == 3
    assert len(trace.mean_optimal_action_probs) == 3
    latest = trace.final_observer_messages_by_state
    assert latest is not None
    assert len(latest) == 3
    assert len(latest[0]) == 4
    assert len(latest[0][0]) == 1


def _r_squared(values: list[float], targets: list[float]) -> float:
    target_mean = fmean(targets)
    total_sum_squares = sum((target - target_mean) ** 2 for target in targets)
    value_mean = fmean(values)
    covariance = sum(
        (value - value_mean) * (target - target_mean)
        for value, target in zip(values, targets, strict=True)
    )
    value_sum_squares = sum((value - value_mean) ** 2 for value in values)
    if value_sum_squares <= 1e-12:
        return 0.0
    return covariance**2 / (value_sum_squares * total_sum_squares)
