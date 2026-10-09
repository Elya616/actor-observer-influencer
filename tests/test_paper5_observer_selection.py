"""Tests for paper Section 5.1 Observer-selection learning."""

from __future__ import annotations

import math

import pytest
import torch

from actor_observer_rl.agents import (
    Actor,
    FixedTableMessageModule,
    FollowingRates,
    MultiObserver,
    Observer,
)
from actor_observer_rl.envs import PartialObservableStateEnv
from actor_observer_rl.policies import SumDecoderPolicy
from actor_observer_rl.training import StagedMultiObserverTrainer
from actor_observer_rl.utils.seeding import set_global_seed


SUM_REWARD_TABLE = [
    [1.0, -0.1, -0.1],
    [-0.1, 1.0, -0.1],
    [-0.1, 1.0, -0.1],
    [-0.1, -0.1, 1.0],
]


def _make_paper51_trainer(
    learning_rate: float = 0.05,
    seed: int = 0,
    observer_messages: list[list[float]] | None = None,
    selection_update_rule: str = "raw_reward",
    selection_baseline_beta: float = 0.01,
    selection_fixed_baseline: float | None = None,
) -> StagedMultiObserverTrainer:
    set_global_seed(seed)
    env = PartialObservableStateEnv(
        reward_table=SUM_REWARD_TABLE,
        state_distribution=[0.25, 0.25, 0.25, 0.25],
        observation_dim=1,
    )
    if observer_messages is None:
        observer_messages = [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0, 2.0],
        ]
    observers = [
        Observer(message_module=FixedTableMessageModule(messages))
        for messages in observer_messages
    ]
    multi_observer = MultiObserver(
        observers=observers,
        following_rates=FollowingRates(num_observers=len(observers)),
    )
    actor = Actor(SumDecoderPolicy(num_actions=3, sharpness=4.0))
    following_optimizer = torch.optim.SGD(
        list(multi_observer.following_parameters()),
        lr=learning_rate,
    )
    return StagedMultiObserverTrainer(
        env=env,
        actor=actor,
        multi_observer=multi_observer,
        following_optimizer=following_optimizer,
        optimal_actions=[0, 1, 1, 2],
        selection_update_rule=selection_update_rule,
        selection_baseline_beta=selection_baseline_beta,
        selection_fixed_baseline=selection_fixed_baseline,
    )


def test_fixed_table_message_module_maps_one_hot_state_to_message() -> None:
    module = FixedTableMessageModule([0.0, 1.0, 1.0, 2.0])

    message = module(torch.tensor([0.0, 0.0, 0.0, 1.0]))

    assert message.shape == torch.Size([1])
    assert float(message.item()) == pytest.approx(2.0)
    assert list(module.parameters()) == []


def test_paper51_evaluation_identifies_sum_observer_as_best() -> None:
    trainer = _make_paper51_trainer()

    expected_reward, per_observer = trainer.evaluate_policy()

    assert len(per_observer) == 3
    assert per_observer[2] > per_observer[1]
    assert per_observer[1] > per_observer[0]
    assert expected_reward == pytest.approx(sum(per_observer) / 3, abs=1e-6)


def test_paper51_selection_concentrates_on_useful_sum_observer() -> None:
    trainer = _make_paper51_trainer(learning_rate=0.05, seed=3)

    trace = trainer.run(num_steps=5000, seed=3, eval_interval=100)

    final_weights = trace.final_following_weights
    assert final_weights is not None
    assert math.isclose(sum(final_weights), 1.0, abs_tol=1e-6)
    assert final_weights[2] > 0.95
    assert trace.final_expected_reward is not None
    assert trace.final_expected_reward > 0.95
    assert trace.final_mean_optimal_action_prob is not None
    assert trace.final_mean_optimal_action_prob > 0.95


def test_paper51_only_following_logits_change() -> None:
    trainer = _make_paper51_trainer(learning_rate=0.05, seed=4)
    before_messages = [
        observer.message_module.message_table.detach().clone()
        for observer in trainer.multi_observer.observers
    ]
    before_logits = trainer.multi_observer.following_rates.logits.detach().clone()

    trainer.run(num_steps=200, seed=4, eval_interval=100)

    after_logits = trainer.multi_observer.following_rates.logits.detach()
    assert not torch.allclose(before_logits, after_logits)
    for before, observer in zip(
        before_messages, trainer.multi_observer.observers, strict=True
    ):
        assert torch.allclose(before, observer.message_module.message_table)


def test_paper51_trace_records_selection_probabilities() -> None:
    trainer = _make_paper51_trainer(learning_rate=0.01, seed=5)

    trace = trainer.run(num_steps=25, seed=5, eval_interval=5)

    assert len(trace.following_weights) == 25
    assert len(trace.selected_observer_indices) == 25
    assert trace.evaluation_steps == [0, 5, 10, 15, 20, 25]
    assert len(trace.mean_optimal_action_probs) == len(trace.evaluation_steps)
    for weights in trace.following_weights:
        assert len(weights) == 3
        assert sum(weights) == pytest.approx(1.0, abs=1e-6)
    assert all(0 <= index < 3 for index in trace.selected_observer_indices)


def test_paper51_raw_reward_is_default_selection_update_rule() -> None:
    trainer = _make_paper51_trainer(learning_rate=0.01, seed=6)

    trace = trainer.run(num_steps=10, seed=6, eval_interval=5)

    assert trace.metadata["selection_update_rule"] == "raw_reward"
    assert trace.selection_update_multipliers == pytest.approx(trace.rewards)
    assert all(baseline == 0.0 for baseline in trace.selection_baselines)
    assert len(trace.following_logits) == 10
    assert len(trace.selection_entropies) == 10


def test_paper51_global_advantage_records_running_baseline() -> None:
    trainer = _make_paper51_trainer(
        learning_rate=0.01,
        seed=7,
        selection_update_rule="advantage_global",
        selection_baseline_beta=0.5,
    )

    trace = trainer.run(num_steps=30, seed=7, eval_interval=10)

    assert trace.metadata["selection_update_rule"] == "advantage_global"
    assert any(abs(baseline) > 0.0 for baseline in trace.selection_baselines)
    assert any(
        abs(multiplier - reward) > 1e-8
        for multiplier, reward in zip(
            trace.selection_update_multipliers,
            trace.rewards,
            strict=True,
        )
    )


def test_paper51_fixed_baseline_uses_reward_minus_baseline() -> None:
    trainer = _make_paper51_trainer(
        learning_rate=0.01,
        seed=8,
        selection_update_rule="advantage_global",
        selection_fixed_baseline=0.25,
    )

    trace = trainer.run(num_steps=10, seed=8, eval_interval=5)

    assert trace.selection_update_multipliers == pytest.approx(
        [reward - 0.25 for reward in trace.rewards]
    )
    assert trace.selection_baselines == pytest.approx([0.0] * 10)
