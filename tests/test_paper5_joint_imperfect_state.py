"""Tests for paper Section 5.3 joint imperfect-state learning."""

from __future__ import annotations

import copy
import math

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
from actor_observer_rl.policies import SoftmaxPolicy
from actor_observer_rl.training import StagedMultiObserverTrainer
from actor_observer_rl.utils.seeding import set_global_seed


STATE_FEATURES = [
    [0.0, 0.0],
    [1.0, 0.0],
    [1.0, 1.0],
]
REWARD_TABLE = [
    [1.0, -0.1, -0.1],
    [-0.1, 1.0, -0.1],
    [-0.1, -0.1, 1.0],
]
OBSERVER_MASKS = [
    [0.0, 0.0],
    [1.0, 0.0],
    [1.0, 1.0],
]


def _make_paper53_trainer(
    *,
    seed: int = 0,
    actor_learning_rate: float = 0.05,
    observer_learning_rate: float = 0.05,
    following_learning_rate: float = 0.005,
) -> StagedMultiObserverTrainer:
    set_global_seed(seed)
    env = FeaturePartialObservableStateEnv(
        state_features=STATE_FEATURES,
        reward_table=REWARD_TABLE,
        state_distribution=[1 / 3, 1 / 3, 1 / 3],
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
        following_rates=FollowingRates(num_observers=len(observers)),
    )
    actor = Actor(SoftmaxPolicy(input_dim=1, num_actions=3))
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
        actor_optimizer=torch.optim.SGD(
            list(actor.parameters()),
            lr=actor_learning_rate,
        ),
        optimal_actions=[0, 1, 2],
    )


def test_paper53_joint_learning_selects_full_state_observer() -> None:
    trainer = _make_paper53_trainer(seed=2, actor_learning_rate=0.05)

    trace = trainer.run(num_steps=12000, seed=2, eval_interval=600)

    final_weights = trace.final_following_weights
    final_messages = trace.final_observer_messages_by_state
    per_observer_reward = trace.final_per_observer_expected_rewards
    assert final_weights is not None
    assert final_messages is not None
    assert per_observer_reward is not None
    assert math.isclose(sum(final_weights), 1.0, abs_tol=1e-6)
    assert final_weights[2] > 0.90
    assert per_observer_reward[2] > per_observer_reward[1] + 0.20
    assert per_observer_reward[1] > per_observer_reward[0] + 0.20
    assert trace.final_expected_reward is not None
    assert trace.final_expected_reward > 0.95
    assert trace.final_mean_optimal_action_prob is not None
    assert trace.final_mean_optimal_action_prob > 0.95
    full_state_messages = [message[0] for message in final_messages[2]]
    assert _message_separation(full_state_messages) > 1.0


def test_paper53_actor_observer_and_selection_parameters_all_update() -> None:
    trainer = _make_paper53_trainer(seed=4)
    before_actor = [parameter.detach().clone() for parameter in trainer.actor.parameters()]
    before_observers = [
        [parameter.detach().clone() for parameter in observer.parameters()]
        for observer in trainer.multi_observer.observers
    ]
    before_logits = trainer.multi_observer.following_rates.logits.detach().clone()

    trace = trainer.run(num_steps=1, seed=4, eval_interval=1)

    after_actor = [parameter.detach().clone() for parameter in trainer.actor.parameters()]
    after_observers = [
        [parameter.detach().clone() for parameter in observer.parameters()]
        for observer in trainer.multi_observer.observers
    ]
    after_logits = trainer.multi_observer.following_rates.logits.detach()
    assert any(
        not torch.allclose(before, after)
        for before, after in zip(before_actor, after_actor, strict=True)
    )
    assert not torch.allclose(before_logits, after_logits)

    selected = trace.selected_observer_indices[0]
    for observer_index, (old_params, new_params) in enumerate(
        zip(before_observers, after_observers, strict=True)
    ):
        moved = any(
            not torch.allclose(old, new)
            for old, new in zip(old_params, new_params, strict=True)
        )
        if observer_index == selected:
            assert moved
        else:
            assert not moved


def test_paper53_one_step_updates_match_single_pre_update_forward_pass() -> None:
    actor_lr = 0.05
    observer_lr = 0.05
    following_lr = 0.005
    trainer = _make_paper53_trainer(
        seed=6,
        actor_learning_rate=actor_lr,
        observer_learning_rate=observer_lr,
        following_learning_rate=following_lr,
    )
    actor_before = copy.deepcopy(trainer.actor.policy)
    observer_before = copy.deepcopy(trainer.multi_observer.observers)
    logits_before = trainer.multi_observer.following_rates.logits.detach().clone()

    trace = trainer.run(num_steps=1, seed=6, eval_interval=1)

    state_index = trace.state_indices[0]
    selected = trace.selected_observer_indices[0]
    action = torch.tensor(trace.actions[0], dtype=torch.long)
    reward = trace.rewards[0]
    full_state = trainer.env.full_state_for_index(state_index)
    sampled_message = torch.tensor(trace.sampled_messages[0], dtype=torch.float32)

    actor_distribution = actor_before.action_distribution(sampled_message)
    actor_log_prob = actor_distribution.log_prob(action)
    assert math.isclose(
        float(actor_log_prob.detach().item()),
        trace.action_log_probs[0],
        rel_tol=1e-6,
        abs_tol=1e-6,
    )
    actor_loss = -reward * actor_log_prob
    actor_loss.backward()
    _assert_sgd_update_matches(
        before_parameters=list(actor_before.parameters()),
        after_parameters=list(trainer.actor.parameters()),
        learning_rate=actor_lr,
    )

    selected_observer_before = observer_before[selected]
    actor_for_observer = copy.deepcopy(actor_before)
    message = selected_observer_before.message(full_state)
    assert torch.allclose(message.detach(), sampled_message, atol=1e-6)
    observer_distribution = actor_for_observer.action_distribution(message)
    observer_loss = -reward * observer_distribution.log_prob(action)
    observer_loss.backward()
    _assert_sgd_update_matches(
        before_parameters=list(selected_observer_before.parameters()),
        after_parameters=list(trainer.multi_observer.observers[selected].parameters()),
        learning_rate=observer_lr,
    )

    for observer_index, old_observer in enumerate(observer_before):
        if observer_index == selected:
            continue
        for old_parameter, new_parameter in zip(
            old_observer.parameters(),
            trainer.multi_observer.observers[observer_index].parameters(),
            strict=True,
        ):
            assert torch.allclose(old_parameter, new_parameter)

    lambda_at_sample = torch.tensor(trace.following_weights_at_sample[0])
    expected_log_prob = torch.log(lambda_at_sample[selected])
    assert math.isclose(
        float(expected_log_prob.item()),
        trace.selection_log_probs[0],
        rel_tol=1e-6,
        abs_tol=1e-6,
    )
    expected_logits = logits_before.clone()
    expected_logits -= following_lr * reward * lambda_at_sample
    expected_logits[selected] += following_lr * reward
    assert torch.allclose(
        trainer.multi_observer.following_rates.logits.detach(),
        expected_logits,
        atol=1e-6,
    )


def test_paper53_trace_records_joint_metrics() -> None:
    trainer = _make_paper53_trainer(seed=5)

    trace = trainer.run(num_steps=20, seed=5, eval_interval=10)

    assert trace.evaluation_steps == [0, 10, 20]
    assert len(trace.mean_optimal_action_probs) == 3
    assert len(trace.per_observer_expected_rewards) == 3
    latest = trace.final_observer_messages_by_state
    assert latest is not None
    assert len(latest) == 3
    assert len(latest[0]) == 3
    assert len(latest[0][0]) == 1


def _message_separation(messages: list[float]) -> float:
    if len(messages) < 2:
        return 0.0
    return min(
        abs(left - right)
        for index, left in enumerate(messages)
        for right in messages[index + 1 :]
    )


def _assert_sgd_update_matches(
    *,
    before_parameters: list[torch.nn.Parameter],
    after_parameters: list[torch.nn.Parameter],
    learning_rate: float,
) -> None:
    for before_parameter, after_parameter in zip(
        before_parameters,
        after_parameters,
        strict=True,
    ):
        grad = before_parameter.grad
        if grad is None:
            expected = before_parameter.detach()
        else:
            expected = before_parameter.detach() - learning_rate * grad.detach()
        assert torch.allclose(after_parameter.detach(), expected, atol=1e-6)
