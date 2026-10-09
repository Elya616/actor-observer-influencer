"""Tests for paper Section 7 group-level imperfect-state learning."""

from __future__ import annotations

import math

import torch

from actor_observer_rl.agents import (
    Actor,
    FixedTableMessageModule,
    FollowingRates,
    MaskedObserverMessageModule,
    Observer,
)
from actor_observer_rl.envs import FeaturePartialObservableStateEnv, PartialObservableStateEnv
from actor_observer_rl.policies import SoftmaxPolicy, SumDecoderPolicy
from actor_observer_rl.training import GroupImperfectStateTrainer
from actor_observer_rl.utils.seeding import set_global_seed


STATE_FEATURES = [
    [0.0, 0.0],
    [1.0, 0.0],
    [1.0, 1.0],
]
STATE_DISTRIBUTION = [1 / 3, 1 / 3, 1 / 3]
REWARD_TABLE = [
    [1.0, -0.1, -0.1],
    [-0.1, 1.0, -0.1],
    [-0.1, -0.1, 1.0],
]
OBSERVER_NAMES = ["zero_input", "first_feature", "full_state"]
OBSERVER_MASKS = [
    [0.0, 0.0],
    [1.0, 0.0],
    [1.0, 1.0],
]


def _make_paper71_trainer(
    *,
    seed: int = 0,
    following_learning_rate: float = 0.02,
    selection_update_rule: str = "raw_reward",
    selection_baseline_learning_rate: float = 0.05,
    selection_baseline_initial: float = 0.0,
    selection_entropy_beta: float = 0.0,
    reward_table: list[list[float]] | None = None,
) -> GroupImperfectStateTrainer:
    set_global_seed(seed)
    table = REWARD_TABLE if reward_table is None else reward_table
    env = PartialObservableStateEnv(
        reward_table=table,
        state_distribution=STATE_DISTRIBUTION,
        observation_dim=1,
    )
    fixed_messages = [
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 1.0],
        [0.0, 1.0, 2.0],
    ]
    observers = [
        Observer(message_module=FixedTableMessageModule(messages))
        for messages in fixed_messages
    ]
    actors = [
        Actor(SumDecoderPolicy(num_actions=3, sharpness=4.0))
        for _ in range(3)
    ]
    following_rates_by_actor = [
        FollowingRates(num_observers=len(observers)) for _ in actors
    ]
    following_optimizer = torch.optim.SGD(
        [
            parameter
            for rates in following_rates_by_actor
            for parameter in rates.parameters()
        ],
        lr=following_learning_rate,
    )
    return GroupImperfectStateTrainer(
        env=env,
        actors=actors,
        observers=observers,
        following_rates_by_actor=following_rates_by_actor,
        actor_selection_probs=[1 / 3, 1 / 3, 1 / 3],
        following_optimizer=following_optimizer,
        optimal_actions=[0, 1, 2],
        selection_update_rule=selection_update_rule,
        selection_baseline_learning_rate=selection_baseline_learning_rate,
        selection_baseline_initial=selection_baseline_initial,
        selection_entropy_beta=selection_entropy_beta,
    )


def _make_learnable_group_trainer(
    *,
    seed: int = 0,
    following_learning_rate: float = 0.005,
    observer_learning_rate: float = 0.02,
    actor_learning_rate: float | None = None,
    selection_update_rule: str = "raw_reward",
    selection_baseline_learning_rate: float = 0.05,
    selection_baseline_initial: float = 0.0,
    selection_entropy_beta: float = 0.0,
) -> GroupImperfectStateTrainer:
    set_global_seed(seed)
    env = FeaturePartialObservableStateEnv(
        state_features=STATE_FEATURES,
        reward_table=REWARD_TABLE,
        state_distribution=STATE_DISTRIBUTION,
        observation_dim=1,
    )
    observers = []
    for input_mask in OBSERVER_MASKS:
        module = MaskedObserverMessageModule(
            input_dim=2,
            message_dim=1,
            input_mask=input_mask,
        )
        with torch.no_grad():
            module.linear.weight.zero_()
            module.linear.bias.zero_()
        observers.append(Observer(message_module=module))

    if actor_learning_rate is None:
        actors = [
            Actor(SumDecoderPolicy(num_actions=3, sharpness=4.0))
            for _ in range(3)
        ]
        actor_optimizers = None
    else:
        actors = [Actor(SoftmaxPolicy(input_dim=1, num_actions=3)) for _ in range(3)]
        actor_optimizers = [
            torch.optim.SGD(actor.parameters(), lr=actor_learning_rate)
            for actor in actors
        ]

    following_rates_by_actor = [
        FollowingRates(num_observers=len(observers)) for _ in actors
    ]
    following_optimizer = torch.optim.SGD(
        [
            parameter
            for rates in following_rates_by_actor
            for parameter in rates.parameters()
        ],
        lr=following_learning_rate,
    )
    observer_optimizer = torch.optim.SGD(
        [parameter for observer in observers for parameter in observer.parameters()],
        lr=observer_learning_rate,
    )
    return GroupImperfectStateTrainer(
        env=env,
        actors=actors,
        observers=observers,
        following_rates_by_actor=following_rates_by_actor,
        actor_selection_probs=[1 / 3, 1 / 3, 1 / 3],
        following_optimizer=following_optimizer,
        observer_optimizer=observer_optimizer,
        actor_optimizers=actor_optimizers,
        optimal_actions=[0, 1, 2],
        selection_update_rule=selection_update_rule,
        selection_baseline_learning_rate=selection_baseline_learning_rate,
        selection_baseline_initial=selection_baseline_initial,
        selection_entropy_beta=selection_entropy_beta,
    )


def test_paper71_updates_only_active_actor_selection_logits() -> None:
    trainer = _make_paper71_trainer(seed=3)
    before_logits = [
        rates.logits.detach().clone() for rates in trainer.following_rates_by_actor
    ]

    trace = trainer.run(num_steps=1, seed=3, eval_interval=1)

    selected_actor = trace.selected_actor_indices[0]
    after_logits = [
        rates.logits.detach().clone() for rates in trainer.following_rates_by_actor
    ]
    for actor_index, (before, after) in enumerate(
        zip(before_logits, after_logits, strict=True)
    ):
        if actor_index == selected_actor:
            assert not torch.allclose(before, after)
        else:
            assert torch.allclose(before, after)


def test_paper7_advantage_selection_uses_per_actor_baseline() -> None:
    trainer = _make_paper71_trainer(
        seed=3,
        following_learning_rate=0.02,
        selection_update_rule="advantage_per_actor",
        selection_baseline_learning_rate=0.1,
        selection_baseline_initial=0.5,
    )
    before_logits = [
        rates.logits.detach().clone() for rates in trainer.following_rates_by_actor
    ]
    before_probs = [
        torch.softmax(logits, dim=-1) for logits in before_logits
    ]

    trace = trainer.run(num_steps=1, seed=3, eval_interval=1)

    selected_actor = trace.selected_actor_indices[0]
    selected_observer = trace.selected_observer_indices[0]
    reward = trace.rewards[0]
    advantage = reward - 0.5
    assert math.isclose(trace.selection_update_multipliers[0], advantage)

    expected_delta = -0.02 * advantage * before_probs[selected_actor]
    expected_delta[selected_observer] = 0.02 * advantage * (
        1.0 - before_probs[selected_actor][selected_observer]
    )
    actual_delta = (
        trainer.following_rates_by_actor[selected_actor].logits.detach()
        - before_logits[selected_actor]
    )
    assert torch.allclose(actual_delta, expected_delta, atol=1e-6)

    expected_baseline = 0.5 + 0.1 * (reward - 0.5)
    for actor_index, baseline in enumerate(trainer.selection_baselines):
        if actor_index == selected_actor:
            assert math.isclose(baseline, expected_baseline, rel_tol=1e-6)
        else:
            assert math.isclose(baseline, 0.5, rel_tol=1e-6)


def test_paper7_entropy_selection_pushes_active_logits_toward_uniform() -> None:
    trainer = _make_paper71_trainer(
        seed=4,
        following_learning_rate=0.2,
        selection_entropy_beta=0.3,
        reward_table=[[0.0, 0.0, 0.0] for _ in range(3)],
    )
    with torch.no_grad():
        for rates in trainer.following_rates_by_actor:
            rates.logits.copy_(torch.tensor([2.0, 0.0, -2.0]))
    before_logits = [
        rates.logits.detach().clone() for rates in trainer.following_rates_by_actor
    ]

    trace = trainer.run(num_steps=1, seed=4, eval_interval=1)

    selected_actor = trace.selected_actor_indices[0]
    for actor_index, rates in enumerate(trainer.following_rates_by_actor):
        after = rates.logits.detach()
        if actor_index == selected_actor:
            before_range = float((before_logits[actor_index].max() - before_logits[actor_index].min()).item())
            after_range = float((after.max() - after.min()).item())
            assert after_range < before_range
        else:
            assert torch.allclose(after, before_logits[actor_index])


def test_paper71_group_selection_concentrates_for_each_actor() -> None:
    trainer = _make_paper71_trainer(seed=1, following_learning_rate=0.02)

    trace = trainer.run(num_steps=7000, seed=1, eval_interval=700)

    final_weights = trace.final_selection_weights_by_actor
    assert final_weights is not None
    assert len(final_weights) == 3
    assert all(math.isclose(sum(row), 1.0, abs_tol=1e-6) for row in final_weights)
    assert all(row[2] > 0.90 for row in final_weights)
    assert trace.final_expected_reward is not None
    assert trace.final_expected_reward > 0.95
    assert trace.final_mean_optimal_action_prob is not None
    assert trace.final_mean_optimal_action_prob > 0.95


def test_paper72_updates_selected_observer_and_active_actor_selector() -> None:
    trainer = _make_learnable_group_trainer(seed=5, observer_learning_rate=0.02)
    before_observers = [
        [parameter.detach().clone() for parameter in observer.parameters()]
        for observer in trainer.observers
    ]
    before_logits = [
        rates.logits.detach().clone() for rates in trainer.following_rates_by_actor
    ]

    trace = trainer.run(num_steps=1, seed=5, eval_interval=1)

    selected_actor = trace.selected_actor_indices[0]
    selected_observer = trace.selected_observer_indices[0]
    after_logits = [
        rates.logits.detach().clone() for rates in trainer.following_rates_by_actor
    ]
    for actor_index, (before, after) in enumerate(
        zip(before_logits, after_logits, strict=True)
    ):
        if actor_index == selected_actor:
            assert not torch.allclose(before, after)
        else:
            assert torch.allclose(before, after)

    for observer_index, observer in enumerate(trainer.observers):
        new_params = [parameter.detach().clone() for parameter in observer.parameters()]
        moved = any(
            not torch.allclose(before, after)
            for before, after in zip(
                before_observers[observer_index],
                new_params,
                strict=True,
            )
        )
        if observer_index == selected_observer:
            assert moved
        else:
            assert not moved


def test_paper72_trace_records_selected_observer_gradient_norm() -> None:
    trainer = _make_learnable_group_trainer(seed=5, observer_learning_rate=0.02)

    trace = trainer.run(num_steps=1, seed=5, eval_interval=1)

    selected_observer = trace.selected_observer_indices[0]
    assert len(trace.observer_gradient_norms) == 1
    assert len(trace.observer_gradient_norms[0]) == len(trainer.observers)
    for observer_index, gradient_norm in enumerate(trace.observer_gradient_norms[0]):
        if observer_index == selected_observer:
            assert gradient_norm > 0.0
        else:
            assert gradient_norm == 0.0


def test_paper72_group_selection_and_compression_learn_full_observer() -> None:
    trainer = _make_learnable_group_trainer(
        seed=2,
        following_learning_rate=0.005,
        observer_learning_rate=0.01,
    )

    trace = trainer.run(num_steps=18000, seed=2, eval_interval=1800)

    final_weights = trace.final_selection_weights_by_actor
    final_messages = trace.final_observer_messages_by_state
    assert final_weights is not None
    assert final_messages is not None
    assert all(row[2] > 0.90 for row in final_weights)
    assert trace.final_expected_reward is not None
    assert trace.final_expected_reward > 0.94
    full_state_messages = [message[0] for message in final_messages[2]]
    assert _message_separation(full_state_messages) > 1.0


def test_paper73_updates_active_actor_selected_observer_and_selector() -> None:
    trainer = _make_learnable_group_trainer(
        seed=6,
        observer_learning_rate=0.05,
        actor_learning_rate=0.05,
    )
    before_actor_params = [
        [parameter.detach().clone() for parameter in actor.parameters()]
        for actor in trainer.actors
    ]
    before_observers = [
        [parameter.detach().clone() for parameter in observer.parameters()]
        for observer in trainer.observers
    ]
    before_logits = [
        rates.logits.detach().clone() for rates in trainer.following_rates_by_actor
    ]

    trace = trainer.run(num_steps=1, seed=6, eval_interval=1)

    selected_actor = trace.selected_actor_indices[0]
    selected_observer = trace.selected_observer_indices[0]
    for actor_index, actor in enumerate(trainer.actors):
        new_params = [parameter.detach().clone() for parameter in actor.parameters()]
        moved = any(
            not torch.allclose(before, after)
            for before, after in zip(
                before_actor_params[actor_index],
                new_params,
                strict=True,
            )
        )
        if actor_index == selected_actor:
            assert moved
        else:
            assert not moved

    for actor_index, rates in enumerate(trainer.following_rates_by_actor):
        if actor_index == selected_actor:
            assert not torch.allclose(before_logits[actor_index], rates.logits)
        else:
            assert torch.allclose(before_logits[actor_index], rates.logits)

    for observer_index, observer in enumerate(trainer.observers):
        new_params = [parameter.detach().clone() for parameter in observer.parameters()]
        moved = any(
            not torch.allclose(before, after)
            for before, after in zip(
                before_observers[observer_index],
                new_params,
                strict=True,
            )
        )
        if observer_index == selected_observer:
            assert moved
        else:
            assert not moved


def test_paper73_group_joint_learning_converges() -> None:
    trainer = _make_learnable_group_trainer(
        seed=4,
        following_learning_rate=0.005,
        observer_learning_rate=0.05,
        actor_learning_rate=0.1,
    )

    trace = trainer.run(num_steps=18000, seed=4, eval_interval=1800)

    final_weights = trace.final_selection_weights_by_actor
    per_actor_observer_rewards = trace.final_per_actor_observer_expected_rewards
    assert final_weights is not None
    assert per_actor_observer_rewards is not None
    assert all(row[2] > 0.90 for row in final_weights)
    assert all(
        observer_rewards[2] > observer_rewards[1] + 0.15
        for observer_rewards in per_actor_observer_rewards
    )
    assert trace.final_expected_reward is not None
    assert trace.final_expected_reward > 0.95
    assert trace.final_mean_optimal_action_prob is not None
    assert trace.final_mean_optimal_action_prob > 0.95


def _message_separation(messages: list[float]) -> float:
    if len(messages) < 2:
        return 0.0
    return min(
        abs(left - right)
        for index, left in enumerate(messages)
        for right in messages[index + 1 :]
    )
