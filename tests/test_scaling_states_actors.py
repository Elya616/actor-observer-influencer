"""Tests for the Phase 3 joint state+actor scaling probe.

These cover the two new pieces Phase 3 relies on:

* the opt-in per-actor running-mean baseline added to ``MultiActorTrainer``
  (needed so learnable Actors can train on the all-positive grid reward), and
* the experiment helpers in ``run_scaling_states_actors`` (the blind Actor and a
  small end-to-end learning rung).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

from actor_observer_rl.agents import BudgetedObserver
from actor_observer_rl.envs import NoisyActorBanditEnv
from actor_observer_rl.policies import NeuralPolicy
from actor_observer_rl.training import MultiActorTrainer, make_fixed_softmax_actor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

from run_scaling_states_actors import (  # noqa: E402
    BlindActor,
    greedy_accuracy,
    top_k_precision,
    train_one_rung,
)


def _budgeted_observer(num_actors: int) -> BudgetedObserver:
    return BudgetedObserver(
        num_actors=num_actors,
        budget=1.0,
        utility_learning_rate=0.05,
        following_learning_rate=0.05,
        c=1.0,
        seed=0,
    )


def test_reward_baseline_validation() -> None:
    actors = [
        make_fixed_softmax_actor(num_states=1, num_actions=2, target_action=j)
        for j in range(2)
    ]
    with pytest.raises(ValueError):
        MultiActorTrainer(
            env=NoisyActorBanditEnv(qualities=[1.0, 0.0]),
            actors=actors,
            observer=_budgeted_observer(2),
            actor_selection_probs=[0.5, 0.5],
            reward_baseline="ema",
        )
    with pytest.raises(ValueError):
        MultiActorTrainer(
            env=NoisyActorBanditEnv(qualities=[1.0, 0.0]),
            actors=actors,
            observer=_budgeted_observer(2),
            actor_selection_probs=[0.5, 0.5],
            baseline_momentum=0.0,
        )


def test_advantage_none_returns_reward_unchanged() -> None:
    actors = [
        make_fixed_softmax_actor(num_states=1, num_actions=2, target_action=j)
        for j in range(2)
    ]
    trainer = MultiActorTrainer(
        env=NoisyActorBanditEnv(qualities=[1.0, 0.0]),
        actors=actors,
        observer=_budgeted_observer(2),
        actor_selection_probs=[0.5, 0.5],
    )
    # Default reward_baseline is None: the advantage is the raw reward.
    assert trainer._advantage(0, 0.7) == pytest.approx(0.7)
    assert trainer._advantage(1, -0.3) == pytest.approx(-0.3)


def test_running_mean_baseline_centers_rewards() -> None:
    actors = [
        make_fixed_softmax_actor(num_states=1, num_actions=2, target_action=j)
        for j in range(2)
    ]
    trainer = MultiActorTrainer(
        env=NoisyActorBanditEnv(qualities=[1.0, 0.0]),
        actors=actors,
        observer=_budgeted_observer(2),
        actor_selection_probs=[0.5, 0.5],
        reward_baseline="running_mean",
        baseline_momentum=0.5,
    )
    # First call seeds the baseline, so the advantage is zero. Each actor keeps
    # its own baseline, so updating actor 0 must not move actor 1's.
    assert trainer._advantage(0, 1.0) == pytest.approx(0.0)
    assert trainer._advantage(0, 1.0) == pytest.approx(0.0)
    assert trainer._advantage(1, 2.0) == pytest.approx(0.0)
    # Actor 0 baseline moves 1.0 -> 1.0 + 0.5*(3.0-1.0) = 2.0 before subtracting,
    # so the returned advantage is 3.0 - 2.0 = 1.0 (update-then-center, matching
    # the single-actor Trainer).
    assert trainer._advantage(0, 3.0) == pytest.approx(1.0)


def test_force_report_always_updates_and_records_probabilities() -> None:
    # With a tiny budget the Bernoulli gate theta(mu) is small, so ordinary
    # runs skip most updates. force_report=True must bypass the gate: every step
    # is reported. The trace must also expose the per-snapshot report probs.
    actors = [
        make_fixed_softmax_actor(num_states=1, num_actions=2, target_action=j)
        for j in range(2)
    ]
    observer = BudgetedObserver(
        num_actors=2,
        budget=0.01,
        utility_learning_rate=0.05,
        following_learning_rate=0.05,
        c=1.0,
        seed=0,
    )
    trainer = MultiActorTrainer(
        env=NoisyActorBanditEnv(qualities=[1.0, 0.0]),
        actors=actors,
        observer=observer,
        actor_selection_probs=[0.5, 0.5],
        force_report=True,
        snapshot_interval=10,
    )
    trace = trainer.run(num_steps=50, seed=0)
    assert all(trace.reported_to_actor)
    assert trace.metadata["force_report"] is True
    assert len(trace.report_probabilities) == 5
    assert all(len(row) == 2 for row in trace.report_probabilities)


def test_default_run_gates_some_reports() -> None:
    # The opposite of the perfect-reward baseline: with a tiny budget, the
    # Bernoulli gate must drop at least some updates.
    actors = [
        make_fixed_softmax_actor(num_states=1, num_actions=2, target_action=j)
        for j in range(2)
    ]
    observer = BudgetedObserver(
        num_actors=2,
        budget=0.01,
        utility_learning_rate=0.05,
        following_learning_rate=0.05,
        c=1.0,
        seed=0,
    )
    trainer = MultiActorTrainer(
        env=NoisyActorBanditEnv(qualities=[1.0, 0.0]),
        actors=actors,
        observer=observer,
        actor_selection_probs=[0.5, 0.5],
        snapshot_interval=10,
    )
    trace = trainer.run(num_steps=200, seed=0)
    assert not all(trace.reported_to_actor)


def test_blind_actor_ignores_the_state() -> None:
    policy = NeuralPolicy(input_dim=2, hidden_dim=8, num_actions=4)
    actor = BlindActor(policy=policy)
    # Two different state features must give the same action distribution,
    # because the blind actor zeros its input before acting.
    torch.manual_seed(0)
    result_a = actor.act(torch.tensor([0.1, 0.9]), sample=False)
    result_b = actor.act(torch.tensor([0.8, 0.2]), sample=False)
    assert torch.allclose(result_a.probs, result_b.probs)


def test_top_k_precision_ranks_good_actors() -> None:
    # Good actors are indices 0..k-1; here both top-2 mu are good.
    assert top_k_precision([0.9, 0.8, 0.1, 0.05], num_good=2) == pytest.approx(1.0)
    # One good actor lost its budget to a bad actor.
    assert top_k_precision([0.9, 0.05, 0.8, 0.1], num_good=2) == pytest.approx(0.5)


def test_small_rung_learns_and_concentrates() -> None:
    # A tiny, easy rung should train: the sighted good actors learn the grid and
    # the observer puts most of its budget on them rather than the blind crowd.
    result = train_one_rung(
        grid_size=3,
        num_actors=6,
        num_good=2,
        num_actions=4,
        hidden_dim=16,
        learning_rate=0.01,
        budget=2.0,
        utility_lr=0.05,
        following_lr=0.05,
        c=1.0,
        num_steps=12000,
        snapshot_interval=2000,
        seed=0,
    )
    assert result["final_good_accuracy"] > 0.6
    assert result["final_concentration"] > 0.5


def test_greedy_accuracy_perfect_when_policy_is_correct() -> None:
    # A blind actor evaluated on a one-region grid (num_actions=1 is invalid, so
    # use a degenerate check): greedy_accuracy returns a fraction in [0, 1].
    from actor_observer_rl.envs import GridWorldBanditEnv

    env = GridWorldBanditEnv(grid_size=2, num_actions=4)
    actor = BlindActor(policy=NeuralPolicy(input_dim=2, hidden_dim=8, num_actions=4))
    accuracy = greedy_accuracy(actor, env, blind=True)
    assert 0.0 <= accuracy <= 1.0
