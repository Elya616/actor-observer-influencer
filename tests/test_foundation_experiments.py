"""Tests for focused foundation experiment helpers."""

from __future__ import annotations

import csv

import pytest
import torch

from actor_observer_rl.agents import Actor
from actor_observer_rl.evaluation import (
    FoundationExperiment,
    expected_reward_under_policy,
    run_foundation_once,
    summarize_foundation_runs,
    write_aggregate_summary_csv,
)
from actor_observer_rl.policies import SoftmaxPolicy


def test_foundation_experiment_validates_state_distribution() -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        FoundationExperiment(
            name="bad",
            display_name="Bad",
            reward_table=((0.0, 1.0), (1.0, 0.0)),
            state_distribution=(0.8, 0.8),
        )


def test_expected_reward_under_policy_uses_action_probabilities() -> None:
    experiment = FoundationExperiment(
        name="single",
        display_name="Single",
        reward_table=((0.0, 1.0),),
        state_distribution=(1.0,),
    )
    policy = SoftmaxPolicy(input_dim=1, num_actions=2)
    torch.nn.init.zeros_(policy.linear.weight)
    torch.nn.init.zeros_(policy.linear.bias)
    actor = Actor(policy)

    expected_reward = expected_reward_under_policy(actor, experiment)

    assert expected_reward == pytest.approx(0.5)


def test_run_foundation_once_records_curve_and_summary() -> None:
    experiment = FoundationExperiment(
        name="single",
        display_name="Single",
        reward_table=((0.0, 1.0),),
        state_distribution=(1.0,),
    )

    curve, summary = run_foundation_once(
        experiment=experiment,
        learning_rate=0.1,
        seed=0,
        num_steps=300,
        eval_interval=100,
        reward_window=50,
        threshold_fraction=0.9,
    )

    assert [point.step for point in curve] == [0, 100, 200, 300]
    assert summary.ceiling_reward == 1.0
    assert summary.threshold_reward == 0.9
    assert summary.final_expected_reward > curve[0].expected_reward
    assert summary.nan_detected is False


def test_summarize_foundation_runs_and_write_csv(tmp_path) -> None:
    experiment = FoundationExperiment(
        name="single",
        display_name="Single",
        reward_table=((0.0, 1.0),),
        state_distribution=(1.0,),
    )
    _, summary = run_foundation_once(
        experiment=experiment,
        learning_rate=0.1,
        seed=0,
        num_steps=300,
        eval_interval=100,
        reward_window=50,
        threshold_fraction=0.9,
    )
    aggregate = summarize_foundation_runs([summary])
    output_path = tmp_path / "summary.csv"

    write_aggregate_summary_csv(aggregate, output_path)

    with output_path.open("r", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert rows[0]["experiment"] == "single"
    assert rows[0]["num_runs"] == "1"
