"""Tests for Case 1 benchmark abstractions."""

from __future__ import annotations

import csv

import pytest

from actor_observer_rl.evaluation import (
    Case1RunResult,
    Case1Scenario,
    convergence_step,
    run_case1_once,
    summarize_case1_results,
    write_case1_results_csv,
)


def test_convergence_step_returns_first_1_based_threshold_crossing() -> None:
    assert convergence_step([0.2, 0.4, 0.91, 0.95], threshold=0.9) == 3


def test_convergence_step_returns_none_when_not_reached() -> None:
    assert convergence_step([0.2, 0.4, 0.89], threshold=0.9) is None


def test_case1_scenario_rejects_wrong_optimal_action() -> None:
    with pytest.raises(ValueError, match="maximum-reward"):
        Case1Scenario(name="bad", reward_table=(0.0, 1.0), optimal_action=0)


def test_run_case1_once_converges_on_easy_binary_scenario() -> None:
    result = run_case1_once(
        scenario=Case1Scenario(
            name="binary_easy",
            reward_table=(0.0, 1.0),
            optimal_action=1,
        ),
        learning_rate=0.1,
        seed=0,
        num_steps=500,
    )

    assert result.converged is True
    assert result.final_optimal_action_prob > 0.9
    assert result.nan_detected is False


def test_summarize_case1_results_aggregates_by_scenario_and_learning_rate() -> None:
    results = [
        Case1RunResult(
            scenario="s",
            num_actions=2,
            reward_table=(0.0, 1.0),
            optimal_action=1,
            learning_rate=0.1,
            seed=0,
            num_steps=10,
            final_reward=1.0,
            final_optimal_action_prob=0.9,
            average_reward_last_window=0.8,
            convergence_step=5,
            converged=True,
            nan_detected=False,
        ),
        Case1RunResult(
            scenario="s",
            num_actions=2,
            reward_table=(0.0, 1.0),
            optimal_action=1,
            learning_rate=0.1,
            seed=1,
            num_steps=10,
            final_reward=1.0,
            final_optimal_action_prob=0.8,
            average_reward_last_window=0.6,
            convergence_step=None,
            converged=False,
            nan_detected=False,
        ),
    ]

    summaries = summarize_case1_results(results)

    assert len(summaries) == 1
    assert summaries[0].success_rate == 0.5
    assert summaries[0].mean_final_optimal_action_prob == pytest.approx(0.85)
    assert summaries[0].mean_average_reward_last_window == pytest.approx(0.7)
    assert summaries[0].mean_convergence_step == 5


def test_write_case1_results_csv_writes_reward_table_as_readable_string(tmp_path) -> None:
    result = Case1RunResult(
        scenario="s",
        num_actions=2,
        reward_table=(0.0, 1.0),
        optimal_action=1,
        learning_rate=0.1,
        seed=0,
        num_steps=10,
        final_reward=1.0,
        final_optimal_action_prob=0.9,
        average_reward_last_window=0.8,
        convergence_step=5,
        converged=True,
        nan_detected=False,
    )
    output_path = tmp_path / "runs.csv"

    write_case1_results_csv([result], output_path)

    with output_path.open("r", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert rows[0]["scenario"] == "s"
    assert rows[0]["reward_table"] == "0.0|1.0"
    assert rows[0]["convergence_step"] == "5"
