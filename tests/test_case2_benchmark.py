"""Tests for Case 2 benchmark abstractions."""

from __future__ import annotations

import csv

import pytest

from actor_observer_rl.evaluation import (
    Case2BenchmarkConfig,
    Case2RunResult,
    Case2Scenario,
    Case2StateDistribution,
    run_case2_benchmark,
    run_case2_once,
    summarize_case2_results,
    write_case2_per_state_csv,
    write_case2_results_csv,
)


def test_case2_scenario_rejects_wrong_optimal_action() -> None:
    with pytest.raises(ValueError, match="maximum-reward"):
        Case2Scenario(
            name="bad",
            reward_table=((1.0, 0.0), (0.0, 1.0)),
            optimal_actions=(1, 1),
        )


def test_case2_state_distribution_rejects_invalid_sum() -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        Case2StateDistribution(name="bad", probabilities=(0.6, 0.6))


def test_default_case2_benchmark_matrix_has_300_compatible_runs() -> None:
    config = Case2BenchmarkConfig()

    compatible_scenario_distribution_pairs = sum(
        1
        for scenario in config.scenarios
        for distribution in config.state_distributions
        if scenario.num_states == distribution.num_states
    )
    num_runs = (
        compatible_scenario_distribution_pairs
        * len(config.learning_rates)
        * len(config.seeds)
    )

    assert num_runs == 300


def test_case2_benchmark_rejects_scenario_without_matching_distribution() -> None:
    with pytest.raises(ValueError, match="matching state distribution"):
        Case2BenchmarkConfig(
            scenarios=(
                Case2Scenario(
                    name="three_state",
                    reward_table=((1.0, 0.0), (0.0, 1.0), (0.5, 0.0)),
                    optimal_actions=(0, 1, 0),
                ),
            ),
            state_distributions=(
                Case2StateDistribution(name="two_state", probabilities=(0.5, 0.5)),
            ),
        )


def test_run_case2_benchmark_skips_nonmatching_state_distributions() -> None:
    config = Case2BenchmarkConfig(
        scenarios=(
            Case2Scenario(
                name="two_state_easy",
                reward_table=((1.0, 0.0), (0.0, 1.0)),
                optimal_actions=(0, 1),
            ),
        ),
        state_distributions=(
            Case2StateDistribution(name="two_state", probabilities=(0.5, 0.5)),
            Case2StateDistribution(name="three_state", probabilities=(0.4, 0.3, 0.3)),
        ),
        learning_rates=(0.1,),
        seeds=(0,),
        num_steps=10,
    )

    run_results, per_state_results = run_case2_benchmark(config)

    assert len(run_results) == 1
    assert run_results[0].distribution == "two_state"
    assert len(per_state_results) == 2


def test_run_case2_once_learns_best_action_for_each_state() -> None:
    scenario = Case2Scenario(
        name="two_state_easy",
        reward_table=((1.0, 0.0), (0.0, 1.0)),
        optimal_actions=(0, 1),
    )
    state_distribution = Case2StateDistribution(
        name="uniform",
        probabilities=(0.5, 0.5),
    )

    result, per_state_results = run_case2_once(
        scenario=scenario,
        state_distribution=state_distribution,
        learning_rate=0.1,
        seed=0,
        num_steps=800,
    )

    assert result.converged is True
    assert result.min_final_optimal_action_prob > 0.9
    assert result.nan_detected is False
    assert sum(result.state_sample_counts) == 800
    assert len(per_state_results) == 2
    assert all(row.final_optimal_action_prob > 0.9 for row in per_state_results)


def test_summarize_case2_results_aggregates_by_distribution_and_learning_rate() -> None:
    results = [
        _case2_run_result(
            distribution="uniform",
            seed=0,
            weighted_prob=0.9,
            min_prob=0.8,
            expected_reward=0.85,
            converged=False,
            convergence_step=None,
        ),
        _case2_run_result(
            distribution="uniform",
            seed=1,
            weighted_prob=1.0,
            min_prob=0.95,
            expected_reward=0.95,
            converged=True,
            convergence_step=50,
        ),
    ]

    summaries = summarize_case2_results(results)

    assert len(summaries) == 1
    assert summaries[0].success_rate == 0.5
    assert summaries[0].mean_weighted_final_optimal_action_prob == pytest.approx(0.95)
    assert summaries[0].mean_min_final_optimal_action_prob == pytest.approx(0.875)
    assert summaries[0].mean_expected_reward_policy == pytest.approx(0.9)
    assert summaries[0].mean_overall_convergence_step == 50


def test_write_case2_results_csv_writes_readable_compound_fields(tmp_path) -> None:
    result = _case2_run_result(
        distribution="uniform",
        seed=0,
        weighted_prob=0.9,
        min_prob=0.8,
        expected_reward=0.85,
        converged=False,
        convergence_step=None,
    )
    output_path = tmp_path / "runs.csv"

    write_case2_results_csv([result], output_path)

    with output_path.open("r", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert rows[0]["state_distribution"] == "0.5|0.5"
    assert rows[0]["reward_table"] == "1|0;0|1"
    assert rows[0]["optimal_actions"] == "0|1"
    assert rows[0]["per_state_convergence_steps"] == "|"


def test_write_case2_per_state_csv_writes_one_row_per_state(tmp_path) -> None:
    _, per_state_results = run_case2_once(
        scenario=Case2Scenario(
            name="two_state_easy",
            reward_table=((1.0, 0.0), (0.0, 1.0)),
            optimal_actions=(0, 1),
        ),
        state_distribution=Case2StateDistribution(
            name="uniform",
            probabilities=(0.5, 0.5),
        ),
        learning_rate=0.1,
        seed=1,
        num_steps=800,
    )
    output_path = tmp_path / "per_state.csv"

    write_case2_per_state_csv(per_state_results, output_path)

    with output_path.open("r", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert [row["state_index"] for row in rows] == ["0", "1"]
    assert [row["optimal_action"] for row in rows] == ["0", "1"]


def _case2_run_result(
    distribution: str,
    seed: int,
    weighted_prob: float,
    min_prob: float,
    expected_reward: float,
    converged: bool,
    convergence_step: int | None,
) -> Case2RunResult:
    return Case2RunResult(
        scenario="two_state_easy",
        distribution=distribution,
        num_states=2,
        num_actions=2,
        state_distribution=(0.5, 0.5),
        reward_table=((1.0, 0.0), (0.0, 1.0)),
        optimal_actions=(0, 1),
        learning_rate=0.1,
        seed=seed,
        num_steps=10,
        final_reward=1.0,
        average_reward_last_window=0.8,
        weighted_final_optimal_action_prob=weighted_prob,
        unweighted_final_optimal_action_prob=weighted_prob,
        min_final_optimal_action_prob=min_prob,
        expected_reward_policy=expected_reward,
        overall_convergence_step=convergence_step,
        converged=converged,
        state_sample_counts=(5, 5),
        per_state_final_optimal_action_probs=(weighted_prob, min_prob),
        per_state_convergence_steps=(None, convergence_step),
        nan_detected=False,
    )
