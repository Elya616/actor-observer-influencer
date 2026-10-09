"""Tests for focused partial-observation experiment helpers."""

from __future__ import annotations

import csv
import math

from actor_observer_rl.evaluation import (
    default_partial_observation_experiment,
    run_partial_observation_once,
    run_partial_observation_sweep,
    summarize_partial_observation_runs,
    write_partial_observation_curve_csv,
    write_partial_observation_run_csv,
    write_partial_observation_summary_csv,
)


def test_default_partial_observation_experiment_has_known_best_actions() -> None:
    experiment = default_partial_observation_experiment()

    assert experiment.num_states == 4
    assert experiment.num_actions == 2
    assert experiment.optimal_actions == (0, 1, 0, 1)
    assert experiment.optimal_expected_reward == 1.0


def test_partial_observation_run_records_initial_and_final_curve_points() -> None:
    experiment = default_partial_observation_experiment()

    curves, summary = run_partial_observation_once(
        experiment=experiment,
        learning_rate=0.05,
        seed=0,
        num_steps=20,
        eval_interval=10,
        reward_window=5,
        threshold_fraction=0.95,
    )

    assert [point.step for point in curves] == [0, 10, 20]
    assert summary.num_steps == 20
    assert summary.final_expected_reward == curves[-1].expected_reward
    assert all(math.isfinite(point.expected_reward) for point in curves)


def test_partial_observation_sweep_aggregates_by_learning_rate() -> None:
    experiment = default_partial_observation_experiment()

    _, run_summaries = run_partial_observation_sweep(
        experiment=experiment,
        learning_rates=[0.01, 0.05],
        seeds=[0, 1],
        num_steps=10,
        eval_interval=10,
        reward_window=5,
        threshold_fraction=0.95,
    )
    aggregate = summarize_partial_observation_runs(run_summaries)

    assert len(run_summaries) == 4
    assert len(aggregate) == 2
    assert {summary.learning_rate for summary in aggregate} == {0.01, 0.05}
    assert all(summary.num_runs == 2 for summary in aggregate)


def test_partial_observation_csv_writers_create_expected_headers(tmp_path) -> None:
    experiment = default_partial_observation_experiment()
    curves, run_summaries = run_partial_observation_sweep(
        experiment=experiment,
        learning_rates=[0.05],
        seeds=[0],
        num_steps=10,
        eval_interval=10,
        reward_window=5,
        threshold_fraction=0.95,
    )
    aggregate = summarize_partial_observation_runs(run_summaries)
    curve_path = tmp_path / "curves.csv"
    run_path = tmp_path / "runs.csv"
    summary_path = tmp_path / "summary.csv"

    write_partial_observation_curve_csv(curves, curve_path)
    write_partial_observation_run_csv(run_summaries, run_path)
    write_partial_observation_summary_csv(aggregate, summary_path)

    with curve_path.open("r", encoding="utf-8") as file:
        curve_header = next(csv.reader(file))
    with run_path.open("r", encoding="utf-8") as file:
        run_header = next(csv.reader(file))
    with summary_path.open("r", encoding="utf-8") as file:
        summary_header = next(csv.reader(file))

    assert "expected_reward" in curve_header
    assert "final_expected_reward" in run_header
    assert "mean_final_expected_reward" in summary_header
