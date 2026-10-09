"""Tests for the Case 4 multi-observer sweep helpers."""

from __future__ import annotations

import csv

import pytest

from actor_observer_rl.evaluation import (
    MultiObserverExperiment,
    MultiObserverRunSummary,
    run_multi_observer_once,
    summarize_multi_observer_runs,
    write_multi_observer_curve_csv,
    write_multi_observer_run_csv,
    write_multi_observer_summary_csv,
)


def _make_experiment(num_observers: int = 2) -> MultiObserverExperiment:
    return MultiObserverExperiment(
        name="case4_smoke",
        display_name="Case 4 smoke",
        reward_table=((1.0, 0.0), (0.0, 1.0)),
        state_distribution=(0.5, 0.5),
        observation_dim=1,
        message_dim=2,
        num_observers=num_observers,
    )


def test_multi_observer_experiment_validates_num_observers() -> None:
    with pytest.raises(ValueError, match="num_observers"):
        MultiObserverExperiment(
            name="bad",
            display_name="Bad",
            reward_table=((1.0, 0.0), (0.0, 1.0)),
            state_distribution=(0.5, 0.5),
            observation_dim=1,
            message_dim=2,
            num_observers=1,
        )


def test_multi_observer_experiment_derives_optimal_actions() -> None:
    experiment = MultiObserverExperiment(
        name="auto_optimal",
        display_name="Auto",
        reward_table=((0.0, 1.0, 0.0), (1.0, 0.0, 0.5)),
        state_distribution=(0.5, 0.5),
        observation_dim=1,
        message_dim=2,
        num_observers=2,
    )

    assert experiment.optimal_actions == (1, 0)
    assert experiment.optimal_expected_reward == pytest.approx(1.0)


def test_run_multi_observer_once_records_curve_and_summary() -> None:
    experiment = _make_experiment(num_observers=3)

    curve, summary = run_multi_observer_once(
        experiment=experiment,
        learning_rate=0.1,
        seed=0,
        num_steps=200,
        eval_interval=50,
        reward_window=50,
        threshold_fraction=0.9,
    )

    assert [point.step for point in curve] == [0, 50, 100, 150, 200]
    for point in curve:
        assert len(point.following_weights) == 3
        assert sum(point.following_weights) == pytest.approx(1.0, abs=1e-5)
    assert summary.ceiling_reward == pytest.approx(1.0)
    assert summary.threshold_reward == pytest.approx(0.9)
    assert summary.num_observers == 3
    assert summary.final_expected_reward > curve[0].expected_reward
    assert summary.nan_detected is False
    assert len(summary.selection_histogram) == 3
    assert sum(summary.selection_histogram) == pytest.approx(1.0, abs=1e-5)
    assert all(0.0 <= fraction <= 1.0 for fraction in summary.selection_histogram)


def test_summarize_multi_observer_runs_averages_following_weights() -> None:
    summaries = [
        MultiObserverRunSummary(
            experiment="case4_smoke",
            learning_rate=0.1,
            seed=0,
            num_steps=10,
            num_observers=2,
            final_expected_reward=0.8,
            final_average_reward_last_window=0.7,
            final_mean_optimal_action_probability=0.85,
            final_following_weights=(0.4, 0.6),
            selection_histogram=(0.3, 0.7),
            ceiling_reward=1.0,
            threshold_reward=0.9,
            convergence_step=None,
            converged=False,
            nan_detected=False,
        ),
        MultiObserverRunSummary(
            experiment="case4_smoke",
            learning_rate=0.1,
            seed=1,
            num_steps=10,
            num_observers=2,
            final_expected_reward=0.95,
            final_average_reward_last_window=0.9,
            final_mean_optimal_action_probability=0.95,
            final_following_weights=(0.6, 0.4),
            selection_histogram=(0.7, 0.3),
            ceiling_reward=1.0,
            threshold_reward=0.9,
            convergence_step=5,
            converged=True,
            nan_detected=False,
        ),
    ]

    aggregate = summarize_multi_observer_runs(summaries)

    assert len(aggregate) == 1
    assert aggregate[0].success_rate == 0.5
    assert aggregate[0].mean_final_expected_reward == pytest.approx(0.875)
    assert aggregate[0].mean_final_following_weights == pytest.approx((0.5, 0.5))
    assert aggregate[0].mean_selection_histogram == pytest.approx((0.5, 0.5))


def test_write_multi_observer_csvs_serialize_following_weights(tmp_path) -> None:
    experiment = _make_experiment(num_observers=2)
    curve, summary = run_multi_observer_once(
        experiment=experiment,
        learning_rate=0.1,
        seed=0,
        num_steps=100,
        eval_interval=50,
        reward_window=50,
        threshold_fraction=0.9,
    )
    aggregate = summarize_multi_observer_runs([summary])

    curve_path = tmp_path / "curves.csv"
    runs_path = tmp_path / "runs.csv"
    summary_path = tmp_path / "summary.csv"
    write_multi_observer_curve_csv(curve, curve_path)
    write_multi_observer_run_csv([summary], runs_path)
    write_multi_observer_summary_csv(aggregate, summary_path)

    with curve_path.open("r", encoding="utf-8") as file:
        curve_rows = list(csv.DictReader(file))
    with runs_path.open("r", encoding="utf-8") as file:
        run_rows = list(csv.DictReader(file))
    with summary_path.open("r", encoding="utf-8") as file:
        summary_rows = list(csv.DictReader(file))

    assert curve_rows[0]["following_weights"].count("|") == 1
    assert run_rows[0]["final_following_weights"].count("|") == 1
    assert run_rows[0]["selection_histogram"].count("|") == 1
    assert summary_rows[0]["mean_final_following_weights"].count("|") == 1
    assert summary_rows[0]["mean_selection_histogram"].count("|") == 1
    assert summary_rows[0]["num_runs"] == "1"
