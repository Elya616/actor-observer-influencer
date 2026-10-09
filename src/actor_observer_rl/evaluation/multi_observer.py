"""Focused experiments for Case 4 multi-observer message learning."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, stdev

import torch

from actor_observer_rl.agents import (
    Actor,
    FollowingRates,
    MultiObserver,
    Observer,
    ObserverMessageModule,
)
from actor_observer_rl.envs import PartialObservableStateEnv
from actor_observer_rl.policies import SoftmaxPolicy
from actor_observer_rl.training import MultiObserverTrainer
from actor_observer_rl.utils.seeding import set_global_seed


@dataclass(frozen=True)
class MultiObserverExperiment:
    """Small setting used to verify learnable following rates."""

    name: str
    display_name: str
    reward_table: tuple[tuple[float, ...], ...]
    state_distribution: tuple[float, ...]
    observation_dim: int
    message_dim: int
    num_observers: int

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("experiment name must be non-empty")
        if not self.reward_table:
            raise ValueError("reward_table must contain at least one state")
        num_actions = len(self.reward_table[0])
        if num_actions <= 1:
            raise ValueError("reward_table must contain at least two actions")
        if any(len(row) != num_actions for row in self.reward_table):
            raise ValueError("reward_table rows must all have the same action count")
        if len(self.state_distribution) != len(self.reward_table):
            raise ValueError("state_distribution length must match reward_table rows")
        if any(probability < 0.0 for probability in self.state_distribution):
            raise ValueError("state_distribution cannot contain negative probabilities")
        if not math.isclose(
            sum(self.state_distribution),
            1.0,
            rel_tol=1e-6,
            abs_tol=1e-6,
        ):
            raise ValueError("state_distribution must sum to 1")
        if self.observation_dim <= 0:
            raise ValueError("observation_dim must be positive")
        if self.message_dim <= 0:
            raise ValueError("message_dim must be positive")
        if self.num_observers <= 1:
            raise ValueError("num_observers must be greater than one")

    @property
    def num_states(self) -> int:
        return len(self.reward_table)

    @property
    def num_actions(self) -> int:
        return len(self.reward_table[0])

    @property
    def optimal_actions(self) -> tuple[int, ...]:
        return tuple(
            max(range(self.num_actions), key=lambda action: rewards[action])
            for rewards in self.reward_table
        )

    @property
    def optimal_expected_reward(self) -> float:
        return sum(
            probability * max(rewards)
            for probability, rewards in zip(
                self.state_distribution,
                self.reward_table,
                strict=True,
            )
        )


@dataclass(frozen=True)
class MultiObserverCurvePoint:
    """One point in a multi-observer learning curve."""

    experiment: str
    learning_rate: float
    seed: int
    step: int
    expected_reward: float
    average_reward_last_window: float
    mean_optimal_action_probability: float
    following_weights: tuple[float, ...]
    ceiling_reward: float
    threshold_reward: float

    def as_csv_row(self) -> dict[str, str | int | float]:
        row = self.__dict__.copy()
        row["following_weights"] = _format_float_sequence(self.following_weights)
        return row


@dataclass(frozen=True)
class MultiObserverRunSummary:
    """Final metrics for one multi-observer run."""

    experiment: str
    learning_rate: float
    seed: int
    num_steps: int
    num_observers: int
    final_expected_reward: float
    final_average_reward_last_window: float
    final_mean_optimal_action_probability: float
    final_following_weights: tuple[float, ...]
    selection_histogram: tuple[float, ...]
    ceiling_reward: float
    threshold_reward: float
    convergence_step: int | None
    converged: bool
    nan_detected: bool

    def as_csv_row(self) -> dict[str, str | int | float | bool]:
        row = self.__dict__.copy()
        row["final_following_weights"] = _format_float_sequence(
            self.final_following_weights
        )
        row["selection_histogram"] = _format_float_sequence(self.selection_histogram)
        row["convergence_step"] = (
            "" if self.convergence_step is None else self.convergence_step
        )
        return row


@dataclass(frozen=True)
class MultiObserverAggregateSummary:
    """Aggregated multi-observer metrics over random seeds."""

    experiment: str
    learning_rate: float
    num_runs: int
    success_rate: float
    mean_final_expected_reward: float
    std_final_expected_reward: float
    mean_final_average_reward_last_window: float
    std_final_average_reward_last_window: float
    mean_final_optimal_action_probability: float
    std_final_optimal_action_probability: float
    mean_final_following_weights: tuple[float, ...]
    mean_selection_histogram: tuple[float, ...]
    ceiling_reward: float
    threshold_reward: float
    mean_convergence_step: float | None
    std_convergence_step: float | None
    num_nan_runs: int

    def as_csv_row(self) -> dict[str, str | int | float]:
        row = self.__dict__.copy()
        row["mean_final_following_weights"] = _format_float_sequence(
            self.mean_final_following_weights
        )
        row["mean_selection_histogram"] = _format_float_sequence(
            self.mean_selection_histogram
        )
        row["mean_convergence_step"] = (
            "" if self.mean_convergence_step is None else self.mean_convergence_step
        )
        row["std_convergence_step"] = (
            "" if self.std_convergence_step is None else self.std_convergence_step
        )
        return row


def default_multi_observer_experiment() -> MultiObserverExperiment:
    """Return the first focused setting for multi-observer following rates."""
    return MultiObserverExperiment(
        name="multi_observer_following_rate_learning",
        display_name="Multi-Observer Following Rate Learning",
        reward_table=(
            (1.0, 0.0),
            (0.0, 1.0),
            (1.0, 0.0),
            (0.0, 1.0),
        ),
        state_distribution=(0.25, 0.25, 0.25, 0.25),
        observation_dim=1,
        message_dim=2,
        num_observers=3,
    )


def run_multi_observer_once(
    experiment: MultiObserverExperiment,
    learning_rate: float,
    seed: int,
    num_steps: int,
    eval_interval: int,
    reward_window: int,
    threshold_fraction: float,
) -> tuple[list[MultiObserverCurvePoint], MultiObserverRunSummary]:
    """Run one learnable-following-rate verification experiment."""
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    if eval_interval <= 0:
        raise ValueError("eval_interval must be positive")
    if reward_window <= 0:
        raise ValueError("reward_window must be positive")
    if not 0.0 < threshold_fraction <= 1.0:
        raise ValueError("threshold_fraction must be in (0, 1]")

    set_global_seed(seed)
    env = PartialObservableStateEnv(
        reward_table=experiment.reward_table,
        state_distribution=experiment.state_distribution,
        observation_dim=experiment.observation_dim,
    )
    observers = [
        Observer(
            message_module=ObserverMessageModule(
                input_dim=env.state_dim,
                message_dim=experiment.message_dim,
            )
        )
        for _ in range(experiment.num_observers)
    ]
    multi_observer = MultiObserver(
        observers=observers,
        following_rates=FollowingRates(num_observers=experiment.num_observers),
    )
    actor = Actor(
        SoftmaxPolicy(
            input_dim=experiment.observation_dim + experiment.message_dim,
            num_actions=experiment.num_actions,
        )
    )
    trainer = MultiObserverTrainer(
        env=env,
        actor=actor,
        multi_observer=multi_observer,
        actor_optimizer=torch.optim.SGD(actor.parameters(), lr=learning_rate),
        observer_optimizer=torch.optim.SGD(
            list(multi_observer.observer_parameters()), lr=learning_rate
        ),
        following_optimizer=torch.optim.SGD(
            list(multi_observer.following_parameters()), lr=learning_rate
        ),
        optimal_actions=experiment.optimal_actions,
    )

    ceiling_reward = experiment.optimal_expected_reward
    threshold_reward = threshold_fraction * ceiling_reward
    initial_expected, initial_optimal_prob = trainer.evaluate_policy()
    if initial_optimal_prob is None:
        raise RuntimeError("multi-observer experiment requires optimal actions")
    initial_weights = tuple(
        float(weight)
        for weight in multi_observer.following_rates.current_weights().tolist()
    )

    trace = trainer.run(
        num_steps=num_steps,
        seed=seed,
        eval_interval=eval_interval,
    )

    curve = [
        MultiObserverCurvePoint(
            experiment=experiment.name,
            learning_rate=float(learning_rate),
            seed=int(seed),
            step=0,
            expected_reward=initial_expected,
            average_reward_last_window=0.0,
            mean_optimal_action_probability=initial_optimal_prob,
            following_weights=initial_weights,
            ceiling_reward=ceiling_reward,
            threshold_reward=threshold_reward,
        )
    ]
    for step, expected_reward, optimal_prob in zip(
        trace.evaluation_steps,
        trace.expected_rewards,
        trace.mean_optimal_action_probs,
        strict=True,
    ):
        weights_at_step = tuple(
            float(weight) for weight in trace.following_weights[step - 1]
        )
        curve.append(
            MultiObserverCurvePoint(
                experiment=experiment.name,
                learning_rate=float(learning_rate),
                seed=int(seed),
                step=int(step),
                expected_reward=expected_reward,
                average_reward_last_window=_average_reward_at_step(
                    trace.rewards,
                    step,
                    reward_window,
                ),
                mean_optimal_action_probability=optimal_prob,
                following_weights=weights_at_step,
                ceiling_reward=ceiling_reward,
                threshold_reward=threshold_reward,
            )
        )

    convergence_step = _first_convergence_step(curve, threshold_reward)
    final_point = curve[-1]
    metric_values = [point.expected_reward for point in curve] + trace.rewards
    selection_histogram = _selection_histogram(
        trace.selected_observer_indices,
        experiment.num_observers,
    )
    summary = MultiObserverRunSummary(
        experiment=experiment.name,
        learning_rate=float(learning_rate),
        seed=int(seed),
        num_steps=int(num_steps),
        num_observers=experiment.num_observers,
        final_expected_reward=final_point.expected_reward,
        final_average_reward_last_window=final_point.average_reward_last_window,
        final_mean_optimal_action_probability=(
            final_point.mean_optimal_action_probability
        ),
        final_following_weights=final_point.following_weights,
        selection_histogram=selection_histogram,
        ceiling_reward=ceiling_reward,
        threshold_reward=threshold_reward,
        convergence_step=convergence_step,
        converged=convergence_step is not None,
        nan_detected=any(not math.isfinite(value) for value in metric_values),
    )
    return curve, summary


def run_multi_observer_sweep(
    experiment: MultiObserverExperiment,
    learning_rates: Sequence[float],
    seeds: Sequence[int],
    num_steps: int,
    eval_interval: int,
    reward_window: int,
    threshold_fraction: float,
) -> tuple[list[MultiObserverCurvePoint], list[MultiObserverRunSummary]]:
    """Run a compact learning-rate sweep for multi-observer learning."""
    curves: list[MultiObserverCurvePoint] = []
    summaries: list[MultiObserverRunSummary] = []
    for learning_rate in learning_rates:
        for seed in seeds:
            curve, summary = run_multi_observer_once(
                experiment=experiment,
                learning_rate=learning_rate,
                seed=seed,
                num_steps=num_steps,
                eval_interval=eval_interval,
                reward_window=reward_window,
                threshold_fraction=threshold_fraction,
            )
            curves.extend(curve)
            summaries.append(summary)
    return curves, summaries


def summarize_multi_observer_runs(
    summaries: Sequence[MultiObserverRunSummary],
) -> list[MultiObserverAggregateSummary]:
    """Aggregate multi-observer run summaries over random seeds."""
    grouped: dict[tuple[str, float], list[MultiObserverRunSummary]] = defaultdict(list)
    for summary in summaries:
        grouped[(summary.experiment, summary.learning_rate)].append(summary)

    aggregate: list[MultiObserverAggregateSummary] = []
    for (experiment, learning_rate), group in sorted(grouped.items()):
        final_expected = [summary.final_expected_reward for summary in group]
        final_average = [summary.final_average_reward_last_window for summary in group]
        final_optimal_probs = [
            summary.final_mean_optimal_action_probability for summary in group
        ]
        convergence_steps = [
            summary.convergence_step
            for summary in group
            if summary.convergence_step is not None
        ]
        mean_following = _mean_following_weights(
            [summary.final_following_weights for summary in group]
        )
        mean_selection = _mean_following_weights(
            [summary.selection_histogram for summary in group]
        )
        aggregate.append(
            MultiObserverAggregateSummary(
                experiment=experiment,
                learning_rate=learning_rate,
                num_runs=len(group),
                success_rate=sum(summary.converged for summary in group) / len(group),
                mean_final_expected_reward=fmean(final_expected),
                std_final_expected_reward=_std(final_expected),
                mean_final_average_reward_last_window=fmean(final_average),
                std_final_average_reward_last_window=_std(final_average),
                mean_final_optimal_action_probability=fmean(final_optimal_probs),
                std_final_optimal_action_probability=_std(final_optimal_probs),
                mean_final_following_weights=mean_following,
                mean_selection_histogram=mean_selection,
                ceiling_reward=group[0].ceiling_reward,
                threshold_reward=group[0].threshold_reward,
                mean_convergence_step=(
                    fmean(convergence_steps) if convergence_steps else None
                ),
                std_convergence_step=(
                    _std(convergence_steps) if convergence_steps else None
                ),
                num_nan_runs=sum(summary.nan_detected for summary in group),
            )
        )
    return aggregate


def write_multi_observer_curve_csv(
    curves: Sequence[MultiObserverCurvePoint],
    path: str | Path,
) -> None:
    _write_csv((point.as_csv_row() for point in curves), path, MultiObserverCurvePoint)


def write_multi_observer_run_csv(
    summaries: Sequence[MultiObserverRunSummary],
    path: str | Path,
) -> None:
    _write_csv(
        (summary.as_csv_row() for summary in summaries),
        path,
        MultiObserverRunSummary,
    )


def write_multi_observer_summary_csv(
    summaries: Sequence[MultiObserverAggregateSummary],
    path: str | Path,
) -> None:
    _write_csv(
        (summary.as_csv_row() for summary in summaries),
        path,
        MultiObserverAggregateSummary,
    )


def _first_convergence_step(
    curve: Sequence[MultiObserverCurvePoint],
    threshold_reward: float,
) -> int | None:
    for point in curve:
        if point.expected_reward >= threshold_reward:
            return point.step
    return None


def _average_reward_at_step(
    rewards: Sequence[float],
    step: int,
    reward_window: int,
) -> float:
    if step <= 0:
        return 0.0
    return fmean(rewards[max(0, step - reward_window) : step])


def _selection_histogram(
    selected_indices: Sequence[int],
    num_observers: int,
) -> tuple[float, ...]:
    if not selected_indices:
        return tuple(0.0 for _ in range(num_observers))
    counts = [0 for _ in range(num_observers)]
    for index in selected_indices:
        if not 0 <= index < num_observers:
            raise ValueError(
                f"selected observer index {index} out of range [0, {num_observers})"
            )
        counts[index] += 1
    total = float(len(selected_indices))
    return tuple(count / total for count in counts)


def _mean_following_weights(
    weights: Sequence[Sequence[float]],
) -> tuple[float, ...]:
    if not weights:
        return ()
    num_observers = len(weights[0])
    if any(len(row) != num_observers for row in weights):
        raise ValueError("all weight rows must have equal length")
    means = [
        fmean(row[observer_index] for row in weights)
        for observer_index in range(num_observers)
    ]
    return tuple(means)


def _format_float_sequence(values: Sequence[float]) -> str:
    return "|".join(f"{value:g}" for value in values)


def _std(values: Sequence[float | int]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def _write_csv(
    rows: Iterable[dict[str, str | int | float | bool]],
    path: str | Path,
    row_type: type,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row_type.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(rows)
