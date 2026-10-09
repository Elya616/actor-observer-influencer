"""Focused experiments for learnable Observer messages under partial observation."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, stdev

import torch

from actor_observer_rl.agents import Actor, Observer, ObserverMessageModule
from actor_observer_rl.envs import PartialObservableStateEnv
from actor_observer_rl.policies import SoftmaxPolicy
from actor_observer_rl.training import PartialObservableTrainer
from actor_observer_rl.utils.seeding import set_global_seed


@dataclass(frozen=True)
class PartialObservationExperiment:
    """Small setting used to verify learnable Observer messages."""

    name: str
    display_name: str
    reward_table: tuple[tuple[float, ...], ...]
    state_distribution: tuple[float, ...]
    observation_dim: int
    message_dim: int

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
class PartialObservationCurvePoint:
    """One point in a partial-observation learning curve."""

    experiment: str
    learning_rate: float
    seed: int
    step: int
    expected_reward: float
    average_reward_last_window: float
    mean_optimal_action_probability: float
    ceiling_reward: float
    threshold_reward: float

    def as_csv_row(self) -> dict[str, str | int | float]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class PartialObservationRunSummary:
    """Final metrics for one partial-observation run."""

    experiment: str
    learning_rate: float
    seed: int
    num_steps: int
    final_expected_reward: float
    final_average_reward_last_window: float
    final_mean_optimal_action_probability: float
    ceiling_reward: float
    threshold_reward: float
    convergence_step: int | None
    converged: bool
    nan_detected: bool

    def as_csv_row(self) -> dict[str, str | int | float | bool]:
        row = self.__dict__.copy()
        row["convergence_step"] = (
            "" if self.convergence_step is None else self.convergence_step
        )
        return row


@dataclass(frozen=True)
class PartialObservationAggregateSummary:
    """Aggregated partial-observation metrics over random seeds."""

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
    ceiling_reward: float
    threshold_reward: float
    mean_convergence_step: float | None
    std_convergence_step: float | None
    num_nan_runs: int

    def as_csv_row(self) -> dict[str, str | int | float]:
        row = self.__dict__.copy()
        row["mean_convergence_step"] = (
            "" if self.mean_convergence_step is None else self.mean_convergence_step
        )
        row["std_convergence_step"] = (
            "" if self.std_convergence_step is None else self.std_convergence_step
        )
        return row


def default_partial_observation_experiment() -> PartialObservationExperiment:
    """Return the first focused setting for partial observation."""
    return PartialObservationExperiment(
        name="partial_observation_message_learning",
        display_name="Partial-Observation Message Learning",
        reward_table=(
            (1.0, 0.0),
            (0.0, 1.0),
            (1.0, 0.0),
            (0.0, 1.0),
        ),
        state_distribution=(0.25, 0.25, 0.25, 0.25),
        observation_dim=1,
        message_dim=2,
    )


def run_partial_observation_once(
    experiment: PartialObservationExperiment,
    learning_rate: float,
    seed: int,
    num_steps: int,
    eval_interval: int,
    reward_window: int,
    threshold_fraction: float,
) -> tuple[list[PartialObservationCurvePoint], PartialObservationRunSummary]:
    """Run one learnable-message verification experiment."""
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
    observer = Observer(
        message_module=ObserverMessageModule(
            input_dim=env.state_dim,
            message_dim=experiment.message_dim,
        )
    )
    actor = Actor(
        SoftmaxPolicy(
            input_dim=experiment.observation_dim + experiment.message_dim,
            num_actions=experiment.num_actions,
        )
    )
    trainer = PartialObservableTrainer(
        env=env,
        actor=actor,
        observer=observer,
        actor_optimizer=torch.optim.SGD(actor.parameters(), lr=learning_rate),
        observer_optimizer=torch.optim.SGD(observer.parameters(), lr=learning_rate),
        optimal_actions=experiment.optimal_actions,
    )

    ceiling_reward = experiment.optimal_expected_reward
    threshold_reward = threshold_fraction * ceiling_reward
    initial_expected, initial_optimal_prob = trainer.evaluate_policy()
    if initial_optimal_prob is None:
        raise RuntimeError("partial observation experiment requires optimal actions")

    trace = trainer.run(
        num_steps=num_steps,
        seed=seed,
        eval_interval=eval_interval,
    )

    curve = [
        PartialObservationCurvePoint(
            experiment=experiment.name,
            learning_rate=float(learning_rate),
            seed=int(seed),
            step=0,
            expected_reward=initial_expected,
            average_reward_last_window=0.0,
            mean_optimal_action_probability=initial_optimal_prob,
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
        curve.append(
            PartialObservationCurvePoint(
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
                ceiling_reward=ceiling_reward,
                threshold_reward=threshold_reward,
            )
        )

    convergence_step = _first_convergence_step(curve, threshold_reward)
    final_point = curve[-1]
    metric_values = [point.expected_reward for point in curve] + trace.rewards
    summary = PartialObservationRunSummary(
        experiment=experiment.name,
        learning_rate=float(learning_rate),
        seed=int(seed),
        num_steps=int(num_steps),
        final_expected_reward=final_point.expected_reward,
        final_average_reward_last_window=final_point.average_reward_last_window,
        final_mean_optimal_action_probability=(
            final_point.mean_optimal_action_probability
        ),
        ceiling_reward=ceiling_reward,
        threshold_reward=threshold_reward,
        convergence_step=convergence_step,
        converged=convergence_step is not None,
        nan_detected=any(not math.isfinite(value) for value in metric_values),
    )
    return curve, summary


def run_partial_observation_sweep(
    experiment: PartialObservationExperiment,
    learning_rates: Sequence[float],
    seeds: Sequence[int],
    num_steps: int,
    eval_interval: int,
    reward_window: int,
    threshold_fraction: float,
) -> tuple[list[PartialObservationCurvePoint], list[PartialObservationRunSummary]]:
    """Run a compact learning-rate sweep for partial observation."""
    curves: list[PartialObservationCurvePoint] = []
    summaries: list[PartialObservationRunSummary] = []
    for learning_rate in learning_rates:
        for seed in seeds:
            curve, summary = run_partial_observation_once(
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


def summarize_partial_observation_runs(
    summaries: Sequence[PartialObservationRunSummary],
) -> list[PartialObservationAggregateSummary]:
    """Aggregate partial-observation run summaries over random seeds."""
    grouped: dict[tuple[str, float], list[PartialObservationRunSummary]] = defaultdict(list)
    for summary in summaries:
        grouped[(summary.experiment, summary.learning_rate)].append(summary)

    aggregate: list[PartialObservationAggregateSummary] = []
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
        aggregate.append(
            PartialObservationAggregateSummary(
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


def write_partial_observation_curve_csv(
    curves: Sequence[PartialObservationCurvePoint],
    path: str | Path,
) -> None:
    _write_csv((point.as_csv_row() for point in curves), path, PartialObservationCurvePoint)


def write_partial_observation_run_csv(
    summaries: Sequence[PartialObservationRunSummary],
    path: str | Path,
) -> None:
    _write_csv(
        (summary.as_csv_row() for summary in summaries),
        path,
        PartialObservationRunSummary,
    )


def write_partial_observation_summary_csv(
    summaries: Sequence[PartialObservationAggregateSummary],
    path: str | Path,
) -> None:
    _write_csv(
        (summary.as_csv_row() for summary in summaries),
        path,
        PartialObservationAggregateSummary,
    )


def _first_convergence_step(
    curve: Sequence[PartialObservationCurvePoint],
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
