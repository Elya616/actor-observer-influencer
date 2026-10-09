"""Focused verification experiments for the first Actor-Observer settings."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, stdev
from typing import Any

import torch

from actor_observer_rl.agents import Actor, Observer
from actor_observer_rl.envs import IIDStateBanditEnv, SingleStateBanditEnv
from actor_observer_rl.policies import SoftmaxPolicy
from actor_observer_rl.training.updates import reinforce_loss
from actor_observer_rl.utils.seeding import set_global_seed


@dataclass(frozen=True)
class FoundationExperiment:
    """Small experiment used to verify one theoretical learning setting."""

    name: str
    display_name: str
    reward_table: tuple[tuple[float, ...], ...]
    state_distribution: tuple[float, ...]

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

    @property
    def num_states(self) -> int:
        return len(self.reward_table)

    @property
    def num_actions(self) -> int:
        return len(self.reward_table[0])

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
class FoundationCurvePoint:
    """One point in a learning curve."""

    experiment: str
    learning_rate: float
    seed: int
    step: int
    expected_reward: float
    average_reward_last_window: float
    ceiling_reward: float
    threshold_reward: float

    def as_csv_row(self) -> dict[str, str | int | float]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class FoundationRunSummary:
    """Final metrics for one seed and learning rate."""

    experiment: str
    learning_rate: float
    seed: int
    num_steps: int
    final_expected_reward: float
    final_average_reward_last_window: float
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
class FoundationAggregateSummary:
    """Aggregated metrics over random seeds."""

    experiment: str
    learning_rate: float
    num_runs: int
    success_rate: float
    mean_final_expected_reward: float
    std_final_expected_reward: float
    mean_final_average_reward_last_window: float
    std_final_average_reward_last_window: float
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


def default_foundation_experiments() -> tuple[FoundationExperiment, ...]:
    """Return the two simple verification settings used in the main report."""
    return (
        FoundationExperiment(
            name="single_state_action_learning",
            display_name="Single-State Action Learning",
            reward_table=((0.0, 1.0, 0.2),),
            state_distribution=(1.0,),
        ),
        FoundationExperiment(
            name="state_dependent_action_learning",
            display_name="State-Dependent Action Learning",
            reward_table=(
                (1.0, 0.0, 0.2),
                (0.1, 1.0, 0.0),
                (0.0, 0.2, 1.0),
            ),
            state_distribution=(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
        ),
    )


def run_foundation_once(
    experiment: FoundationExperiment,
    learning_rate: float,
    seed: int,
    num_steps: int,
    eval_interval: int,
    reward_window: int,
    threshold_fraction: float,
) -> tuple[list[FoundationCurvePoint], FoundationRunSummary]:
    """Run one focused verification experiment."""
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
    env = _make_environment(experiment)
    policy = SoftmaxPolicy(input_dim=env.state_dim, num_actions=experiment.num_actions)
    actor = Actor(policy)
    observer = Observer()
    optimizer = torch.optim.SGD(actor.parameters(), lr=learning_rate)

    ceiling_reward = experiment.optimal_expected_reward
    threshold_reward = threshold_fraction * ceiling_reward
    rewards: list[float] = []
    curve: list[FoundationCurvePoint] = []
    convergence_step: int | None = None

    observation, _ = env.reset(seed=seed)
    _record_curve_point(
        curve=curve,
        experiment=experiment,
        actor=actor,
        learning_rate=learning_rate,
        seed=seed,
        step=0,
        rewards=rewards,
        reward_window=reward_window,
        ceiling_reward=ceiling_reward,
        threshold_reward=threshold_reward,
    )

    for step in range(1, num_steps + 1):
        actor_input = torch.as_tensor(observation, dtype=torch.float32)
        action_result = actor.act(actor_input, sample=True)
        action = int(action_result.action.detach().cpu().reshape(-1)[0].item())
        step_result = env.step(action)
        reported_reward = observer.report_reward(step_result.reward)
        loss = reinforce_loss(reported_reward, action_result.log_prob)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        rewards.append(float(reported_reward))
        observation = env.reset(seed=None)[0] if step_result.done else step_result.observation

        should_evaluate = step % eval_interval == 0 or step == num_steps
        if should_evaluate:
            point = _record_curve_point(
                curve=curve,
                experiment=experiment,
                actor=actor,
                learning_rate=learning_rate,
                seed=seed,
                step=step,
                rewards=rewards,
                reward_window=reward_window,
                ceiling_reward=ceiling_reward,
                threshold_reward=threshold_reward,
            )
            if convergence_step is None and point.expected_reward >= threshold_reward:
                convergence_step = step

    final_point = curve[-1]
    metric_values = [point.expected_reward for point in curve] + rewards
    summary = FoundationRunSummary(
        experiment=experiment.name,
        learning_rate=float(learning_rate),
        seed=int(seed),
        num_steps=int(num_steps),
        final_expected_reward=final_point.expected_reward,
        final_average_reward_last_window=final_point.average_reward_last_window,
        ceiling_reward=ceiling_reward,
        threshold_reward=threshold_reward,
        convergence_step=convergence_step,
        converged=convergence_step is not None,
        nan_detected=any(not math.isfinite(value) for value in metric_values),
    )
    return curve, summary


def run_foundation_sweep(
    experiments: Sequence[FoundationExperiment],
    learning_rates: Sequence[float],
    seeds: Sequence[int],
    num_steps: int,
    eval_interval: int,
    reward_window: int,
    threshold_fraction: float,
) -> tuple[list[FoundationCurvePoint], list[FoundationRunSummary]]:
    """Run all focused verification experiments."""
    curves: list[FoundationCurvePoint] = []
    summaries: list[FoundationRunSummary] = []
    for experiment in experiments:
        for learning_rate in learning_rates:
            for seed in seeds:
                curve, summary = run_foundation_once(
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


def summarize_foundation_runs(
    summaries: Sequence[FoundationRunSummary],
) -> list[FoundationAggregateSummary]:
    """Aggregate focused experiment summaries over random seeds."""
    grouped: dict[tuple[str, float], list[FoundationRunSummary]] = defaultdict(list)
    for summary in summaries:
        grouped[(summary.experiment, summary.learning_rate)].append(summary)

    aggregate: list[FoundationAggregateSummary] = []
    for (experiment, learning_rate), group in sorted(grouped.items()):
        final_expected = [summary.final_expected_reward for summary in group]
        final_average = [summary.final_average_reward_last_window for summary in group]
        convergence_steps = [
            summary.convergence_step
            for summary in group
            if summary.convergence_step is not None
        ]
        aggregate.append(
            FoundationAggregateSummary(
                experiment=experiment,
                learning_rate=learning_rate,
                num_runs=len(group),
                success_rate=sum(summary.converged for summary in group) / len(group),
                mean_final_expected_reward=fmean(final_expected),
                std_final_expected_reward=_std(final_expected),
                mean_final_average_reward_last_window=fmean(final_average),
                std_final_average_reward_last_window=_std(final_average),
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


def expected_reward_under_policy(
    actor: Actor,
    experiment: FoundationExperiment,
) -> float:
    """Compute exact expected reward under the current policy."""
    with torch.no_grad():
        inputs = _state_inputs(experiment)
        probs = actor.policy.action_distribution(inputs).probs.detach().cpu()

    expected_reward = 0.0
    for state_index, state_probability in enumerate(experiment.state_distribution):
        state_expected_reward = 0.0
        for action_index, reward in enumerate(experiment.reward_table[state_index]):
            state_expected_reward += float(probs[state_index, action_index].item()) * reward
        expected_reward += state_probability * state_expected_reward
    return expected_reward


def write_curve_csv(curves: Sequence[FoundationCurvePoint], path: str | Path) -> None:
    _write_csv((point.as_csv_row() for point in curves), path, FoundationCurvePoint)


def write_run_summary_csv(
    summaries: Sequence[FoundationRunSummary],
    path: str | Path,
) -> None:
    _write_csv((summary.as_csv_row() for summary in summaries), path, FoundationRunSummary)


def write_aggregate_summary_csv(
    summaries: Sequence[FoundationAggregateSummary],
    path: str | Path,
) -> None:
    _write_csv(
        (summary.as_csv_row() for summary in summaries),
        path,
        FoundationAggregateSummary,
    )


def _make_environment(experiment: FoundationExperiment) -> Any:
    if experiment.num_states == 1:
        return SingleStateBanditEnv(reward_table=experiment.reward_table[0])
    return IIDStateBanditEnv(
        reward_table=experiment.reward_table,
        state_distribution=experiment.state_distribution,
        observation_mode="one_hot",
    )


def _state_inputs(experiment: FoundationExperiment) -> torch.Tensor:
    if experiment.num_states == 1:
        return torch.ones((1, 1), dtype=torch.float32)
    return torch.eye(experiment.num_states, dtype=torch.float32)


def _record_curve_point(
    curve: list[FoundationCurvePoint],
    experiment: FoundationExperiment,
    actor: Actor,
    learning_rate: float,
    seed: int,
    step: int,
    rewards: Sequence[float],
    reward_window: int,
    ceiling_reward: float,
    threshold_reward: float,
) -> FoundationCurvePoint:
    average_reward = fmean(rewards[-reward_window:]) if rewards else 0.0
    point = FoundationCurvePoint(
        experiment=experiment.name,
        learning_rate=float(learning_rate),
        seed=int(seed),
        step=int(step),
        expected_reward=expected_reward_under_policy(actor, experiment),
        average_reward_last_window=average_reward,
        ceiling_reward=ceiling_reward,
        threshold_reward=threshold_reward,
    )
    curve.append(point)
    return point


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
