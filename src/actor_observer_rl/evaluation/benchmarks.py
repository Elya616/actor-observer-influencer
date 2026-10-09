"""Benchmark orchestration for report-ready experiment sweeps."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean, stdev

import torch

from actor_observer_rl.agents import Actor, Observer
from actor_observer_rl.envs import IIDStateBanditEnv, SingleStateBanditEnv
from actor_observer_rl.policies import SoftmaxPolicy
from actor_observer_rl.training import Trainer
from actor_observer_rl.utils.seeding import set_global_seed


@dataclass(frozen=True)
class Case1Scenario:
    """One reward-table instance of the PDF's single-state objective."""

    name: str
    reward_table: Sequence[float]
    optimal_action: int

    def __post_init__(self) -> None:
        rewards = tuple(float(reward) for reward in self.reward_table)
        if not self.name:
            raise ValueError("scenario name must be non-empty")
        if len(rewards) <= 1:
            raise ValueError("Case 1 scenarios need at least two actions")
        if not 0 <= self.optimal_action < len(rewards):
            raise ValueError("optimal_action must be a valid action index")
        best_reward = max(rewards)
        if rewards[self.optimal_action] != best_reward:
            raise ValueError("optimal_action must identify a maximum-reward action")
        object.__setattr__(self, "reward_table", rewards)

    @property
    def num_actions(self) -> int:
        return len(self.reward_table)


def default_case1_scenarios() -> tuple[Case1Scenario, ...]:
    """Return interpretable Case 1 scenarios of increasing difficulty."""
    return (
        Case1Scenario(
            name="binary_easy",
            reward_table=(0.0, 1.0),
            optimal_action=1,
        ),
        Case1Scenario(
            name="three_action_default",
            reward_table=(0.0, 1.0, 0.2),
            optimal_action=1,
        ),
        Case1Scenario(
            name="five_action_sparse",
            reward_table=(0.0, 0.1, 1.0, 0.2, 0.0),
            optimal_action=2,
        ),
        Case1Scenario(
            name="ten_action_sparse",
            reward_table=(0.0, 0.1, 0.0, 0.2, 1.0, 0.0, 0.3, 0.0, 0.1, 0.0),
            optimal_action=4,
        ),
        Case1Scenario(
            name="three_action_small_gap",
            reward_table=(0.8, 1.0, 0.9),
            optimal_action=1,
        ),
    )


@dataclass(frozen=True)
class Case1BenchmarkConfig:
    """Configuration for a Case 1 scenario x learning-rate x seed sweep."""

    scenarios: Sequence[Case1Scenario] = field(default_factory=default_case1_scenarios)
    learning_rates: Sequence[float] = (0.001, 0.01, 0.05, 0.1, 0.2)
    seeds: Sequence[int] = (0, 1, 2, 3, 4)
    num_steps: int = 1000
    convergence_threshold: float = 0.9
    reward_window: int = 100

    def __post_init__(self) -> None:
        if not self.scenarios:
            raise ValueError("at least one scenario is required")
        if not self.learning_rates:
            raise ValueError("at least one learning rate is required")
        if not self.seeds:
            raise ValueError("at least one seed is required")
        if self.num_steps <= 0:
            raise ValueError("num_steps must be positive")
        if not 0.0 < self.convergence_threshold < 1.0:
            raise ValueError("convergence_threshold must be between 0 and 1")
        if self.reward_window <= 0:
            raise ValueError("reward_window must be positive")
        if any(learning_rate <= 0.0 for learning_rate in self.learning_rates):
            raise ValueError("learning rates must be positive")

        object.__setattr__(self, "scenarios", tuple(self.scenarios))
        object.__setattr__(
            self,
            "learning_rates",
            tuple(float(learning_rate) for learning_rate in self.learning_rates),
        )
        object.__setattr__(self, "seeds", tuple(int(seed) for seed in self.seeds))


@dataclass(frozen=True)
class Case1RunResult:
    """Metrics from one Case 1 training run."""

    scenario: str
    num_actions: int
    reward_table: tuple[float, ...]
    optimal_action: int
    learning_rate: float
    seed: int
    num_steps: int
    final_reward: float
    final_optimal_action_prob: float
    average_reward_last_window: float
    convergence_step: int | None
    converged: bool
    nan_detected: bool

    def as_csv_row(self) -> dict[str, str | int | float | bool]:
        return {
            "scenario": self.scenario,
            "num_actions": self.num_actions,
            "reward_table": "|".join(str(reward) for reward in self.reward_table),
            "optimal_action": self.optimal_action,
            "learning_rate": self.learning_rate,
            "seed": self.seed,
            "num_steps": self.num_steps,
            "final_reward": self.final_reward,
            "final_optimal_action_prob": self.final_optimal_action_prob,
            "average_reward_last_window": self.average_reward_last_window,
            "convergence_step": "" if self.convergence_step is None else self.convergence_step,
            "converged": self.converged,
            "nan_detected": self.nan_detected,
        }


@dataclass(frozen=True)
class Case1SummaryResult:
    """Aggregated metrics for one scenario and learning rate."""

    scenario: str
    learning_rate: float
    num_runs: int
    success_rate: float
    mean_final_optimal_action_prob: float
    std_final_optimal_action_prob: float
    mean_average_reward_last_window: float
    std_average_reward_last_window: float
    mean_convergence_step: float | None
    std_convergence_step: float | None
    num_nan_runs: int

    def as_csv_row(self) -> dict[str, str | int | float]:
        return {
            "scenario": self.scenario,
            "learning_rate": self.learning_rate,
            "num_runs": self.num_runs,
            "success_rate": self.success_rate,
            "mean_final_optimal_action_prob": self.mean_final_optimal_action_prob,
            "std_final_optimal_action_prob": self.std_final_optimal_action_prob,
            "mean_average_reward_last_window": self.mean_average_reward_last_window,
            "std_average_reward_last_window": self.std_average_reward_last_window,
            "mean_convergence_step": (
                "" if self.mean_convergence_step is None else self.mean_convergence_step
            ),
            "std_convergence_step": (
                "" if self.std_convergence_step is None else self.std_convergence_step
            ),
            "num_nan_runs": self.num_nan_runs,
        }


def convergence_step(probabilities: Sequence[float], threshold: float) -> int | None:
    """Return first 1-based step where optimal-action probability reaches threshold."""
    for step, probability in enumerate(probabilities, start=1):
        if probability >= threshold:
            return step
    return None


def run_case1_once(
    scenario: Case1Scenario,
    learning_rate: float,
    seed: int,
    num_steps: int,
    convergence_threshold: float = 0.9,
    reward_window: int = 100,
) -> Case1RunResult:
    """Train one Case 1 policy and return report-ready metrics."""
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    if reward_window <= 0:
        raise ValueError("reward_window must be positive")

    set_global_seed(seed)
    env = SingleStateBanditEnv(reward_table=scenario.reward_table)
    policy = SoftmaxPolicy(input_dim=env.state_dim, num_actions=scenario.num_actions)
    actor = Actor(policy)
    observer = Observer()
    optimizer = torch.optim.SGD(actor.parameters(), lr=learning_rate)
    trainer = Trainer(
        env=env,
        actor=actor,
        actor_optimizer=optimizer,
        observer=observer,
        optimal_action=scenario.optimal_action,
    )
    trace = trainer.run(num_steps=num_steps, seed=seed)

    step = convergence_step(trace.optimal_action_probs, convergence_threshold)
    metric_values = trace.rewards + trace.losses + trace.optimal_action_probs
    nan_detected = any(not math.isfinite(value) for value in metric_values)

    return Case1RunResult(
        scenario=scenario.name,
        num_actions=scenario.num_actions,
        reward_table=tuple(scenario.reward_table),
        optimal_action=scenario.optimal_action,
        learning_rate=float(learning_rate),
        seed=int(seed),
        num_steps=int(num_steps),
        final_reward=float(trace.final_reward),
        final_optimal_action_prob=float(trace.final_optimal_action_prob),
        average_reward_last_window=_average_last(trace.rewards, reward_window),
        convergence_step=step,
        converged=step is not None,
        nan_detected=nan_detected,
    )


def run_case1_benchmark(config: Case1BenchmarkConfig) -> list[Case1RunResult]:
    """Run a scenario x learning-rate x seed Case 1 benchmark sweep."""
    results: list[Case1RunResult] = []
    for scenario in config.scenarios:
        for learning_rate in config.learning_rates:
            for seed in config.seeds:
                results.append(
                    run_case1_once(
                        scenario=scenario,
                        learning_rate=learning_rate,
                        seed=seed,
                        num_steps=config.num_steps,
                        convergence_threshold=config.convergence_threshold,
                        reward_window=config.reward_window,
                    )
                )
    return results


def summarize_case1_results(results: Sequence[Case1RunResult]) -> list[Case1SummaryResult]:
    """Aggregate repeated-seed results by scenario and learning rate."""
    grouped: dict[tuple[str, float], list[Case1RunResult]] = defaultdict(list)
    for result in results:
        grouped[(result.scenario, result.learning_rate)].append(result)

    summaries: list[Case1SummaryResult] = []
    for (scenario, learning_rate), group in sorted(grouped.items()):
        final_probs = [result.final_optimal_action_prob for result in group]
        average_rewards = [result.average_reward_last_window for result in group]
        convergence_steps = [
            result.convergence_step
            for result in group
            if result.convergence_step is not None
        ]
        summaries.append(
            Case1SummaryResult(
                scenario=scenario,
                learning_rate=learning_rate,
                num_runs=len(group),
                success_rate=sum(result.converged for result in group) / len(group),
                mean_final_optimal_action_prob=fmean(final_probs),
                std_final_optimal_action_prob=_std(final_probs),
                mean_average_reward_last_window=fmean(average_rewards),
                std_average_reward_last_window=_std(average_rewards),
                mean_convergence_step=(
                    fmean(convergence_steps) if convergence_steps else None
                ),
                std_convergence_step=(
                    _std(convergence_steps) if convergence_steps else None
                ),
                num_nan_runs=sum(result.nan_detected for result in group),
            )
        )
    return summaries


def write_case1_results_csv(results: Sequence[Case1RunResult], path: str | Path) -> None:
    """Write per-run benchmark results to CSV."""
    fieldnames = list(Case1RunResult.__dataclass_fields__)
    _write_csv((result.as_csv_row() for result in results), path, fieldnames)


def write_case1_summary_csv(results: Sequence[Case1SummaryResult], path: str | Path) -> None:
    """Write aggregate benchmark summaries to CSV."""
    fieldnames = list(Case1SummaryResult.__dataclass_fields__)
    _write_csv((result.as_csv_row() for result in results), path, fieldnames)


@dataclass(frozen=True)
class Case2Scenario:
    """One reward-table instance of the PDF's i.i.d. states objective."""

    name: str
    reward_table: Sequence[Sequence[float]]
    optimal_actions: Sequence[int]

    def __post_init__(self) -> None:
        rewards = tuple(tuple(float(reward) for reward in row) for row in self.reward_table)
        optimal_actions = tuple(int(action) for action in self.optimal_actions)

        if not self.name:
            raise ValueError("scenario name must be non-empty")
        if len(rewards) <= 1:
            raise ValueError("Case 2 scenarios need at least two states")

        num_actions = len(rewards[0])
        if num_actions <= 1:
            raise ValueError("Case 2 scenarios need at least two actions")
        if any(len(row) != num_actions for row in rewards):
            raise ValueError("reward_table rows must all have the same action count")
        if len(optimal_actions) != len(rewards):
            raise ValueError("optimal_actions length must match reward_table rows")

        for state_index, optimal_action in enumerate(optimal_actions):
            if not 0 <= optimal_action < num_actions:
                raise ValueError("optimal_actions must be valid action indices")
            best_reward = max(rewards[state_index])
            if rewards[state_index][optimal_action] != best_reward:
                raise ValueError("optimal_actions must identify maximum-reward actions")

        object.__setattr__(self, "reward_table", rewards)
        object.__setattr__(self, "optimal_actions", optimal_actions)

    @property
    def num_states(self) -> int:
        return len(self.reward_table)

    @property
    def num_actions(self) -> int:
        return len(self.reward_table[0])


@dataclass(frozen=True)
class Case2StateDistribution:
    """Named i.i.d. state distribution `D` for a Case 2 benchmark."""

    name: str
    probabilities: Sequence[float]

    def __post_init__(self) -> None:
        probabilities = tuple(float(probability) for probability in self.probabilities)

        if not self.name:
            raise ValueError("distribution name must be non-empty")
        if len(probabilities) <= 1:
            raise ValueError("Case 2 distributions need at least two states")
        if any(probability < 0.0 for probability in probabilities):
            raise ValueError("state probabilities cannot be negative")
        if not math.isclose(sum(probabilities), 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("state probabilities must sum to 1")

        object.__setattr__(self, "probabilities", probabilities)

    @property
    def num_states(self) -> int:
        return len(self.probabilities)


def default_case2_scenarios() -> tuple[Case2Scenario, ...]:
    """Return interpretable Case 2 reward-table scenarios."""
    return (
        Case2Scenario(
            name="three_state_diagonal",
            reward_table=(
                (1.0, 0.0, 0.2),
                (0.1, 1.0, 0.0),
                (0.0, 0.2, 1.0),
            ),
            optimal_actions=(0, 1, 2),
        ),
        Case2Scenario(
            name="three_state_small_gap",
            reward_table=(
                (1.0, 0.9, 0.8),
                (0.8, 1.0, 0.9),
                (0.9, 0.8, 1.0),
            ),
            optimal_actions=(0, 1, 2),
        ),
        Case2Scenario(
            name="three_state_low_scale",
            reward_table=(
                (0.10, 0.00, 0.02),
                (0.01, 0.10, 0.00),
                (0.00, 0.02, 0.10),
            ),
            optimal_actions=(0, 1, 2),
        ),
        Case2Scenario(
            name="three_state_high_scale",
            reward_table=(
                (5.0, 0.0, 1.0),
                (0.5, 5.0, 0.0),
                (0.0, 1.0, 5.0),
            ),
            optimal_actions=(0, 1, 2),
        ),
        Case2Scenario(
            name="five_state_diagonal",
            reward_table=(
                (1.0, 0.2, 0.0, 0.0, 0.0),
                (0.0, 1.0, 0.2, 0.0, 0.0),
                (0.0, 0.0, 1.0, 0.2, 0.0),
                (0.0, 0.0, 0.0, 1.0, 0.2),
                (0.2, 0.0, 0.0, 0.0, 1.0),
            ),
            optimal_actions=(0, 1, 2, 3, 4),
        ),
    )


def default_case2_state_distributions() -> tuple[Case2StateDistribution, ...]:
    """Return state distributions that expose frequent and rare-state behavior."""
    return (
        Case2StateDistribution(
            name="uniform_3",
            probabilities=(0.34, 0.33, 0.33),
        ),
        Case2StateDistribution(
            name="mild_skew_3",
            probabilities=(0.60, 0.30, 0.10),
        ),
        Case2StateDistribution(
            name="rare_state_3",
            probabilities=(0.85, 0.10, 0.05),
        ),
        Case2StateDistribution(
            name="uniform_5",
            probabilities=(0.20, 0.20, 0.20, 0.20, 0.20),
        ),
        Case2StateDistribution(
            name="mild_skew_5",
            probabilities=(0.45, 0.25, 0.15, 0.10, 0.05),
        ),
        Case2StateDistribution(
            name="rare_state_5",
            probabilities=(0.70, 0.15, 0.07, 0.05, 0.03),
        ),
    )


@dataclass(frozen=True)
class Case2BenchmarkConfig:
    """Configuration for a Case 2 scenario x distribution x LR x seed sweep."""

    scenarios: Sequence[Case2Scenario] = field(default_factory=default_case2_scenarios)
    state_distributions: Sequence[Case2StateDistribution] = field(
        default_factory=default_case2_state_distributions
    )
    learning_rates: Sequence[float] = (0.001, 0.01, 0.05, 0.1)
    seeds: Sequence[int] = (0, 1, 2, 3, 4)
    num_steps: int = 2000
    convergence_threshold: float = 0.9
    reward_window: int = 100

    def __post_init__(self) -> None:
        if not self.scenarios:
            raise ValueError("at least one scenario is required")
        if not self.state_distributions:
            raise ValueError("at least one state distribution is required")
        if not self.learning_rates:
            raise ValueError("at least one learning rate is required")
        if not self.seeds:
            raise ValueError("at least one seed is required")
        if self.num_steps <= 0:
            raise ValueError("num_steps must be positive")
        if not 0.0 < self.convergence_threshold < 1.0:
            raise ValueError("convergence_threshold must be between 0 and 1")
        if self.reward_window <= 0:
            raise ValueError("reward_window must be positive")
        if any(learning_rate <= 0.0 for learning_rate in self.learning_rates):
            raise ValueError("learning rates must be positive")

        scenarios = tuple(self.scenarios)
        state_distributions = tuple(self.state_distributions)
        for scenario in scenarios:
            if not any(
                distribution.num_states == scenario.num_states
                for distribution in state_distributions
            ):
                raise ValueError(
                    "each scenario needs at least one matching state distribution"
                )

        object.__setattr__(self, "scenarios", scenarios)
        object.__setattr__(self, "state_distributions", state_distributions)
        object.__setattr__(
            self,
            "learning_rates",
            tuple(float(learning_rate) for learning_rate in self.learning_rates),
        )
        object.__setattr__(self, "seeds", tuple(int(seed) for seed in self.seeds))


@dataclass(frozen=True)
class Case2RunResult:
    """Run-level metrics from one Case 2 training run."""

    scenario: str
    distribution: str
    num_states: int
    num_actions: int
    state_distribution: tuple[float, ...]
    reward_table: tuple[tuple[float, ...], ...]
    optimal_actions: tuple[int, ...]
    learning_rate: float
    seed: int
    num_steps: int
    final_reward: float
    average_reward_last_window: float
    weighted_final_optimal_action_prob: float
    unweighted_final_optimal_action_prob: float
    min_final_optimal_action_prob: float
    expected_reward_policy: float
    overall_convergence_step: int | None
    converged: bool
    state_sample_counts: tuple[int, ...]
    per_state_final_optimal_action_probs: tuple[float, ...]
    per_state_convergence_steps: tuple[int | None, ...]
    nan_detected: bool

    def as_csv_row(self) -> dict[str, str | int | float | bool]:
        return {
            "scenario": self.scenario,
            "distribution": self.distribution,
            "num_states": self.num_states,
            "num_actions": self.num_actions,
            "state_distribution": _format_float_sequence(self.state_distribution),
            "reward_table": _format_reward_table(self.reward_table),
            "optimal_actions": _format_int_sequence(self.optimal_actions),
            "learning_rate": self.learning_rate,
            "seed": self.seed,
            "num_steps": self.num_steps,
            "final_reward": self.final_reward,
            "average_reward_last_window": self.average_reward_last_window,
            "weighted_final_optimal_action_prob": self.weighted_final_optimal_action_prob,
            "unweighted_final_optimal_action_prob": self.unweighted_final_optimal_action_prob,
            "min_final_optimal_action_prob": self.min_final_optimal_action_prob,
            "expected_reward_policy": self.expected_reward_policy,
            "overall_convergence_step": (
                ""
                if self.overall_convergence_step is None
                else self.overall_convergence_step
            ),
            "converged": self.converged,
            "state_sample_counts": _format_int_sequence(self.state_sample_counts),
            "per_state_final_optimal_action_probs": _format_float_sequence(
                self.per_state_final_optimal_action_probs
            ),
            "per_state_convergence_steps": _format_optional_int_sequence(
                self.per_state_convergence_steps
            ),
            "nan_detected": self.nan_detected,
        }


@dataclass(frozen=True)
class Case2PerStateRunResult:
    """Per-state metrics from one Case 2 training run."""

    scenario: str
    distribution: str
    learning_rate: float
    seed: int
    num_steps: int
    state_index: int
    state_probability: float
    optimal_action: int
    sample_count: int
    final_optimal_action_prob: float
    convergence_step: int | None

    def as_csv_row(self) -> dict[str, str | int | float]:
        return {
            "scenario": self.scenario,
            "distribution": self.distribution,
            "learning_rate": self.learning_rate,
            "seed": self.seed,
            "num_steps": self.num_steps,
            "state_index": self.state_index,
            "state_probability": self.state_probability,
            "optimal_action": self.optimal_action,
            "sample_count": self.sample_count,
            "final_optimal_action_prob": self.final_optimal_action_prob,
            "convergence_step": (
                "" if self.convergence_step is None else self.convergence_step
            ),
        }


@dataclass(frozen=True)
class Case2SummaryResult:
    """Aggregate metrics for one Case 2 scenario, distribution, and learning rate."""

    scenario: str
    distribution: str
    learning_rate: float
    num_runs: int
    success_rate: float
    mean_weighted_final_optimal_action_prob: float
    std_weighted_final_optimal_action_prob: float
    mean_unweighted_final_optimal_action_prob: float
    std_unweighted_final_optimal_action_prob: float
    mean_min_final_optimal_action_prob: float
    std_min_final_optimal_action_prob: float
    mean_expected_reward_policy: float
    std_expected_reward_policy: float
    mean_average_reward_last_window: float
    std_average_reward_last_window: float
    mean_overall_convergence_step: float | None
    std_overall_convergence_step: float | None
    num_nan_runs: int

    def as_csv_row(self) -> dict[str, str | int | float]:
        return {
            "scenario": self.scenario,
            "distribution": self.distribution,
            "learning_rate": self.learning_rate,
            "num_runs": self.num_runs,
            "success_rate": self.success_rate,
            "mean_weighted_final_optimal_action_prob": (
                self.mean_weighted_final_optimal_action_prob
            ),
            "std_weighted_final_optimal_action_prob": (
                self.std_weighted_final_optimal_action_prob
            ),
            "mean_unweighted_final_optimal_action_prob": (
                self.mean_unweighted_final_optimal_action_prob
            ),
            "std_unweighted_final_optimal_action_prob": (
                self.std_unweighted_final_optimal_action_prob
            ),
            "mean_min_final_optimal_action_prob": self.mean_min_final_optimal_action_prob,
            "std_min_final_optimal_action_prob": self.std_min_final_optimal_action_prob,
            "mean_expected_reward_policy": self.mean_expected_reward_policy,
            "std_expected_reward_policy": self.std_expected_reward_policy,
            "mean_average_reward_last_window": self.mean_average_reward_last_window,
            "std_average_reward_last_window": self.std_average_reward_last_window,
            "mean_overall_convergence_step": (
                ""
                if self.mean_overall_convergence_step is None
                else self.mean_overall_convergence_step
            ),
            "std_overall_convergence_step": (
                ""
                if self.std_overall_convergence_step is None
                else self.std_overall_convergence_step
            ),
            "num_nan_runs": self.num_nan_runs,
        }


def run_case2_once(
    scenario: Case2Scenario,
    state_distribution: Case2StateDistribution,
    learning_rate: float,
    seed: int,
    num_steps: int,
    convergence_threshold: float = 0.9,
    reward_window: int = 100,
) -> tuple[Case2RunResult, list[Case2PerStateRunResult]]:
    """Train one Case 2 policy and return run-level and per-state metrics."""
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    if reward_window <= 0:
        raise ValueError("reward_window must be positive")
    if len(state_distribution.probabilities) != scenario.num_states:
        raise ValueError("state distribution length must match scenario num_states")

    set_global_seed(seed)
    env = IIDStateBanditEnv(
        reward_table=scenario.reward_table,
        state_distribution=state_distribution.probabilities,
        observation_mode="one_hot",
    )
    policy = SoftmaxPolicy(input_dim=env.state_dim, num_actions=scenario.num_actions)
    actor = Actor(policy)
    observer = Observer()
    optimizer = torch.optim.SGD(actor.parameters(), lr=learning_rate)
    trainer = Trainer(
        env=env,
        actor=actor,
        actor_optimizer=optimizer,
        observer=observer,
        optimal_actions=scenario.optimal_actions,
    )
    trace = trainer.run(num_steps=num_steps, seed=seed)

    if trace.final_reward is None:
        raise RuntimeError("Case 2 trace unexpectedly has no final reward")
    if len(trace.state_indices) != num_steps:
        raise RuntimeError("Case 2 trace must record one state index per step")

    per_state_probs = _final_optimal_action_probs_by_state(actor, scenario)
    state_counts = _state_sample_counts(trace.state_indices, scenario.num_states)
    per_state_steps = _per_state_convergence_steps(
        state_indices=trace.state_indices,
        optimal_action_probs=trace.optimal_action_probs,
        num_states=scenario.num_states,
        threshold=convergence_threshold,
    )
    overall_step = (
        max(step for step in per_state_steps if step is not None)
        if all(step is not None for step in per_state_steps)
        else None
    )
    expected_reward = _expected_reward_under_policy(
        actor=actor,
        scenario=scenario,
        state_distribution=state_distribution,
    )
    weighted_final_prob = sum(
        probability * probability_optimal
        for probability, probability_optimal in zip(
            state_distribution.probabilities,
            per_state_probs,
            strict=True,
        )
    )
    metric_values = (
        trace.rewards
        + trace.losses
        + trace.optimal_action_probs
        + list(per_state_probs)
        + [expected_reward, weighted_final_prob]
    )
    nan_detected = any(not math.isfinite(value) for value in metric_values)
    min_final_prob = min(per_state_probs)

    run_result = Case2RunResult(
        scenario=scenario.name,
        distribution=state_distribution.name,
        num_states=scenario.num_states,
        num_actions=scenario.num_actions,
        state_distribution=tuple(state_distribution.probabilities),
        reward_table=tuple(tuple(row) for row in scenario.reward_table),
        optimal_actions=tuple(scenario.optimal_actions),
        learning_rate=float(learning_rate),
        seed=int(seed),
        num_steps=int(num_steps),
        final_reward=float(trace.final_reward),
        average_reward_last_window=_average_last(trace.rewards, reward_window),
        weighted_final_optimal_action_prob=float(weighted_final_prob),
        unweighted_final_optimal_action_prob=fmean(per_state_probs),
        min_final_optimal_action_prob=float(min_final_prob),
        expected_reward_policy=float(expected_reward),
        overall_convergence_step=overall_step,
        converged=min_final_prob >= convergence_threshold,
        state_sample_counts=tuple(state_counts),
        per_state_final_optimal_action_probs=tuple(per_state_probs),
        per_state_convergence_steps=tuple(per_state_steps),
        nan_detected=nan_detected,
    )

    per_state_results = [
        Case2PerStateRunResult(
            scenario=scenario.name,
            distribution=state_distribution.name,
            learning_rate=float(learning_rate),
            seed=int(seed),
            num_steps=int(num_steps),
            state_index=state_index,
            state_probability=float(state_distribution.probabilities[state_index]),
            optimal_action=int(scenario.optimal_actions[state_index]),
            sample_count=int(state_counts[state_index]),
            final_optimal_action_prob=float(per_state_probs[state_index]),
            convergence_step=per_state_steps[state_index],
        )
        for state_index in range(scenario.num_states)
    ]
    return run_result, per_state_results


def run_case2_benchmark(
    config: Case2BenchmarkConfig,
) -> tuple[list[Case2RunResult], list[Case2PerStateRunResult]]:
    """Run a scenario x distribution x learning-rate x seed Case 2 sweep."""
    run_results: list[Case2RunResult] = []
    per_state_results: list[Case2PerStateRunResult] = []
    for scenario in config.scenarios:
        for state_distribution in config.state_distributions:
            if state_distribution.num_states != scenario.num_states:
                continue
            for learning_rate in config.learning_rates:
                for seed in config.seeds:
                    run_result, per_state_result = run_case2_once(
                        scenario=scenario,
                        state_distribution=state_distribution,
                        learning_rate=learning_rate,
                        seed=seed,
                        num_steps=config.num_steps,
                        convergence_threshold=config.convergence_threshold,
                        reward_window=config.reward_window,
                    )
                    run_results.append(run_result)
                    per_state_results.extend(per_state_result)
    return run_results, per_state_results


def summarize_case2_results(results: Sequence[Case2RunResult]) -> list[Case2SummaryResult]:
    """Aggregate repeated-seed Case 2 results by scenario, distribution, and LR."""
    grouped: dict[tuple[str, str, float], list[Case2RunResult]] = defaultdict(list)
    for result in results:
        grouped[(result.scenario, result.distribution, result.learning_rate)].append(result)

    summaries: list[Case2SummaryResult] = []
    for (scenario, distribution, learning_rate), group in sorted(grouped.items()):
        weighted_probs = [result.weighted_final_optimal_action_prob for result in group]
        unweighted_probs = [
            result.unweighted_final_optimal_action_prob for result in group
        ]
        min_probs = [result.min_final_optimal_action_prob for result in group]
        expected_rewards = [result.expected_reward_policy for result in group]
        average_rewards = [result.average_reward_last_window for result in group]
        convergence_steps = [
            result.overall_convergence_step
            for result in group
            if result.overall_convergence_step is not None
        ]
        summaries.append(
            Case2SummaryResult(
                scenario=scenario,
                distribution=distribution,
                learning_rate=learning_rate,
                num_runs=len(group),
                success_rate=sum(result.converged for result in group) / len(group),
                mean_weighted_final_optimal_action_prob=fmean(weighted_probs),
                std_weighted_final_optimal_action_prob=_std(weighted_probs),
                mean_unweighted_final_optimal_action_prob=fmean(unweighted_probs),
                std_unweighted_final_optimal_action_prob=_std(unweighted_probs),
                mean_min_final_optimal_action_prob=fmean(min_probs),
                std_min_final_optimal_action_prob=_std(min_probs),
                mean_expected_reward_policy=fmean(expected_rewards),
                std_expected_reward_policy=_std(expected_rewards),
                mean_average_reward_last_window=fmean(average_rewards),
                std_average_reward_last_window=_std(average_rewards),
                mean_overall_convergence_step=(
                    fmean(convergence_steps) if convergence_steps else None
                ),
                std_overall_convergence_step=(
                    _std(convergence_steps) if convergence_steps else None
                ),
                num_nan_runs=sum(result.nan_detected for result in group),
            )
        )
    return summaries


def write_case2_results_csv(results: Sequence[Case2RunResult], path: str | Path) -> None:
    """Write per-run Case 2 benchmark results to CSV."""
    fieldnames = list(Case2RunResult.__dataclass_fields__)
    _write_csv((result.as_csv_row() for result in results), path, fieldnames)


def write_case2_per_state_csv(
    results: Sequence[Case2PerStateRunResult],
    path: str | Path,
) -> None:
    """Write per-state Case 2 benchmark results to CSV."""
    fieldnames = list(Case2PerStateRunResult.__dataclass_fields__)
    _write_csv((result.as_csv_row() for result in results), path, fieldnames)


def write_case2_summary_csv(results: Sequence[Case2SummaryResult], path: str | Path) -> None:
    """Write aggregate Case 2 benchmark summaries to CSV."""
    fieldnames = list(Case2SummaryResult.__dataclass_fields__)
    _write_csv((result.as_csv_row() for result in results), path, fieldnames)


def run_learning_rate_sweep() -> list[Case1SummaryResult]:
    """Compatibility wrapper for the initial benchmark placeholder."""
    results = run_case1_benchmark(Case1BenchmarkConfig())
    return summarize_case1_results(results)


def _average_last(values: Sequence[float], window: int) -> float:
    tail = values[-window:]
    return fmean(tail)


def _std(values: Sequence[float | int]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def _state_sample_counts(state_indices: Sequence[int], num_states: int) -> list[int]:
    counts = [0] * num_states
    for state_index in state_indices:
        counts[state_index] += 1
    return counts


def _per_state_convergence_steps(
    state_indices: Sequence[int],
    optimal_action_probs: Sequence[float],
    num_states: int,
    threshold: float,
) -> list[int | None]:
    steps: list[int | None] = [None] * num_states
    for global_step, (state_index, probability) in enumerate(
        zip(state_indices, optimal_action_probs, strict=True),
        start=1,
    ):
        if steps[state_index] is None and probability >= threshold:
            steps[state_index] = global_step
    return steps


def _final_optimal_action_probs_by_state(
    actor: Actor,
    scenario: Case2Scenario,
) -> tuple[float, ...]:
    with torch.no_grad():
        state_inputs = torch.eye(scenario.num_states, dtype=torch.float32)
        distribution = actor.policy.action_distribution(state_inputs)
        probs = distribution.probs.detach().cpu()
    return tuple(
        float(probs[state_index, optimal_action].item())
        for state_index, optimal_action in enumerate(scenario.optimal_actions)
    )


def _expected_reward_under_policy(
    actor: Actor,
    scenario: Case2Scenario,
    state_distribution: Case2StateDistribution,
) -> float:
    with torch.no_grad():
        state_inputs = torch.eye(scenario.num_states, dtype=torch.float32)
        action_probs = actor.policy.action_distribution(state_inputs).probs.detach().cpu()

    expected_reward = 0.0
    for state_index, state_probability in enumerate(state_distribution.probabilities):
        state_expected_reward = 0.0
        for action_index in range(scenario.num_actions):
            state_expected_reward += (
                float(action_probs[state_index, action_index].item())
                * scenario.reward_table[state_index][action_index]
            )
        expected_reward += state_probability * state_expected_reward
    return expected_reward


def _format_float_sequence(values: Sequence[float]) -> str:
    return "|".join(f"{value:g}" for value in values)


def _format_int_sequence(values: Sequence[int]) -> str:
    return "|".join(str(value) for value in values)


def _format_optional_int_sequence(values: Sequence[int | None]) -> str:
    return "|".join("" if value is None else str(value) for value in values)


def _format_reward_table(values: Sequence[Sequence[float]]) -> str:
    return ";".join(_format_float_sequence(row) for row in values)


def _write_csv(
    rows: Iterable[dict[str, str | int | float | bool]],
    path: str | Path,
    fieldnames: Sequence[str],
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
