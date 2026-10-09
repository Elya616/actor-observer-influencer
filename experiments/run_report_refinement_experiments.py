"""Generate report figures and CSVs for mechanism-level experiments.

The experiments report both aggregate reward behavior and the internal learned
quantities needed to verify each mechanism: action probabilities, compression
weights, messages, and Observer-selection probabilities.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict, deque
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, stdev
from typing import Any

import matplotlib.pyplot as plt
import torch

from actor_observer_rl.agents import Actor, Observer, ObserverMessageModule
from actor_observer_rl.evaluation import (
    default_multi_observer_experiment,
    run_multi_observer_sweep,
)
from actor_observer_rl.policies import SoftmaxPolicy
from actor_observer_rl.training import build_actor_message_input
from actor_observer_rl.training.updates import reinforce_loss
from actor_observer_rl.utils.seeding import set_global_seed


STATE_FEATURES = (
    (0.0, 0.0),
    (1.0, 0.0),
    (0.0, 1.0),
    (1.0, 1.0),
)
STATE_SUMS = tuple(int(sum(state)) for state in STATE_FEATURES)
SUM_REWARD_TABLE = tuple(
    tuple(1.0 if action == state_sum else -0.1 for action in range(3))
    for state_sum in STATE_SUMS
)


@dataclass(frozen=True)
class FoundationDiagnosticConfig:
    name: str
    display_name: str
    reward_table: tuple[tuple[float, ...], ...]
    state_distribution: tuple[float, ...]

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
    def ceiling_reward(self) -> float:
        return sum(
            probability * max(rewards)
            for probability, rewards in zip(
                self.state_distribution,
                self.reward_table,
                strict=True,
            )
        )


class SumDecoderPolicy(torch.nn.Module):
    """Fixed differentiable decoder used to test sum compression.

    The decoder expects a scalar message close to one of the action labels
    0, 1, or 2. It is not trained. Gradients flow through the message, so the
    Observer can learn a compression that makes the fixed decoder choose the
    correct action.
    """

    def __init__(self, num_actions: int = 3, sharpness: float = 2.0) -> None:
        super().__init__()
        self.num_actions = int(num_actions)
        self.sharpness = float(sharpness)
        self.register_buffer(
            "action_values",
            torch.arange(self.num_actions, dtype=torch.float32),
        )

    def action_distribution(
        self,
        message: torch.Tensor,
    ) -> torch.distributions.Categorical:
        message = message.float().reshape(-1)[0]
        logits = -self.sharpness * (message - self.action_values).pow(2)
        return torch.distributions.Categorical(logits=logits)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/report_refinement"),
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=Path("docs/figures"),
    )
    parser.add_argument("--num-steps", type=int, default=5000)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--reward-window", type=int, default=100)
    parser.add_argument(
        "--learning-rates",
        type=float,
        nargs="+",
        default=[0.001, 0.01, 0.05, 0.1],
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    foundation_rows = run_foundation_diagnostics(
        learning_rates=args.learning_rates,
        seeds=args.seeds,
        num_steps=args.num_steps,
        eval_interval=args.eval_interval,
        reward_window=args.reward_window,
    )
    case3_rows = run_case3_sum_diagnostics(
        learning_rates=args.learning_rates,
        seeds=args.seeds,
        num_steps=args.num_steps,
        eval_interval=args.eval_interval,
        reward_window=args.reward_window,
    )
    case4_rows = run_case4_fixed_selection_diagnostics(
        learning_rates=args.learning_rates,
        seeds=args.seeds,
        num_steps=args.num_steps,
        eval_interval=args.eval_interval,
        reward_window=args.reward_window,
    )
    tie_rows = run_appendix_tie_diagnostics(
        learning_rates=args.learning_rates,
        seeds=args.seeds,
        num_steps=args.num_steps,
        eval_interval=args.eval_interval,
        reward_window=args.reward_window,
    )
    symmetric_rows = run_appendix_symmetric_multi_observer(
        learning_rates=args.learning_rates,
        seeds=args.seeds,
        num_steps=args.num_steps,
        eval_interval=args.eval_interval,
        reward_window=args.reward_window,
    )

    write_csv(args.output_dir / "foundation_policy_diagnostics.csv", foundation_rows)
    write_csv(args.output_dir / "case3_sum_compression_diagnostics.csv", case3_rows)
    write_csv(args.output_dir / "case4_fixed_observer_selection_diagnostics.csv", case4_rows)
    write_csv(args.output_dir / "appendix_tie_diagnostics.csv", tie_rows)
    write_csv(
        args.output_dir / "appendix_symmetric_multi_observer_diagnostics.csv",
        symmetric_rows,
    )
    write_csv(
        args.output_dir / "summary.csv",
        build_summary_rows(foundation_rows, case3_rows, case4_rows),
    )

    write_foundation_figures(foundation_rows, args.figure_dir)
    write_case3_figure(case3_rows, args.figure_dir)
    write_case4_figure(case4_rows, args.figure_dir)
    write_appendix_tie_figure(tie_rows, args.figure_dir)
    write_appendix_symmetric_figure(symmetric_rows, args.figure_dir)

    print(f"wrote_output_dir={args.output_dir}")
    print(f"wrote_figure_dir={args.figure_dir}")
    for row in build_summary_rows(foundation_rows, case3_rows, case4_rows):
        print(
            f"{row['experiment']} lr={row['learning_rate']}: "
            f"final_expected={row['mean_final_expected_reward']:.3f}, "
            f"final_sampled={row['mean_final_average_reward']:.3f}"
        )


def foundation_configs() -> tuple[FoundationDiagnosticConfig, ...]:
    return (
        FoundationDiagnosticConfig(
            name="case1_single_state",
            display_name="Case 1: Single-State Action Learning",
            reward_table=((0.0, 1.0, 0.2),),
            state_distribution=(1.0,),
        ),
        FoundationDiagnosticConfig(
            name="case2_state_dependent",
            display_name="Case 2: State-Dependent Action Learning",
            reward_table=(
                (1.0, 0.0, 0.2),
                (0.1, 1.0, 0.0),
                (0.0, 0.2, 1.0),
            ),
            state_distribution=(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
        ),
    )


def run_foundation_diagnostics(
    learning_rates: Sequence[float],
    seeds: Sequence[int],
    num_steps: int,
    eval_interval: int,
    reward_window: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for config in foundation_configs():
        for learning_rate in learning_rates:
            for seed in seeds:
                rows.extend(
                    run_foundation_once(
                        config=config,
                        learning_rate=learning_rate,
                        seed=seed,
                        num_steps=num_steps,
                        eval_interval=eval_interval,
                        reward_window=reward_window,
                    )
                )
    return rows


def run_foundation_once(
    config: FoundationDiagnosticConfig,
    learning_rate: float,
    seed: int,
    num_steps: int,
    eval_interval: int,
    reward_window: int,
) -> list[dict[str, Any]]:
    set_global_seed(seed)
    state_inputs = foundation_state_inputs(config)
    policy = SoftmaxPolicy(
        input_dim=state_inputs.shape[-1],
        num_actions=config.num_actions,
    )
    actor = Actor(policy)
    observer = Observer()
    optimizer = torch.optim.SGD(actor.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(seed)
    rewards: deque[float] = deque(maxlen=reward_window)
    rows: list[dict[str, Any]] = []

    record_foundation_rows(
        rows=rows,
        config=config,
        actor=actor,
        learning_rate=learning_rate,
        seed=seed,
        step=0,
        average_reward=0.0,
    )

    for step in range(1, num_steps + 1):
        state_index = sample_index(config.state_distribution, generator)
        actor_input = state_inputs[state_index]
        action_result = actor.act(actor_input, sample=True)
        action = int(action_result.action.detach().cpu().reshape(-1)[0].item())
        reward = observer.report_reward(config.reward_table[state_index][action])
        loss = reinforce_loss(reward, action_result.log_prob)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        rewards.append(float(reward))

        if step % eval_interval == 0 or step == num_steps:
            record_foundation_rows(
                rows=rows,
                config=config,
                actor=actor,
                learning_rate=learning_rate,
                seed=seed,
                step=step,
                average_reward=fmean(rewards) if rewards else 0.0,
            )
    return rows


def record_foundation_rows(
    rows: list[dict[str, Any]],
    config: FoundationDiagnosticConfig,
    actor: Actor,
    learning_rate: float,
    seed: int,
    step: int,
    average_reward: float,
) -> None:
    state_inputs = foundation_state_inputs(config)
    probs = action_probs(actor, state_inputs)
    expected_reward = expected_reward_from_probs(
        probs,
        config.reward_table,
        config.state_distribution,
    )
    for state_index in range(config.num_states):
        for action_index in range(config.num_actions):
            rows.append(
                {
                    "experiment": config.name,
                    "learning_rate": float(learning_rate),
                    "seed": int(seed),
                    "step": int(step),
                    "state_index": int(state_index),
                    "action_index": int(action_index),
                    "probability": float(probs[state_index, action_index].item()),
                    "is_optimal_action": int(
                        action_index == config.optimal_actions[state_index]
                    ),
                    "expected_reward": float(expected_reward),
                    "average_reward": float(average_reward),
                    "ceiling_reward": float(config.ceiling_reward),
                }
            )


def run_case3_sum_diagnostics(
    learning_rates: Sequence[float],
    seeds: Sequence[int],
    num_steps: int,
    eval_interval: int,
    reward_window: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mode in ("controlled_compression", "joint_actor_observer"):
        for learning_rate in learning_rates:
            for seed in seeds:
                rows.extend(
                    run_case3_sum_once(
                        mode=mode,
                        learning_rate=learning_rate,
                        seed=seed,
                        num_steps=num_steps,
                        eval_interval=eval_interval,
                        reward_window=reward_window,
                    )
                )
    return rows


def run_case3_sum_once(
    mode: str,
    learning_rate: float,
    seed: int,
    num_steps: int,
    eval_interval: int,
    reward_window: int,
) -> list[dict[str, Any]]:
    set_global_seed(seed)
    observer = Observer(ObserverMessageModule(input_dim=2, message_dim=1))
    with torch.no_grad():
        observer.message_module.linear.weight.fill_(0.0)
        observer.message_module.linear.bias.fill_(0.0)

    if mode == "controlled_compression":
        policy: Any = SumDecoderPolicy(sharpness=4.0)
        actor = None
        optimizer = torch.optim.SGD(observer.parameters(), lr=learning_rate)
    elif mode == "joint_actor_observer":
        actor = Actor(SoftmaxPolicy(input_dim=2, num_actions=3))
        policy = actor.policy
        optimizer = torch.optim.SGD(
            list(actor.parameters()) + list(observer.parameters()),
            lr=learning_rate,
        )
    else:
        raise ValueError(f"unknown Case 3 mode: {mode}")

    generator = torch.Generator().manual_seed(seed)
    rewards: deque[float] = deque(maxlen=reward_window)
    rows: list[dict[str, Any]] = []
    record_case3_sum_rows(
        rows=rows,
        mode=mode,
        observer=observer,
        actor=actor,
        policy=policy,
        learning_rate=learning_rate,
        seed=seed,
        step=0,
        average_reward=0.0,
    )

    for step in range(1, num_steps + 1):
        state_index = int(torch.randint(0, len(STATE_FEATURES), (1,), generator=generator))
        state = state_tensor(state_index)
        message = observer.message(state)
        if mode == "controlled_compression":
            distribution = policy.action_distribution(message)
        else:
            actor_input = build_actor_message_input(torch.zeros(1), message)
            distribution = policy.action_distribution(actor_input)
        action = distribution.sample()
        action_index = int(action.detach().cpu().reshape(-1)[0].item())
        reward = SUM_REWARD_TABLE[state_index][action_index]
        loss = reinforce_loss(reward, distribution.log_prob(action))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        rewards.append(float(reward))

        if step % eval_interval == 0 or step == num_steps:
            record_case3_sum_rows(
                rows=rows,
                mode=mode,
                observer=observer,
                actor=actor,
                policy=policy,
                learning_rate=learning_rate,
                seed=seed,
                step=step,
                average_reward=fmean(rewards) if rewards else 0.0,
            )
    return rows


def record_case3_sum_rows(
    rows: list[dict[str, Any]],
    mode: str,
    observer: Observer,
    actor: Actor | None,
    policy: Any,
    learning_rate: float,
    seed: int,
    step: int,
    average_reward: float,
) -> None:
    del actor
    with torch.no_grad():
        weight = observer.message_module.linear.weight.detach().cpu().reshape(-1)
        bias = float(observer.message_module.linear.bias.detach().cpu().reshape(-1)[0].item())
        messages = []
        probs_by_state = []
        for state_index in range(len(STATE_FEATURES)):
            state = state_tensor(state_index)
            message = observer.message(state)
            messages.append(float(message.detach().cpu().reshape(-1)[0].item()))
            if mode == "controlled_compression":
                distribution = policy.action_distribution(message)
            else:
                actor_input = build_actor_message_input(torch.zeros(1), message)
                distribution = policy.action_distribution(actor_input)
            probs_by_state.append(distribution.probs.detach().cpu())

    probs = torch.stack(probs_by_state)
    expected_reward = expected_reward_from_probs(
        probs,
        SUM_REWARD_TABLE,
        (0.25, 0.25, 0.25, 0.25),
    )
    compression_r2 = r_squared(messages, STATE_SUMS)
    weight_cosine = cosine_to_sum_direction(weight)
    for state_index, message_value in enumerate(messages):
        for action_index in range(3):
            rows.append(
                {
                    "experiment": f"case3_sum_{mode}",
                    "learning_rate": float(learning_rate),
                    "seed": int(seed),
                    "step": int(step),
                    "state_index": int(state_index),
                    "state_x1": float(STATE_FEATURES[state_index][0]),
                    "state_x2": float(STATE_FEATURES[state_index][1]),
                    "state_sum": int(STATE_SUMS[state_index]),
                    "action_index": int(action_index),
                    "probability": float(probs[state_index, action_index].item()),
                    "message": float(message_value),
                    "weight_x1": float(weight[0].item()),
                    "weight_x2": float(weight[1].item()),
                    "bias": float(bias),
                    "compression_r2": float(compression_r2),
                    "weight_cosine_to_sum": float(weight_cosine),
                    "expected_reward": float(expected_reward),
                    "average_reward": float(average_reward),
                    "ceiling_reward": 1.0,
                }
            )


def run_case4_fixed_selection_diagnostics(
    learning_rates: Sequence[float],
    seeds: Sequence[int],
    num_steps: int,
    eval_interval: int,
    reward_window: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for learning_rate in learning_rates:
        for seed in seeds:
            rows.extend(
                run_case4_fixed_selection_once(
                    learning_rate=learning_rate,
                    seed=seed,
                    num_steps=num_steps,
                    eval_interval=eval_interval,
                    reward_window=reward_window,
                )
            )
    return rows


def run_case4_fixed_selection_once(
    learning_rate: float,
    seed: int,
    num_steps: int,
    eval_interval: int,
    reward_window: int,
) -> list[dict[str, Any]]:
    set_global_seed(seed)
    logits = torch.nn.Parameter(torch.zeros(3))
    optimizer = torch.optim.SGD([logits], lr=learning_rate)
    policy = SumDecoderPolicy(sharpness=4.0)
    generator = torch.Generator().manual_seed(seed)
    rewards: deque[float] = deque(maxlen=reward_window)
    rows: list[dict[str, Any]] = []
    observer_names = ("zero", "first_feature", "sum")
    observer_functions = (message_zero, message_first_feature, message_sum)

    record_case4_rows(
        rows=rows,
        logits=logits,
        observer_names=observer_names,
        observer_functions=observer_functions,
        policy=policy,
        learning_rate=learning_rate,
        seed=seed,
        step=0,
        average_reward=0.0,
    )

    for step in range(1, num_steps + 1):
        state_index = int(torch.randint(0, len(STATE_FEATURES), (1,), generator=generator))
        weights = torch.softmax(logits, dim=-1)
        distribution = torch.distributions.Categorical(probs=weights)
        selected = distribution.sample()
        observer_index = int(selected.detach().cpu().item())
        message = observer_functions[observer_index](state_tensor(state_index))
        action_distribution = policy.action_distribution(message)
        action = action_distribution.sample()
        action_index = int(action.detach().cpu().item())
        reward = SUM_REWARD_TABLE[state_index][action_index]
        loss = reinforce_loss(reward, distribution.log_prob(selected))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        rewards.append(float(reward))

        if step % eval_interval == 0 or step == num_steps:
            record_case4_rows(
                rows=rows,
                logits=logits,
                observer_names=observer_names,
                observer_functions=observer_functions,
                policy=policy,
                learning_rate=learning_rate,
                seed=seed,
                step=step,
                average_reward=fmean(rewards) if rewards else 0.0,
            )
    return rows


def run_appendix_tie_diagnostics(
    learning_rates: Sequence[float],
    seeds: Sequence[int],
    num_steps: int,
    eval_interval: int,
    reward_window: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    tie_config = FoundationDiagnosticConfig(
        name="appendix_case1_action_tie",
        display_name="Appendix: Case 1 Action Tie",
        reward_table=((1.0, 1.0, 0.0),),
        state_distribution=(1.0,),
    )
    for learning_rate in learning_rates:
        for seed in seeds:
            rows.extend(
                run_foundation_once(
                    config=tie_config,
                    learning_rate=learning_rate,
                    seed=seed,
                    num_steps=num_steps,
                    eval_interval=eval_interval,
                    reward_window=reward_window,
                )
            )
            rows.extend(
                run_case4_observer_tie_once(
                    learning_rate=learning_rate,
                    seed=seed,
                    num_steps=num_steps,
                    eval_interval=eval_interval,
                    reward_window=reward_window,
                )
            )
    return rows


def run_case4_observer_tie_once(
    learning_rate: float,
    seed: int,
    num_steps: int,
    eval_interval: int,
    reward_window: int,
) -> list[dict[str, Any]]:
    set_global_seed(seed)
    logits = torch.nn.Parameter(torch.zeros(3))
    optimizer = torch.optim.SGD([logits], lr=learning_rate)
    policy = SumDecoderPolicy(sharpness=4.0)
    generator = torch.Generator().manual_seed(seed)
    rewards: deque[float] = deque(maxlen=reward_window)
    rows: list[dict[str, Any]] = []
    observer_names = ("zero", "sum_a", "sum_b")
    observer_functions = (message_zero, message_sum, message_sum)

    record_case4_rows(
        rows=rows,
        logits=logits,
        observer_names=observer_names,
        observer_functions=observer_functions,
        policy=policy,
        learning_rate=learning_rate,
        seed=seed,
        step=0,
        average_reward=0.0,
        experiment="appendix_case4_observer_tie",
    )

    for step in range(1, num_steps + 1):
        state_index = int(torch.randint(0, len(STATE_FEATURES), (1,), generator=generator))
        weights = torch.softmax(logits, dim=-1)
        distribution = torch.distributions.Categorical(probs=weights)
        selected = distribution.sample()
        observer_index = int(selected.detach().cpu().item())
        message = observer_functions[observer_index](state_tensor(state_index))
        action_distribution = policy.action_distribution(message)
        action = action_distribution.sample()
        action_index = int(action.detach().cpu().item())
        reward = SUM_REWARD_TABLE[state_index][action_index]
        loss = reinforce_loss(reward, distribution.log_prob(selected))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        rewards.append(float(reward))

        if step % eval_interval == 0 or step == num_steps:
            record_case4_rows(
                rows=rows,
                logits=logits,
                observer_names=observer_names,
                observer_functions=observer_functions,
                policy=policy,
                learning_rate=learning_rate,
                seed=seed,
                step=step,
                average_reward=fmean(rewards) if rewards else 0.0,
                experiment="appendix_case4_observer_tie",
            )
    return rows


def run_appendix_symmetric_multi_observer(
    learning_rates: Sequence[float],
    seeds: Sequence[int],
    num_steps: int,
    eval_interval: int,
    reward_window: int,
) -> list[dict[str, Any]]:
    experiment = default_multi_observer_experiment()
    curves, _ = run_multi_observer_sweep(
        experiment=experiment,
        learning_rates=learning_rates,
        seeds=seeds,
        num_steps=num_steps,
        eval_interval=eval_interval,
        reward_window=reward_window,
        threshold_fraction=0.95,
    )
    rows: list[dict[str, Any]] = []
    for point in curves:
        rows.append(
            {
                "experiment": "appendix_symmetric_multi_observer",
                "learning_rate": float(point.learning_rate),
                "seed": int(point.seed),
                "step": int(point.step),
                "expected_reward": float(point.expected_reward),
                "average_reward": float(point.average_reward_last_window),
                "mean_optimal_action_probability": float(
                    point.mean_optimal_action_probability
                ),
                "following_weights": _format_float_sequence(point.following_weights),
                "max_following_weight": max(point.following_weights),
                "ceiling_reward": float(point.ceiling_reward),
            }
        )
    return rows


def record_case4_rows(
    rows: list[dict[str, Any]],
    logits: torch.nn.Parameter,
    observer_names: Sequence[str],
    observer_functions: Sequence[Callable[[torch.Tensor], torch.Tensor]],
    policy: SumDecoderPolicy,
    learning_rate: float,
    seed: int,
    step: int,
    average_reward: float,
    experiment: str = "case4_fixed_observer_selection",
) -> None:
    with torch.no_grad():
        weights = torch.softmax(logits, dim=-1).detach().cpu()
        expected_reward = 0.0
        observer_expected_rewards = []
        for observer_index, observer_function in enumerate(observer_functions):
            observer_reward = expected_reward_for_fixed_observer(observer_function, policy)
            observer_expected_rewards.append(observer_reward)
            expected_reward += float(weights[observer_index].item()) * observer_reward
    for observer_index, observer_name in enumerate(observer_names):
        rows.append(
            {
                "experiment": experiment,
                "learning_rate": float(learning_rate),
                "seed": int(seed),
                "step": int(step),
                "observer_index": int(observer_index),
                "observer_name": observer_name,
                "selection_probability": float(weights[observer_index].item()),
                "observer_expected_reward": float(observer_expected_rewards[observer_index]),
                "expected_reward": float(expected_reward),
                "average_reward": float(average_reward),
                "ceiling_reward": max(observer_expected_rewards),
                "best_observer_index": int(
                    max(
                        range(len(observer_expected_rewards)),
                        key=lambda index: observer_expected_rewards[index],
                    )
                ),
            }
        )


def write_foundation_figures(rows: Sequence[dict[str, Any]], figure_dir: Path) -> None:
    write_foundation_case_reward_figure(
        rows,
        figure_dir / "report_case1_reward_curves",
        experiment="case1_single_state",
        title="Case 1: single-state reward learning",
    )
    write_foundation_case_reward_figure(
        rows,
        figure_dir / "report_case2_reward_curves",
        experiment="case2_state_dependent",
        title="Case 2: state-dependent reward learning",
    )
    write_case1_probability_figure(rows, figure_dir / "report_case1_action_probabilities")
    write_case2_probability_figure(rows, figure_dir / "report_case2_state_probabilities")


def write_foundation_case_reward_figure(
    rows: Sequence[dict[str, Any]],
    output_base: Path,
    experiment: str,
    title: str,
) -> None:
    unique_rows = unique_state_action_rows(rows)
    metrics = (
        ("expected_reward", "Expected reward"),
        ("average_reward", "Average sampled reward"),
    )
    experiment_rows = [row for row in unique_rows if row["experiment"] == experiment]
    ceiling = max(float(row["ceiling_reward"]) for row in experiment_rows)
    threshold = 0.95 * ceiling
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.2), sharex=True)

    for ax, (metric, metric_title) in zip(axes, metrics, strict=True):
        for learning_rate in sorted({row["learning_rate"] for row in experiment_rows}):
            series = mean_series(
                experiment_rows,
                key_filter=lambda row, learning_rate=learning_rate: (
                    row["learning_rate"] == learning_rate
                ),
                value_key=metric,
            )
            ax.plot(
                [point[0] for point in series],
                [point[1] for point in series],
                linewidth=1.7,
                label=f"lr={learning_rate:g}",
            )
        ax.axhline(ceiling, color="black", linewidth=1.0, alpha=0.55)
        ax.axhline(threshold, color="black", linestyle="--", linewidth=0.9, alpha=0.45)
        ax.set_title(metric_title)
        ax.set_xlabel("training step")
        ax.set_ylabel("reward")
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(frameon=False, fontsize=7)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, output_base)


def write_case1_probability_figure(rows: Sequence[dict[str, Any]], output_base: Path) -> None:
    filtered = [
        row
        for row in rows
        if row["experiment"] == "case1_single_state" and row["learning_rate"] == 0.05
    ]
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    for action_index in sorted({row["action_index"] for row in filtered}):
        series = mean_series(
            filtered,
            key_filter=lambda row, action_index=action_index: (
                row["action_index"] == action_index
            ),
            value_key="probability",
        )
        ax.plot(
            [point[0] for point in series],
            [point[1] for point in series],
            linewidth=1.9,
            label=f"action {action_index}",
        )
    ax.set_title("Case 1: action probabilities (lr=0.05)")
    ax.set_xlabel("training step")
    ax.set_ylabel(r"$\pi_\theta(a \mid s_{\mathrm{static}})$")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    save_figure(fig, output_base)


def write_appendix_tie_figure(rows: Sequence[dict[str, Any]], figure_dir: Path) -> None:
    output_base = figure_dir / "report_appendix_tie_cases"
    lr = 0.05
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.2))

    case1_rows = [
        row
        for row in rows
        if row["experiment"] == "appendix_case1_action_tie"
        and row["learning_rate"] == lr
    ]
    for action_index in sorted({row["action_index"] for row in case1_rows}):
        series = mean_series(
            case1_rows,
            key_filter=lambda row, action_index=action_index: (
                row["action_index"] == action_index
            ),
            value_key="probability",
        )
        axes[0].plot(
            [point[0] for point in series],
            [point[1] for point in series],
            linewidth=1.8,
            label=f"action {action_index}",
        )
    axes[0].set_title("Action tie: actions 0 and 1 optimal")
    axes[0].set_xlabel("training step")
    axes[0].set_ylabel("probability")
    axes[0].set_ylim(0.0, 1.02)
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)

    tie_rows = [
        row
        for row in rows
        if row["experiment"] == "appendix_case4_observer_tie"
        and row["learning_rate"] == lr
    ]
    for observer_name in ("zero", "sum_a", "sum_b"):
        series = mean_series(
            tie_rows,
            key_filter=lambda row, observer_name=observer_name: (
                row["observer_name"] == observer_name
            ),
            value_key="selection_probability",
        )
        axes[1].plot(
            [point[0] for point in series],
            [point[1] for point in series],
            linewidth=1.8,
            label=observer_name,
        )
    axes[1].set_title("Observer tie: two useful sum Observers")
    axes[1].set_xlabel("training step")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].grid(True, axis="y", alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)

    useful_by_seed_step: dict[tuple[int, int], float] = defaultdict(float)
    for row in tie_rows:
        if row["observer_name"] in {"sum_a", "sum_b"}:
            useful_by_seed_step[(int(row["seed"]), int(row["step"]))] += float(
                row["selection_probability"]
            )
    grouped: dict[int, list[float]] = defaultdict(list)
    for (seed, step), value in sorted(useful_by_seed_step.items()):
        grouped[step].append(value)
    for seed in sorted({int(row["seed"]) for row in tie_rows}):
        series = [
            (step, value)
            for (series_seed, step), value in sorted(useful_by_seed_step.items())
            if series_seed == seed
        ]
        axes[2].plot(
            [point[0] for point in series],
            [point[1] for point in series],
            color="gray",
            alpha=0.35,
            linewidth=1.0,
        )
    axes[2].plot(
        sorted(grouped),
        [fmean(grouped[step]) for step in sorted(grouped)],
        color="black",
        linewidth=2.0,
        label="mean",
    )
    axes[2].set_title("Tie case: probability on useful set")
    axes[2].set_xlabel("training step")
    axes[2].set_ylim(0.0, 1.02)
    axes[2].grid(True, axis="y", alpha=0.25)
    axes[2].legend(frameon=False, fontsize=8)

    fig.suptitle("Appendix tie-case results", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_figure(fig, output_base)


def write_appendix_symmetric_figure(
    rows: Sequence[dict[str, Any]],
    figure_dir: Path,
) -> None:
    output_base = figure_dir / "report_appendix_symmetric_multi_observer"
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.2))

    for learning_rate in sorted({row["learning_rate"] for row in rows}):
        series = mean_series(
            rows,
            key_filter=lambda row, learning_rate=learning_rate: (
                row["learning_rate"] == learning_rate
            ),
            value_key="expected_reward",
        )
        axes[0].plot(
            [point[0] for point in series],
            [point[1] for point in series],
            linewidth=1.7,
            label=f"lr={learning_rate:g}",
        )
    axes[0].set_title("Symmetric joint run: expected reward")
    axes[0].set_xlabel("training step")
    axes[0].set_ylabel("reward")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=7)

    for learning_rate in sorted({row["learning_rate"] for row in rows}):
        series = mean_series(
            rows,
            key_filter=lambda row, learning_rate=learning_rate: (
                row["learning_rate"] == learning_rate
            ),
            value_key="max_following_weight",
        )
        axes[1].plot(
            [point[0] for point in series],
            [point[1] for point in series],
            linewidth=1.7,
            label=f"lr={learning_rate:g}",
        )
    axes[1].axhline(1.0 / 3.0, color="black", linestyle=":", linewidth=1.0)
    axes[1].set_title(r"Mean collapse indicator $\max_i \rho_i$")
    axes[1].set_xlabel("training step")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].grid(True, axis="y", alpha=0.25)
    axes[1].legend(frameon=False, fontsize=7)

    lr = 0.05
    filtered = [row for row in rows if row["learning_rate"] == lr]
    for seed in sorted({int(row["seed"]) for row in filtered}):
        series = [
            (int(row["step"]), float(row["max_following_weight"]))
            for row in filtered
            if int(row["seed"]) == seed
        ]
        axes[2].plot(
            [point[0] for point in series],
            [point[1] for point in series],
            color="gray",
            alpha=0.45,
            linewidth=1.0,
        )
    mean = mean_series(
        filtered,
        key_filter=lambda row: True,
        value_key="max_following_weight",
    )
    axes[2].plot(
        [point[0] for point in mean],
        [point[1] for point in mean],
        color="black",
        linewidth=2.0,
        label="mean",
    )
    axes[2].axhline(1.0 / 3.0, color="black", linestyle=":", linewidth=1.0)
    axes[2].set_title(r"Per-seed collapse, lr=0.05")
    axes[2].set_xlabel("training step")
    axes[2].set_ylim(0.0, 1.05)
    axes[2].grid(True, axis="y", alpha=0.25)
    axes[2].legend(frameon=False, fontsize=8)

    fig.suptitle("Appendix symmetric multi-Observer result", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_figure(fig, output_base)


def write_case2_probability_figure(rows: Sequence[dict[str, Any]], output_base: Path) -> None:
    filtered = [
        row
        for row in rows
        if row["experiment"] == "case2_state_dependent" and row["learning_rate"] == 0.05
    ]
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    for state_index in sorted({row["state_index"] for row in filtered}):
        series = mean_series(
            filtered,
            key_filter=lambda row, state_index=state_index: (
                row["state_index"] == state_index and row["is_optimal_action"] == 1
            ),
            value_key="probability",
        )
        ax.plot(
            [point[0] for point in series],
            [point[1] for point in series],
            linewidth=1.9,
            label=f"state {state_index}",
        )
    ax.set_title("Case 2: optimal-action probability by state (lr=0.05)")
    ax.set_xlabel("training step")
    ax.set_ylabel(r"$\pi_\theta(a^*(s) \mid s)$")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    save_figure(fig, output_base)


def write_case3_figure(rows: Sequence[dict[str, Any]], figure_dir: Path) -> None:
    output_base = figure_dir / "report_case3_sum_compression"
    lr = 0.01
    controlled = [
        row
        for row in rows
        if row["experiment"] == "case3_sum_controlled_compression"
        and row["learning_rate"] == lr
    ]
    all_controlled = [
        row
        for row in rows
        if row["experiment"] == "case3_sum_controlled_compression"
    ]

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.2))
    for learning_rate in sorted({row["learning_rate"] for row in all_controlled}):
        series = mean_series(
            unique_state_action_rows(all_controlled),
            key_filter=lambda row, learning_rate=learning_rate: (
                row["learning_rate"] == learning_rate
            ),
            value_key="expected_reward",
        )
        axes[0].plot(
            [point[0] for point in series],
            [point[1] for point in series],
            linewidth=1.8,
            label=f"lr={learning_rate:g}",
        )
    axes[0].axhline(1.0, color="black", linewidth=1.0, alpha=0.5)
    axes[0].set_title("Expected reward")
    axes[0].set_xlabel("training step")
    axes[0].set_ylabel("reward")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)

    for key, label in (
        ("weight_x1", r"$w_1$"),
        ("weight_x2", r"$w_2$"),
        ("bias", "bias"),
    ):
        series = mean_series(
            unique_state_action_rows(controlled),
            key_filter=lambda row: True,
            value_key=key,
        )
        axes[1].plot(
            [point[0] for point in series],
            [point[1] for point in series],
            linewidth=1.8,
            label=label,
        )
    axes[1].set_title(r"Controlled test: $m=w_1x_1+w_2x_2+b$ (lr=0.01)")
    axes[1].set_xlabel("training step")
    axes[1].grid(True, axis="y", alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)

    for state_sum in sorted(set(STATE_SUMS)):
        series = mean_series(
            controlled,
            key_filter=lambda row, state_sum=state_sum: (
                row["state_sum"] == state_sum and row["action_index"] == 0
            ),
            value_key="message",
        )
        axes[2].plot(
            [point[0] for point in series],
            [point[1] for point in series],
            linewidth=1.8,
            label=f"true sum {state_sum}",
        )
    axes[2].plot(
        [0, max(row["step"] for row in controlled)],
        [0, 0],
        color="black",
        linewidth=0.8,
        alpha=0.25,
    )
    axes[2].set_title("Message by true sum (lr=0.01)")
    axes[2].set_xlabel("training step")
    axes[2].set_ylabel("message")
    axes[2].grid(True, axis="y", alpha=0.25)
    axes[2].legend(frameon=False, fontsize=8)

    fig.suptitle("Case 3 sum-compression results", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_figure(fig, output_base)


def write_case4_figure(rows: Sequence[dict[str, Any]], figure_dir: Path) -> None:
    output_base = figure_dir / "report_case4_fixed_observer_selection"
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.2))
    unique_rows = unique_observer_rows(rows)

    for learning_rate in sorted({row["learning_rate"] for row in rows}):
        series = mean_series(
            unique_rows,
            key_filter=lambda row, learning_rate=learning_rate: (
                row["learning_rate"] == learning_rate
            ),
            value_key="expected_reward",
        )
        axes[0].plot(
            [point[0] for point in series],
            [point[1] for point in series],
            linewidth=1.7,
            label=f"lr={learning_rate:g}",
        )
    axes[0].set_title("Expected reward")
    axes[0].set_xlabel("training step")
    axes[0].set_ylabel("reward")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=7)

    lr = 0.05
    filtered = [row for row in rows if row["learning_rate"] == lr]
    for observer_name in ("zero", "first_feature", "sum"):
        series = mean_series(
            filtered,
            key_filter=lambda row, observer_name=observer_name: (
                row["observer_name"] == observer_name
            ),
            value_key="selection_probability",
        )
        axes[1].plot(
            [point[0] for point in series],
            [point[1] for point in series],
            linewidth=1.9,
            label=observer_name,
        )
    axes[1].set_title(r"Selection probabilities $\lambda_i$ (lr=0.05)")
    axes[1].set_xlabel("training step")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].grid(True, axis="y", alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)

    for learning_rate in sorted({row["learning_rate"] for row in rows}):
        series = mean_series(
            rows,
            key_filter=lambda row, learning_rate=learning_rate: (
                row["learning_rate"] == learning_rate
                and row["observer_name"] == "sum"
            ),
            value_key="selection_probability",
        )
        axes[2].plot(
            [point[0] for point in series],
            [point[1] for point in series],
            linewidth=1.7,
            label=f"lr={learning_rate:g}",
        )
    axes[2].set_title("Probability of useful sum Observer")
    axes[2].set_xlabel("training step")
    axes[2].set_ylim(0.0, 1.02)
    axes[2].grid(True, axis="y", alpha=0.25)
    axes[2].legend(frameon=False, fontsize=7)

    fig.suptitle("Case 4 fixed-Observer selection results", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_figure(fig, output_base)


def build_summary_rows(
    foundation_rows: Sequence[dict[str, Any]],
    case3_rows: Sequence[dict[str, Any]],
    case4_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    rows.extend(summary_for_rows(unique_state_action_rows(foundation_rows)))
    rows.extend(summary_for_rows(unique_state_action_rows(case3_rows)))
    rows.extend(summary_for_rows(unique_observer_rows(case4_rows)))
    return rows


def summary_for_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    final_by_seed: dict[tuple[str, float, int], dict[str, Any]] = {}
    for row in rows:
        key = (row["experiment"], row["learning_rate"], row["seed"])
        if key not in final_by_seed or row["step"] > final_by_seed[key]["step"]:
            final_by_seed[key] = row
    for row in final_by_seed.values():
        grouped[(row["experiment"], row["learning_rate"])].append(row)

    summary_rows = []
    for (experiment, learning_rate), group in sorted(grouped.items()):
        final_expected = [row["expected_reward"] for row in group]
        final_sampled = [row["average_reward"] for row in group]
        summary: dict[str, Any] = {
            "experiment": experiment,
            "learning_rate": learning_rate,
            "num_runs": len(group),
            "mean_final_expected_reward": fmean(final_expected),
            "std_final_expected_reward": std(final_expected),
            "mean_final_average_reward": fmean(final_sampled),
            "std_final_average_reward": std(final_sampled),
        }
        if "compression_r2" in group[0]:
            summary["mean_final_compression_r2"] = fmean(
                [row["compression_r2"] for row in group]
            )
            summary["mean_final_weight_cosine_to_sum"] = fmean(
                [row["weight_cosine_to_sum"] for row in group]
            )
        if "selection_probability" in group[0]:
            best_rows = [
                row for row in group if row.get("observer_name") == "sum"
            ]
            if best_rows:
                summary["mean_final_useful_observer_probability"] = fmean(
                    [row["selection_probability"] for row in best_rows]
                )
        summary_rows.append(summary)
    return summary_rows


def foundation_state_inputs(config: FoundationDiagnosticConfig) -> torch.Tensor:
    if config.num_states == 1:
        return torch.ones(1, 1, dtype=torch.float32)
    return torch.eye(config.num_states, dtype=torch.float32)


def action_probs(actor: Actor, state_inputs: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return actor.policy.action_distribution(state_inputs).probs.detach().cpu()


def expected_reward_from_probs(
    probs: torch.Tensor,
    reward_table: Sequence[Sequence[float]],
    state_distribution: Sequence[float],
) -> float:
    total = 0.0
    for state_index, probability in enumerate(state_distribution):
        state_reward = 0.0
        for action_index, reward in enumerate(reward_table[state_index]):
            state_reward += float(probs[state_index, action_index].item()) * float(reward)
        total += float(probability) * state_reward
    return total


def expected_reward_for_fixed_observer(
    observer_function: Callable[[torch.Tensor], torch.Tensor],
    policy: SumDecoderPolicy,
) -> float:
    total = 0.0
    with torch.no_grad():
        for state_index in range(len(STATE_FEATURES)):
            message = observer_function(state_tensor(state_index))
            probs = policy.action_distribution(message).probs.detach().cpu()
            total += 0.25 * sum(
                float(probs[action_index].item()) * reward
                for action_index, reward in enumerate(SUM_REWARD_TABLE[state_index])
            )
    return total


def sample_index(distribution: Sequence[float], generator: torch.Generator) -> int:
    probabilities = torch.tensor(distribution, dtype=torch.float32)
    return int(torch.multinomial(probabilities, 1, generator=generator).item())


def state_tensor(state_index: int) -> torch.Tensor:
    return torch.tensor(STATE_FEATURES[state_index], dtype=torch.float32)


def message_zero(state: torch.Tensor) -> torch.Tensor:
    del state
    return torch.tensor([0.0], dtype=torch.float32)


def message_first_feature(state: torch.Tensor) -> torch.Tensor:
    return state[0:1].float()


def message_sum(state: torch.Tensor) -> torch.Tensor:
    return state.sum().reshape(1).float()


def cosine_to_sum_direction(weight: torch.Tensor) -> float:
    target = torch.ones_like(weight)
    denominator = float(weight.norm().item() * target.norm().item())
    if denominator == 0.0:
        return 0.0
    return float(torch.dot(weight, target).item() / denominator)


def r_squared(values: Sequence[float], targets: Sequence[float | int]) -> float:
    x = torch.tensor(list(values), dtype=torch.float64)
    y = torch.tensor([float(target) for target in targets], dtype=torch.float64)
    if torch.var(x) <= 1e-12:
        return 0.0
    x_design = torch.stack((x, torch.ones_like(x)), dim=1)
    solution = torch.linalg.lstsq(x_design, y).solution
    predicted = x_design @ solution
    ss_res = torch.sum((y - predicted).pow(2))
    ss_tot = torch.sum((y - y.mean()).pow(2))
    if float(ss_tot.item()) == 0.0:
        return 1.0
    return float((1.0 - ss_res / ss_tot).item())


def unique_state_action_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    unique = []
    for row in rows:
        key = (
            row["experiment"],
            row["learning_rate"],
            row["seed"],
            row["step"],
            row.get("state_index"),
        )
        if key not in seen:
            seen.add(key)
            unique.append(row)
    return unique


def unique_observer_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    unique = []
    for row in rows:
        key = (row["experiment"], row["learning_rate"], row["seed"], row["step"])
        if key not in seen:
            seen.add(key)
            unique.append(row)
    return unique


def mean_series(
    rows: Sequence[dict[str, Any]],
    key_filter: Callable[[dict[str, Any]], bool],
    value_key: str,
) -> list[tuple[int, float]]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        if key_filter(row):
            grouped[int(row["step"])].append(float(row[value_key]))
    return [(step, fmean(values)) for step, values in sorted(grouped.items())]


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"no rows to write for {path}")
    fieldnames = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _format_float_sequence(values: Sequence[float]) -> str:
    return "|".join(f"{float(value):.10g}" for value in values)


def save_figure(fig: plt.Figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png", ".svg"):
        fig.savefig(output_base.with_suffix(suffix), bbox_inches="tight")
    plt.close(fig)


def std(values: Sequence[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


if __name__ == "__main__":
    main()
