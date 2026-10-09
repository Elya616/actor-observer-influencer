"""Run paper Section 5.2 selection-plus-compression experiments."""

from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path("/private/tmp/actor_observer_rl_mplcache").resolve()),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from actor_observer_rl.agents import (
    Actor,
    FollowingRates,
    MaskedObserverMessageModule,
    MultiObserver,
    Observer,
)
from actor_observer_rl.envs import FeaturePartialObservableStateEnv
from actor_observer_rl.policies import SumDecoderPolicy
from actor_observer_rl.training import StagedMultiObserverTrainer
from actor_observer_rl.utils.config import Paper52Config, load_yaml_config
from actor_observer_rl.utils.seeding import set_global_seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/paper_5_2_selection_compression.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/paper_5_2"),
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=Path("docs/figures"),
    )
    parser.add_argument(
        "--observer-learning-rates",
        type=float,
        nargs="+",
        default=[0.005, 0.01, 0.02, 0.05],
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = parser.parse_args()

    config = Paper52Config.from_mapping(load_yaml_config(args.config))
    rows = run_sweep(
        config,
        observer_learning_rates=args.observer_learning_rates,
        seeds=args.seeds,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    curve_path = args.output_dir / "selection_compression_curves.csv"
    summary_path = args.output_dir / "selection_compression_summary.csv"
    write_csv(curve_path, rows)
    summaries = summarize(rows)
    write_csv(summary_path, summaries)
    write_figure(rows, args.figure_dir / "paper5_2_selection_compression")

    print(f"wrote_curves={curve_path}")
    print(f"wrote_summary={summary_path}")
    print(
        "wrote_figure="
        f"{args.figure_dir / 'paper5_2_selection_compression'}.[pdf|png|svg]"
    )
    print_summary(summaries)


def run_sweep(
    config: Paper52Config,
    observer_learning_rates: list[float],
    seeds: list[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for observer_learning_rate in observer_learning_rates:
        for seed in seeds:
            trainer = build_trainer(
                config,
                observer_learning_rate=observer_learning_rate,
                seed=seed,
            )
            trace = trainer.run(
                num_steps=config.num_steps,
                seed=seed,
                eval_interval=config.eval_interval,
                reward_window=config.reward_window,
            )
            rows.extend(trace_to_rows(config, trace, observer_learning_rate, seed))
    return rows


def build_trainer(
    config: Paper52Config,
    observer_learning_rate: float,
    seed: int,
) -> StagedMultiObserverTrainer:
    set_global_seed(seed)
    env = FeaturePartialObservableStateEnv(
        state_features=config.state_features,
        reward_table=config.reward_table,
        state_distribution=config.state_distribution,
        observation_dim=config.observation_dim,
    )
    observers = []
    for input_mask in config.observer_input_masks:
        message_module = MaskedObserverMessageModule(
            input_dim=config.state_dim,
            message_dim=1,
            input_mask=input_mask,
        )
        with torch.no_grad():
            message_module.linear.weight.zero_()
            message_module.linear.bias.zero_()
        observers.append(Observer(message_module=message_module))

    multi_observer = MultiObserver(
        observers=observers,
        following_rates=FollowingRates(num_observers=config.num_observers),
    )
    actor = Actor(
        SumDecoderPolicy(
            num_actions=config.num_actions,
            sharpness=config.decoder_sharpness,
        )
    )
    following_optimizer = torch.optim.SGD(
        list(multi_observer.following_parameters()),
        lr=config.following_learning_rate,
    )
    observer_optimizer = torch.optim.SGD(
        list(multi_observer.observer_parameters()),
        lr=observer_learning_rate,
    )
    return StagedMultiObserverTrainer(
        env=env,
        actor=actor,
        multi_observer=multi_observer,
        following_optimizer=following_optimizer,
        observer_optimizer=observer_optimizer,
        optimal_actions=optimal_actions(config.reward_table),
    )


def trace_to_rows(
    config: Paper52Config,
    trace,
    observer_learning_rate: float,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    target_sums = [sum(features) for features in config.state_features]
    ceiling = ideal_sum_decoder_ceiling(config)
    for eval_index, step in enumerate(trace.evaluation_steps):
        if step == 0:
            weights = [1.0 / config.num_observers] * config.num_observers
        else:
            weights = trace.following_weights[step - 1]
        messages_by_observer = trace.observer_messages_by_state[eval_index]
        for observer_index, observer_name in enumerate(config.observer_names):
            messages = [
                float(message[0]) for message in messages_by_observer[observer_index]
            ]
            r2 = r_squared(messages, target_sums)
            rows.append(
                {
                    "experiment": "paper_5_2_selection_compression",
                    "observer_learning_rate": float(observer_learning_rate),
                    "following_learning_rate": float(config.following_learning_rate),
                    "seed": int(seed),
                    "step": int(step),
                    "observer_index": int(observer_index),
                    "observer_name": observer_name,
                    "selection_probability": float(weights[observer_index]),
                    "observer_expected_reward": float(
                        trace.per_observer_expected_rewards[eval_index][
                            observer_index
                        ]
                    ),
                    "observer_optimal_action_probability": float(
                        trace.per_observer_optimal_action_probs[eval_index][
                            observer_index
                        ]
                    ),
                    "messages_by_state": format_float_sequence(messages),
                    "state_sums": format_float_sequence(target_sums),
                    "compression_r2": float(r2),
                    "expected_reward": float(trace.expected_rewards[eval_index]),
                    "mean_optimal_action_probability": float(
                        trace.mean_optimal_action_probs[eval_index]
                    ),
                    "average_reward": float(trace.average_rewards[eval_index]),
                    "ceiling_reward": float(ceiling),
                }
            )
    return rows


def ideal_sum_decoder_ceiling(config: Paper52Config) -> float:
    return sum(
        probability * max(rewards)
        for probability, rewards in zip(
            config.state_distribution,
            config.reward_table,
            strict=True,
        )
    )


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    final_step = max(int(row["step"]) for row in rows)
    grouped: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if (
            int(row["step"]) == final_step
            and int(row["observer_index"]) == 0
        ):
            grouped[float(row["observer_learning_rate"])].append(row)

    summaries: list[dict[str, Any]] = []
    for observer_learning_rate, lr_rows in sorted(grouped.items()):
        seed_rows = list(lr_rows)
        final_expected = fmean(float(row["expected_reward"]) for row in seed_rows)
        final_average = fmean(float(row["average_reward"]) for row in seed_rows)
        final_optimal_action_probability = fmean(
            float(row["mean_optimal_action_probability"]) for row in seed_rows
        )
        ceiling = fmean(float(row["ceiling_reward"]) for row in seed_rows)
        threshold = 0.95 * ceiling
        success = fmean(
            1.0 if float(row["expected_reward"]) >= threshold else 0.0
            for row in seed_rows
        )
        full_sum_rows = [
            row
            for row in rows
            if float(row["observer_learning_rate"]) == observer_learning_rate
            and int(row["step"]) == final_step
            and str(row["observer_name"]) == "full_sum"
        ]
        final_full_sum_probability = fmean(
            float(row["selection_probability"]) for row in full_sum_rows
        )
        final_full_sum_r2 = fmean(float(row["compression_r2"]) for row in full_sum_rows)
        summaries.append(
            {
                "observer_learning_rate": observer_learning_rate,
                "following_learning_rate": float(seed_rows[0]["following_learning_rate"]),
                "final_expected_reward": final_expected,
                "final_average_reward": final_average,
                "final_optimal_action_probability": final_optimal_action_probability,
                "success_rate": success,
                "final_full_sum_probability": final_full_sum_probability,
                "final_full_sum_r2": final_full_sum_r2,
                "ceiling_reward": ceiling,
            }
        )
    return summaries


def write_figure(rows: list[dict[str, Any]], output_base: Path) -> None:
    unique_rows = [row for row in rows if int(row["observer_index"]) == 0]
    observer_learning_rates = sorted(
        {float(row["observer_learning_rate"]) for row in unique_rows}
    )
    ceiling = max(float(row["ceiling_reward"]) for row in unique_rows)
    display_lr = 0.01 if 0.01 in observer_learning_rates else observer_learning_rates[-1]
    display_rows = [
        row for row in rows if float(row["observer_learning_rate"]) == display_lr
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 7.0))

    for axis, metric, title, ylabel in (
        (axes[0, 0], "expected_reward", "Expected reward", "reward"),
        (
            axes[0, 1],
            "mean_optimal_action_probability",
            "High-reward action probability",
            "probability",
        ),
    ):
        for learning_rate in observer_learning_rates:
            series = mean_series(
                unique_rows,
                lambda row, learning_rate=learning_rate: (
                    float(row["observer_learning_rate"]) == learning_rate
                ),
                metric,
            )
            axis.plot(
                [point[0] for point in series],
                [point[1] for point in series],
                linewidth=1.7,
                label=rf"$\alpha_\phi$={learning_rate:g}",
            )
        reference = ceiling if metric == "expected_reward" else 1.0
        axis.axhline(reference, color="black", linewidth=1.0, alpha=0.55)
        axis.axhline(
            0.95 * reference,
            color="black",
            linestyle="--",
            linewidth=0.9,
            alpha=0.45,
        )
        axis.set_title(title)
        axis.set_xlabel("training step")
        axis.set_ylabel(ylabel)
        axis.set_ylim(-0.15, 1.05)
        axis.grid(True, axis="y", alpha=0.25)
        axis.legend(frameon=False, fontsize=8)

    observer_order = sorted(
        {
            (int(row["observer_index"]), str(row["observer_name"]))
            for row in display_rows
        }
    )
    for _, observer_name in observer_order:
        series = mean_series(
            [
                row
                for row in display_rows
                if str(row["observer_name"]) == observer_name
            ],
            lambda row: True,
            "selection_probability",
        )
        axes[0, 2].plot(
            [point[0] for point in series],
            [point[1] for point in series],
            linewidth=1.8,
            label=observer_name,
        )
    axes[0, 2].set_title(rf"Selection probabilities $\lambda_i$ ($\alpha_\phi$={display_lr:g})")
    axes[0, 2].set_xlabel("training step")
    axes[0, 2].set_ylabel("probability")
    axes[0, 2].set_ylim(0.0, 1.02)
    axes[0, 2].grid(True, axis="y", alpha=0.25)
    axes[0, 2].legend(frameon=False, fontsize=8)

    for _, observer_name in observer_order:
        series = mean_series(
            [
                row
                for row in display_rows
                if str(row["observer_name"]) == observer_name
            ],
            lambda row: True,
            "compression_r2",
        )
        axes[1, 0].plot(
            [point[0] for point in series],
            [point[1] for point in series],
            linewidth=1.8,
            label=observer_name,
        )
    axes[1, 0].set_title(r"Compression fit $R^2(m, x_1+x_2)$")
    axes[1, 0].set_xlabel("training step")
    axes[1, 0].set_ylabel(r"$R^2$")
    axes[1, 0].set_ylim(-0.05, 1.05)
    axes[1, 0].grid(True, axis="y", alpha=0.25)
    axes[1, 0].legend(frameon=False, fontsize=8)

    for _, observer_name in observer_order:
        series = mean_series(
            [
                row
                for row in display_rows
                if str(row["observer_name"]) == observer_name
            ],
            lambda row: True,
            "observer_expected_reward",
        )
        axes[1, 1].plot(
            [point[0] for point in series],
            [point[1] for point in series],
            linewidth=1.8,
            label=observer_name,
        )
    axes[1, 1].set_title("Per-Observer expected reward")
    axes[1, 1].set_xlabel("training step")
    axes[1, 1].set_ylabel("reward")
    axes[1, 1].set_ylim(-0.15, 1.05)
    axes[1, 1].grid(True, axis="y", alpha=0.25)
    axes[1, 1].legend(frameon=False, fontsize=8)

    final_step = max(int(row["step"]) for row in display_rows)
    final_rows = [
        row
        for row in display_rows
        if int(row["step"]) == final_step and str(row["observer_name"]) == "full_sum"
    ]
    grouped_messages: dict[float, list[float]] = defaultdict(list)
    for row in final_rows:
        state_sums = parse_float_sequence(str(row["state_sums"]))
        messages = parse_float_sequence(str(row["messages_by_state"]))
        for state_sum, message in zip(state_sums, messages, strict=True):
            grouped_messages[state_sum].append(message)
    target_values = sorted(grouped_messages)
    mean_messages = [fmean(grouped_messages[value]) for value in target_values]
    y_min = min(min(mean_messages), -0.05) - 0.15
    y_max = max(max(mean_messages), 2.05) + 0.15
    axes[1, 2].plot(
        [-0.05, 2.05],
        [-0.05, 2.05],
        color="black",
        linewidth=1.0,
        alpha=0.45,
    )
    axes[1, 2].scatter(target_values, mean_messages, s=55)
    axes[1, 2].set_title("Final full-sum messages")
    axes[1, 2].set_xlabel(r"target $x_1+x_2$")
    axes[1, 2].set_ylabel("message")
    axes[1, 2].set_xlim(-0.1, 2.1)
    axes[1, 2].set_ylim(y_min, y_max)
    axes[1, 2].grid(True, alpha=0.25)

    fig.suptitle("Observer Selection and Compression", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, output_base)


def mean_series(
    rows: list[dict[str, Any]],
    predicate,
    metric: str,
) -> list[tuple[int, float]]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        if predicate(row):
            grouped[int(row["step"])].append(float(row[metric]))
    return [(step, fmean(values)) for step, values in sorted(grouped.items())]


def optimal_actions(reward_table: list[list[float]]) -> list[int]:
    return [
        max(range(len(rewards)), key=lambda action: rewards[action])
        for rewards in reward_table
    ]


def r_squared(values: list[float], targets: list[float]) -> float:
    if len(values) != len(targets):
        raise ValueError("values and targets must have the same length")
    target_mean = fmean(targets)
    total_sum_squares = sum((target - target_mean) ** 2 for target in targets)
    if total_sum_squares == 0.0:
        return 1.0
    value_mean = fmean(values)
    covariance = sum(
        (value - value_mean) * (target - target_mean)
        for value, target in zip(values, targets, strict=True)
    )
    value_sum_squares = sum((value - value_mean) ** 2 for value in values)
    if value_sum_squares <= 1e-12:
        return 0.0
    return max(0.0, min(1.0, covariance**2 / (value_sum_squares * total_sum_squares)))


def format_float_sequence(values: list[float]) -> str:
    return ";".join(f"{value:.8g}" for value in values)


def parse_float_sequence(value: str) -> list[float]:
    return [float(part) for part in value.split(";") if part]


def save_figure(fig: plt.Figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png", ".svg"):
        output_path = output_base.with_suffix(suffix)
        fig.savefig(output_path, bbox_inches="tight")
        if suffix == ".svg":
            strip_trailing_whitespace(output_path)
    plt.close(fig)


def strip_trailing_whitespace(path: Path) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text(
        "\n".join(line.rstrip() for line in lines) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def print_summary(summaries: list[dict[str, Any]]) -> None:
    print(
        "alpha_phi | final_expected | final_sampled | p_action | success | p_full | r2_full"
    )
    print(
        "--------- | -------------- | ------------- | -------- | ------- | ------ | -------"
    )
    for row in summaries:
        print(
            f"{row['observer_learning_rate']:g} | "
            f"{row['final_expected_reward']:.3f} | "
            f"{row['final_average_reward']:.3f} | "
            f"{row['final_optimal_action_probability']:.3f} | "
            f"{row['success_rate']:.2f} | "
            f"{row['final_full_sum_probability']:.3f} | "
            f"{row['final_full_sum_r2']:.3f}"
        )


if __name__ == "__main__":
    main()
