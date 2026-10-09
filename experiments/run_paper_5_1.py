"""Run paper Section 5.1 fixed-Observer selection experiments."""

from __future__ import annotations

import argparse
import csv
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
    FixedTableMessageModule,
    FollowingRates,
    MultiObserver,
    Observer,
)
from actor_observer_rl.envs import PartialObservableStateEnv
from actor_observer_rl.policies import SumDecoderPolicy
from actor_observer_rl.training import StagedMultiObserverTrainer
from actor_observer_rl.utils.config import Paper51Config, load_yaml_config
from actor_observer_rl.utils.seeding import set_global_seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/paper_5_1_observer_selection.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/paper_5_1"),
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=Path("docs/figures"),
    )
    parser.add_argument(
        "--learning-rates",
        type=float,
        nargs="+",
        default=[0.001, 0.01, 0.05, 0.1],
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = parser.parse_args()

    config = Paper51Config.from_mapping(load_yaml_config(args.config))
    rows = run_sweep(config, learning_rates=args.learning_rates, seeds=args.seeds)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    curve_path = args.output_dir / "observer_selection_curves.csv"
    summary_path = args.output_dir / "observer_selection_summary.csv"
    write_csv(curve_path, rows)
    summaries = summarize(rows)
    write_csv(summary_path, summaries)
    write_figure(rows, args.figure_dir / "paper5_1_observer_selection")

    print(f"wrote_curves={curve_path}")
    print(f"wrote_summary={summary_path}")
    print(f"wrote_figure={args.figure_dir / 'paper5_1_observer_selection'}.[pdf|png|svg]")
    print_summary(summaries)


def run_sweep(
    config: Paper51Config,
    learning_rates: list[float],
    seeds: list[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for learning_rate in learning_rates:
        for seed in seeds:
            trainer = build_trainer(config, learning_rate=learning_rate, seed=seed)
            trace = trainer.run(
                num_steps=config.num_steps,
                seed=seed,
                eval_interval=config.eval_interval,
                reward_window=config.reward_window,
            )
            rows.extend(trace_to_rows(config, trace, learning_rate, seed))
    return rows


def build_trainer(
    config: Paper51Config,
    learning_rate: float,
    seed: int,
) -> StagedMultiObserverTrainer:
    set_global_seed(seed)
    env = PartialObservableStateEnv(
        reward_table=config.reward_table,
        state_distribution=config.state_distribution,
        observation_dim=config.observation_dim,
    )
    observers = [
        Observer(message_module=FixedTableMessageModule(messages))
        for messages in config.observer_messages
    ]
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
        lr=learning_rate,
    )
    return StagedMultiObserverTrainer(
        env=env,
        actor=actor,
        multi_observer=multi_observer,
        following_optimizer=following_optimizer,
        optimal_actions=optimal_actions(config.reward_table),
    )


def trace_to_rows(
    config: Paper51Config,
    trace,
    learning_rate: float,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ceiling = max(trace.per_observer_expected_rewards[0])
    for eval_index, step in enumerate(trace.evaluation_steps):
        if step == 0:
            weights = [1.0 / config.num_observers] * config.num_observers
        else:
            weights = trace.following_weights[step - 1]
        for observer_index, observer_name in enumerate(config.observer_names):
            rows.append(
                {
                    "experiment": "paper_5_1_observer_selection",
                    "learning_rate": float(learning_rate),
                    "seed": int(seed),
                    "step": int(step),
                    "observer_index": int(observer_index),
                    "observer_name": observer_name,
                    "selection_probability": float(weights[observer_index]),
                    "observer_expected_reward": float(
                        trace.per_observer_expected_rewards[eval_index][observer_index]
                    ),
                    "observer_optimal_action_probability": float(
                        trace.per_observer_optimal_action_probs[eval_index][
                            observer_index
                        ]
                    ),
                    "expected_reward": float(trace.expected_rewards[eval_index]),
                    "mean_optimal_action_probability": float(
                        trace.mean_optimal_action_probs[eval_index]
                    ),
                    "average_reward": float(trace.average_rewards[eval_index]),
                    "ceiling_reward": float(ceiling),
                }
            )
    return rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    final_step = max(int(row["step"]) for row in rows)
    grouped: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if int(row["step"]) == final_step and int(row["observer_index"]) == 0:
            grouped[float(row["learning_rate"])].append(row)

    summaries: list[dict[str, Any]] = []
    for learning_rate, lr_rows in sorted(grouped.items()):
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
        sum_prob_rows = [
            row
            for row in rows
            if float(row["learning_rate"]) == learning_rate
            and int(row["step"]) == final_step
            and str(row["observer_name"]) == "sum"
        ]
        final_sum_probability = fmean(
            float(row["selection_probability"]) for row in sum_prob_rows
        )
        summaries.append(
            {
                "learning_rate": learning_rate,
                "final_expected_reward": final_expected,
                "final_average_reward": final_average,
                "final_optimal_action_probability": final_optimal_action_probability,
                "success_rate": success,
                "final_sum_observer_probability": final_sum_probability,
                "ceiling_reward": ceiling,
            }
        )
    return summaries


def write_figure(rows: list[dict[str, Any]], output_base: Path) -> None:
    unique_rows = [row for row in rows if int(row["observer_index"]) == 0]
    learning_rates = sorted({float(row["learning_rate"]) for row in unique_rows})
    ceiling = max(float(row["ceiling_reward"]) for row in unique_rows)
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.4))

    for axis, metric, label, ylabel in [
        (axes[0], "expected_reward", "Expected reward", "reward"),
        (
            axes[1],
            "mean_optimal_action_probability",
            "High-reward action probability",
            "probability",
        ),
    ]:
        for learning_rate in learning_rates:
            series = mean_series(
                unique_rows,
                lambda row, learning_rate=learning_rate: (
                    float(row["learning_rate"]) == learning_rate
                ),
                metric,
            )
            axis.plot(
                [point[0] for point in series],
                [point[1] for point in series],
                linewidth=1.7,
                label=f"lr={learning_rate:g}",
            )
        reference = ceiling if metric == "expected_reward" else 1.0
        axis.axhline(reference, color="black", linewidth=1.0, alpha=0.55)
        axis.axhline(0.95 * reference, color="black", linestyle="--", linewidth=0.9, alpha=0.45)
        axis.set_title(label)
        axis.set_xlabel("training step")
        axis.set_ylabel(ylabel)
        axis.set_ylim(-0.15, 1.05)
        axis.grid(True, axis="y", alpha=0.25)
        axis.legend(frameon=False, fontsize=7)

    display_lr = 0.05 if 0.05 in learning_rates else learning_rates[-1]
    selection_rows = [
        row for row in rows if float(row["learning_rate"]) == display_lr
    ]
    observer_order = sorted(
        {
            (int(row["observer_index"]), str(row["observer_name"]))
            for row in selection_rows
        }
    )
    for _, observer_name in observer_order:
        series = mean_series(
            selection_rows,
            lambda row, observer_name=observer_name: (
                str(row["observer_name"]) == observer_name
            ),
            "selection_probability",
        )
        axes[2].plot(
            [point[0] for point in series],
            [point[1] for point in series],
            linewidth=1.9,
            label=observer_name,
        )
    axes[2].set_title(rf"Selection probabilities $\lambda_i$ (lr={display_lr:g})")
    axes[2].set_xlabel("training step")
    axes[2].set_ylabel("probability")
    axes[2].set_ylim(0.0, 1.02)
    axes[2].grid(True, axis="y", alpha=0.25)
    axes[2].legend(frameon=False, fontsize=8)

    fig.suptitle("Fixed-Observer Selection", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
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
    print("lr | final_expected | final_sampled | p_action | success | p_sum | ceiling")
    print("-- | -------------- | ------------- | -------- | ------- | ----- | -------")
    for row in summaries:
        print(
            f"{row['learning_rate']:g} | "
            f"{row['final_expected_reward']:.3f} | "
            f"{row['final_average_reward']:.3f} | "
            f"{row['final_optimal_action_probability']:.3f} | "
            f"{row['success_rate']:.2f} | "
            f"{row['final_sum_observer_probability']:.3f} | "
            f"{row['ceiling_reward']:.3f}"
        )


if __name__ == "__main__":
    main()
