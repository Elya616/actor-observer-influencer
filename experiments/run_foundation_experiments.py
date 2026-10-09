"""Run focused Actor-Observer verification experiments."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import fmean

import matplotlib.pyplot as plt

from actor_observer_rl.evaluation import (
    default_foundation_experiments,
    run_foundation_sweep,
    summarize_foundation_runs,
    write_aggregate_summary_csv,
    write_curve_csv,
    write_run_summary_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("outputs/foundation"),
        help="Prefix for generated CSV outputs.",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("docs/figures/foundation_learning_curves.svg"),
        help="Path for the generated learning-curve figure.",
    )
    parser.add_argument("--num-steps", type=int, default=5000)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--reward-window", type=int, default=100)
    parser.add_argument("--threshold-fraction", type=float, default=0.95)
    parser.add_argument(
        "--learning-rates",
        type=float,
        nargs="+",
        default=[0.001, 0.01, 0.05, 0.1],
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = parser.parse_args()

    experiments = default_foundation_experiments()
    curves, run_summaries = run_foundation_sweep(
        experiments=experiments,
        learning_rates=args.learning_rates,
        seeds=args.seeds,
        num_steps=args.num_steps,
        eval_interval=args.eval_interval,
        reward_window=args.reward_window,
        threshold_fraction=args.threshold_fraction,
    )
    aggregate_summaries = summarize_foundation_runs(run_summaries)

    curve_path = args.output_prefix.with_name(f"{args.output_prefix.name}_curves.csv")
    run_path = args.output_prefix.with_name(f"{args.output_prefix.name}_runs.csv")
    summary_path = args.output_prefix.with_name(f"{args.output_prefix.name}_summary.csv")
    write_curve_csv(curves, curve_path)
    write_run_summary_csv(run_summaries, run_path)
    write_aggregate_summary_csv(aggregate_summaries, summary_path)
    _write_plot(curve_path, args.figure)

    print(f"wrote_curves={curve_path}")
    print(f"wrote_runs={run_path}")
    print(f"wrote_summary={summary_path}")
    print(f"wrote_figure={args.figure}")
    _print_summary_table(aggregate_summaries)


def _write_plot(curve_path: Path, figure_path: Path) -> None:
    rows = _read_curve_rows(curve_path)
    experiments = [
        "single_state_action_learning",
        "state_dependent_action_learning",
    ]
    experiment_labels = {
        "single_state_action_learning": "Single-State Action Learning",
        "state_dependent_action_learning": "State-Dependent Action Learning",
    }
    metric_specs = [
        ("expected_reward", "Expected reward"),
        ("average_reward_last_window", "Average sampled reward"),
    ]

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        len(experiments),
        len(metric_specs),
        figsize=(9.0, 5.8),
        sharex=True,
        sharey=True,
    )
    for row_index, experiment in enumerate(experiments):
        for column_index, (metric, metric_label) in enumerate(metric_specs):
            ax = axes[row_index][column_index]
            for learning_rate in sorted(
                {
                    row["learning_rate"]
                    for row in rows
                    if row["experiment"] == experiment
                }
            ):
                series = _mean_curve(rows, experiment, learning_rate, metric)
                ax.plot(
                    [point[0] for point in series],
                    [point[1] for point in series],
                    linewidth=1.7,
                    label=f"lr={learning_rate:g}",
                )

            reference = next(row for row in rows if row["experiment"] == experiment)
            ax.axhline(
                reference["ceiling_reward"],
                color="black",
                linewidth=1.0,
                linestyle="-",
                alpha=0.65,
                label="ceiling" if row_index == 0 and column_index == 0 else None,
            )
            ax.axhline(
                reference["threshold_reward"],
                color="black",
                linewidth=1.0,
                linestyle="--",
                alpha=0.55,
                label="95% threshold" if row_index == 0 and column_index == 0 else None,
            )
            ax.set_title(f"{experiment_labels[experiment]}: {metric_label}", fontsize=10)
            ax.set_ylim(0.0, 1.05)
            ax.grid(True, axis="y", alpha=0.25)
            if column_index == 0:
                ax.set_ylabel("reward")
            if row_index == len(experiments) - 1:
                ax.set_xlabel("training step")
            if row_index == 0 and column_index == len(metric_specs) - 1:
                ax.legend(fontsize=7, frameon=False, loc="lower right")

    fig.suptitle("Foundational Actor-Observer Learning Curves", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(figure_path, format="svg")
    plt.close(fig)


def _read_curve_rows(path: Path) -> list[dict[str, float | str]]:
    with path.open("r", encoding="utf-8") as file:
        rows = []
        for row in csv.DictReader(file):
            rows.append(
                {
                    "experiment": row["experiment"],
                    "learning_rate": float(row["learning_rate"]),
                    "seed": int(row["seed"]),
                    "step": int(row["step"]),
                    "expected_reward": float(row["expected_reward"]),
                    "average_reward_last_window": float(
                        row["average_reward_last_window"]
                    ),
                    "ceiling_reward": float(row["ceiling_reward"]),
                    "threshold_reward": float(row["threshold_reward"]),
                }
            )
    return rows


def _mean_curve(
    rows: list[dict[str, float | str]],
    experiment: str,
    learning_rate: float,
    metric: str,
) -> list[tuple[int, float]]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        if row["experiment"] == experiment and row["learning_rate"] == learning_rate:
            grouped[int(row["step"])].append(float(row[metric]))
    return [(step, fmean(values)) for step, values in sorted(grouped.items())]


def _print_summary_table(summaries) -> None:
    header = (
        "experiment",
        "lr",
        "success",
        "final_expected",
        "final_avg_reward",
        "mean_conv_step",
    )
    print(" | ".join(header))
    print(" | ".join("-" * len(column) for column in header))
    for summary in summaries:
        convergence = (
            "NA"
            if summary.mean_convergence_step is None
            else f"{summary.mean_convergence_step:.1f}"
        )
        print(
            " | ".join(
                [
                    summary.experiment,
                    f"{summary.learning_rate:g}",
                    f"{summary.success_rate:.2f}",
                    f"{summary.mean_final_expected_reward:.3f}",
                    f"{summary.mean_final_average_reward_last_window:.3f}",
                    convergence,
                ]
            )
        )


if __name__ == "__main__":
    main()
