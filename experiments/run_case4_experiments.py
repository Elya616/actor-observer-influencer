"""Run focused multi-observer Actor-Observer experiments."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import fmean

import matplotlib.pyplot as plt

from actor_observer_rl.evaluation import (
    default_multi_observer_experiment,
    run_multi_observer_sweep,
    summarize_multi_observer_runs,
    write_multi_observer_curve_csv,
    write_multi_observer_run_csv,
    write_multi_observer_summary_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("outputs/case4"),
        help="Prefix for generated CSV outputs.",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("docs/figures/case4_learning_curves.svg"),
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

    experiment = default_multi_observer_experiment()
    curves, run_summaries = run_multi_observer_sweep(
        experiment=experiment,
        learning_rates=args.learning_rates,
        seeds=args.seeds,
        num_steps=args.num_steps,
        eval_interval=args.eval_interval,
        reward_window=args.reward_window,
        threshold_fraction=args.threshold_fraction,
    )
    aggregate_summaries = summarize_multi_observer_runs(run_summaries)

    curve_path = args.output_prefix.with_name(f"{args.output_prefix.name}_curves.csv")
    run_path = args.output_prefix.with_name(f"{args.output_prefix.name}_runs.csv")
    summary_path = args.output_prefix.with_name(f"{args.output_prefix.name}_summary.csv")
    write_multi_observer_curve_csv(curves, curve_path)
    write_multi_observer_run_csv(run_summaries, run_path)
    write_multi_observer_summary_csv(aggregate_summaries, summary_path)
    _write_plot(curve_path, args.figure, num_observers=experiment.num_observers)

    print(f"wrote_curves={curve_path}")
    print(f"wrote_runs={run_path}")
    print(f"wrote_summary={summary_path}")
    print(f"wrote_figure={args.figure}")
    _print_summary_table(aggregate_summaries)


def _write_plot(curve_path: Path, figure_path: Path, num_observers: int) -> None:
    rows = _read_curve_rows(curve_path, num_observers=num_observers)
    metric_specs = [
        ("expected_reward", "Expected reward"),
        ("average_reward_last_window", "Average sampled reward"),
    ]

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.4))

    learning_rates = sorted({row["learning_rate"] for row in rows})
    for axis, (metric, metric_label) in zip(axes[:2], metric_specs, strict=True):
        for learning_rate in learning_rates:
            series = _mean_curve(rows, learning_rate, metric)
            axis.plot(
                [point[0] for point in series],
                [point[1] for point in series],
                linewidth=1.7,
                label=f"lr={learning_rate:g}",
            )

        reference = rows[0]
        axis.axhline(
            reference["ceiling_reward"],
            color="black",
            linewidth=1.0,
            linestyle="-",
            alpha=0.65,
            label="ceiling" if metric == "expected_reward" else None,
        )
        axis.axhline(
            reference["threshold_reward"],
            color="black",
            linewidth=1.0,
            linestyle="--",
            alpha=0.55,
            label="95% threshold" if metric == "expected_reward" else None,
        )
        axis.set_title(metric_label, fontsize=10)
        axis.set_ylim(0.0, 1.05)
        axis.grid(True, axis="y", alpha=0.25)
        axis.set_xlabel("training step")

    axes[0].set_ylabel("reward")
    axes[1].legend(fontsize=7, frameon=False, loc="lower right")

    weight_axis = axes[2]
    for learning_rate in learning_rates:
        series = _mean_max_following_curve(rows, learning_rate)
        weight_axis.plot(
            [point[0] for point in series],
            [point[1] for point in series],
            linewidth=1.7,
            label=f"lr={learning_rate:g}",
        )
    weight_axis.axhline(
        1.0 / num_observers,
        color="black",
        linewidth=1.0,
        linestyle=":",
        alpha=0.6,
        label="uniform",
    )
    weight_axis.set_title(
        "Max following weight (collapse indicator)",
        fontsize=10,
    )
    weight_axis.set_ylim(0.0, 1.05)
    weight_axis.grid(True, axis="y", alpha=0.25)
    weight_axis.set_xlabel("training step")
    weight_axis.set_ylabel("max softmax(lambda)")
    weight_axis.legend(fontsize=7, frameon=False, loc="lower right")

    fig.suptitle("Multi-Observer Following Rate Learning", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(figure_path, format="svg")
    plt.close(fig)


def _read_curve_rows(
    path: Path,
    num_observers: int,
) -> list[dict[str, float | str | list[float]]]:
    with path.open("r", encoding="utf-8") as file:
        rows: list[dict[str, float | str | list[float]]] = []
        for row in csv.DictReader(file):
            weights = [float(value) for value in row["following_weights"].split("|")]
            if len(weights) != num_observers:
                raise ValueError(
                    f"expected {num_observers} following weights, got {len(weights)}"
                )
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
                    "mean_optimal_action_probability": float(
                        row["mean_optimal_action_probability"]
                    ),
                    "following_weights": weights,
                    "ceiling_reward": float(row["ceiling_reward"]),
                    "threshold_reward": float(row["threshold_reward"]),
                }
            )
    return rows


def _mean_curve(
    rows: list[dict[str, float | str | list[float]]],
    learning_rate: float,
    metric: str,
) -> list[tuple[int, float]]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        if row["learning_rate"] == learning_rate:
            grouped[int(row["step"])].append(float(row[metric]))
    return [(step, fmean(values)) for step, values in sorted(grouped.items())]


def _mean_max_following_curve(
    rows: list[dict[str, float | str | list[float]]],
    learning_rate: float,
) -> list[tuple[int, float]]:
    """Return mean (over seeds) of max_i softmax(lambda)_i per step.

    A flat curve at ``1 / num_observers`` means the following rates stay
    uniform; a curve rising toward ``1.0`` means each seed collapses onto
    a single observer.
    """
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        if row["learning_rate"] == learning_rate:
            weights = row["following_weights"]
            assert isinstance(weights, list)
            grouped[int(row["step"])].append(max(weights))
    return [(step, fmean(values)) for step, values in sorted(grouped.items())]


def _print_summary_table(summaries) -> None:
    header = (
        "experiment",
        "lr",
        "success",
        "final_expected",
        "final_avg_reward",
        "mean_conv_step",
        "mean_following_weights",
        "mean_selection_histogram",
    )
    print(" | ".join(header))
    print(" | ".join("-" * len(column) for column in header))
    for summary in summaries:
        convergence = (
            "NA"
            if summary.mean_convergence_step is None
            else f"{summary.mean_convergence_step:.1f}"
        )
        weights = "|".join(
            f"{weight:.3f}" for weight in summary.mean_final_following_weights
        )
        selection = "|".join(
            f"{value:.3f}" for value in summary.mean_selection_histogram
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
                    weights,
                    selection,
                ]
            )
        )


if __name__ == "__main__":
    main()
