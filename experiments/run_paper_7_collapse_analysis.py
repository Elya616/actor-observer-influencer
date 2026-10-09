"""Analyze partial-Observer collapse in paper Section 7 experiments."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from statistics import fmean
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path("/private/tmp/actor_observer_rl_mplcache").resolve()),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from actor_observer_rl.utils.config import Paper72Config, Paper73Config, load_yaml_config
from experiments.run_paper_7 import save_figure
from experiments.run_paper_7_stabilization import (
    FULL_OBSERVER_NAME,
    PARTIAL_OBSERVER_NAME,
    build_section_7_2_trainer,
    build_section_7_3_trainer,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-7-2",
        type=Path,
        default=Path("configs/paper_7_2_group_selection_compression.yaml"),
    )
    parser.add_argument(
        "--config-7-3",
        type=Path,
        default=Path("configs/paper_7_3_group_joint_imperfect_state.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/paper_7_collapse_analysis"),
    )
    parser.add_argument("--figure-dir", type=Path, default=Path("docs/figures"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = parser.parse_args()

    config_7_2 = replace(
        Paper72Config.from_mapping(load_yaml_config(args.config_7_2)),
        following_learning_rate=0.005,
        observer_learning_rate=0.01,
        num_steps=20_000,
    )
    config_7_3 = replace(
        Paper73Config.from_mapping(load_yaml_config(args.config_7_3)),
        following_learning_rate=0.005,
        observer_learning_rate=0.05,
        actor_learning_rate=0.1,
        num_steps=20_000,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    curve_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []

    experiments = [
        (
            "7.2",
            "7.2 fixed Actor decoder",
            config_7_2,
            lambda seed: build_section_7_2_trainer(
                config_7_2,
                seed=seed,
                selection_update_rule="raw_reward",
            ),
        ),
        (
            "7.3",
            "7.3 raw reward",
            config_7_3,
            lambda seed: build_section_7_3_trainer(
                config_7_3,
                seed=seed,
                selection_update_rule="raw_reward",
            ),
        ),
        (
            "7.3",
            "7.3 per-Actor advantage",
            config_7_3,
            lambda seed: build_section_7_3_trainer(
                config_7_3,
                seed=seed,
                selection_update_rule="advantage_per_actor",
            ),
        ),
    ]

    for section, experiment, config, build_trainer in experiments:
        print(f"running {experiment}", flush=True)
        for seed in args.seeds:
            trainer = build_trainer(seed)
            trace = trainer.run(
                num_steps=config.num_steps,
                seed=seed,
                eval_interval=config.eval_interval,
                reward_window=config.reward_window,
            )
            curve_rows.extend(
                collapse_rows_from_trace(
                    config=config,
                    trace=trace,
                    section=section,
                    experiment=experiment,
                    seed=seed,
                )
            )
            seed_rows.append(
                seed_summary_from_trace(
                    config=config,
                    trace=trace,
                    section=section,
                    experiment=experiment,
                    seed=seed,
                )
            )

    summary_rows = summarize_seed_rows(seed_rows)
    write_csv(args.output_dir / "collapse_curves.csv", curve_rows)
    write_csv(args.output_dir / "collapse_seed_summary.csv", seed_rows)
    write_csv(args.output_dir / "collapse_summary.csv", summary_rows)

    write_matrix_figure(
        curve_rows,
        args.figure_dir / "paper7_collapse_matrices",
    )
    write_failed_timeline_figure(
        curve_rows,
        seed_rows,
        args.figure_dir / "paper7_collapse_failed_timeline",
    )

    for row in summary_rows:
        print(
            "{experiment}: final_expected={final_expected_reward:.3f} "
            "min_p_full={final_min_full_probability:.3f} "
            "min_full_exposure={final_min_full_exposure:.3f} "
            "collapse_pairs={collapsed_actor_seed_pairs:.0f}".format(**row)
        )


def collapse_rows_from_trace(
    *,
    config: Any,
    trace: Any,
    section: str,
    experiment: str,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    num_actors = int(config.num_actors)
    num_observers = int(config.num_observers)
    active_counts = [0] * num_actors
    selected_counts = [[0] * num_observers for _ in range(num_actors)]
    gradient_sums = [[0.0] * num_observers for _ in range(num_actors)]
    step_cursor = 0

    for eval_index, step in enumerate(trace.evaluation_steps):
        while step_cursor < int(step):
            actor_index = trace.selected_actor_indices[step_cursor]
            observer_index = trace.selected_observer_indices[step_cursor]
            active_counts[actor_index] += 1
            selected_counts[actor_index][observer_index] += 1
            gradient_norms = trace.observer_gradient_norms[step_cursor]
            for gradient_observer_index, gradient_norm in enumerate(gradient_norms):
                gradient_sums[actor_index][gradient_observer_index] += float(
                    gradient_norm
                )
            step_cursor += 1

        for actor_index in range(num_actors):
            actor_gradient_total = sum(gradient_sums[actor_index])
            for observer_index, observer_name in enumerate(config.observer_names):
                active_count = active_counts[actor_index]
                selection_count = selected_counts[actor_index][observer_index]
                exposure = selection_count / active_count if active_count else 0.0
                gradient_sum = gradient_sums[actor_index][observer_index]
                gradient_share = (
                    gradient_sum / actor_gradient_total
                    if actor_gradient_total > 0.0
                    else 0.0
                )
                rows.append(
                    {
                        "section": section,
                        "experiment": experiment,
                        "seed": int(seed),
                        "step": int(step),
                        "actor_index": int(actor_index),
                        "observer_index": int(observer_index),
                        "observer_name": observer_name,
                        "active_count": int(active_count),
                        "selection_count": int(selection_count),
                        "selection_exposure": float(exposure),
                        "selection_probability": float(
                            trace.selection_weights_by_actor[eval_index][actor_index][
                                observer_index
                            ]
                        ),
                        "actor_observer_expected_reward": float(
                            trace.per_actor_observer_expected_rewards[eval_index][
                                actor_index
                            ][observer_index]
                        ),
                        "actor_expected_reward": float(
                            trace.per_actor_expected_rewards[eval_index][actor_index]
                        ),
                        "observer_gradient_norm_sum": float(gradient_sum),
                        "observer_gradient_norm_share": float(gradient_share),
                        "expected_reward": float(trace.expected_rewards[eval_index]),
                        "mean_optimal_action_probability": float(
                            trace.mean_optimal_action_probs[eval_index]
                        ),
                    }
                )
    return rows


def seed_summary_from_trace(
    *,
    config: Any,
    trace: Any,
    section: str,
    experiment: str,
    seed: int,
) -> dict[str, Any]:
    final_selection = trace.final_selection_weights_by_actor
    final_observer_rewards = trace.final_per_actor_observer_expected_rewards
    if final_selection is None or final_observer_rewards is None:
        raise ValueError("trace is missing final evaluation data")
    full_index = config.observer_names.index(FULL_OBSERVER_NAME)
    partial_index = config.observer_names.index(PARTIAL_OBSERVER_NAME)

    active_counts = [0] * config.num_actors
    selected_counts = [[0] * config.num_observers for _ in range(config.num_actors)]
    gradient_sums = [[0.0] * config.num_observers for _ in range(config.num_actors)]
    for step_index, actor_index in enumerate(trace.selected_actor_indices):
        observer_index = trace.selected_observer_indices[step_index]
        active_counts[actor_index] += 1
        selected_counts[actor_index][observer_index] += 1
        for gradient_observer_index, gradient_norm in enumerate(
            trace.observer_gradient_norms[step_index]
        ):
            gradient_sums[actor_index][gradient_observer_index] += float(gradient_norm)

    full_probs = [row[full_index] for row in final_selection]
    partial_probs = [row[partial_index] for row in final_selection]
    full_exposures = [
        selected_counts[actor_index][full_index] / active_counts[actor_index]
        for actor_index in range(config.num_actors)
    ]
    partial_exposures = [
        selected_counts[actor_index][partial_index] / active_counts[actor_index]
        for actor_index in range(config.num_actors)
    ]
    full_reward_edges = [
        final_observer_rewards[actor_index][full_index]
        - final_observer_rewards[actor_index][partial_index]
        for actor_index in range(config.num_actors)
    ]
    collapsed_actors = [
        actor_index
        for actor_index in range(config.num_actors)
        if full_probs[actor_index] < 0.5
        or partial_probs[actor_index] > full_probs[actor_index]
    ]

    return {
        "section": section,
        "experiment": experiment,
        "seed": int(seed),
        "final_expected_reward": float(trace.final_expected_reward or 0.0),
        "final_optimal_action_probability": float(
            trace.final_mean_optimal_action_prob or 0.0
        ),
        "final_mean_full_probability": float(fmean(full_probs)),
        "final_min_full_probability": float(min(full_probs)),
        "final_mean_full_exposure": float(fmean(full_exposures)),
        "final_min_full_exposure": float(min(full_exposures)),
        "final_mean_partial_exposure": float(fmean(partial_exposures)),
        "final_min_full_reward_edge": float(min(full_reward_edges)),
        "collapsed_actor_count": int(len(collapsed_actors)),
        "collapsed_actor_indices": ";".join(str(index) for index in collapsed_actors),
        "total_full_gradient_norm": float(
            sum(row[full_index] for row in gradient_sums)
        ),
        "total_partial_gradient_norm": float(
            sum(row[partial_index] for row in gradient_sums)
        ),
    }


def summarize_seed_rows(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in seed_rows:
        grouped[str(row["experiment"])].append(row)

    summaries: list[dict[str, Any]] = []
    for experiment, rows in grouped.items():
        summaries.append(
            {
                "section": rows[0]["section"],
                "experiment": experiment,
                "num_seeds": len(rows),
                "final_expected_reward": fmean(
                    float(row["final_expected_reward"]) for row in rows
                ),
                "final_optimal_action_probability": fmean(
                    float(row["final_optimal_action_probability"]) for row in rows
                ),
                "final_mean_full_probability": fmean(
                    float(row["final_mean_full_probability"]) for row in rows
                ),
                "final_min_full_probability": min(
                    float(row["final_min_full_probability"]) for row in rows
                ),
                "final_mean_full_exposure": fmean(
                    float(row["final_mean_full_exposure"]) for row in rows
                ),
                "final_min_full_exposure": min(
                    float(row["final_min_full_exposure"]) for row in rows
                ),
                "final_mean_partial_exposure": fmean(
                    float(row["final_mean_partial_exposure"]) for row in rows
                ),
                "final_min_full_reward_edge": min(
                    float(row["final_min_full_reward_edge"]) for row in rows
                ),
                "collapsed_actor_seed_pairs": sum(
                    int(row["collapsed_actor_count"]) for row in rows
                ),
                "total_full_gradient_norm": sum(
                    float(row["total_full_gradient_norm"]) for row in rows
                ),
                "total_partial_gradient_norm": sum(
                    float(row["total_partial_gradient_norm"]) for row in rows
                ),
            }
        )
    return summaries


def write_matrix_figure(rows: list[dict[str, Any]], output_base: Path) -> None:
    experiments = [
        "7.2 fixed Actor decoder",
        "7.3 raw reward",
        "7.3 per-Actor advantage",
    ]
    metrics = [
        ("selection_probability", r"final $\lambda_{j,i}$"),
        ("actor_observer_expected_reward", r"quality $Q_{j,i}$"),
        ("observer_gradient_norm_share", "compression-gradient share"),
    ]
    fig, axes = plt.subplots(
        len(experiments),
        len(metrics),
        figsize=(12.2, 8.2),
        constrained_layout=True,
    )
    for row_index, experiment in enumerate(experiments):
        for col_index, (metric, title) in enumerate(metrics):
            axis = axes[row_index, col_index]
            matrix = final_mean_matrix(rows, experiment, metric)
            axis.imshow(matrix, vmin=0.0, vmax=1.0, cmap="Blues")
            for actor_index, values in enumerate(matrix):
                for observer_index, value in enumerate(values):
                    axis.text(
                        observer_index,
                        actor_index,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                    )
            if row_index == 0:
                axis.set_title(title)
            axis.set_xticks([0, 1, 2])
            axis.set_xticklabels(["zero", "partial", "full"], rotation=25)
            axis.set_yticks([0, 1, 2])
            axis.set_yticklabels(["Actor 0", "Actor 1", "Actor 2"])
            if col_index == 0:
                axis.set_ylabel(experiment)
    fig.suptitle("Section 7 Collapse Analysis: Selection, Quality, and Training Signal")
    save_figure(fig, output_base)


def final_mean_matrix(
    rows: list[dict[str, Any]],
    experiment: str,
    metric: str,
) -> list[list[float]]:
    experiment_rows = [row for row in rows if str(row["experiment"]) == experiment]
    final_step = max(int(row["step"]) for row in experiment_rows)
    grouped: dict[tuple[int, int], list[float]] = defaultdict(list)
    for row in experiment_rows:
        if int(row["step"]) == final_step:
            grouped[(int(row["actor_index"]), int(row["observer_index"]))].append(
                float(row[metric])
            )
    return [
        [fmean(grouped[(actor_index, observer_index)]) for observer_index in range(3)]
        for actor_index in range(3)
    ]


def write_failed_timeline_figure(
    rows: list[dict[str, Any]],
    seed_rows: list[dict[str, Any]],
    output_base: Path,
) -> None:
    candidates = [
        row for row in seed_rows if str(row["experiment"]) == "7.3 raw reward"
    ]
    worst_seed = min(candidates, key=lambda row: float(row["final_min_full_probability"]))[
        "seed"
    ]
    raw_rows = [
        row
        for row in rows
        if str(row["experiment"]) == "7.3 raw reward" and int(row["seed"]) == worst_seed
    ]
    final_step = max(int(row["step"]) for row in raw_rows)
    final_rows = [row for row in raw_rows if int(row["step"]) == final_step]
    full_rows = [row for row in final_rows if str(row["observer_name"]) == FULL_OBSERVER_NAME]
    worst_actor = min(full_rows, key=lambda row: float(row["selection_probability"]))[
        "actor_index"
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 6.8))
    plot_actor_observer_series(
        axes[0, 0],
        raw_rows,
        int(worst_actor),
        "selection_probability",
        r"selection probability $\lambda_{j,i}$",
    )
    plot_actor_observer_series(
        axes[0, 1],
        raw_rows,
        int(worst_actor),
        "selection_exposure",
        "empirical selection exposure",
    )
    plot_actor_observer_series(
        axes[1, 0],
        raw_rows,
        int(worst_actor),
        "actor_observer_expected_reward",
        r"Observer quality $Q_{j,i}$",
    )
    plot_actor_observer_series(
        axes[1, 1],
        raw_rows,
        int(worst_actor),
        "observer_gradient_norm_sum",
        "cumulative compression-gradient norm",
    )
    fig.suptitle(
        f"Failed-Seed Timeline: 7.3 Raw Reward, Seed {worst_seed}, Actor {worst_actor}"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, output_base)


def plot_actor_observer_series(
    axis: plt.Axes,
    rows: list[dict[str, Any]],
    actor_index: int,
    metric: str,
    title: str,
) -> None:
    labels = [
        ("zero_input", "zero"),
        (PARTIAL_OBSERVER_NAME, "partial"),
        (FULL_OBSERVER_NAME, "full"),
    ]
    for observer_name, label in labels:
        series = [
            (int(row["step"]), float(row[metric]))
            for row in rows
            if int(row["actor_index"]) == actor_index
            and str(row["observer_name"]) == observer_name
        ]
        series.sort()
        axis.plot(
            [point[0] for point in series],
            [point[1] for point in series],
            linewidth=1.8,
            label=label,
        )
    axis.set_title(title)
    axis.set_xlabel("training step")
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend(frameon=False, fontsize=8)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV to {path}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
