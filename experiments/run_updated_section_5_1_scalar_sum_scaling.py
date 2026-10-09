"""Updated-paper Section 5.1 scalar-sum compression scaling.

Section 5.1 has one Actor-compressor and one Observer. The Observer policy is
fixed. The only learned object is the Actor-compressor message map ``phi``.

This script uses the current controlled imperfect-state representation:

``state k = cumulative binary vector whose sum is k``

and checks whether a scalar linear message can learn the state sum as K grows.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

import matplotlib.pyplot as plt
import torch

from actor_observer_rl.envs import FeaturePartialObservableStateEnv
from actor_observer_rl.training import (
    UpdatedImperfectStateTrainer,
    build_scalar_sum_actor_compressors,
    cumulative_binary_state_features,
    diagonal_reward_table,
    fixed_sum_observer_policy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-counts", nargs="+", type=int, default=[3, 5, 8, 16, 32])
    parser.add_argument("--learning-rates", nargs="+", type=float, default=[0.01, 0.02, 0.05, 0.1])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--eval-interval", type=int, default=1000)
    parser.add_argument("--sharpness", type=float, default=4.0)
    parser.add_argument("--action-exploration", type=float, default=0.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/updated_section_5_1_scalar_sum_scaling"),
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=Path("docs/figures"),
    )
    parser.add_argument(
        "--figure-prefix",
        type=str,
        default=None,
        help="Prefix for generated figure filenames. Defaults to output directory name.",
    )
    parser.add_argument(
        "--doc-path",
        type=Path,
        default=Path("docs/updated_section_5_1_scalar_sum_results.md"),
    )
    return parser.parse_args()


def make_env(num_states: int) -> FeaturePartialObservableStateEnv:
    return FeaturePartialObservableStateEnv(
        state_features=cumulative_binary_state_features(num_states),
        reward_table=diagonal_reward_table(num_states),
        state_distribution=[1.0 / num_states for _ in range(num_states)],
        observation_dim=1,
    )


def run_one(
    *,
    num_states: int,
    learning_rate: float,
    seed: int,
    steps: int,
    eval_interval: int,
    sharpness: float,
    action_exploration: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    env = make_env(num_states)
    actor_compressor = build_scalar_sum_actor_compressors(
        num_states,
        learnable=True,
    )[2]
    fixed_policy = fixed_sum_observer_policy(
        num_actions=num_states,
        sharpness=sharpness,
    )
    trainer = UpdatedImperfectStateTrainer(
        env=env,
        actor_compressors=[actor_compressor],
        observer_policies=[fixed_policy],
        compression_optimizer=torch.optim.SGD(
            actor_compressor.parameters(),
            lr=learning_rate,
        ),
        action_exploration_probability=action_exploration,
        action_log_prob_source="learned",
        sample_actions=True,
    )
    trace = trainer.run(
        num_steps=steps,
        seed=seed,
        eval_interval=eval_interval,
    )

    curve_rows: list[dict[str, Any]] = []
    previous_messages: list[float] | None = None
    previous_weights: list[float] | None = None
    for eval_index, step in enumerate(trace.evaluation_steps):
        messages = [
            float(message[0])
            for message in trace.actor_messages_by_state[eval_index][0]
        ]
        weights = trace.actor_compressor_linear_weights[eval_index][0]
        biases = trace.actor_compressor_linear_biases[eval_index][0]
        correct_probs = correct_action_probabilities(fixed_policy, messages)
        metrics = message_metrics(messages)
        message_movement = (
            0.0
            if previous_messages is None
            else l2_distance(messages, previous_messages)
        )
        weight_vector = weights + biases
        weight_movement = (
            0.0
            if previous_weights is None
            else l2_distance(weight_vector, previous_weights)
        )
        previous_messages = messages
        previous_weights = weight_vector
        curve_rows.append(
            {
                "state_count": num_states,
                "learning_rate": learning_rate,
                "seed": seed,
                "step": step,
                "expected_reward": trace.expected_rewards[eval_index],
                "mean_correct_action_prob": fmean(correct_probs),
                "min_correct_action_prob": min(correct_probs),
                "missed_states_095": sum(prob < 0.95 for prob in correct_probs),
                "message_mae": metrics["mae"],
                "message_max_abs_error": metrics["max_abs_error"],
                "message_r2": metrics["r2"],
                "message_round_accuracy": metrics["round_accuracy"],
                "message_min_gap": metrics["min_gap"],
                "message_movement": message_movement,
                "weight_mean_abs_error": weight_mean_abs_error(weights),
                "weight_max_abs_error": weight_max_abs_error(weights),
                "weight_movement": weight_movement,
                "bias": biases[0] if biases else 0.0,
                "messages": ";".join(f"{value:.6g}" for value in messages),
                "weights": ";".join(f"{value:.6g}" for value in weights),
                "correct_action_probs": ";".join(
                    f"{value:.6g}" for value in correct_probs
                ),
            }
        )

    final = curve_rows[-1]
    run_row = dict(final)
    run_row.update(
        {
            "success_reward": float(final["expected_reward"] >= 0.95),
            "success_internal": float(
                final["min_correct_action_prob"] >= 0.95
                and final["message_round_accuracy"] >= 1.0
            ),
        }
    )
    return curve_rows, run_row


def correct_action_probabilities(
    policy: torch.nn.Module,
    messages: list[float],
) -> list[float]:
    probs: list[float] = []
    with torch.no_grad():
        for action, message in enumerate(messages):
            distribution = policy.action_distribution(torch.tensor([message]))
            probs.append(float(distribution.probs[action].item()))
    return probs


def message_metrics(messages: list[float]) -> dict[str, float]:
    targets = [float(index) for index in range(len(messages))]
    errors = [value - target for value, target in zip(messages, targets, strict=True)]
    rounded_matches = 0
    for value, target in zip(messages, targets, strict=True):
        rounded = int(round(value))
        rounded = max(0, min(len(messages) - 1, rounded))
        rounded_matches += int(rounded == int(target))
    gaps = [
        abs(messages[index + 1] - messages[index])
        for index in range(len(messages) - 1)
    ]
    return {
        "mae": fmean(abs(error) for error in errors),
        "max_abs_error": max(abs(error) for error in errors),
        "r2": r_squared(messages, targets),
        "round_accuracy": rounded_matches / len(messages),
        "min_gap": min(gaps) if gaps else 0.0,
    }


def r_squared(values: list[float], targets: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    x = torch.tensor(targets, dtype=torch.float32)
    y = torch.tensor(values, dtype=torch.float32)
    design = torch.stack([x, torch.ones_like(x)], dim=1)
    solution = torch.linalg.lstsq(design, y.reshape(-1, 1)).solution.reshape(-1)
    predicted = design @ solution
    total = torch.sum((y - y.mean()) ** 2)
    if float(total.item()) <= 1e-12:
        return 0.0
    residual = torch.sum((y - predicted) ** 2)
    return float((1.0 - residual / total).item())


def weight_mean_abs_error(weights: list[float]) -> float:
    return fmean(abs(weight - 1.0) for weight in weights)


def weight_max_abs_error(weights: list[float]) -> float:
    return max(abs(weight - 1.0) for weight in weights)


def l2_distance(left: list[float], right: list[float]) -> float:
    return math.sqrt(
        sum((lval - rval) ** 2 for lval, rval in zip(left, right, strict=True))
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"no rows to write for {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_runs(run_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[int, float], list[dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        grouped[(int(row["state_count"]), float(row["learning_rate"]))].append(row)

    summary_rows: list[dict[str, Any]] = []
    for (state_count, learning_rate), rows in sorted(grouped.items()):
        summary_rows.append(
            {
                "state_count": state_count,
                "learning_rate": learning_rate,
                "num_seeds": len(rows),
                "mean_final_expected_reward": mean(rows, "expected_reward"),
                "std_final_expected_reward": std(rows, "expected_reward"),
                "mean_min_correct_action_prob": mean(rows, "min_correct_action_prob"),
                "mean_missed_states_095": mean(rows, "missed_states_095"),
                "mean_message_mae": mean(rows, "message_mae"),
                "mean_message_r2": mean(rows, "message_r2"),
                "mean_round_accuracy": mean(rows, "message_round_accuracy"),
                "mean_weight_abs_error": mean(rows, "weight_mean_abs_error"),
                "mean_bias": mean(rows, "bias"),
                "reward_success_rate": mean(rows, "success_reward"),
                "internal_success_rate": mean(rows, "success_internal"),
            }
        )

    best_rows: list[dict[str, Any]] = []
    by_state: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        by_state[int(row["state_count"])].append(row)
    for state_count, rows in sorted(by_state.items()):
        best = max(
            rows,
            key=lambda row: (
                row["internal_success_rate"],
                row["reward_success_rate"],
                row["mean_final_expected_reward"],
                -row["mean_message_mae"],
            ),
        )
        best_rows.append(dict(best))
    return summary_rows, best_rows


def mean(rows: list[dict[str, Any]], key: str) -> float:
    return fmean(float(row[key]) for row in rows)


def std(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    return pstdev(values) if len(values) > 1 else 0.0


def plot_curves(
    curve_rows: list[dict[str, Any]],
    best_rows: list[dict[str, Any]],
    figure_dir: Path,
    figure_prefix: str,
) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    best_lr_by_state = {
        int(row["state_count"]): float(row["learning_rate"])
        for row in best_rows
    }
    rows = [
        row
        for row in curve_rows
        if float(row["learning_rate"]) == best_lr_by_state[int(row["state_count"])]
    ]
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["state_count"]), int(row["step"]))].append(row)

    by_state: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for (state_count, step), group_rows in grouped.items():
        by_state[state_count].append(
            {
                "step": step,
                "expected_reward": mean(group_rows, "expected_reward"),
                "message_mae": mean(group_rows, "message_mae"),
                "message_movement": mean(group_rows, "message_movement"),
                "weight_mean_abs_error": mean(group_rows, "weight_mean_abs_error"),
            }
        )
    for state_count in by_state:
        by_state[state_count].sort(key=lambda row: row["step"])

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    panels = [
        ("expected_reward", "expected reward"),
        ("message_mae", "message MAE"),
        ("message_movement", "message movement"),
        ("weight_mean_abs_error", "weight mean abs error"),
    ]
    for axis, (metric, label) in zip(axes.reshape(-1), panels, strict=True):
        for state_count, state_rows in sorted(by_state.items()):
            axis.plot(
                [row["step"] for row in state_rows],
                [row[metric] for row in state_rows],
                label=f"K={state_count}, lr={best_lr_by_state[state_count]:g}",
            )
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    axes[1, 0].set_xlabel("training step")
    axes[1, 1].set_xlabel("training step")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Updated Section 5.1: scalar-sum compression")
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(
            figure_dir / f"{figure_prefix}_curves.{suffix}",
            dpi=180,
        )
    plt.close(fig)


def plot_final_weights(
    run_rows: list[dict[str, Any]],
    best_rows: list[dict[str, Any]],
    figure_dir: Path,
    figure_prefix: str,
) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    largest = max(int(row["state_count"]) for row in best_rows)
    best_lr = next(
        float(row["learning_rate"])
        for row in best_rows
        if int(row["state_count"]) == largest
    )
    selected = [
        row
        for row in run_rows
        if int(row["state_count"]) == largest
        and float(row["learning_rate"]) == best_lr
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for row in selected:
        weights = [float(value) for value in str(row["weights"]).split(";")]
        messages = [float(value) for value in str(row["messages"]).split(";")]
        axes[0].plot(range(largest), weights, alpha=0.55, marker="o", label=f"seed {row['seed']}")
        axes[1].plot(range(largest), messages, alpha=0.55, marker="o", label=f"seed {row['seed']}")
    axes[0].axhline(1.0, color="black", linestyle="--", linewidth=1, label="ideal weight")
    axes[1].plot(range(largest), range(largest), color="black", linestyle="--", linewidth=1, label="target")
    axes[0].set_title(f"K={largest} learned weights, lr={best_lr:g}")
    axes[0].set_xlabel("state-vector coordinate")
    axes[0].set_ylabel("weight")
    axes[1].set_title(f"K={largest} final messages")
    axes[1].set_xlabel("state ID")
    axes[1].set_ylabel("message")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(
            figure_dir / f"{figure_prefix}_final_weights.{suffix}",
            dpi=180,
        )
    plt.close(fig)


def write_report(
    path: Path,
    args: argparse.Namespace,
    summary_rows: list[dict[str, Any]],
    best_rows: list[dict[str, Any]],
    figure_prefix: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Updated Section 5.1 Scalar-Sum Compression Results",
        "",
        "This is the updated-paper dyadic imperfect-state check.",
        "",
        "Setup:",
        "",
        "```text",
        "one Actor-compressor",
        "one Observer",
        "fixed Observer policy theta",
        "learn compression phi only",
        "scalar message m = f_phi(s)",
        "cumulative binary state vector, where sum(s) is the state ID",
        "```",
        "",
        "A run is counted as internally successful when the minimum correct-action probability is at least 0.95 and every final message rounds to the correct state ID.",
        "",
        "## Best Learning Rate By State Count",
        "",
        "| K | best alpha_phi | reward | min correct prob. | missed states | message MAE | message R2 | weight error | internal success |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in best_rows:
        lines.append(
            "| {state_count} | {learning_rate:.4g} | {mean_final_expected_reward:.3f} | "
            "{mean_min_correct_action_prob:.3f} | {mean_missed_states_095:.2f} | "
            "{mean_message_mae:.3f} | {mean_message_r2:.3f} | "
            "{mean_weight_abs_error:.3f} | {internal_success_rate:.2f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## All Learning-Rate Summary",
            "",
            "| K | alpha_phi | reward | min correct prob. | missed states | message MAE | message R2 | internal success |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary_rows:
        lines.append(
            "| {state_count} | {learning_rate:.4g} | {mean_final_expected_reward:.3f} | "
            "{mean_min_correct_action_prob:.3f} | {mean_missed_states_095:.2f} | "
            "{mean_message_mae:.3f} | {mean_message_r2:.3f} | "
            "{internal_success_rate:.2f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "```text",
            f"outputs: {args.output_dir}",
            "figures:",
            f"  {args.figure_dir / (figure_prefix + '_curves.png')}",
            f"  {args.figure_dir / (figure_prefix + '_final_weights.png')}",
            "```",
            "",
            "## Current Read",
            "",
            "This stage is only about whether the message map itself can learn the scalar state ID. If K fails here, later selection and group experiments should wait. If K works here, move to Section 5.2 and add Observer selection.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    figure_prefix = args.figure_prefix or args.output_dir.name
    with (args.output_dir / "config.json").open("w") as handle:
        json.dump(
            vars(args)
            | {
                "output_dir": str(args.output_dir),
                "figure_dir": str(args.figure_dir),
                "doc_path": str(args.doc_path),
                "figure_prefix": figure_prefix,
            },
            handle,
            indent=2,
        )

    curve_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    total = len(args.state_counts) * len(args.learning_rates) * len(args.seeds)
    done = 0
    for state_count in args.state_counts:
        for learning_rate in args.learning_rates:
            for seed in args.seeds:
                done += 1
                print(
                    f"[{done}/{total}] K={state_count} "
                    f"alpha_phi={learning_rate:g} seed={seed}",
                    flush=True,
                )
                curves, run = run_one(
                    num_states=state_count,
                    learning_rate=learning_rate,
                    seed=seed,
                    steps=args.steps,
                    eval_interval=args.eval_interval,
                    sharpness=args.sharpness,
                    action_exploration=args.action_exploration,
                )
                curve_rows.extend(curves)
                run_rows.append(run)

    summary_rows, best_rows = summarize_runs(run_rows)
    write_csv(args.output_dir / "curves.csv", curve_rows)
    write_csv(args.output_dir / "runs.csv", run_rows)
    write_csv(args.output_dir / "summary.csv", summary_rows)
    write_csv(args.output_dir / "best_by_state_count.csv", best_rows)
    plot_curves(curve_rows, best_rows, args.figure_dir, figure_prefix)
    plot_final_weights(run_rows, best_rows, args.figure_dir, figure_prefix)
    write_report(args.doc_path, args, summary_rows, best_rows, figure_prefix)

    print("\nBest by state count:")
    for row in best_rows:
        print(
            f"K={row['state_count']:>2} alpha={row['learning_rate']:<6g} "
            f"reward={row['mean_final_expected_reward']:.3f} "
            f"min_prob={row['mean_min_correct_action_prob']:.3f} "
            f"missed={row['mean_missed_states_095']:.2f} "
            f"success={row['internal_success_rate']:.2f}"
        )


if __name__ == "__main__":
    main()
