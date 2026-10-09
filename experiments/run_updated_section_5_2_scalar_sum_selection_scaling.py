"""Updated-paper Section 5.2 scalar-sum selection and compression scaling.

Section 5.2 has multiple Actor-compressors and one Observer. The Observer
policy is fixed. The learned objects are:

* Observer selection logits ``w`` over Actor-compressors.
* The selected Actor-compressor message map ``phi``.

The controlled setup uses cumulative binary states and scalar messages. The
full Actor-compressor can learn the state sum. The zero and partial
Actor-compressors cannot identify every state.
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
    SelectionLogits,
    UpdatedImperfectStateTrainer,
    build_scalar_sum_actor_compressors,
    cumulative_binary_state_features,
    diagonal_reward_table,
    fixed_sum_observer_policy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-counts", nargs="+", type=int, default=[3, 5, 8, 16])
    parser.add_argument(
        "--compression-learning-rates",
        nargs="+",
        type=float,
        default=[0.005, 0.01, 0.02, 0.05],
    )
    parser.add_argument(
        "--selection-learning-rates",
        nargs="+",
        type=float,
        default=None,
        help="Defaults to the compression learning-rate list.",
    )
    parser.add_argument(
        "--same-timescale-only",
        action="store_true",
        default=False,
        help="Only run alpha_phi == alpha_w pairs.",
    )
    parser.add_argument("--selection-exploration", type=float, default=0.05)
    parser.add_argument("--action-exploration", type=float, default=0.0)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--eval-interval", type=int, default=1000)
    parser.add_argument("--sharpness", type=float, default=4.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/updated_section_5_2_scalar_sum_selection_scaling"),
    )
    parser.add_argument("--figure-dir", type=Path, default=Path("docs/figures"))
    parser.add_argument(
        "--figure-prefix",
        type=str,
        default=None,
        help="Prefix for generated figure filenames. Defaults to output directory name.",
    )
    parser.add_argument(
        "--doc-path",
        type=Path,
        default=Path("docs/updated_section_5_2_scalar_sum_selection_results.md"),
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
    alpha_phi: float,
    alpha_w: float,
    selection_exploration: float,
    action_exploration: float,
    seed: int,
    steps: int,
    eval_interval: int,
    sharpness: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    env = make_env(num_states)
    actor_compressors = build_scalar_sum_actor_compressors(num_states, learnable=True)
    selection = SelectionLogits(num_actor_compressors=len(actor_compressors))
    fixed_policy = fixed_sum_observer_policy(num_actions=num_states, sharpness=sharpness)
    compression_parameters = [
        parameter for actor in actor_compressors for parameter in actor.parameters()
    ]
    trainer = UpdatedImperfectStateTrainer(
        env=env,
        actor_compressors=actor_compressors,
        observer_policies=[fixed_policy],
        selection_logits_by_observer=[selection],
        compression_optimizer=torch.optim.SGD(compression_parameters, lr=alpha_phi),
        selection_optimizer=torch.optim.SGD(selection.parameters(), lr=alpha_w),
        selection_exploration_probability=selection_exploration,
        selection_log_prob_source="learned",
        action_exploration_probability=action_exploration,
        action_log_prob_source="learned",
        sample_selection=True,
        sample_actions=True,
    )
    trace = trainer.run(num_steps=steps, seed=seed, eval_interval=eval_interval)

    curve_rows: list[dict[str, Any]] = []
    previous_full_messages: list[float] | None = None
    previous_selection: list[float] | None = None
    exposure_by_step = cumulative_exposure(trace.compression_feedback_counts)
    for eval_index, step in enumerate(trace.evaluation_steps):
        messages_by_actor = trace.actor_messages_by_state[eval_index]
        full_messages = [float(message[0]) for message in messages_by_actor[2]]
        full_weights = trace.actor_compressor_linear_weights[eval_index][2]
        full_biases = trace.actor_compressor_linear_biases[eval_index][2]
        selection_probs = trace.selection_probabilities_by_observer[eval_index][0]
        mixture_correct_probs = mixture_correct_action_probabilities(
            fixed_policy,
            messages_by_actor,
            selection_probs,
        )
        full_correct_probs = correct_action_probabilities(fixed_policy, full_messages)
        metrics = message_metrics(full_messages)
        full_message_movement = (
            0.0
            if previous_full_messages is None
            else l2_distance(full_messages, previous_full_messages)
        )
        selection_movement = (
            0.0
            if previous_selection is None
            else l2_distance(selection_probs, previous_selection)
        )
        previous_full_messages = full_messages
        previous_selection = list(selection_probs)
        exposure = exposure_by_step[min(step, len(exposure_by_step) - 1)]
        curve_rows.append(
            {
                "state_count": num_states,
                "alpha_phi": alpha_phi,
                "alpha_w": alpha_w,
                "selection_exploration": selection_exploration,
                "action_exploration": action_exploration,
                "seed": seed,
                "step": step,
                "expected_reward": trace.expected_rewards[eval_index],
                "mean_correct_action_prob": fmean(mixture_correct_probs),
                "min_correct_action_prob": min(mixture_correct_probs),
                "missed_states_095": sum(prob < 0.95 for prob in mixture_correct_probs),
                "full_mean_correct_action_prob": fmean(full_correct_probs),
                "full_min_correct_action_prob": min(full_correct_probs),
                "full_missed_states_095": sum(prob < 0.95 for prob in full_correct_probs),
                "p_zero": selection_probs[0],
                "p_partial": selection_probs[1],
                "p_full": selection_probs[2],
                "selection_entropy": entropy(selection_probs),
                "selection_movement": selection_movement,
                "full_message_mae": metrics["mae"],
                "full_message_max_abs_error": metrics["max_abs_error"],
                "full_message_r2": metrics["r2"],
                "full_message_round_accuracy": metrics["round_accuracy"],
                "full_message_min_gap": metrics["min_gap"],
                "full_message_movement": full_message_movement,
                "full_weight_mean_abs_error": weight_mean_abs_error(full_weights),
                "full_bias": full_biases[0] if full_biases else 0.0,
                "exposure_zero": exposure[0],
                "exposure_partial": exposure[1],
                "exposure_full": exposure[2],
                "exposure_full_share": (
                    exposure[2] / sum(exposure) if sum(exposure) > 0 else 0.0
                ),
                "selection_probs": ";".join(f"{value:.6g}" for value in selection_probs),
                "full_messages": ";".join(f"{value:.6g}" for value in full_messages),
                "full_weights": ";".join(f"{value:.6g}" for value in full_weights),
                "mixture_correct_action_probs": ";".join(
                    f"{value:.6g}" for value in mixture_correct_probs
                ),
                "full_correct_action_probs": ";".join(
                    f"{value:.6g}" for value in full_correct_probs
                ),
            }
        )
    final = curve_rows[-1]
    run_row = dict(final)
    run_row.update(
        {
            "success_reward": float(final["expected_reward"] >= 0.95),
            "success_internal": float(
                final["expected_reward"] >= 0.95
                and final["p_full"] >= 0.95
                and final["min_correct_action_prob"] >= 0.95
                and final["full_message_round_accuracy"] >= 1.0
            ),
        }
    )
    return curve_rows, run_row


def cumulative_exposure(feedback_counts: list[list[int]]) -> list[list[int]]:
    totals = [[0, 0, 0]]
    current = [0, 0, 0]
    for counts in feedback_counts:
        current = [old + int(new) for old, new in zip(current, counts, strict=True)]
        totals.append(list(current))
    return totals


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


def mixture_correct_action_probabilities(
    policy: torch.nn.Module,
    messages_by_actor: list[list[list[float]]],
    selection_probs: list[float],
) -> list[float]:
    probs: list[float] = []
    with torch.no_grad():
        for state_index in range(len(messages_by_actor[0])):
            total = 0.0
            for actor_index, actor_messages in enumerate(messages_by_actor):
                message = torch.tensor(actor_messages[state_index], dtype=torch.float32)
                distribution = policy.action_distribution(message)
                total += float(selection_probs[actor_index]) * float(
                    distribution.probs[state_index].item()
                )
            probs.append(total)
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


def l2_distance(left: list[float], right: list[float]) -> float:
    return math.sqrt(
        sum((lval - rval) ** 2 for lval, rval in zip(left, right, strict=True))
    )


def entropy(probabilities: list[float]) -> float:
    return -sum(float(prob) * math.log(max(float(prob), 1e-12)) for prob in probabilities)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"no rows to write for {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def mean(rows: list[dict[str, Any]], key: str) -> float:
    return fmean(float(row[key]) for row in rows)


def std(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    return pstdev(values) if len(values) > 1 else 0.0


def summarize_runs(run_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[int, float, float, float, float], list[dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        grouped[
            (
                int(row["state_count"]),
                float(row["alpha_phi"]),
                float(row["alpha_w"]),
                float(row["selection_exploration"]),
                float(row["action_exploration"]),
            )
        ].append(row)

    summary_rows: list[dict[str, Any]] = []
    for (state_count, alpha_phi, alpha_w, epsilon_w, epsilon_a), rows in sorted(grouped.items()):
        summary_rows.append(
            {
                "state_count": state_count,
                "alpha_phi": alpha_phi,
                "alpha_w": alpha_w,
                "selection_exploration": epsilon_w,
                "action_exploration": epsilon_a,
                "num_seeds": len(rows),
                "mean_final_expected_reward": mean(rows, "expected_reward"),
                "std_final_expected_reward": std(rows, "expected_reward"),
                "mean_min_correct_action_prob": mean(rows, "min_correct_action_prob"),
                "mean_missed_states_095": mean(rows, "missed_states_095"),
                "mean_p_full": mean(rows, "p_full"),
                "mean_p_partial": mean(rows, "p_partial"),
                "mean_p_zero": mean(rows, "p_zero"),
                "mean_full_message_mae": mean(rows, "full_message_mae"),
                "mean_full_message_r2": mean(rows, "full_message_r2"),
                "mean_full_weight_abs_error": mean(rows, "full_weight_mean_abs_error"),
                "mean_full_exposure_share": mean(rows, "exposure_full_share"),
                "reward_success_rate": mean(rows, "success_reward"),
                "internal_success_rate": mean(rows, "success_internal"),
            }
        )
    best_rows: list[dict[str, Any]] = []
    for state_count in sorted({int(row["state_count"]) for row in summary_rows}):
        candidates = [row for row in summary_rows if int(row["state_count"]) == state_count]
        best_rows.append(
            max(
                candidates,
                key=lambda row: (
                    float(row["internal_success_rate"]),
                    float(row["reward_success_rate"]),
                    float(row["mean_final_expected_reward"]),
                    float(row["mean_p_full"]),
                    -float(row["mean_missed_states_095"]),
                ),
            )
        )
    return summary_rows, best_rows


def plot_curves(
    curve_rows: list[dict[str, Any]],
    best_rows: list[dict[str, Any]],
    figure_dir: Path,
    figure_prefix: str,
) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    best_by_state = {
        int(row["state_count"]): (
            float(row["alpha_phi"]),
            float(row["alpha_w"]),
            float(row["selection_exploration"]),
            float(row["action_exploration"]),
        )
        for row in best_rows
    }
    rows = [
        row
        for row in curve_rows
        if (
            float(row["alpha_phi"]),
            float(row["alpha_w"]),
            float(row["selection_exploration"]),
            float(row["action_exploration"]),
        )
        == best_by_state[int(row["state_count"])]
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
                "p_full": mean(group_rows, "p_full"),
                "full_message_mae": mean(group_rows, "full_message_mae"),
                "min_correct_action_prob": mean(group_rows, "min_correct_action_prob"),
            }
        )
    for state_count in by_state:
        by_state[state_count].sort(key=lambda row: row["step"])

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    panels = [
        ("expected_reward", "expected reward"),
        ("p_full", "P(full compressor)"),
        ("full_message_mae", "full-message MAE"),
        ("min_correct_action_prob", "min correct-action prob."),
    ]
    for axis, (metric, label) in zip(axes.reshape(-1), panels, strict=True):
        for state_count, state_rows in sorted(by_state.items()):
            alpha_phi, alpha_w, epsilon_w, epsilon_a = best_by_state[state_count]
            axis.plot(
                [row["step"] for row in state_rows],
                [row[metric] for row in state_rows],
                label=(
                    f"K={state_count}, "
                    f"phi={alpha_phi:g}, w={alpha_w:g}, "
                    f"eps_w={epsilon_w:g}, eps_a={epsilon_a:g}"
                ),
            )
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    axes[1, 0].set_xlabel("training step")
    axes[1, 1].set_xlabel("training step")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Updated Section 5.2: selection plus scalar compression")
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(figure_dir / f"{figure_prefix}_curves.{suffix}", dpi=180)
    plt.close(fig)


def plot_selection_and_messages(
    run_rows: list[dict[str, Any]],
    best_rows: list[dict[str, Any]],
    figure_dir: Path,
    figure_prefix: str,
) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    largest = max(int(row["state_count"]) for row in best_rows)
    best = next(row for row in best_rows if int(row["state_count"]) == largest)
    selected = [
        row
        for row in run_rows
        if int(row["state_count"]) == largest
        and float(row["alpha_phi"]) == float(best["alpha_phi"])
        and float(row["alpha_w"]) == float(best["alpha_w"])
        and float(row["selection_exploration"]) == float(best["selection_exploration"])
        and float(row["action_exploration"]) == float(best["action_exploration"])
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for row in selected:
        probs = [float(value) for value in str(row["selection_probs"]).split(";")]
        messages = [float(value) for value in str(row["full_messages"]).split(";")]
        axes[0].plot(["zero", "partial", "full"], probs, alpha=0.65, marker="o", label=f"seed {row['seed']}")
        axes[1].plot(range(largest), messages, alpha=0.55, marker="o", label=f"seed {row['seed']}")
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].set_title(f"K={largest} final selection")
    axes[0].set_ylabel("probability")
    axes[1].plot(range(largest), range(largest), color="black", linestyle="--", linewidth=1, label="target")
    axes[1].set_title(f"K={largest} full-compressor messages")
    axes[1].set_xlabel("state ID")
    axes[1].set_ylabel("message")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(figure_dir / f"{figure_prefix}_final_selection_messages.{suffix}", dpi=180)
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
        "# Updated Section 5.2 Scalar-Sum Selection and Compression Results",
        "",
        "This is the updated-paper one-Observer, multiple-Actor-compressor imperfect-state check.",
        "",
        "Setup:",
        "",
        "```text",
        "one Observer",
        "three Actor-compressors: zero, partial, full",
        "fixed Observer policy theta",
        "learn Observer selection w",
        "learn selected Actor-compressor compression phi",
        "scalar message m = f_phi(s)",
        "cumulative binary state vector, where sum(s) is the state ID",
        "```",
        "",
        "The main question is whether the Observer learns to select the full compressor while the full compressor learns the scalar state sum.",
        "",
        "## Best Setting By State Count",
        "",
        "| K | alpha_phi | alpha_w | eps_w | eps_a | reward | min correct prob. | P(full) | missed states | full msg MAE | full msg R2 | internal success |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in best_rows:
        lines.append(
            "| {state_count} | {alpha_phi:.4g} | {alpha_w:.4g} | "
            "{selection_exploration:.3g} | {action_exploration:.3g} | "
            "{mean_final_expected_reward:.3f} | "
            "{mean_min_correct_action_prob:.3f} | {mean_p_full:.3f} | "
            "{mean_missed_states_095:.2f} | {mean_full_message_mae:.3f} | "
            "{mean_full_message_r2:.3f} | {internal_success_rate:.2f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## All Settings",
            "",
            "| K | alpha_phi | alpha_w | eps_w | eps_a | reward | P(full) | missed states | full msg MAE | internal success |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary_rows:
        lines.append(
            "| {state_count} | {alpha_phi:.4g} | {alpha_w:.4g} | "
            "{selection_exploration:.3g} | {action_exploration:.3g} | "
            "{mean_final_expected_reward:.3f} | "
            "{mean_p_full:.3f} | {mean_missed_states_095:.2f} | "
            "{mean_full_message_mae:.3f} | {internal_success_rate:.2f} |".format(**row)
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
            f"  {args.figure_dir / (figure_prefix + '_final_selection_messages.png')}",
            "```",
            "",
            "## Current Read",
            "",
            "This stage adds Observer selection on top of the Section 5.1 compression check. If this works, move to Section 5.3 and add learned Observer policies.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def learning_rate_pairs(args: argparse.Namespace) -> list[tuple[float, float]]:
    alpha_ws = args.selection_learning_rates or args.compression_learning_rates
    pairs = [
        (float(alpha_phi), float(alpha_w))
        for alpha_phi in args.compression_learning_rates
        for alpha_w in alpha_ws
    ]
    if args.same_timescale_only:
        pairs = [(alpha_phi, alpha_w) for alpha_phi, alpha_w in pairs if alpha_phi == alpha_w]
    return pairs


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

    pairs = learning_rate_pairs(args)
    curve_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    total = len(args.state_counts) * len(pairs) * len(args.seeds)
    done = 0
    for state_count in args.state_counts:
        for alpha_phi, alpha_w in pairs:
            for seed in args.seeds:
                done += 1
                print(
                    f"[{done}/{total}] K={state_count} "
                    f"alpha_phi={alpha_phi:g} alpha_w={alpha_w:g} "
                    f"eps_w={args.selection_exploration:g} "
                    f"eps_a={args.action_exploration:g} seed={seed}",
                    flush=True,
                )
                curves, run = run_one(
                    num_states=state_count,
                    alpha_phi=alpha_phi,
                    alpha_w=alpha_w,
                    selection_exploration=args.selection_exploration,
                    action_exploration=args.action_exploration,
                    seed=seed,
                    steps=args.steps,
                    eval_interval=args.eval_interval,
                    sharpness=args.sharpness,
                )
                curve_rows.extend(curves)
                run_rows.append(run)

    summary_rows, best_rows = summarize_runs(run_rows)
    write_csv(args.output_dir / "curves.csv", curve_rows)
    write_csv(args.output_dir / "runs.csv", run_rows)
    write_csv(args.output_dir / "summary.csv", summary_rows)
    write_csv(args.output_dir / "best_by_state_count.csv", best_rows)
    plot_curves(curve_rows, best_rows, args.figure_dir, figure_prefix)
    plot_selection_and_messages(run_rows, best_rows, args.figure_dir, figure_prefix)
    write_report(args.doc_path, args, summary_rows, best_rows, figure_prefix)

    print("\nBest by state count:")
    for row in best_rows:
        print(
            f"K={row['state_count']:>2} phi={row['alpha_phi']:<6g} "
            f"w={row['alpha_w']:<6g} reward={row['mean_final_expected_reward']:.3f} "
            f"p_full={row['mean_p_full']:.3f} "
            f"missed={row['mean_missed_states_095']:.2f} "
            f"success={row['internal_success_rate']:.2f}"
        )


if __name__ == "__main__":
    main()
