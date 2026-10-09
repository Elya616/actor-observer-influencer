"""Run a structured state-scaling check for the technical report.

This is not a new Observer mechanism. It is a small sanity check requested by
the meeting notes: when scaling the number of states, use state features with a
smooth rule for the best action rather than assigning IID-random best actions.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict, deque
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
import torch

from actor_observer_rl.agents import Actor
from actor_observer_rl.policies import NeuralPolicy
from actor_observer_rl.training.updates import reinforce_loss
from actor_observer_rl.utils.seeding import set_global_seed
from experiments.run_paper_7 import save_figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-sizes", type=int, nargs="+", default=[8, 16])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--num-steps", type=int, default=80000)
    parser.add_argument("--eval-interval", type=int, default=800)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--hidden-dim", type=int, default=48)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/structured_scaling"),
    )
    parser.add_argument("--figure-dir", type=Path, default=Path("docs/figures"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for grid_size in args.grid_sizes:
        for seed in args.seeds:
            rows.extend(
                run_one(
                    grid_size=grid_size,
                    seed=seed,
                    num_steps=args.num_steps,
                    eval_interval=args.eval_interval,
                    learning_rate=args.learning_rate,
                    hidden_dim=args.hidden_dim,
                )
            )

    summaries = summarize(rows)
    write_csv(args.output_dir / "structured_scaling_curves.csv", rows)
    write_csv(args.output_dir / "structured_scaling_summary.csv", summaries)
    write_figure(rows, args.figure_dir / "structured_scaling_check")

    for row in summaries:
        print(
            "grid={grid_size}: final_expected={final_expected_reward:.3f} "
            "mean_opt_prob={final_mean_optimal_probability:.3f} "
            "min_rep_prob={final_min_representative_probability:.3f}".format(**row)
        )


def run_one(
    *,
    grid_size: int,
    seed: int,
    num_steps: int,
    eval_interval: int,
    learning_rate: float,
    hidden_dim: int,
) -> list[dict[str, Any]]:
    set_global_seed(seed)
    states, optimal_actions = build_grid_task(grid_size)
    reward_table = build_reward_table(optimal_actions)
    actor = Actor(NeuralPolicy(input_dim=2, hidden_dim=hidden_dim, num_actions=4))
    optimizer = torch.optim.Adam(actor.parameters(), lr=learning_rate)
    generator = torch.Generator()
    generator.manual_seed(seed + 31_337)
    reward_window: deque[float] = deque(maxlen=100)
    rows: list[dict[str, Any]] = []

    representative_indices = representative_state_indices(grid_size)

    def record(step: int) -> None:
        expected_reward, mean_optimal, representative = evaluate(
            actor,
            states,
            reward_table,
            optimal_actions,
            representative_indices,
        )
        rows.append(
            {
                "grid_size": int(grid_size),
                "seed": int(seed),
                "step": int(step),
                "expected_reward": float(expected_reward),
                "mean_optimal_probability": float(mean_optimal),
                "min_representative_probability": float(min(representative.values())),
                "average_sampled_reward": (
                    0.0 if not reward_window else float(fmean(reward_window))
                ),
                **{
                    f"rep_{name}_probability": float(value)
                    for name, value in representative.items()
                },
            }
        )

    record(0)
    for step_index in range(num_steps):
        state_index = int(
            torch.randint(
                low=0,
                high=len(states),
                size=(1,),
                generator=generator,
            ).item()
        )
        state = states[state_index]
        action_result = actor.act(state, sample=True)
        action = int(action_result.action.detach().cpu().item())
        reward = reward_table[state_index][action]
        loss = reinforce_loss(reward, action_result.log_prob)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        reward_window.append(float(reward))

        step = step_index + 1
        if step % eval_interval == 0 or step == num_steps:
            record(step)

    return rows


def build_grid_task(grid_size: int) -> tuple[torch.Tensor, list[int]]:
    if grid_size < 2:
        raise ValueError("grid_size must be at least 2")
    states: list[list[float]] = []
    optimal_actions: list[int] = []
    denom = float(grid_size - 1)
    for y_index in range(grid_size):
        for x_index in range(grid_size):
            x_value = x_index / denom
            y_value = y_index / denom
            states.append([x_value, y_value])
            optimal_actions.append(region_action(x_value, y_value))
    return torch.tensor(states, dtype=torch.float32), optimal_actions


def region_action(x_value: float, y_value: float) -> int:
    right = 1 if x_value >= 0.5 else 0
    upper = 1 if y_value >= 0.5 else 0
    return right + 2 * upper


def build_reward_table(optimal_actions: list[int]) -> list[list[float]]:
    table: list[list[float]] = []
    for optimal_action in optimal_actions:
        rewards = [-0.1] * 4
        rewards[optimal_action] = 1.0
        table.append(rewards)
    return table


def representative_state_indices(grid_size: int) -> dict[str, int]:
    def index(x_index: int, y_index: int) -> int:
        return y_index * grid_size + x_index

    last = grid_size - 1
    mid_low = max(0, grid_size // 2 - 1)
    mid_high = min(last, grid_size // 2)
    return {
        "lower_left": index(0, 0),
        "lower_right": index(last, 0),
        "upper_left": index(0, last),
        "upper_right": index(last, last),
        "boundary_low": index(mid_low, mid_low),
        "boundary_high": index(mid_high, mid_high),
    }


def evaluate(
    actor: Actor,
    states: torch.Tensor,
    reward_table: list[list[float]],
    optimal_actions: list[int],
    representative_indices: dict[str, int],
) -> tuple[float, float, dict[str, float]]:
    expected_reward = 0.0
    mean_optimal = 0.0
    representative: dict[str, float] = {}
    with torch.no_grad():
        for state_index, state in enumerate(states):
            distribution = actor.policy.action_distribution(state)
            probs = distribution.probs.detach().cpu()
            rewards = torch.tensor(reward_table[state_index], dtype=torch.float32)
            expected_reward += float((probs * rewards).sum().item())
            mean_optimal += float(probs[optimal_actions[state_index]].item())
        for name, state_index in representative_indices.items():
            distribution = actor.policy.action_distribution(states[state_index])
            probs = distribution.probs.detach().cpu()
            representative[name] = float(probs[optimal_actions[state_index]].item())
    num_states = len(states)
    return expected_reward / num_states, mean_optimal / num_states, representative


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    final_step = max(int(row["step"]) for row in rows)
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if int(row["step"]) == final_step:
            grouped[int(row["grid_size"])].append(row)

    summaries: list[dict[str, Any]] = []
    for grid_size, grid_rows in sorted(grouped.items()):
        summaries.append(
            {
                "grid_size": grid_size,
                "num_states": grid_size * grid_size,
                "final_expected_reward": fmean(
                    float(row["expected_reward"]) for row in grid_rows
                ),
                "final_sampled_reward": fmean(
                    float(row["average_sampled_reward"]) for row in grid_rows
                ),
                "final_mean_optimal_probability": fmean(
                    float(row["mean_optimal_probability"]) for row in grid_rows
                ),
                "final_min_representative_probability": min(
                    float(row["min_representative_probability"]) for row in grid_rows
                ),
                "success_rate": fmean(
                    1.0 if float(row["expected_reward"]) >= 0.95 else 0.0
                    for row in grid_rows
                ),
            }
        )
    return summaries


def write_figure(rows: list[dict[str, Any]], output_base: Path) -> None:
    grid_sizes = sorted({int(row["grid_size"]) for row in rows})
    largest_grid = max(grid_sizes)
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.0))

    for grid_size in grid_sizes:
        plot_mean_series(
            axes[0],
            rows,
            grid_size,
            "expected_reward",
            label=f"{grid_size}x{grid_size}",
        )
        plot_mean_series(
            axes[1],
            rows,
            grid_size,
            "mean_optimal_probability",
            label=f"{grid_size}x{grid_size}",
        )

    representative_keys = [
        "rep_lower_left_probability",
        "rep_lower_right_probability",
        "rep_upper_left_probability",
        "rep_upper_right_probability",
        "rep_boundary_low_probability",
        "rep_boundary_high_probability",
    ]
    for key in representative_keys:
        label = key.removeprefix("rep_").removesuffix("_probability")
        plot_mean_series(axes[2], rows, largest_grid, key, label=label)

    axes[0].set_title("Expected reward")
    axes[0].set_ylabel("reward")
    axes[1].set_title("Mean correct-action probability")
    axes[1].set_ylabel("probability")
    axes[2].set_title(f"Representative states ({largest_grid}x{largest_grid})")
    axes[2].set_ylabel("correct-action probability")
    for axis in axes:
        axis.set_xlabel("training step")
        axis.set_ylim(-0.12, 1.05)
        axis.axhline(1.0, color="black", linewidth=0.8, alpha=0.4)
        axis.axhline(0.95, color="black", linestyle="--", linewidth=0.8, alpha=0.4)
        axis.grid(True, axis="y", alpha=0.25)
        axis.legend(frameon=False, fontsize=8)
    fig.suptitle("Structured State-Scaling Check")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_figure(fig, output_base)


def plot_mean_series(
    axis: plt.Axes,
    rows: list[dict[str, Any]],
    grid_size: int,
    metric: str,
    *,
    label: str,
) -> None:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        if int(row["grid_size"]) == grid_size:
            grouped[int(row["step"])].append(float(row[metric]))
    series = [(step, fmean(values)) for step, values in sorted(grouped.items())]
    axis.plot(
        [point[0] for point in series],
        [point[1] for point in series],
        linewidth=1.6,
        label=label,
    )


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
