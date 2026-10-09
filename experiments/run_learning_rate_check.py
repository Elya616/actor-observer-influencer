"""Learning-rate trajectory check for the imperfect-reward side (Sections 4 & 6).

Professor Marbach's first debugging step whenever a case "does not work" is to
look at the *trajectory* of the learnable quantities and ask whether the
learning rate is simply wrong: too high and the curve oscillates without
settling, too low and it barely moves. This script produces exactly that
evidence for the reward-side scaling task.

To isolate the learning rate from the budget, it runs under the *perfect-reward*
baseline (``force_report=True``): every acting Actor is always updated, so a
good Actor's progress is governed only by its own learning rate and how often it
is selected -- not by the Observer's report gate. That is the right setting to
answer "is the residual gap under perfect reward a learning-rate problem?".

For each actor learning rate in the sweep it traces the good-actor greedy
accuracy over training and reports:

* the final accuracy (mean over seeds), and
* a late-training wobble measure (std of the accuracy over the last snapshots),
  which is large when the rate is too high and the curve keeps bouncing.

Figures (pdf/png/svg) go to ``docs/figures``; a summary CSV goes to ``outputs``.
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".mplcache").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_reward_budget_sanity import train_one_condition


def _save_figure(fig: plt.Figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png", ".svg"):
        fig.savefig(output_base.with_suffix(suffix), bbox_inches="tight")
    plt.close(fig)


def late_wobble(curve: list[float], window: int = 5) -> float:
    """Std of the accuracy over the last ``window`` snapshots (settling check)."""

    tail = curve[-window:] if len(curve) >= window else curve
    return statistics.pstdev(tail) if len(tail) > 1 else 0.0


def average_curves(curves: list[list[float]]) -> list[float]:
    n = len(curves)
    num_points = min(len(c) for c in curves)
    return [sum(c[k] for c in curves) / n for k in range(num_points)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure-dir", type=Path, default=Path("docs/figures"))
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/learning_rate_check/learning_rate_check_summary.csv"),
    )
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--grid-size", type=int, default=10)
    parser.add_argument("--num-actors", type=int, default=64)
    parser.add_argument(
        "--learning-rates",
        type=float,
        nargs="+",
        default=[0.001, 0.003, 0.01, 0.03, 0.1],
    )
    parser.add_argument("--num-good", type=int, default=2)
    parser.add_argument("--num-actions", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--budget", type=float, default=2.0)
    parser.add_argument("--utility-lr", type=float, default=0.05)
    parser.add_argument("--following-lr", type=float, default=0.05)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--num-steps", type=int, default=60000)
    parser.add_argument("--snapshot-interval", type=int, default=2000)
    args = parser.parse_args()

    seeds = list(range(args.seeds))
    results: list[dict[str, object]] = []
    for lr in args.learning_rates:
        runs = [
            train_one_condition(
                grid_size=args.grid_size,
                num_actors=args.num_actors,
                num_good=args.num_good,
                num_actions=args.num_actions,
                hidden_dim=args.hidden_dim,
                learning_rate=lr,
                budget=args.budget,
                utility_lr=args.utility_lr,
                following_lr=args.following_lr,
                c=args.c,
                force_report=True,
                num_steps=args.num_steps,
                snapshot_interval=args.snapshot_interval,
                seed=seed,
            )
            for seed in seeds
        ]
        num_points = min(len(r["steps_axis"]) for r in runs)
        mean_curve = average_curves([r["good_accuracy_curve"] for r in runs])
        results.append(
            {
                "learning_rate": lr,
                "steps_axis": runs[0]["steps_axis"][:num_points],
                "good_accuracy_curve": mean_curve,
                "final_good_accuracy": sum(r["final_good_accuracy"] for r in runs)
                / len(runs),
                "late_wobble": sum(late_wobble(r["good_accuracy_curve"]) for r in runs)
                / len(runs),
            }
        )

    # Trajectory plot: this is the figure the professor asks for -- you can see
    # directly whether a rate oscillates (too high) or crawls (too low).
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for result in results:
        ax.plot(
            result["steps_axis"],
            result["good_accuracy_curve"],
            linewidth=1.9,
            label=f"lr={result['learning_rate']:g}",
        )
    ax.set_title(
        f"Learning-rate check under perfect reward "
        f"({args.grid_size ** 2} states, {args.num_actors} actors)"
    )
    ax.set_xlabel("training step")
    ax.set_ylabel("good-actor greedy accuracy")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    _save_figure(fig, args.figure_dir / "learning_rate_check_trajectories")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["learning_rate", "final_good_accuracy", "late_wobble"],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "learning_rate": result["learning_rate"],
                    "final_good_accuracy": result["final_good_accuracy"],
                    "late_wobble": result["late_wobble"],
                }
            )

    print(
        f"seeds={args.seeds}, states={args.grid_size ** 2}, "
        f"actors={args.num_actors}, perfect reward, num_steps={args.num_steps}"
    )
    print(f"{'lr':>8} {'final_acc':>10} {'late_wobble':>12}")
    for result in results:
        print(
            f"{result['learning_rate']:>8g} "
            f"{result['final_good_accuracy']:>10.3f} "
            f"{result['late_wobble']:>12.4f}"
        )


if __name__ == "__main__":
    main()
