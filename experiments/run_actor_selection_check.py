"""Actor-selection diagnostic for the residual perfect-reward gap (Sections 4 & 6).

Even under the perfect-reward baseline (every acting Actor is always updated),
good-actor accuracy on the large joint rung tops out around 0.8 rather than 1.0.
The hypothesis is that this is *not* a reward-mechanism problem but a compute-
per-actor one: with uniform selection over ``N`` Actors, each of the two good
Actors only acts ``1/N`` of the time, so at equal total steps it takes far fewer
gradient steps as the crowd grows -- too few to fully learn a 100-state policy.

This script tests that directly. It fixes the collapsed rung (100 states /
64 actors) under perfect reward and sweeps the *share* of the selection
probability given to the good Actors, from the uniform baseline up to a large
share. If accuracy climbs toward 1.0 as the good Actors get more turns, the
residual gap is the ``1/N`` selection dilution, not the mechanism.

The good Actors (indices ``0..k-1``) split ``good_share`` of the probability
mass equally; the remaining ``1 - good_share`` is split equally among the blind
Actors. ``good_share = k / N`` reproduces the uniform baseline.

Figures (pdf/png/svg) go to ``docs/figures``; a summary CSV goes to ``outputs``.
"""

from __future__ import annotations

import argparse
import csv
import os
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


def build_selection_probs(num_actors: int, num_good: int, good_share: float) -> list[float]:
    """Give the good Actors ``good_share`` of the mass, split equally; the blind
    Actors split the rest equally."""

    num_blind = num_actors - num_good
    good_each = good_share / num_good
    blind_each = (1.0 - good_share) / num_blind if num_blind else 0.0
    return [good_each] * num_good + [blind_each] * num_blind


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure-dir", type=Path, default=Path("docs/figures"))
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/actor_selection_check/actor_selection_check_summary.csv"),
    )
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--grid-size", type=int, default=10)
    parser.add_argument("--num-actors", type=int, default=64)
    parser.add_argument(
        "--good-shares",
        type=float,
        nargs="+",
        default=[0.03125, 0.1, 0.25, 0.5],
        help="Total selection mass on the good actors (0.03125 = 2/64 uniform).",
    )
    parser.add_argument("--num-good", type=int, default=2)
    parser.add_argument("--num-actions", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--budget", type=float, default=2.0)
    parser.add_argument("--utility-lr", type=float, default=0.05)
    parser.add_argument("--following-lr", type=float, default=0.05)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--num-steps", type=int, default=60000)
    parser.add_argument("--snapshot-interval", type=int, default=2000)
    args = parser.parse_args()

    uniform_share = args.num_good / args.num_actors
    seeds = list(range(args.seeds))
    results: list[dict[str, object]] = []
    for good_share in args.good_shares:
        probs = build_selection_probs(args.num_actors, args.num_good, good_share)
        runs = [
            train_one_condition(
                grid_size=args.grid_size,
                num_actors=args.num_actors,
                num_good=args.num_good,
                num_actions=args.num_actions,
                hidden_dim=args.hidden_dim,
                learning_rate=args.learning_rate,
                budget=args.budget,
                utility_lr=args.utility_lr,
                following_lr=args.following_lr,
                c=args.c,
                force_report=True,
                num_steps=args.num_steps,
                snapshot_interval=args.snapshot_interval,
                seed=seed,
                actor_selection_probs=probs,
            )
            for seed in seeds
        ]
        turns_per_good = args.num_steps * (good_share / args.num_good)
        results.append(
            {
                "good_share": good_share,
                "turns_per_good": turns_per_good,
                "final_good_accuracy": sum(r["final_good_accuracy"] for r in runs)
                / len(runs),
            }
        )

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.plot(
        [r["good_share"] for r in results],
        [r["final_good_accuracy"] for r in results],
        "o-",
        linewidth=2.0,
        markersize=7,
    )
    ax.axvline(
        uniform_share,
        color="grey",
        linestyle=":",
        linewidth=1.6,
        label=f"uniform ({args.num_good}/{args.num_actors})",
    )
    ax.set_title(
        f"Perfect reward: accuracy vs good-actor selection share "
        f"({args.grid_size ** 2} states, {args.num_actors} actors)"
    )
    ax.set_xlabel("selection mass on good actors")
    ax.set_ylabel("good-actor greedy accuracy")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    _save_figure(fig, args.figure_dir / "actor_selection_check")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["good_share", "turns_per_good", "final_good_accuracy"],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(result)

    print(
        f"seeds={args.seeds}, states={args.grid_size ** 2}, "
        f"actors={args.num_actors}, perfect reward, lr={args.learning_rate}, "
        f"num_steps={args.num_steps} (uniform share = {uniform_share:.4g})"
    )
    print(f"{'good_share':>11} {'turns/good':>11} {'final_acc':>10}")
    for result in results:
        print(
            f"{result['good_share']:>11.4g} "
            f"{result['turns_per_good']:>11.0f} "
            f"{result['final_good_accuracy']:>10.3f}"
        )


if __name__ == "__main__":
    main()
