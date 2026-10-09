"""Reward-budget sanity check for the imperfect-reward side (Sections 4 & 6).

Professor Marbach's question after the scaling report was: if we increase the
attention budget ``B``, the following rates ``mu_j`` should grow, the report
probabilities ``theta(mu_j) = 1 - exp(-c mu_j)`` should grow, and in the limit
every Actor is always told its reward -- which is exactly *perfect* reward
information. So performance should climb toward the perfect-reward upper bound
as ``B`` grows, never fall. This script makes that limit explicit and checks it.

It fixes one joint task (a ``GridWorldBanditEnv`` rung with ``k`` sighted good
Actors and ``N - k`` blind Actors, as in Phase 3) and runs it under a sequence
of conditions on the *same* task, at equal compute:

* ``perfect`` -- the report probability is hard-coded to 1 (``force_report``),
  so every acting Actor is always updated. This is the perfect-reward baseline
  and the performance ceiling.
* ``B = b`` for a sweep of budgets -- the ordinary imperfect-reward mechanism,
  where a larger ``B`` should push report probabilities up toward 1.

For every condition it traces, alongside the good-actor accuracy and the
budget concentration, the two quantities the professor wants to see move:

* the mean report probability on the good Actors (``theta`` on indices < k), and
* the mean following rate ``mu`` on the good Actors.

The claim to verify is monotone: as ``B`` grows, mean report probability and
good-actor accuracy should both rise toward the ``perfect`` baseline.

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
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from actor_observer_rl.agents import BudgetedObserver
from actor_observer_rl.envs import GridWorldBanditEnv
from actor_observer_rl.training import MultiActorTrainer
from actor_observer_rl.utils.seeding import set_global_seed
from run_scaling_states_actors import (
    build_actor,
    concentration,
    greedy_accuracy,
    top_k_precision,
)

PERFECT = "perfect"


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def train_one_condition(
    *,
    grid_size: int,
    num_actors: int,
    num_good: int,
    num_actions: int,
    hidden_dim: int,
    learning_rate: float,
    budget: float,
    utility_lr: float,
    following_lr: float,
    c: float,
    force_report: bool,
    num_steps: int,
    snapshot_interval: int,
    seed: int,
    actor_selection_probs: list[float] | None = None,
) -> dict[str, object]:
    """Train the fixed joint task under one budget / perfect-reward condition.

    ``actor_selection_probs`` defaults to a uniform distribution over the crowd
    (each Actor acts ``1/N`` of the time). Passing a non-uniform vector lets a
    caller give the good Actors more turns, which isolates the "each good Actor
    only acts 1/N of the time" compute bottleneck from the reward mechanism.
    """

    set_global_seed(seed)
    env = GridWorldBanditEnv(grid_size=grid_size, num_actions=num_actions)

    actors = [
        build_actor(num_actions, hidden_dim, blind=j >= num_good)
        for j in range(num_actors)
    ]
    optimizers = [
        torch.optim.Adam(actor.parameters(), lr=learning_rate) for actor in actors
    ]
    observer = BudgetedObserver(
        num_actors=num_actors,
        budget=budget,
        utility_learning_rate=utility_lr,
        following_learning_rate=following_lr,
        c=c,
        seed=seed,
    )
    if actor_selection_probs is None:
        actor_selection_probs = [1.0 / num_actors] * num_actors
    trainer = MultiActorTrainer(
        env=env,
        actors=actors,
        observer=observer,
        actor_selection_probs=actor_selection_probs,
        actor_optimizers=optimizers,
        reward_baseline="running_mean",
        force_report=force_report,
        snapshot_interval=snapshot_interval,
    )

    steps_axis: list[int] = []
    good_accuracy_curve: list[float] = []
    report_prob_good_curve: list[float] = []
    mu_good_curve: list[float] = []
    concentration_curve: list[float] = []

    trained = 0
    is_first = True
    while trained < num_steps:
        segment = min(snapshot_interval, num_steps - trained)
        trainer.run(num_steps=segment, seed=seed if is_first else None)
        is_first = False
        trained += segment

        mu = observer.current_following_rates()
        theta = [float(p) for p in observer.report_probabilities()]
        good_acc = _mean(
            [greedy_accuracy(actors[j], env, blind=False) for j in range(num_good)]
        )
        # Under the perfect-reward baseline every Actor is always updated, so the
        # *effective* report probability that gates learning is 1 regardless of mu.
        effective_report_good = 1.0 if force_report else _mean(theta[:num_good])

        steps_axis.append(trained)
        good_accuracy_curve.append(good_acc)
        report_prob_good_curve.append(effective_report_good)
        mu_good_curve.append(_mean(mu[:num_good]))
        concentration_curve.append(concentration(mu, num_good, budget))

    final_mu = observer.current_following_rates()
    per_actor_accuracy = [
        greedy_accuracy(actors[j], env, blind=j >= num_good)
        for j in range(num_actors)
    ]
    system_score = sum(
        (final_mu[j] / budget) * per_actor_accuracy[j] for j in range(num_actors)
    )
    return {
        "grid_size": grid_size,
        "num_states": env.num_states,
        "num_actors": num_actors,
        "budget": budget,
        "force_report": force_report,
        "steps_axis": steps_axis,
        "good_accuracy_curve": good_accuracy_curve,
        "report_prob_good_curve": report_prob_good_curve,
        "mu_good_curve": mu_good_curve,
        "concentration_curve": concentration_curve,
        "final_good_accuracy": good_accuracy_curve[-1],
        "final_report_prob_good": report_prob_good_curve[-1],
        "final_mu_good": mu_good_curve[-1],
        "final_concentration": concentration(final_mu, num_good, budget),
        "final_top_k_precision": top_k_precision(final_mu, num_good),
        "final_system_score": system_score,
    }


def average_conditions(
    runs_by_seed: list[list[dict[str, object]]],
) -> list[dict[str, object]]:
    """Average per-condition results across seeds (outer list = seeds)."""

    num_conditions = len(runs_by_seed[0])
    averaged: list[dict[str, object]] = []
    for c in range(num_conditions):
        group = [seed_runs[c] for seed_runs in runs_by_seed]
        n = len(group)
        num_points = min(len(r["good_accuracy_curve"]) for r in group)

        def curve(key: str) -> list[float]:
            return [
                sum(r[key][k] for r in group) / n for k in range(num_points)
            ]

        def final(key: str) -> float:
            return sum(r[key] for r in group) / n

        averaged.append(
            {
                "label": condition_label(group[0]),
                "budget": group[0]["budget"],
                "force_report": group[0]["force_report"],
                "num_states": group[0]["num_states"],
                "num_actors": group[0]["num_actors"],
                "steps_axis": group[0]["steps_axis"][:num_points],
                "good_accuracy_curve": curve("good_accuracy_curve"),
                "report_prob_good_curve": curve("report_prob_good_curve"),
                "mu_good_curve": curve("mu_good_curve"),
                "concentration_curve": curve("concentration_curve"),
                "final_good_accuracy": final("final_good_accuracy"),
                "final_report_prob_good": final("final_report_prob_good"),
                "final_mu_good": final("final_mu_good"),
                "final_concentration": final("final_concentration"),
                "final_top_k_precision": final("final_top_k_precision"),
                "final_system_score": final("final_system_score"),
            }
        )
    return averaged


def condition_label(result: dict[str, object]) -> str:
    if result["force_report"]:
        return "perfect (θ=1)"
    return f"B={result['budget']:g}"


def plot_trajectories(results: list[dict[str, object]], output_base: Path) -> None:
    fig, (ax_prob, ax_acc) = plt.subplots(1, 2, figsize=(11.0, 4.2))
    for result in results:
        ax_prob.plot(
            result["steps_axis"],
            result["report_prob_good_curve"],
            linewidth=1.9,
            label=result["label"],
        )
        ax_acc.plot(
            result["steps_axis"],
            result["good_accuracy_curve"],
            linewidth=1.9,
            label=result["label"],
        )
    ax_prob.set_title("Report probability on good actors")
    ax_prob.set_xlabel("training step")
    ax_prob.set_ylabel("mean θ(μ) on good actors")
    ax_prob.set_ylim(0.0, 1.02)
    ax_prob.grid(True, axis="y", alpha=0.25)
    ax_prob.legend(frameon=False, fontsize=8)

    ax_acc.set_title("Good-actor accuracy")
    ax_acc.set_xlabel("training step")
    ax_acc.set_ylabel("greedy accuracy of good actors")
    ax_acc.set_ylim(0.0, 1.02)
    ax_acc.grid(True, axis="y", alpha=0.25)
    ax_acc.legend(frameon=False, fontsize=8)

    fig.suptitle(
        f"Reward-budget sanity: {results[0]['num_states']} states, "
        f"{results[0]['num_actors']} actors"
    )
    fig.tight_layout()
    _save_figure(fig, output_base)


def plot_summary(results: list[dict[str, object]], output_base: Path) -> None:
    # Order the budget sweep on the x-axis; the perfect baseline is a flat line.
    sweep = [r for r in results if not r["force_report"]]
    perfect = next((r for r in results if r["force_report"]), None)
    budgets = [r["budget"] for r in sweep]

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.plot(
        budgets,
        [r["final_report_prob_good"] for r in sweep],
        "o--",
        linewidth=2.0,
        markersize=7,
        label="report prob on good actors",
    )
    ax.plot(
        budgets,
        [r["final_good_accuracy"] for r in sweep],
        "s-",
        linewidth=2.0,
        markersize=7,
        label="good-actor accuracy",
    )
    ax.plot(
        budgets,
        [r["final_system_score"] for r in sweep],
        "^-",
        linewidth=1.6,
        markersize=7,
        label="system score",
    )
    if perfect is not None:
        ax.axhline(
            perfect["final_good_accuracy"],
            color="grey",
            linestyle=":",
            linewidth=1.6,
            label="perfect-reward accuracy",
        )
    ax.set_xscale("log", base=2)
    ax.set_title("Final quality vs budget B (approaching perfect reward)")
    ax.set_xlabel("budget B (log scale)")
    ax.set_ylabel("final value")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    _save_figure(fig, output_base)


def _save_figure(fig: plt.Figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png", ".svg"):
        fig.savefig(output_base.with_suffix(suffix), bbox_inches="tight")
    plt.close(fig)


def write_csv(results: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "label",
        "budget",
        "force_report",
        "num_states",
        "num_actors",
        "final_report_prob_good",
        "final_mu_good",
        "final_good_accuracy",
        "final_concentration",
        "final_top_k_precision",
        "final_system_score",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow({field: result[field] for field in fields})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure-dir", type=Path, default=Path("docs/figures"))
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/reward_budget_sanity/reward_budget_sanity_summary.csv"),
    )
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument(
        "--grid-size",
        type=int,
        default=10,
        help="Grid side; number of states is its square (default 10 -> 100).",
    )
    parser.add_argument(
        "--num-actors",
        type=int,
        default=64,
        help="Crowd size for the fixed task (default 64, where Phase 3 collapsed).",
    )
    parser.add_argument(
        "--budgets",
        type=float,
        nargs="+",
        default=[2.0, 8.0, 32.0, 128.0],
        help="Budget sweep for the imperfect-reward condition.",
    )
    parser.add_argument("--num-good", type=int, default=2)
    parser.add_argument("--num-actions", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--utility-lr", type=float, default=0.05)
    parser.add_argument("--following-lr", type=float, default=0.05)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--num-steps", type=int, default=60000)
    parser.add_argument("--snapshot-interval", type=int, default=2000)
    args = parser.parse_args()

    if args.num_actors <= args.num_good:
        raise ValueError("num_actors must be greater than num_good")

    # Perfect-reward baseline uses the smallest budget only to size mu; the
    # gating is bypassed anyway. Use the first budget for it.
    baseline_budget = args.budgets[0]

    seeds = list(range(args.seeds))
    runs_by_seed: list[list[dict[str, object]]] = []
    for seed in seeds:
        seed_runs: list[dict[str, object]] = []
        seed_runs.append(
            train_one_condition(
                grid_size=args.grid_size,
                num_actors=args.num_actors,
                num_good=args.num_good,
                num_actions=args.num_actions,
                hidden_dim=args.hidden_dim,
                learning_rate=args.learning_rate,
                budget=baseline_budget,
                utility_lr=args.utility_lr,
                following_lr=args.following_lr,
                c=args.c,
                force_report=True,
                num_steps=args.num_steps,
                snapshot_interval=args.snapshot_interval,
                seed=seed,
            )
        )
        for budget in args.budgets:
            seed_runs.append(
                train_one_condition(
                    grid_size=args.grid_size,
                    num_actors=args.num_actors,
                    num_good=args.num_good,
                    num_actions=args.num_actions,
                    hidden_dim=args.hidden_dim,
                    learning_rate=args.learning_rate,
                    budget=budget,
                    utility_lr=args.utility_lr,
                    following_lr=args.following_lr,
                    c=args.c,
                    force_report=False,
                    num_steps=args.num_steps,
                    snapshot_interval=args.snapshot_interval,
                    seed=seed,
                )
            )
        runs_by_seed.append(seed_runs)

    results = average_conditions(runs_by_seed)
    plot_trajectories(results, args.figure_dir / "reward_budget_sanity_trajectories")
    plot_summary(results, args.figure_dir / "reward_budget_sanity_summary")
    write_csv(results, args.output_csv)

    print(
        f"seeds={args.seeds}, states={args.grid_size ** 2}, "
        f"actors={args.num_actors}, num_steps={args.num_steps}"
    )
    header = (
        f"{'condition':>14} {'rep_good':>9} {'mu_good':>8} {'good_acc':>9} "
        f"{'concentr':>9} {'system':>7}"
    )
    print(header)
    for result in results:
        print(
            f"{result['label']:>14} "
            f"{result['final_report_prob_good']:>9.3f} "
            f"{result['final_mu_good']:>8.3f} "
            f"{result['final_good_accuracy']:>9.3f} "
            f"{result['final_concentration']:>9.3f} "
            f"{result['final_system_score']:>7.3f}"
        )


if __name__ == "__main__":
    main()
