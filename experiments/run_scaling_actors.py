"""Phase 2 scaling: one Observer, grow the number of Actors (Section 4 setting).

Phase 1 grew the number of states with a single Actor and Observer. Phase 2
keeps the task tiny (one state, one action per Actor) and instead grows the
number of Actors that a single ``BudgetedObserver`` has to watch.

The setup is the paper's Section 4.1 mechanism, used as a scaling probe:

* ``num_actors = N`` fixed-policy Actors. Actor ``j`` always plays action ``j``,
  so "follow Actor ``j``" just means "the Observer puts budget ``mu_j`` on it".
* A fixed set of ``k`` Actors are *good* (mean reward ``good_quality``); the
  remaining ``N - k`` are *bad* (mean reward ``bad_quality``). Rewards are noisy
  (``NoisyActorBanditEnv``), so the Observer needs several samples per Actor to
  tell them apart.
* The Observer holds a fixed attention budget ``B`` on the capped simplex and
  does projected gradient ascent on ``sum_j theta(mu_j) U_hat_j``.

The crowd ``N`` grows while ``k``, ``B`` and the total step budget stay fixed.
The question: can the budgeted-following mechanism still concentrate its fixed
budget on the fixed set of good Actors as the crowd grows, at equal compute?

Diagnostics (per N, averaged over seeds):

* budget concentration -- fraction of the budget sitting on the good Actors,
  ``sum_{j good} mu_j / B`` (ideal 1.0);
* top-k precision -- of the ``k`` Actors with the largest ``mu``, the fraction
  that are truly good (ideal 1.0);
* mean ``mu`` on good vs bad Actors.

Figures (pdf/png/svg) are written to ``docs/figures``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".mplcache").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from actor_observer_rl.agents import BudgetedObserver
from actor_observer_rl.envs import NoisyActorBanditEnv
from actor_observer_rl.training import MultiActorTrainer, make_fixed_softmax_actor
from actor_observer_rl.utils.seeding import set_global_seed


def save_figure(fig: plt.Figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png", ".svg"):
        fig.savefig(output_base.with_suffix(suffix), bbox_inches="tight")
    plt.close(fig)


def concentration(mu: list[float], num_good: int, budget: float) -> float:
    """Fraction of the budget that sits on the good Actors (indices 0..k-1)."""

    return sum(mu[:num_good]) / budget


def top_k_precision(mu: list[float], num_good: int) -> float:
    """Fraction of the top-k highest-mu Actors that are truly good."""

    ranked = sorted(range(len(mu)), key=lambda j: mu[j], reverse=True)
    top = ranked[:num_good]
    hits = sum(1 for j in top if j < num_good)
    return hits / num_good


def train_one_crowd(
    num_actors: int,
    num_good: int,
    good_quality: float,
    bad_quality: float,
    noise_std: float,
    budget: float,
    utility_lr: float,
    following_lr: float,
    c: float,
    sharpness: float,
    num_steps: int,
    snapshot_interval: int,
    seed: int,
) -> dict[str, object]:
    set_global_seed(seed)
    qualities = [good_quality] * num_good + [bad_quality] * (num_actors - num_good)
    env = NoisyActorBanditEnv(qualities=qualities, noise_std=noise_std)
    actors = [
        make_fixed_softmax_actor(
            num_states=1,
            num_actions=num_actors,
            target_action=j,
            sharpness=sharpness,
        )
        for j in range(num_actors)
    ]
    observer = BudgetedObserver(
        num_actors=num_actors,
        budget=budget,
        utility_learning_rate=utility_lr,
        following_learning_rate=following_lr,
        c=c,
        seed=seed,
    )
    trainer = MultiActorTrainer(
        env=env,
        actors=actors,
        observer=observer,
        actor_selection_probs=[1.0 / num_actors] * num_actors,
        actor_optimizers=None,
        snapshot_interval=snapshot_interval,
    )
    trace = trainer.run(num_steps=num_steps, seed=seed)

    steps_axis = [
        min((i + 1) * snapshot_interval, num_steps)
        for i in range(len(trace.following_rates))
    ]
    concentration_curve = [
        concentration(mu, num_good, budget) for mu in trace.following_rates
    ]

    final_mu = trace.final_following_rates or [0.0] * num_actors
    mean_mu_good = sum(final_mu[:num_good]) / num_good
    num_bad = num_actors - num_good
    mean_mu_bad = sum(final_mu[num_good:]) / num_bad if num_bad else 0.0
    return {
        "num_actors": num_actors,
        "steps_axis": steps_axis,
        "concentration_curve": concentration_curve,
        "final_concentration": concentration(final_mu, num_good, budget),
        "final_top_k_precision": top_k_precision(final_mu, num_good),
        "mean_mu_good": mean_mu_good,
        "mean_mu_bad": mean_mu_bad,
    }


def average_runs(runs: list[list[dict[str, object]]]) -> list[dict[str, object]]:
    """Average per-crowd results across seeds (outer list = seeds)."""

    num_crowds = len(runs[0])
    averaged: list[dict[str, object]] = []
    for g in range(num_crowds):
        crowd_runs = [seed_runs[g] for seed_runs in runs]
        num_points = min(len(r["concentration_curve"]) for r in crowd_runs)
        steps_axis = crowd_runs[0]["steps_axis"][:num_points]
        concentration_curve = [
            sum(r["concentration_curve"][k] for r in crowd_runs) / len(crowd_runs)
            for k in range(num_points)
        ]
        averaged.append(
            {
                "num_actors": crowd_runs[0]["num_actors"],
                "steps_axis": steps_axis,
                "concentration_curve": concentration_curve,
                "final_concentration": sum(r["final_concentration"] for r in crowd_runs)
                / len(crowd_runs),
                "final_top_k_precision": sum(
                    r["final_top_k_precision"] for r in crowd_runs
                )
                / len(crowd_runs),
                "mean_mu_good": sum(r["mean_mu_good"] for r in crowd_runs)
                / len(crowd_runs),
                "mean_mu_bad": sum(r["mean_mu_bad"] for r in crowd_runs)
                / len(crowd_runs),
            }
        )
    return averaged


def plot_curves(results: list[dict[str, object]], output_base: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for result in results:
        ax.plot(
            result["steps_axis"],
            result["concentration_curve"],
            linewidth=1.9,
            label=f"{result['num_actors']} actors",
        )
    ax.set_title("Phase 2: budget concentration while scaling actors (1 Observer)")
    ax.set_xlabel("training step")
    ax.set_ylabel("fraction of budget on good actors")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save_figure(fig, output_base)


def plot_summary(results: list[dict[str, object]], output_base: Path) -> None:
    num_actors = [result["num_actors"] for result in results]
    concentration_values = [result["final_concentration"] for result in results]
    precision_values = [result["final_top_k_precision"] for result in results]

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(
        num_actors,
        concentration_values,
        "o--",
        linewidth=2.4,
        markersize=8,
        label="budget concentration on good actors",
    )
    ax.plot(
        num_actors,
        precision_values,
        "s-",
        linewidth=1.6,
        label="top-k precision",
    )
    ax.set_xscale("log", base=2)
    ax.set_title("Phase 2: final quality vs number of actors (equal compute)")
    ax.set_xlabel("number of actors (log scale)")
    ax.set_ylabel("final value")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save_figure(fig, output_base)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure-dir", type=Path, default=Path("docs/figures"))
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument(
        "--num-actors",
        type=int,
        nargs="+",
        default=[4, 8, 16, 32, 64, 128, 256],
    )
    parser.add_argument("--num-good", type=int, default=2)
    parser.add_argument("--good-quality", type=float, default=1.0)
    parser.add_argument("--bad-quality", type=float, default=0.0)
    parser.add_argument("--noise-std", type=float, default=1.5)
    parser.add_argument("--budget", type=float, default=2.0)
    parser.add_argument("--utility-lr", type=float, default=0.05)
    parser.add_argument("--following-lr", type=float, default=0.05)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--sharpness", type=float, default=8.0)
    parser.add_argument(
        "--num-steps",
        type=int,
        default=12000,
        help="Fixed total-step budget used for every crowd size, so the "
        "scaling comparison is at equal compute.",
    )
    parser.add_argument("--snapshot-interval", type=int, default=1000)
    args = parser.parse_args()

    seeds = list(range(args.seeds))
    runs_by_seed: list[list[dict[str, object]]] = []
    for seed in seeds:
        seed_runs: list[dict[str, object]] = []
        for num_actors in args.num_actors:
            if num_actors <= args.num_good:
                raise ValueError("num_actors must be greater than num_good")
            seed_runs.append(
                train_one_crowd(
                    num_actors=num_actors,
                    num_good=args.num_good,
                    good_quality=args.good_quality,
                    bad_quality=args.bad_quality,
                    noise_std=args.noise_std,
                    budget=args.budget,
                    utility_lr=args.utility_lr,
                    following_lr=args.following_lr,
                    c=args.c,
                    sharpness=args.sharpness,
                    num_steps=args.num_steps,
                    snapshot_interval=args.snapshot_interval,
                    seed=seed,
                )
            )
        runs_by_seed.append(seed_runs)

    results = average_runs(runs_by_seed)
    plot_curves(results, args.figure_dir / "scaling_actors_curves")
    plot_summary(results, args.figure_dir / "scaling_actors_summary")

    print(
        f"seeds={args.seeds}, num_good={args.num_good}, budget={args.budget}, "
        f"noise_std={args.noise_std}, num_steps={args.num_steps}"
    )
    print(
        f"{'actors':>7} {'concentration':>14} {'top_k_prec':>11} "
        f"{'mu_good':>9} {'mu_bad':>9}"
    )
    for result in results:
        print(
            f"{result['num_actors']:>7} "
            f"{result['final_concentration']:>14.3f} "
            f"{result['final_top_k_precision']:>11.3f} "
            f"{result['mean_mu_good']:>9.3f} "
            f"{result['mean_mu_bad']:>9.4f}"
        )


if __name__ == "__main__":
    main()
