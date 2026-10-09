"""Phase 3 scaling: grow the number of states *and* actors together.

Phase 1 grew the state count with a single Actor and Observer. Phase 2 grew the
Actor count with a tiny one-state task. Phase 3 is the combination the professor
asked for: scale both axes at once and see whether the one-Observer / many-Actor
machinery still works when the task has many states *and* the crowd is large.

The setup reuses the existing pieces rather than building new ones:

* a structured ``GridWorldBanditEnv`` with ``grid_size ** 2`` states (Phase 1);
* ``N`` Actors watched by one ``BudgetedObserver`` with a fixed attention budget
  ``B`` (Phase 2 / Section 4.2);
* ``k`` *good* Actors that see the true 2-D grid feature and learn a state-
  dependent ``NeuralPolicy``; the remaining ``N - k`` *blind* Actors see a zeroed
  feature, so they can only learn a single state-independent action. This mirrors
  the good/bad split of Phase 2, but now "good" means "can use the state".

Each rung of the ladder grows ``grid_size`` and ``N`` together, at a fixed total
step budget ``T`` (equal compute). One Actor is sampled uniformly per step; it
acts on the current state, the Observer updates its utility estimate, and -- when
the Observer's Bernoulli report ``theta(mu_j)`` fires -- that Actor takes a
REINFORCE step (with a per-actor running-mean baseline, needed because the grid
reward is all-positive).

Diagnostics (per rung, averaged over seeds):

* good-actor accuracy -- greedy per-state accuracy of the good Actors over all
  states (does the followed Actor actually solve the grid?);
* budget concentration -- ``sum_{j good} mu_j / B`` (did the Observer put its
  budget on the Actors that can use the state?);
* top-k precision -- of the ``k`` highest-``mu`` Actors, the fraction truly good;
* system score -- budget-weighted greedy accuracy ``sum_j (mu_j / B) * acc_j``,
  which is high only when the good Actors learn *and* the Observer follows them.

Figures (pdf/png/svg) go to ``docs/figures``; a summary CSV goes to ``outputs``.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".mplcache").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from actor_observer_rl.agents import Actor, BudgetedObserver
from actor_observer_rl.envs import GridWorldBanditEnv
from actor_observer_rl.policies import NeuralPolicy
from actor_observer_rl.training import MultiActorTrainer
from actor_observer_rl.utils.seeding import set_global_seed


class BlindActor(Actor):
    """Actor that cannot see the state: it zeros its input before acting.

    This is the low-information ("bad") Actor of the Phase 3 crowd. With a zero
    feature it can only learn a single state-independent action, so its ceiling
    is the best constant action averaged over the grid -- below what a good Actor
    reaches once it has learned the per-region policy.
    """

    def act(self, input_tensor: torch.Tensor, sample: bool = True):
        return self.policy.act(torch.zeros_like(input_tensor), sample=sample)


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


@torch.no_grad()
def greedy_accuracy(actor: Actor, env: GridWorldBanditEnv, blind: bool) -> float:
    """Greedy per-state accuracy over all states.

    A blind Actor sees a zero feature, so its distribution is the same for every
    state; we evaluate it on the same zero feature it acts on.
    """

    optimal = env.optimal_actions()
    correct = 0
    for index in range(env.num_states):
        feature = env.feature(index)
        if blind:
            feature = torch.zeros_like(feature)
        distribution = actor.policy.action_distribution(feature)
        if int(distribution.probs.argmax().item()) == optimal[index]:
            correct += 1
    return correct / env.num_states


def build_actor(num_actions: int, hidden_dim: int, blind: bool) -> Actor:
    policy = NeuralPolicy(input_dim=2, hidden_dim=hidden_dim, num_actions=num_actions)
    return BlindActor(policy=policy) if blind else Actor(policy=policy)


def train_one_rung(
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
    num_steps: int,
    snapshot_interval: int,
    seed: int,
) -> dict[str, object]:
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
    trainer = MultiActorTrainer(
        env=env,
        actors=actors,
        observer=observer,
        actor_selection_probs=[1.0 / num_actors] * num_actors,
        actor_optimizers=optimizers,
        reward_baseline="running_mean",
        snapshot_interval=snapshot_interval,
    )

    # Train in segments so we can trace good-actor accuracy alongside the
    # Observer's budget concentration.
    steps_axis: list[int] = []
    concentration_curve: list[float] = []
    good_accuracy_curve: list[float] = []
    trained = 0
    is_first = True
    while trained < num_steps:
        segment = min(snapshot_interval, num_steps - trained)
        trainer.run(num_steps=segment, seed=seed if is_first else None)
        is_first = False
        trained += segment
        mu = observer.current_following_rates()
        good_acc = sum(
            greedy_accuracy(actors[j], env, blind=False) for j in range(num_good)
        ) / num_good
        steps_axis.append(trained)
        concentration_curve.append(concentration(mu, num_good, budget))
        good_accuracy_curve.append(good_acc)

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
        "steps_axis": steps_axis,
        "concentration_curve": concentration_curve,
        "good_accuracy_curve": good_accuracy_curve,
        "final_good_accuracy": good_accuracy_curve[-1],
        "final_concentration": concentration(final_mu, num_good, budget),
        "final_top_k_precision": top_k_precision(final_mu, num_good),
        "final_system_score": system_score,
    }


def average_runs(runs: list[list[dict[str, object]]]) -> list[dict[str, object]]:
    """Average per-rung results across seeds (outer list = seeds)."""

    num_rungs = len(runs[0])
    averaged: list[dict[str, object]] = []
    for g in range(num_rungs):
        rung = [seed_runs[g] for seed_runs in runs]
        n = len(rung)
        num_points = min(len(r["good_accuracy_curve"]) for r in rung)
        averaged.append(
            {
                "grid_size": rung[0]["grid_size"],
                "num_states": rung[0]["num_states"],
                "num_actors": rung[0]["num_actors"],
                "steps_axis": rung[0]["steps_axis"][:num_points],
                "good_accuracy_curve": [
                    sum(r["good_accuracy_curve"][k] for r in rung) / n
                    for k in range(num_points)
                ],
                "concentration_curve": [
                    sum(r["concentration_curve"][k] for r in rung) / n
                    for k in range(num_points)
                ],
                "final_good_accuracy": sum(r["final_good_accuracy"] for r in rung) / n,
                "final_concentration": sum(r["final_concentration"] for r in rung) / n,
                "final_top_k_precision": sum(
                    r["final_top_k_precision"] for r in rung
                )
                / n,
                "final_system_score": sum(r["final_system_score"] for r in rung) / n,
            }
        )
    return averaged


def rung_label(result: dict[str, object]) -> str:
    return f"{result['num_states']} states, {result['num_actors']} actors"


def plot_curves(results: list[dict[str, object]], output_base: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for result in results:
        ax.plot(
            result["steps_axis"],
            result["good_accuracy_curve"],
            linewidth=1.9,
            label=rung_label(result),
        )
    ax.set_title("Phase 3: good-actor accuracy while scaling states and actors")
    ax.set_xlabel("training step")
    ax.set_ylabel("greedy accuracy of good actors")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save_figure(fig, output_base)


def plot_summary(results: list[dict[str, object]], output_base: Path) -> None:
    labels = [rung_label(result) for result in results]
    x = list(range(len(results)))

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot(
        x,
        [r["final_good_accuracy"] for r in results],
        "o--",
        linewidth=2.4,
        markersize=8,
        label="good-actor accuracy",
    )
    ax.plot(
        x,
        [r["final_concentration"] for r in results],
        "s-",
        linewidth=1.6,
        label="budget concentration on good actors",
    )
    ax.plot(
        x,
        [r["final_system_score"] for r in results],
        "^-",
        linewidth=1.6,
        label="system score (budget-weighted accuracy)",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_title("Phase 3: final quality along the joint ladder (equal compute)")
    ax.set_ylabel("final value")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save_figure(fig, output_base)


def write_csv(results: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "grid_size",
        "num_states",
        "num_actors",
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
        default=Path("outputs/scaling_states_actors/scaling_states_actors_summary.csv"),
    )
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument(
        "--grid-sizes",
        type=int,
        nargs="+",
        default=[4, 6, 8, 10, 12],
        help="Grid side per rung; number of states is the square of this.",
    )
    parser.add_argument(
        "--num-actors",
        type=int,
        nargs="+",
        default=[8, 16, 32, 64, 128],
        help="Crowd size per rung; must match --grid-sizes in length.",
    )
    parser.add_argument("--num-good", type=int, default=2)
    parser.add_argument("--num-actions", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--budget", type=float, default=2.0)
    parser.add_argument("--utility-lr", type=float, default=0.05)
    parser.add_argument("--following-lr", type=float, default=0.05)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument(
        "--num-steps",
        type=int,
        default=60000,
        help="Fixed total-step budget used for every rung, so the joint scaling "
        "comparison is at equal compute.",
    )
    parser.add_argument("--snapshot-interval", type=int, default=2000)
    args = parser.parse_args()

    if len(args.grid_sizes) != len(args.num_actors):
        raise ValueError("--grid-sizes and --num-actors must have equal length")

    seeds = list(range(args.seeds))
    runs_by_seed: list[list[dict[str, object]]] = []
    for seed in seeds:
        seed_runs: list[dict[str, object]] = []
        for grid_size, num_actors in zip(args.grid_sizes, args.num_actors):
            if num_actors <= args.num_good:
                raise ValueError("num_actors must be greater than num_good")
            seed_runs.append(
                train_one_rung(
                    grid_size=grid_size,
                    num_actors=num_actors,
                    num_good=args.num_good,
                    num_actions=args.num_actions,
                    hidden_dim=args.hidden_dim,
                    learning_rate=args.learning_rate,
                    budget=args.budget,
                    utility_lr=args.utility_lr,
                    following_lr=args.following_lr,
                    c=args.c,
                    num_steps=args.num_steps,
                    snapshot_interval=args.snapshot_interval,
                    seed=seed,
                )
            )
        runs_by_seed.append(seed_runs)

    results = average_runs(runs_by_seed)
    plot_curves(results, args.figure_dir / "scaling_states_actors_curves")
    plot_summary(results, args.figure_dir / "scaling_states_actors_summary")
    write_csv(results, args.output_csv)

    print(
        f"seeds={args.seeds}, num_good={args.num_good}, budget={args.budget}, "
        f"num_steps={args.num_steps}"
    )
    header = (
        f"{'states':>7} {'actors':>7} {'good_acc':>9} {'concentr':>9} "
        f"{'top_k':>6} {'system':>7}"
    )
    print(header)
    for result in results:
        print(
            f"{result['num_states']:>7} {result['num_actors']:>7} "
            f"{result['final_good_accuracy']:>9.3f} "
            f"{result['final_concentration']:>9.3f} "
            f"{result['final_top_k_precision']:>6.2f} "
            f"{result['final_system_score']:>7.3f}"
        )


if __name__ == "__main__":
    main()
