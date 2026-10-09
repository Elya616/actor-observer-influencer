"""Phase 1 scaling: one Actor, one Observer, grow the number of states.

This is the first scaling step the professor asked for: keep the simplest
1-Actor / 1-Observer setting, keep the action count fixed, and only increase the
number of states. The states live on a structured 2-D grid (see
``GridWorldBanditEnv``) so the best-action map is region-based rather than
i.i.d.-random, which is what makes larger state counts learnable.

For each grid size we train a small MLP policy whose reward is routed through the
Observer, then check two things:

* overall behavior -- greedy accuracy and mean optimal-action probability over
  *all* states;
* representative states -- the optimal-action probability at one cell per region
  (the cell nearest each region center), so we can eyeball individual states even
  when there are hundreds or thousands of them.

The goal is not a single number but a curve: how far can we scale the state count
before the mechanism stops cleanly reaching the correct per-state action?

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
import torch

from actor_observer_rl.agents import Actor, Observer
from actor_observer_rl.envs import GridWorldBanditEnv
from actor_observer_rl.policies import NeuralPolicy
from actor_observer_rl.training import Trainer
from actor_observer_rl.utils.seeding import set_global_seed


def save_figure(fig: plt.Figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png", ".svg"):
        fig.savefig(output_base.with_suffix(suffix), bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def evaluate_all_states(actor: Actor, env: GridWorldBanditEnv) -> tuple[float, float]:
    """Return (greedy accuracy, mean optimal-action probability) over all states."""

    optimal = env.optimal_actions()
    correct = 0
    prob_sum = 0.0
    for index in range(env.num_states):
        feature = env.feature(index)
        distribution = actor.policy.action_distribution(feature)
        probs = distribution.probs
        best_action = optimal[index]
        prob_sum += float(probs[best_action].item())
        if int(probs.argmax().item()) == best_action:
            correct += 1
    return correct / env.num_states, prob_sum / env.num_states


@torch.no_grad()
def evaluate_representative_states(actor: Actor, env: GridWorldBanditEnv) -> list[float]:
    """Optimal-action probability at one representative cell per region."""

    optimal = env.optimal_actions()
    probs_per_region: list[float] = []
    for index in env.representative_states():
        distribution = actor.policy.action_distribution(env.feature(index))
        probs_per_region.append(float(distribution.probs[optimal[index]].item()))
    return probs_per_region


def train_one_grid(
    grid_size: int,
    num_actions: int,
    hidden_dim: int,
    learning_rate: float,
    num_steps: int,
    eval_interval: int,
    seed: int,
) -> dict[str, object]:
    set_global_seed(seed)
    env = GridWorldBanditEnv(grid_size=grid_size, num_actions=num_actions)
    policy = NeuralPolicy(input_dim=2, hidden_dim=hidden_dim, num_actions=num_actions)
    actor = Actor(policy=policy)
    optimizer = torch.optim.Adam(actor.parameters(), lr=learning_rate)
    trainer = Trainer(
        env=env,
        actor=actor,
        actor_optimizer=optimizer,
        observer=Observer(),
        optimal_actions=env.optimal_actions(),
        reward_baseline="running_mean",
    )

    steps_axis: list[int] = []
    accuracy_curve: list[float] = []
    trained = 0
    is_first = True
    while trained < num_steps:
        segment = min(eval_interval, num_steps - trained)
        trainer.run(num_steps=segment, seed=seed if is_first else None)
        is_first = False
        trained += segment
        accuracy, _ = evaluate_all_states(actor, env)
        steps_axis.append(trained)
        accuracy_curve.append(accuracy)

    final_accuracy, final_mean_prob = evaluate_all_states(actor, env)
    representative = evaluate_representative_states(actor, env)
    return {
        "grid_size": grid_size,
        "num_states": env.num_states,
        "steps_axis": steps_axis,
        "accuracy_curve": accuracy_curve,
        "final_accuracy": final_accuracy,
        "final_mean_prob": final_mean_prob,
        "representative_mean": sum(representative) / len(representative),
        "representative": representative,
    }


def average_runs(runs: list[list[dict[str, object]]]) -> list[dict[str, object]]:
    """Average per-grid results across seeds (outer list = seeds)."""

    num_grids = len(runs[0])
    averaged: list[dict[str, object]] = []
    for g in range(num_grids):
        grid_runs = [seed_runs[g] for seed_runs in runs]
        num_points = min(len(r["accuracy_curve"]) for r in grid_runs)
        steps_axis = grid_runs[0]["steps_axis"][:num_points]
        accuracy_curve = [
            sum(r["accuracy_curve"][k] for r in grid_runs) / len(grid_runs)
            for k in range(num_points)
        ]
        averaged.append(
            {
                "grid_size": grid_runs[0]["grid_size"],
                "num_states": grid_runs[0]["num_states"],
                "steps_axis": steps_axis,
                "accuracy_curve": accuracy_curve,
                "final_accuracy": sum(r["final_accuracy"] for r in grid_runs)
                / len(grid_runs),
                "final_mean_prob": sum(r["final_mean_prob"] for r in grid_runs)
                / len(grid_runs),
                "representative_mean": sum(r["representative_mean"] for r in grid_runs)
                / len(grid_runs),
            }
        )
    return averaged


def plot_curves(results: list[dict[str, object]], output_base: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for result in results:
        ax.plot(
            result["steps_axis"],
            result["accuracy_curve"],
            linewidth=1.9,
            label=f"{result['grid_size']}x{result['grid_size']} "
            f"({result['num_states']} states)",
        )
    ax.set_title("Phase 1: greedy accuracy while scaling states (1 Actor, 1 Observer)")
    ax.set_xlabel("training step")
    ax.set_ylabel("fraction of states with correct greedy action")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save_figure(fig, output_base)


def plot_summary(results: list[dict[str, object]], output_base: Path) -> None:
    num_states = [result["num_states"] for result in results]
    final_accuracy = [result["final_accuracy"] for result in results]
    final_mean_prob = [result["final_mean_prob"] for result in results]
    representative_mean = [result["representative_mean"] for result in results]

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(
        num_states,
        final_accuracy,
        "o--",
        linewidth=2.4,
        markersize=8,
        label="greedy accuracy",
    )
    ax.plot(num_states, final_mean_prob, "s-", linewidth=1.6, label="mean P(correct action)")
    ax.plot(
        num_states,
        representative_mean,
        "^-",
        linewidth=1.9,
        label="representative-state mean P(correct)",
    )
    ax.set_xscale("log", base=2)
    ax.set_title("Phase 1: final quality vs number of states (equal compute)")
    ax.set_xlabel("number of states (log scale)")
    ax.set_ylabel("final value")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save_figure(fig, output_base)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure-dir", type=Path, default=Path("docs/figures"))
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument(
        "--grid-sizes",
        type=int,
        nargs="+",
        default=[2, 4, 8, 16, 32],
    )
    parser.add_argument("--num-actions", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--eval-interval", type=int, default=2000)
    parser.add_argument(
        "--num-steps",
        type=int,
        default=40000,
        help="Fixed training-step budget used for every grid size, so the "
        "scaling comparison is at equal compute.",
    )
    args = parser.parse_args()

    seeds = list(range(args.seeds))
    runs_by_seed: list[list[dict[str, object]]] = []
    for seed in seeds:
        seed_runs: list[dict[str, object]] = []
        for grid_size in args.grid_sizes:
            seed_runs.append(
                train_one_grid(
                    grid_size=grid_size,
                    num_actions=args.num_actions,
                    hidden_dim=args.hidden_dim,
                    learning_rate=args.learning_rate,
                    num_steps=args.num_steps,
                    eval_interval=args.eval_interval,
                    seed=seed,
                )
            )
        runs_by_seed.append(seed_runs)

    results = average_runs(runs_by_seed)
    plot_curves(results, args.figure_dir / "scaling_states_curves")
    plot_summary(results, args.figure_dir / "scaling_states_summary")

    print(f"seeds={args.seeds}, num_actions={args.num_actions}")
    print(
        f"{'grid':>7} {'states':>7} {'greedy_acc':>11} "
        f"{'mean_P_opt':>11} {'rep_P_opt':>10}"
    )
    for result in results:
        print(
            f"{result['grid_size']:>2}x{result['grid_size']:<4} "
            f"{result['num_states']:>7} "
            f"{result['final_accuracy']:>11.3f} "
            f"{result['final_mean_prob']:>11.3f} "
            f"{result['representative_mean']:>10.3f}"
        )


if __name__ == "__main__":
    main()
