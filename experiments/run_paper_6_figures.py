"""Mechanism-level figures for paper Sections 6.1 and 6.2.

The group-level imperfect-reward track has ``N`` Observers, each with its own
reward function ``R_i`` and its own budgeted following rate ``mu_{i,j}``. This
script runs controlled experiments with known answers and plots the internal
quantities over time:

* ``paper6_1_assortative`` -- N Observers, single state, Actor ``j`` pinned to
  action ``j``. Observer ``i`` most values Actor ``i`` (diagonal reward), so
  each ``mu_i`` should concentrate on the matching Actor -- a "specialization"
  pattern. One subplot per Observer shows ``mu_{i,j}(t)``.
* ``paper6_2_joint`` -- N Observers share a diagonal reward; Actors learn
  jointly via the Eq. (29) Observer-aggregated reward. Tracks per-Actor optimal
  action probability and Observer 0's ``mu_{0,j}(t)`` / ``U_hat_{0,j}(t)``.

Figures are written (pdf/png/svg) to ``docs/figures``.
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

from actor_observer_rl.agents import Actor, BudgetedObserver
from actor_observer_rl.envs import IIDStateBanditEnv
from actor_observer_rl.policies import SoftmaxPolicy
from actor_observer_rl.training import (
    GroupMultiActorTrainer,
    GroupMultiActorTrainingTrace,
    make_fixed_softmax_actor,
)
from actor_observer_rl.utils.seeding import set_global_seed


def save_figure(fig: plt.Figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png", ".svg"):
        fig.savefig(output_base.with_suffix(suffix), bbox_inches="tight")
    plt.close(fig)


def average_group_series(
    traces: list[GroupMultiActorTrainingTrace],
    attribute: str,
    snapshot_interval: int,
) -> tuple[list[int], list[list[list[float]]]]:
    """Average a ``[snapshot][observer][actor]`` quantity across seeds.

    Returns ``(steps, series[observer][actor][snapshot])``.
    """

    snapshots = [getattr(trace, attribute) for trace in traces]
    num_snapshots = min(len(snapshot) for snapshot in snapshots)
    num_observers = len(snapshots[0][0])
    num_actors = len(snapshots[0][0][0])
    steps = [(k + 1) * snapshot_interval for k in range(num_snapshots)]
    series = [
        [[0.0] * num_snapshots for _ in range(num_actors)]
        for _ in range(num_observers)
    ]
    for k in range(num_snapshots):
        for i in range(num_observers):
            for j in range(num_actors):
                values = [snapshot[k][i][j] for snapshot in snapshots]
                series[i][j][k] = sum(values) / len(values)
    return steps, series


def average_per_actor_series(
    traces: list[GroupMultiActorTrainingTrace],
    snapshot_interval: int,
) -> tuple[list[int], list[list[float]]]:
    snapshots = [trace.per_actor_optimal_action_probs for trace in traces]
    num_snapshots = min(len(snapshot) for snapshot in snapshots)
    num_actors = len(snapshots[0][0])
    steps = [(k + 1) * snapshot_interval for k in range(num_snapshots)]
    series = [[0.0] * num_snapshots for _ in range(num_actors)]
    for k in range(num_snapshots):
        for j in range(num_actors):
            values = [snapshot[k][j] for snapshot in snapshots]
            series[j][k] = sum(values) / len(values)
    return steps, series


def run_section_6_1_assortative(
    observer_reward_tables: list[list[list[float]]],
    seeds: list[int],
    budget: float,
    c: float,
    utility_lr: float,
    following_lr: float,
    num_steps: int,
    snapshot_interval: int,
) -> list[GroupMultiActorTrainingTrace]:
    num_observers = len(observer_reward_tables)
    num_actions = len(observer_reward_tables[0][0])
    traces: list[GroupMultiActorTrainingTrace] = []
    for seed in seeds:
        set_global_seed(seed)
        env = IIDStateBanditEnv(
            reward_table=observer_reward_tables[0],
            state_distribution=[1.0],
            observation_mode="one_hot",
        )
        actors = [
            make_fixed_softmax_actor(
                num_states=1, num_actions=num_actions, target_action=j, sharpness=8.0
            )
            for j in range(num_actions)
        ]
        observers = [
            BudgetedObserver(
                num_actors=num_actions,
                budget=budget,
                utility_learning_rate=utility_lr,
                following_learning_rate=following_lr,
                c=c,
                seed=seed + i,
            )
            for i in range(num_observers)
        ]
        trainer = GroupMultiActorTrainer(
            env=env,
            actors=actors,
            observers=observers,
            observer_reward_tables=observer_reward_tables,
            actor_selection_probs=[1.0 / num_actions] * num_actions,
            actor_optimizers=None,
            snapshot_interval=snapshot_interval,
        )
        traces.append(trainer.run(num_steps=num_steps, seed=seed))
    return traces


def run_section_6_2_joint(
    reward_table: list[list[float]],
    num_observers: int,
    optimal_actions: list[int],
    seeds: list[int],
    budget: float,
    c: float,
    utility_lr: float,
    following_lr: float,
    actor_lr: float,
    num_steps: int,
    snapshot_interval: int,
) -> list[GroupMultiActorTrainingTrace]:
    num_states = len(reward_table)
    num_actions = len(reward_table[0])
    num_actors = num_states
    observer_reward_tables = [reward_table for _ in range(num_observers)]
    traces: list[GroupMultiActorTrainingTrace] = []
    for seed in seeds:
        set_global_seed(seed)
        env = IIDStateBanditEnv(
            reward_table=reward_table,
            state_distribution=[1.0 / num_states] * num_states,
            observation_mode="one_hot",
        )
        actors = [
            Actor(policy=SoftmaxPolicy(input_dim=num_states, num_actions=num_actions))
            for _ in range(num_actors)
        ]
        optimizers = [
            torch.optim.SGD(actor.parameters(), lr=actor_lr) for actor in actors
        ]
        observers = [
            BudgetedObserver(
                num_actors=num_actors,
                budget=budget,
                utility_learning_rate=utility_lr,
                following_learning_rate=following_lr,
                c=c,
                init_mu=[budget / num_actors] * num_actors,
                seed=seed + i,
            )
            for i in range(num_observers)
        ]
        trainer = GroupMultiActorTrainer(
            env=env,
            actors=actors,
            observers=observers,
            observer_reward_tables=observer_reward_tables,
            actor_selection_probs=[1.0 / num_actors] * num_actors,
            actor_optimizers=optimizers,
            optimal_actions_per_state=optimal_actions,
            snapshot_interval=snapshot_interval,
        )
        traces.append(trainer.run(num_steps=num_steps, seed=seed))
    return traces


def plot_assortative(
    traces: list[GroupMultiActorTrainingTrace],
    observer_reward_tables: list[list[list[float]]],
    snapshot_interval: int,
    output_base: Path,
) -> None:
    steps, mu_series = average_group_series(traces, "following_rates", snapshot_interval)
    num_observers = len(mu_series)
    num_actors = len(mu_series[0])

    fig, axes = plt.subplots(
        1, num_observers, figsize=(4.4 * num_observers, 3.4), sharey=True
    )
    if num_observers == 1:
        axes = [axes]
    for i in range(num_observers):
        utilities = observer_reward_tables[i][0]
        preferred = max(range(num_actors), key=utilities.__getitem__)
        for j in range(num_actors):
            axes[i].plot(
                steps,
                mu_series[i][j],
                linewidth=1.9,
                label=rf"actor {j} ($U={utilities[j]:.2f}$)",
            )
        axes[i].set_title(f"Observer {i} (prefers actor {preferred})")
        axes[i].set_xlabel("training step")
        axes[i].grid(True, axis="y", alpha=0.25)
        axes[i].legend(frameon=False, fontsize=8)
    axes[0].set_ylabel(r"$\mu_{i,j}$")

    fig.suptitle(
        f"§6.1 assortative following (mean over {len(traces)} seeds): "
        "each Observer concentrates on the Actor it values most",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_figure(fig, output_base)


def plot_joint(
    traces: list[GroupMultiActorTrainingTrace],
    snapshot_interval: int,
    output_base: Path,
) -> None:
    steps_p, p_series = average_per_actor_series(traces, snapshot_interval)
    steps_mu, mu_series = average_group_series(
        traces, "following_rates", snapshot_interval
    )
    steps_u, u_series = average_group_series(
        traces, "utility_estimates", snapshot_interval
    )

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.4))

    for j, series in enumerate(p_series):
        axes[0].plot(steps_p, series, linewidth=1.9, label=f"actor {j}")
    axes[0].set_title("§6.2: optimal-action prob per actor")
    axes[0].set_xlabel("training step")
    axes[0].set_ylabel(r"$\overline{P}(a^\star \mid s)$")
    axes[0].set_ylim(0.0, 1.02)
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)

    for j, series in enumerate(mu_series[0]):
        axes[1].plot(steps_mu, series, linewidth=1.9, label=f"actor {j}")
    axes[1].set_title(r"§6.2: Observer 0 following rate $\mu_{0,j}(t)$")
    axes[1].set_xlabel("training step")
    axes[1].set_ylabel(r"$\mu_{0,j}$")
    axes[1].grid(True, axis="y", alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)

    for j, series in enumerate(u_series[0]):
        axes[2].plot(steps_u, series, linewidth=1.9, label=f"actor {j}")
    axes[2].set_title(r"§6.2: Observer 0 utility estimate $\hat U_{0,j}(t)$")
    axes[2].set_xlabel("training step")
    axes[2].set_ylabel(r"$\hat U_{0,j}$")
    axes[2].grid(True, axis="y", alpha=0.25)
    axes[2].legend(frameon=False, fontsize=8)

    fig.suptitle(f"§6.2 group joint learning (mean over {len(traces)} seeds)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_figure(fig, output_base)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure-dir", type=Path, default=Path("docs/figures"))
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()

    seeds = list(range(args.seeds))

    assortative_tables = [
        [[0.9, 0.4, 0.1]],
        [[0.1, 0.9, 0.4]],
        [[0.4, 0.1, 0.9]],
    ]
    assortative_traces = run_section_6_1_assortative(
        observer_reward_tables=assortative_tables,
        seeds=seeds,
        budget=2.0,
        c=1.0,
        utility_lr=0.1,
        following_lr=0.05,
        num_steps=8000,
        snapshot_interval=80,
    )
    plot_assortative(
        assortative_traces,
        observer_reward_tables=assortative_tables,
        snapshot_interval=80,
        output_base=args.figure_dir / "paper6_1_assortative",
    )

    joint_traces = run_section_6_2_joint(
        reward_table=[[1.0, 0.1, 0.2], [0.1, 1.0, 0.0], [0.0, 0.2, 1.0]],
        num_observers=3,
        optimal_actions=[0, 1, 2],
        seeds=seeds,
        budget=3.0,
        c=2.0,
        utility_lr=0.2,
        following_lr=0.01,
        actor_lr=0.1,
        num_steps=30000,
        snapshot_interval=300,
    )
    plot_joint(
        joint_traces,
        snapshot_interval=300,
        output_base=args.figure_dir / "paper6_2_joint",
    )

    _print_assortative_summary(assortative_traces, assortative_tables)
    _print_joint_summary(joint_traces)


def _print_assortative_summary(
    traces: list[GroupMultiActorTrainingTrace],
    observer_reward_tables: list[list[list[float]]],
) -> None:
    num_observers = len(observer_reward_tables)
    num_actors = len(observer_reward_tables[0][0])
    for i in range(num_observers):
        final_mu = [0.0] * num_actors
        for trace in traces:
            for j in range(num_actors):
                final_mu[j] += trace.final_following_rates[i][j] / len(traces)
        preferred = max(range(num_actors), key=observer_reward_tables[i][0].__getitem__)
        argmax_mu = max(range(num_actors), key=final_mu.__getitem__)
        mu_str = " | ".join(f"{v:.3f}" for v in final_mu)
        print(
            f"[§6.1 Observer {i}] final_mu = {mu_str}  "
            f"(prefers actor {preferred}, argmax(mu)={argmax_mu})"
        )


def _print_joint_summary(traces: list[GroupMultiActorTrainingTrace]) -> None:
    num_actors = len(traces[0].per_actor_optimal_action_probs[-1])
    final_p = [0.0] * num_actors
    for trace in traces:
        snap = trace.per_actor_optimal_action_probs[-1]
        for j in range(num_actors):
            final_p[j] += snap[j] / len(traces)
    p_str = " | ".join(f"{v:.3f}" for v in final_p)
    print(f"[§6.2 joint] final_optimal_prob = {p_str}")


if __name__ == "__main__":
    main()
