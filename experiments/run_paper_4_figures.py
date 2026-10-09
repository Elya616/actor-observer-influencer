"""Mechanism-level figures for paper Sections 4.1 and 4.2.

For the imperfect-reward track (one Observer, many Actors), the decision
variable is the budgeted following rate ``mu_j`` and the supporting estimate is
``U_hat_j``. This script runs controlled experiments whose correct answer is
known in advance and plots those internal quantities over time:

* ``paper4_1_single_best``  -- one strictly-best Actor; mu should concentrate.
* ``paper4_1_tie``          -- two equally good Actors; mu should split, the
                               useless Actor should lose budget.
* ``paper4_1_timescale_comparison``
                             -- compare utility/following-rate timescale
                                separation in the §4.1 controlled setup.
* ``paper4_2_joint``        -- Actors learn jointly; track per-Actor optimal
                               action probability together with mu and U_hat.

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
    MultiActorTrainer,
    MultiActorTrainingTrace,
    make_fixed_softmax_actor,
)
from actor_observer_rl.utils.seeding import set_global_seed


def save_figure(fig: plt.Figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png", ".svg"):
        fig.savefig(output_base.with_suffix(suffix), bbox_inches="tight")
    plt.close(fig)


def average_snapshot_series(
    traces: list[MultiActorTrainingTrace],
    attribute: str,
    snapshot_interval: int,
) -> tuple[list[int], list[list[float]]]:
    """Average a per-snapshot list-of-lists quantity across seeds.

    Returns ``(steps, series_per_actor)`` where ``series_per_actor[j][k]`` is
    the mean over seeds of actor ``j`` at snapshot ``k``.
    """

    snapshots = [getattr(trace, attribute) for trace in traces]
    num_snapshots = min(len(snapshot) for snapshot in snapshots)
    num_actors = len(snapshots[0][0])
    steps = [(k + 1) * snapshot_interval for k in range(num_snapshots)]
    series_per_actor = [[0.0] * num_snapshots for _ in range(num_actors)]
    for k in range(num_snapshots):
        for j in range(num_actors):
            values = [snapshot[k][j] for snapshot in snapshots]
            series_per_actor[j][k] = sum(values) / len(values)
    return steps, series_per_actor


def run_section_4_1_fixed(
    reward_row: list[float],
    seeds: list[int],
    budget: float,
    c: float,
    utility_lr: float,
    following_lr: float,
    num_steps: int,
    snapshot_interval: int,
) -> tuple[list[MultiActorTrainingTrace], list[float]]:
    """Run §4.1 with one state, fixed actors, controlled per-actor utilities.

    With a single state and Actor ``j`` using a sharp softmax around action
    ``j``, the analytical utility of Actor ``j`` rounds to ``reward_row[j]``.
    """

    num_actions = len(reward_row)
    traces: list[MultiActorTrainingTrace] = []
    for seed in seeds:
        set_global_seed(seed)
        env = IIDStateBanditEnv(
            reward_table=[reward_row],
            state_distribution=[1.0],
            observation_mode="one_hot",
        )
        actors = [
            make_fixed_softmax_actor(
                num_states=1,
                num_actions=num_actions,
                target_action=j,
                sharpness=8.0,
            )
            for j in range(num_actions)
        ]
        observer = BudgetedObserver(
            num_actors=num_actions,
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
            actor_selection_probs=[1.0 / num_actions] * num_actions,
            actor_optimizers=None,
            snapshot_interval=snapshot_interval,
        )
        traces.append(trainer.run(num_steps=num_steps, seed=seed))
    return traces, list(reward_row)


def run_section_4_2_joint(
    reward_table: list[list[float]],
    optimal_actions: list[int],
    seeds: list[int],
    budget: float,
    c: float,
    utility_lr: float,
    following_lr: float,
    actor_lr: float,
    num_steps: int,
    snapshot_interval: int,
) -> list[MultiActorTrainingTrace]:
    num_states = len(reward_table)
    num_actions = len(reward_table[0])
    num_actors = num_states
    traces: list[MultiActorTrainingTrace] = []
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
        observer = BudgetedObserver(
            num_actors=num_actors,
            budget=budget,
            utility_learning_rate=utility_lr,
            following_learning_rate=following_lr,
            c=c,
            init_mu=[budget / num_actors] * num_actors,
            seed=seed,
        )
        trainer = MultiActorTrainer(
            env=env,
            actors=actors,
            observer=observer,
            actor_selection_probs=[1.0 / num_actors] * num_actors,
            actor_optimizers=optimizers,
            optimal_actions_per_state=optimal_actions,
            snapshot_interval=snapshot_interval,
        )
        traces.append(trainer.run(num_steps=num_steps, seed=seed))
    return traces


def plot_following_and_utility(
    traces: list[MultiActorTrainingTrace],
    expected_utilities: list[float],
    snapshot_interval: int,
    title: str,
    output_base: Path,
) -> None:
    steps_mu, mu_series = average_snapshot_series(
        traces, "following_rates", snapshot_interval
    )
    steps_u, u_series = average_snapshot_series(
        traces, "utility_estimates", snapshot_interval
    )
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))

    for j, series in enumerate(mu_series):
        axes[0].plot(
            steps_mu,
            series,
            linewidth=1.9,
            label=rf"actor {j} ($U={expected_utilities[j]:.2f}$)",
        )
    axes[0].set_title(f"{title}: following rate $\\mu_j(t)$")
    axes[0].set_xlabel("training step")
    axes[0].set_ylabel(r"$\mu_j$")
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)

    for j, series in enumerate(u_series):
        axes[1].plot(steps_u, series, linewidth=1.9, label=f"actor {j}")
        axes[1].axhline(
            expected_utilities[j],
            color=f"C{j}",
            linestyle=":",
            linewidth=1.0,
            alpha=0.7,
        )
    axes[1].set_title(f"{title}: utility estimate $\\hat U_j(t)$")
    axes[1].set_xlabel("training step")
    axes[1].set_ylabel(r"$\hat U_j$")
    axes[1].grid(True, axis="y", alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)

    fig.suptitle(
        f"{title} (mean over {len(traces)} seeds; dotted = analytical utility)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_figure(fig, output_base)


def plot_joint(
    traces: list[MultiActorTrainingTrace],
    snapshot_interval: int,
    output_base: Path,
) -> None:
    steps_p, p_series = average_snapshot_series(
        traces, "per_actor_optimal_action_probs", snapshot_interval
    )
    steps_mu, mu_series = average_snapshot_series(
        traces, "following_rates", snapshot_interval
    )
    steps_u, u_series = average_snapshot_series(
        traces, "utility_estimates", snapshot_interval
    )

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.4))

    for j, series in enumerate(p_series):
        axes[0].plot(steps_p, series, linewidth=1.9, label=f"actor {j}")
    axes[0].set_title("§4.2: optimal-action prob per actor")
    axes[0].set_xlabel("training step")
    axes[0].set_ylabel(r"$\overline{P}(a^\star \mid s)$")
    axes[0].set_ylim(0.0, 1.02)
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)

    for j, series in enumerate(mu_series):
        axes[1].plot(steps_mu, series, linewidth=1.9, label=f"actor {j}")
    axes[1].set_title(r"§4.2: following rate $\mu_j(t)$")
    axes[1].set_xlabel("training step")
    axes[1].set_ylabel(r"$\mu_j$")
    axes[1].grid(True, axis="y", alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)

    for j, series in enumerate(u_series):
        axes[2].plot(steps_u, series, linewidth=1.9, label=f"actor {j}")
    axes[2].set_title(r"§4.2: utility estimate $\hat U_j(t)$")
    axes[2].set_xlabel("training step")
    axes[2].set_ylabel(r"$\hat U_j$")
    axes[2].grid(True, axis="y", alpha=0.25)
    axes[2].legend(frameon=False, fontsize=8)

    fig.suptitle(f"§4.2 joint learning (mean over {len(traces)} seeds)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_figure(fig, output_base)


def plot_timescale_comparison(
    settings: list[tuple[str, list[MultiActorTrainingTrace], list[float], int]],
    output_base: Path,
) -> None:
    """Plot utility-tracking and following-rate behavior across timescales."""

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.4))

    for label, traces, expected_utilities, snapshot_interval in settings:
        num_snapshots = min(len(trace.utility_estimates) for trace in traces)
        steps = [(k + 1) * snapshot_interval for k in range(num_snapshots)]
        best_actor = max(range(len(expected_utilities)), key=expected_utilities.__getitem__)
        utility_error: list[float] = []
        best_mu: list[float] = []
        for k in range(num_snapshots):
            error_total = 0.0
            mu_total = 0.0
            for trace in traces:
                utility_snapshot = trace.utility_estimates[k]
                error_total += sum(
                    abs(estimate - target)
                    for estimate, target in zip(
                        utility_snapshot, expected_utilities, strict=True
                    )
                ) / len(expected_utilities)
                mu_total += trace.following_rates[k][best_actor]
            utility_error.append(error_total / len(traces))
            best_mu.append(mu_total / len(traces))

        axes[0].plot(steps, utility_error, linewidth=1.9, label=label)
        axes[1].plot(steps, best_mu, linewidth=1.9, label=label)

    axes[0].set_title(r"utility tracking error")
    axes[0].set_xlabel("training step")
    axes[0].set_ylabel(r"$N^{-1}\sum_j |\hat U_j-U_j|$")
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].set_title(r"following rate on best Actor")
    axes[1].set_xlabel("training step")
    axes[1].set_ylabel(r"$\mu_{\mathrm{best}}$")
    axes[1].grid(True, axis="y", alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)

    fig.suptitle("§4.1 timescale comparison (mean over seeds)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_figure(fig, output_base)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure-dir", type=Path, default=Path("docs/figures"))
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()

    seeds = list(range(args.seeds))

    single_best_traces, single_best_u = run_section_4_1_fixed(
        reward_row=[0.9, 0.5, 0.1],
        seeds=seeds,
        budget=1.0,
        c=2.0,
        utility_lr=0.1,
        following_lr=0.02,
        num_steps=4000,
        snapshot_interval=40,
    )
    plot_following_and_utility(
        single_best_traces,
        expected_utilities=single_best_u,
        snapshot_interval=40,
        title="§4.1 single best actor",
        output_base=args.figure_dir / "paper4_1_single_best",
    )

    tie_traces, tie_u = run_section_4_1_fixed(
        reward_row=[0.9, 0.9, 0.1],
        seeds=seeds,
        budget=1.0,
        c=2.0,
        utility_lr=0.1,
        following_lr=0.02,
        num_steps=4000,
        snapshot_interval=40,
    )
    plot_following_and_utility(
        tie_traces,
        expected_utilities=tie_u,
        snapshot_interval=40,
        title="§4.1 tie (two equal actors)",
        output_base=args.figure_dir / "paper4_1_tie",
    )

    timescale_settings: list[
        tuple[str, list[MultiActorTrainingTrace], list[float], int]
    ] = []
    for label, utility_lr, following_lr in [
        (r"$\alpha_U=0.10,\ \alpha_\mu=0.02$", 0.10, 0.02),
        (r"$\alpha_U=0.02,\ \alpha_\mu=0.02$", 0.02, 0.02),
        (r"$\alpha_U=0.005,\ \alpha_\mu=0.02$", 0.005, 0.02),
    ]:
        traces, expected_u = run_section_4_1_fixed(
            reward_row=[0.9, 0.5, 0.1],
            seeds=seeds,
            budget=1.0,
            c=2.0,
            utility_lr=utility_lr,
            following_lr=following_lr,
            num_steps=4000,
            snapshot_interval=40,
        )
        timescale_settings.append((label, traces, expected_u, 40))
    plot_timescale_comparison(
        timescale_settings,
        output_base=args.figure_dir / "paper4_1_timescale_comparison",
    )

    joint_traces = run_section_4_2_joint(
        reward_table=[[1.0, 0.1, 0.2], [0.1, 1.0, 0.0], [0.0, 0.2, 1.0]],
        optimal_actions=[0, 1, 2],
        seeds=seeds,
        budget=3.0,
        c=2.0,
        utility_lr=0.2,
        following_lr=0.01,
        actor_lr=0.2,
        num_steps=30000,
        snapshot_interval=300,
    )
    plot_joint(
        joint_traces,
        snapshot_interval=300,
        output_base=args.figure_dir / "paper4_2_joint",
    )

    _print_summary("§4.1 single best", single_best_traces, single_best_u)
    _print_summary("§4.1 tie", tie_traces, tie_u)
    _print_timescale_summary(timescale_settings)
    _print_joint_summary(joint_traces)


def _print_summary(
    label: str,
    traces: list[MultiActorTrainingTrace],
    expected_u: list[float],
) -> None:
    num_actors = len(expected_u)
    final_mu = [0.0] * num_actors
    final_u = [0.0] * num_actors
    for trace in traces:
        for j in range(num_actors):
            final_mu[j] += trace.final_following_rates[j] / len(traces)
            final_u[j] += trace.final_utilities[j] / len(traces)
    mu_str = " | ".join(f"{v:.3f}" for v in final_mu)
    u_str = " | ".join(f"{v:.3f}" for v in final_u)
    exp_str = " | ".join(f"{v:.3f}" for v in expected_u)
    print(f"[{label}] expected_U = {exp_str}")
    print(f"[{label}] final_Uhat  = {u_str}")
    print(f"[{label}] final_mu    = {mu_str}  (sum={sum(final_mu):.3f})")


def _print_joint_summary(traces: list[MultiActorTrainingTrace]) -> None:
    num_actors = len(traces[0].final_following_rates)
    final_p = [0.0] * num_actors
    final_mu = [0.0] * num_actors
    for trace in traces:
        snap = trace.per_actor_optimal_action_probs[-1]
        for j in range(num_actors):
            final_p[j] += snap[j] / len(traces)
            final_mu[j] += trace.final_following_rates[j] / len(traces)
    p_str = " | ".join(f"{v:.3f}" for v in final_p)
    mu_str = " | ".join(f"{v:.3f}" for v in final_mu)
    print(f"[§4.2 joint] final_optimal_prob = {p_str}")
    print(f"[§4.2 joint] final_mu           = {mu_str}  (sum={sum(final_mu):.3f})")


def _print_timescale_summary(
    settings: list[tuple[str, list[MultiActorTrainingTrace], list[float], int]]
) -> None:
    for label, traces, expected_u, _ in settings:
        num_actors = len(expected_u)
        best_actor = max(range(num_actors), key=expected_u.__getitem__)
        final_error = 0.0
        final_mu_best = 0.0
        for trace in traces:
            final_error += sum(
                abs(estimate - target)
                for estimate, target in zip(trace.final_utilities, expected_u, strict=True)
            ) / (num_actors * len(traces))
            final_mu_best += trace.final_following_rates[best_actor] / len(traces)
        print(
            f"[§4.1 timescale] {label}: "
            f"final_utility_mae={final_error:.4f}, "
            f"final_mu_best={final_mu_best:.3f}"
        )


if __name__ == "__main__":
    main()
