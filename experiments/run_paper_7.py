"""Run paper Section 7 group-level imperfect-state experiments."""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path("/private/tmp/actor_observer_rl_mplcache").resolve()),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from actor_observer_rl.agents import (
    Actor,
    FixedTableMessageModule,
    FollowingRates,
    MaskedObserverMessageModule,
    Observer,
)
from actor_observer_rl.envs import FeaturePartialObservableStateEnv, PartialObservableStateEnv
from actor_observer_rl.policies import SoftmaxPolicy, SumDecoderPolicy
from actor_observer_rl.training import GroupImperfectStateTrainer
from actor_observer_rl.utils.config import (
    Paper71Config,
    Paper72Config,
    Paper73Config,
    load_yaml_config,
)
from actor_observer_rl.utils.seeding import set_global_seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-7-1",
        type=Path,
        default=Path("configs/paper_7_1_group_observer_selection.yaml"),
    )
    parser.add_argument(
        "--config-7-2",
        type=Path,
        default=Path("configs/paper_7_2_group_selection_compression.yaml"),
    )
    parser.add_argument(
        "--config-7-3",
        type=Path,
        default=Path("configs/paper_7_3_group_joint_imperfect_state.yaml"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/paper_7"))
    parser.add_argument("--figure-dir", type=Path, default=Path("docs/figures"))
    parser.add_argument(
        "--following-learning-rates",
        type=float,
        nargs="+",
        default=[0.005, 0.01, 0.02, 0.05],
    )
    parser.add_argument(
        "--observer-learning-rates",
        type=float,
        nargs="+",
        default=[0.005, 0.01, 0.02, 0.05],
    )
    parser.add_argument(
        "--actor-learning-rates",
        type=float,
        nargs="+",
        default=[0.05, 0.1, 0.2],
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = parser.parse_args()

    config_7_1 = Paper71Config.from_mapping(load_yaml_config(args.config_7_1))
    config_7_2 = Paper72Config.from_mapping(load_yaml_config(args.config_7_2))
    config_7_3 = Paper73Config.from_mapping(load_yaml_config(args.config_7_3))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    rows_7_1 = run_section_7_1(
        config_7_1,
        following_learning_rates=args.following_learning_rates,
        seeds=args.seeds,
    )
    rows_7_2 = run_section_7_2(
        config_7_2,
        observer_learning_rates=args.observer_learning_rates,
        seeds=args.seeds,
    )
    rows_7_3 = run_section_7_3(
        config_7_3,
        actor_learning_rates=args.actor_learning_rates,
        seeds=args.seeds,
    )

    write_outputs(
        args.output_dir / "group_observer_selection",
        rows_7_1,
        summarize(rows_7_1, "following_learning_rate", "full_state"),
    )
    write_outputs(
        args.output_dir / "group_selection_compression",
        rows_7_2,
        summarize(rows_7_2, "observer_learning_rate", "full_state"),
    )
    write_outputs(
        args.output_dir / "group_joint_imperfect_state",
        rows_7_3,
        summarize(rows_7_3, "actor_learning_rate", "full_state"),
    )

    write_figure_7_1(rows_7_1, args.figure_dir / "paper7_1_group_observer_selection")
    write_figure_7_2(rows_7_2, args.figure_dir / "paper7_2_group_selection_compression")
    write_figure_7_3(rows_7_3, args.figure_dir / "paper7_3_group_joint_imperfect_state")

    print_summary("7.1", summarize(rows_7_1, "following_learning_rate", "full_state"))
    print_summary("7.2", summarize(rows_7_2, "observer_learning_rate", "full_state"))
    print_summary("7.3", summarize(rows_7_3, "actor_learning_rate", "full_state"))


def run_section_7_1(
    config: Paper71Config,
    following_learning_rates: list[float],
    seeds: list[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for learning_rate in following_learning_rates:
        for seed in seeds:
            trainer = build_section_7_1_trainer(
                config,
                following_learning_rate=learning_rate,
                seed=seed,
            )
            trace = trainer.run(
                num_steps=config.num_steps,
                seed=seed,
                eval_interval=config.eval_interval,
                reward_window=config.reward_window,
            )
            rows.extend(
                trace_to_rows(
                    config=config,
                    trace=trace,
                    experiment="paper_7_1_group_observer_selection",
                    seed=seed,
                    varied_rate_name="following_learning_rate",
                    varied_rate_value=learning_rate,
                    following_learning_rate=learning_rate,
                    observer_learning_rate=None,
                    actor_learning_rate=None,
                    target_values=[float(action) for action in optimal_actions(config.reward_table)],
                )
            )
    return rows


def run_section_7_2(
    config: Paper72Config,
    observer_learning_rates: list[float],
    seeds: list[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for learning_rate in observer_learning_rates:
        for seed in seeds:
            trainer = build_section_7_2_trainer(
                config,
                observer_learning_rate=learning_rate,
                seed=seed,
            )
            trace = trainer.run(
                num_steps=config.num_steps,
                seed=seed,
                eval_interval=config.eval_interval,
                reward_window=config.reward_window,
            )
            rows.extend(
                trace_to_rows(
                    config=config,
                    trace=trace,
                    experiment="paper_7_2_group_selection_compression",
                    seed=seed,
                    varied_rate_name="observer_learning_rate",
                    varied_rate_value=learning_rate,
                    following_learning_rate=config.following_learning_rate,
                    observer_learning_rate=learning_rate,
                    actor_learning_rate=None,
                    target_values=[sum(features) for features in config.state_features],
                )
            )
    return rows


def run_section_7_3(
    config: Paper73Config,
    actor_learning_rates: list[float],
    seeds: list[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for learning_rate in actor_learning_rates:
        for seed in seeds:
            trainer = build_section_7_3_trainer(
                config,
                actor_learning_rate=learning_rate,
                seed=seed,
            )
            trace = trainer.run(
                num_steps=config.num_steps,
                seed=seed,
                eval_interval=config.eval_interval,
                reward_window=config.reward_window,
            )
            rows.extend(
                trace_to_rows(
                    config=config,
                    trace=trace,
                    experiment="paper_7_3_group_joint_imperfect_state",
                    seed=seed,
                    varied_rate_name="actor_learning_rate",
                    varied_rate_value=learning_rate,
                    following_learning_rate=config.following_learning_rate,
                    observer_learning_rate=config.observer_learning_rate,
                    actor_learning_rate=learning_rate,
                    target_values=[sum(features) for features in config.state_features],
                )
            )
    return rows


def build_section_7_1_trainer(
    config: Paper71Config,
    following_learning_rate: float,
    seed: int,
) -> GroupImperfectStateTrainer:
    set_global_seed(seed)
    env = PartialObservableStateEnv(
        reward_table=config.reward_table,
        state_distribution=config.state_distribution,
        observation_dim=config.observation_dim,
    )
    observers = [
        Observer(message_module=FixedTableMessageModule(messages))
        for messages in config.observer_messages
    ]
    actors = [
        Actor(
            SumDecoderPolicy(
                num_actions=config.num_actions,
                sharpness=config.decoder_sharpness,
            )
        )
        for _ in range(config.num_actors)
    ]
    following_rates_by_actor = [
        FollowingRates(num_observers=config.num_observers)
        for _ in range(config.num_actors)
    ]
    following_optimizer = torch.optim.SGD(
        following_parameters(following_rates_by_actor),
        lr=following_learning_rate,
    )
    return GroupImperfectStateTrainer(
        env=env,
        actors=actors,
        observers=observers,
        following_rates_by_actor=following_rates_by_actor,
        actor_selection_probs=config.actor_selection_probs,
        following_optimizer=following_optimizer,
        optimal_actions=optimal_actions(config.reward_table),
    )


def build_section_7_2_trainer(
    config: Paper72Config,
    observer_learning_rate: float,
    seed: int,
) -> GroupImperfectStateTrainer:
    set_global_seed(seed)
    env = FeaturePartialObservableStateEnv(
        state_features=config.state_features,
        reward_table=config.reward_table,
        state_distribution=config.state_distribution,
        observation_dim=config.observation_dim,
    )
    observers = build_masked_observers(
        state_dim=config.state_dim,
        input_masks=config.observer_input_masks,
    )
    actors = [
        Actor(
            SumDecoderPolicy(
                num_actions=config.num_actions,
                sharpness=config.decoder_sharpness,
            )
        )
        for _ in range(config.num_actors)
    ]
    following_rates_by_actor = [
        FollowingRates(num_observers=config.num_observers)
        for _ in range(config.num_actors)
    ]
    return GroupImperfectStateTrainer(
        env=env,
        actors=actors,
        observers=observers,
        following_rates_by_actor=following_rates_by_actor,
        actor_selection_probs=config.actor_selection_probs,
        following_optimizer=torch.optim.SGD(
            following_parameters(following_rates_by_actor),
            lr=config.following_learning_rate,
        ),
        observer_optimizer=torch.optim.SGD(
            observer_parameters(observers),
            lr=observer_learning_rate,
        ),
        optimal_actions=optimal_actions(config.reward_table),
    )


def build_section_7_3_trainer(
    config: Paper73Config,
    actor_learning_rate: float,
    seed: int,
) -> GroupImperfectStateTrainer:
    set_global_seed(seed)
    env = FeaturePartialObservableStateEnv(
        state_features=config.state_features,
        reward_table=config.reward_table,
        state_distribution=config.state_distribution,
        observation_dim=config.observation_dim,
    )
    observers = build_masked_observers(
        state_dim=config.state_dim,
        input_masks=config.observer_input_masks,
    )
    actors = [
        Actor(SoftmaxPolicy(input_dim=1, num_actions=config.num_actions))
        for _ in range(config.num_actors)
    ]
    following_rates_by_actor = [
        FollowingRates(num_observers=config.num_observers)
        for _ in range(config.num_actors)
    ]
    return GroupImperfectStateTrainer(
        env=env,
        actors=actors,
        observers=observers,
        following_rates_by_actor=following_rates_by_actor,
        actor_selection_probs=config.actor_selection_probs,
        following_optimizer=torch.optim.SGD(
            following_parameters(following_rates_by_actor),
            lr=config.following_learning_rate,
        ),
        observer_optimizer=torch.optim.SGD(
            observer_parameters(observers),
            lr=config.observer_learning_rate,
        ),
        actor_optimizers=[
            torch.optim.SGD(actor.parameters(), lr=actor_learning_rate)
            for actor in actors
        ],
        optimal_actions=optimal_actions(config.reward_table),
    )


def build_masked_observers(
    state_dim: int,
    input_masks: list[list[float]],
) -> list[Observer]:
    observers = []
    for input_mask in input_masks:
        module = MaskedObserverMessageModule(
            input_dim=state_dim,
            message_dim=1,
            input_mask=input_mask,
        )
        with torch.no_grad():
            module.linear.weight.zero_()
            module.linear.bias.zero_()
        observers.append(Observer(message_module=module))
    return observers


def trace_to_rows(
    *,
    config: Any,
    trace: Any,
    experiment: str,
    seed: int,
    varied_rate_name: str,
    varied_rate_value: float,
    following_learning_rate: float,
    observer_learning_rate: float | None,
    actor_learning_rate: float | None,
    target_values: list[float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ceiling = ideal_reward_ceiling(config)
    target_actions = optimal_actions(config.reward_table)
    for eval_index, step in enumerate(trace.evaluation_steps):
        selection_by_actor = trace.selection_weights_by_actor[eval_index]
        messages_by_observer = trace.observer_messages_by_state[eval_index]
        for actor_index in range(config.num_actors):
            for observer_index, observer_name in enumerate(config.observer_names):
                messages = [
                    float(message[0]) for message in messages_by_observer[observer_index]
                ]
                rows.append(
                    {
                        "experiment": experiment,
                        "varied_rate_name": varied_rate_name,
                        "varied_rate_value": float(varied_rate_value),
                        "following_learning_rate": float(following_learning_rate),
                        "observer_learning_rate": (
                            ""
                            if observer_learning_rate is None
                            else float(observer_learning_rate)
                        ),
                        "actor_learning_rate": (
                            ""
                            if actor_learning_rate is None
                            else float(actor_learning_rate)
                        ),
                        "seed": int(seed),
                        "step": int(step),
                        "actor_index": int(actor_index),
                        "observer_index": int(observer_index),
                        "observer_name": observer_name,
                        "actor_selection_probability": float(
                            config.actor_selection_probs[actor_index]
                        ),
                        "selection_probability": float(
                            selection_by_actor[actor_index][observer_index]
                        ),
                        "actor_expected_reward": float(
                            trace.per_actor_expected_rewards[eval_index][actor_index]
                        ),
                        "actor_optimal_action_probability": float(
                            trace.per_actor_optimal_action_probs[eval_index][actor_index]
                        ),
                        "actor_observer_expected_reward": float(
                            trace.per_actor_observer_expected_rewards[eval_index][
                                actor_index
                            ][observer_index]
                        ),
                        "actor_observer_optimal_action_probability": float(
                            trace.per_actor_observer_optimal_action_probs[eval_index][
                                actor_index
                            ][observer_index]
                        ),
                        "messages_by_state": format_float_sequence(messages),
                        "target_values": format_float_sequence(target_values),
                        "target_actions": format_float_sequence(
                            [float(action) for action in target_actions]
                        ),
                        "message_separation": float(message_separation(messages)),
                        "compression_r2": float(r_squared(messages, target_values)),
                        "expected_reward": float(trace.expected_rewards[eval_index]),
                        "mean_optimal_action_probability": float(
                            trace.mean_optimal_action_probs[eval_index]
                        ),
                        "average_reward": float(trace.average_rewards[eval_index]),
                        "ceiling_reward": float(ceiling),
                    }
                )
    return rows


def summarize(
    rows: list[dict[str, Any]],
    varied_rate_name: str,
    full_observer_name: str,
) -> list[dict[str, Any]]:
    final_step = max(int(row["step"]) for row in rows)
    grouped: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if (
            int(row["step"]) == final_step
            and int(row["actor_index"]) == 0
            and int(row["observer_index"]) == 0
        ):
            grouped[float(row["varied_rate_value"])].append(row)

    summaries: list[dict[str, Any]] = []
    for rate_value, seed_rows in sorted(grouped.items()):
        final_expected = fmean(float(row["expected_reward"]) for row in seed_rows)
        final_sampled = fmean(float(row["average_reward"]) for row in seed_rows)
        final_action_prob = fmean(
            float(row["mean_optimal_action_probability"]) for row in seed_rows
        )
        ceiling = fmean(float(row["ceiling_reward"]) for row in seed_rows)
        threshold = 0.95 * ceiling
        success_rate = fmean(
            1.0 if float(row["expected_reward"]) >= threshold else 0.0
            for row in seed_rows
        )
        full_rows = [
            row
            for row in rows
            if int(row["step"]) == final_step
            and float(row["varied_rate_value"]) == rate_value
            and str(row["observer_name"]) == full_observer_name
        ]
        full_probs = [float(row["selection_probability"]) for row in full_rows]
        final_mean_full_probability = fmean(full_probs)
        final_min_full_probability = min(full_probs)
        final_full_r2 = fmean(float(row["compression_r2"]) for row in full_rows)
        final_full_reward = fmean(
            float(row["actor_observer_expected_reward"]) for row in full_rows
        )
        summaries.append(
            {
                varied_rate_name: rate_value,
                "following_learning_rate": float(seed_rows[0]["following_learning_rate"]),
                "observer_learning_rate": seed_rows[0]["observer_learning_rate"],
                "actor_learning_rate": seed_rows[0]["actor_learning_rate"],
                "final_expected_reward": final_expected,
                "final_sampled_reward": final_sampled,
                "final_optimal_action_probability": final_action_prob,
                "success_rate": success_rate,
                "final_mean_full_probability": final_mean_full_probability,
                "final_min_full_probability": final_min_full_probability,
                "final_full_observer_reward": final_full_reward,
                "final_full_observer_r2": final_full_r2,
                "ceiling_reward": ceiling,
            }
        )
    return summaries


def write_figure_7_1(rows: list[dict[str, Any]], output_base: Path) -> None:
    unique_rows = [row for row in rows if int(row["actor_index"]) == 0 and int(row["observer_index"]) == 0]
    learning_rates = sorted({float(row["varied_rate_value"]) for row in unique_rows})
    display_lr = 0.02 if 0.02 in learning_rates else learning_rates[-1]
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.0))
    plot_metric_sweep(
        axes[0],
        unique_rows,
        "expected_reward",
        "Expected reward",
        "reward",
        learning_rates,
        "varied_rate_value",
        r"$\alpha_w$",
    )
    plot_metric_sweep(
        axes[1],
        unique_rows,
        "mean_optimal_action_probability",
        "High-reward action probability",
        "probability",
        learning_rates,
        "varied_rate_value",
        r"$\alpha_w$",
    )
    plot_per_actor_full_probability(
        axes[2],
        rows,
        display_lr,
        title=rf"Per-Actor $P(\mathrm{{full}})$ ($\alpha_w$={display_lr:g})",
    )
    fig.suptitle("Group Observer Selection with Fixed Messages", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_figure(fig, output_base)


def write_figure_7_2(rows: list[dict[str, Any]], output_base: Path) -> None:
    unique_rows = [row for row in rows if int(row["actor_index"]) == 0 and int(row["observer_index"]) == 0]
    learning_rates = sorted({float(row["varied_rate_value"]) for row in unique_rows})
    display_lr = 0.01 if 0.01 in learning_rates else learning_rates[0]
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 7.0))
    plot_metric_sweep(
        axes[0, 0],
        unique_rows,
        "expected_reward",
        "Expected reward",
        "reward",
        learning_rates,
        "varied_rate_value",
        r"$\alpha_\phi$",
    )
    plot_metric_sweep(
        axes[0, 1],
        unique_rows,
        "mean_optimal_action_probability",
        "High-reward action probability",
        "probability",
        learning_rates,
        "varied_rate_value",
        r"$\alpha_\phi$",
    )
    plot_per_actor_full_probability(
        axes[0, 2],
        rows,
        display_lr,
        title=rf"Per-Actor $P(\mathrm{{full}})$ ($\alpha_\phi$={display_lr:g})",
    )
    plot_observer_metric(
        axes[1, 0],
        rows,
        display_lr,
        "actor_observer_expected_reward",
        "Mean per-Observer expected reward",
        "reward",
    )
    plot_full_observer_sweep(
        axes[1, 1],
        rows,
        learning_rates,
        "compression_r2",
        r"Full Observer $R^2(m,z)$",
        r"$R^2$",
        r"$\alpha_\phi$",
    )
    plot_final_messages(axes[1, 2], rows, display_lr, "Final full-state messages")
    fig.suptitle("Group Observer Selection and Compression", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, output_base)


def write_figure_7_3(rows: list[dict[str, Any]], output_base: Path) -> None:
    unique_rows = [row for row in rows if int(row["actor_index"]) == 0 and int(row["observer_index"]) == 0]
    learning_rates = sorted({float(row["varied_rate_value"]) for row in unique_rows})
    display_lr = 0.1 if 0.1 in learning_rates else learning_rates[-1]
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 7.0))
    plot_metric_sweep(
        axes[0, 0],
        unique_rows,
        "expected_reward",
        "Expected reward",
        "reward",
        learning_rates,
        "varied_rate_value",
        r"$\alpha_\theta$",
    )
    plot_metric_sweep(
        axes[0, 1],
        unique_rows,
        "mean_optimal_action_probability",
        "High-reward action probability",
        "probability",
        learning_rates,
        "varied_rate_value",
        r"$\alpha_\theta$",
    )
    plot_per_actor_full_probability(
        axes[0, 2],
        rows,
        display_lr,
        title=rf"Per-Actor $P(\mathrm{{full}})$ ($\alpha_\theta$={display_lr:g})",
    )
    plot_observer_metric(
        axes[1, 0],
        rows,
        display_lr,
        "actor_observer_expected_reward",
        "Mean per-Observer expected reward",
        "reward",
    )
    plot_full_observer_sweep(
        axes[1, 1],
        rows,
        learning_rates,
        "message_separation",
        "Full Observer message separation",
        "minimum distance",
        r"$\alpha_\theta$",
    )
    plot_final_messages(axes[1, 2], rows, display_lr, "Final full-state messages")
    fig.suptitle("Group Observer Selection, Compression, and Actor Policies", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, output_base)


def plot_metric_sweep(
    axis: plt.Axes,
    rows: list[dict[str, Any]],
    metric: str,
    title: str,
    ylabel: str,
    learning_rates: list[float],
    rate_key: str,
    rate_label: str,
) -> None:
    ceiling = max(float(row["ceiling_reward"]) for row in rows)
    for learning_rate in learning_rates:
        series = mean_series(
            rows,
            lambda row, learning_rate=learning_rate: (
                float(row[rate_key]) == learning_rate
            ),
            metric,
        )
        axis.plot(
            [point[0] for point in series],
            [point[1] for point in series],
            linewidth=1.7,
            label=rf"{rate_label}={learning_rate:g}",
        )
    reference = ceiling if metric == "expected_reward" else 1.0
    axis.axhline(reference, color="black", linewidth=1.0, alpha=0.55)
    axis.axhline(0.95 * reference, color="black", linestyle="--", linewidth=0.9, alpha=0.45)
    axis.set_title(title)
    axis.set_xlabel("training step")
    axis.set_ylabel(ylabel)
    axis.set_ylim(-0.15, 1.05)
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend(frameon=False, fontsize=8)


def plot_per_actor_full_probability(
    axis: plt.Axes,
    rows: list[dict[str, Any]],
    display_rate: float,
    title: str,
) -> None:
    full_rows = [
        row
        for row in rows
        if float(row["varied_rate_value"]) == display_rate
        and str(row["observer_name"]) == "full_state"
    ]
    actor_indices = sorted({int(row["actor_index"]) for row in full_rows})
    for actor_index in actor_indices:
        series = mean_series(
            full_rows,
            lambda row, actor_index=actor_index: int(row["actor_index"]) == actor_index,
            "selection_probability",
        )
        axis.plot(
            [point[0] for point in series],
            [point[1] for point in series],
            linewidth=1.8,
            label=f"Actor {actor_index}",
        )
    axis.set_title(title)
    axis.set_xlabel("training step")
    axis.set_ylabel("probability")
    axis.set_ylim(0.0, 1.02)
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend(frameon=False, fontsize=8)


def plot_observer_metric(
    axis: plt.Axes,
    rows: list[dict[str, Any]],
    display_rate: float,
    metric: str,
    title: str,
    ylabel: str,
) -> None:
    display_rows = [row for row in rows if float(row["varied_rate_value"]) == display_rate]
    observer_names = sorted({str(row["observer_name"]) for row in display_rows})
    for observer_name in observer_names:
        series = mean_series(
            [
                row
                for row in display_rows
                if str(row["observer_name"]) == observer_name
            ],
            lambda row: True,
            metric,
        )
        axis.plot(
            [point[0] for point in series],
            [point[1] for point in series],
            linewidth=1.8,
            label=observer_name,
        )
    axis.set_title(title)
    axis.set_xlabel("training step")
    axis.set_ylabel(ylabel)
    axis.set_ylim(-0.15, 1.05)
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend(frameon=False, fontsize=8)


def plot_full_observer_sweep(
    axis: plt.Axes,
    rows: list[dict[str, Any]],
    learning_rates: list[float],
    metric: str,
    title: str,
    ylabel: str,
    rate_label: str,
) -> None:
    full_rows = [row for row in rows if str(row["observer_name"]) == "full_state"]
    for learning_rate in learning_rates:
        series = mean_series(
            full_rows,
            lambda row, learning_rate=learning_rate: (
                float(row["varied_rate_value"]) == learning_rate
            ),
            metric,
        )
        axis.plot(
            [point[0] for point in series],
            [point[1] for point in series],
            linewidth=1.8,
            label=rf"{rate_label}={learning_rate:g}",
        )
    axis.set_title(title)
    axis.set_xlabel("training step")
    axis.set_ylabel(ylabel)
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend(frameon=False, fontsize=8)


def plot_final_messages(
    axis: plt.Axes,
    rows: list[dict[str, Any]],
    display_rate: float,
    title: str,
) -> None:
    full_rows = [
        row
        for row in rows
        if float(row["varied_rate_value"]) == display_rate
        and str(row["observer_name"]) == "full_state"
    ]
    final_step = max(int(row["step"]) for row in full_rows)
    final_rows = [row for row in full_rows if int(row["step"]) == final_step]
    grouped_messages: dict[float, list[float]] = defaultdict(list)
    for row in final_rows:
        targets = parse_float_sequence(str(row["target_values"]))
        messages = parse_float_sequence(str(row["messages_by_state"]))
        for target, message in zip(targets, messages, strict=True):
            grouped_messages[target].append(message)
    target_values = sorted(grouped_messages)
    mean_messages = [fmean(grouped_messages[value]) for value in target_values]
    axis.scatter(target_values, mean_messages, s=55)
    axis.plot(target_values, mean_messages, linewidth=1.0, alpha=0.55)
    axis.set_title(title)
    axis.set_xlabel("target state summary")
    axis.set_ylabel("message")
    axis.set_xticks(target_values)
    if mean_messages:
        pad = max(0.25, 0.1 * (max(mean_messages) - min(mean_messages)))
        axis.set_ylim(min(mean_messages) - pad, max(mean_messages) + pad)
    axis.grid(True, alpha=0.25)


def mean_series(
    rows: list[dict[str, Any]],
    predicate,
    metric: str,
) -> list[tuple[int, float]]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        if predicate(row):
            grouped[int(row["step"])].append(float(row[metric]))
    return [(step, fmean(values)) for step, values in sorted(grouped.items())]


def following_parameters(
    following_rates_by_actor: list[FollowingRates],
) -> list[torch.nn.Parameter]:
    return [
        parameter
        for rates in following_rates_by_actor
        for parameter in rates.parameters()
    ]


def observer_parameters(observers: list[Observer]) -> list[torch.nn.Parameter]:
    return [
        parameter
        for observer in observers
        for parameter in observer.parameters()
    ]


def optimal_actions(reward_table: list[list[float]]) -> list[int]:
    return [
        max(range(len(rewards)), key=lambda action: rewards[action])
        for rewards in reward_table
    ]


def ideal_reward_ceiling(config: Any) -> float:
    return sum(
        probability * max(rewards)
        for probability, rewards in zip(
            config.state_distribution,
            config.reward_table,
            strict=True,
        )
    )


def message_separation(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return min(
        abs(left - right)
        for index, left in enumerate(values)
        for right in values[index + 1 :]
    )


def r_squared(values: list[float], targets: list[float]) -> float:
    if len(values) != len(targets):
        raise ValueError("values and targets must have the same length")
    target_mean = fmean(targets)
    total_sum_squares = sum((target - target_mean) ** 2 for target in targets)
    if total_sum_squares == 0.0:
        return 1.0
    value_mean = fmean(values)
    covariance = sum(
        (value - value_mean) * (target - target_mean)
        for value, target in zip(values, targets, strict=True)
    )
    value_sum_squares = sum((value - value_mean) ** 2 for value in values)
    if value_sum_squares <= 1e-12:
        return 0.0
    return max(0.0, min(1.0, covariance**2 / (value_sum_squares * total_sum_squares)))


def format_float_sequence(values: list[float]) -> str:
    return ";".join(f"{value:.8g}" for value in values)


def parse_float_sequence(value: str) -> list[float]:
    return [float(part) for part in value.split(";") if part]


def write_outputs(output_base: Path, rows: list[dict[str, Any]], summaries: list[dict[str, Any]]) -> None:
    write_csv(output_base.with_name(output_base.name + "_curves.csv"), rows)
    write_csv(output_base.with_name(output_base.name + "_summary.csv"), summaries)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig: plt.Figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png", ".svg"):
        output_path = output_base.with_suffix(suffix)
        fig.savefig(output_path, bbox_inches="tight")
        if suffix == ".svg":
            strip_trailing_whitespace(output_path)
    plt.close(fig)


def strip_trailing_whitespace(path: Path) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text(
        "\n".join(line.rstrip() for line in lines) + "\n",
        encoding="utf-8",
    )


def print_summary(section: str, summaries: list[dict[str, Any]]) -> None:
    for row in summaries:
        rate_keys = [
            key
            for key in ("following_learning_rate", "observer_learning_rate", "actor_learning_rate")
            if key in row and row[key] != ""
        ]
        varied = ", ".join(f"{key}={row[key]}" for key in rate_keys)
        print(
            f"section={section} {varied} "
            "final_expected={final_expected_reward:.3f} "
            "final_action_prob={final_optimal_action_probability:.3f} "
            "success={success_rate:.2f} "
            "mean_p_full={final_mean_full_probability:.3f} "
            "min_p_full={final_min_full_probability:.3f}".format(**row)
        )


if __name__ == "__main__":
    main()
