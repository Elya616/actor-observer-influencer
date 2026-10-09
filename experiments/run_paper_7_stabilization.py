"""Run stabilization checks for paper Section 7.2 and 7.3."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter, defaultdict
from dataclasses import replace
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

from actor_observer_rl.agents import Actor, FollowingRates
from actor_observer_rl.envs import FeaturePartialObservableStateEnv
from actor_observer_rl.policies import SoftmaxPolicy, SumDecoderPolicy
from actor_observer_rl.training import GroupImperfectStateTrainer
from actor_observer_rl.utils.config import Paper72Config, Paper73Config, load_yaml_config
from actor_observer_rl.utils.seeding import set_global_seed
from experiments.run_paper_7 import (
    build_masked_observers,
    format_float_sequence,
    ideal_reward_ceiling,
    observer_parameters,
    optimal_actions,
    r_squared,
    save_figure,
    trace_to_rows,
)

torch.distributions.Distribution.set_default_validate_args(False)


FULL_OBSERVER_NAME = "full_state"
PARTIAL_OBSERVER_NAME = "first_feature"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/paper_7_stabilization"),
    )
    parser.add_argument("--figure-dir", type=Path, default=Path("docs/figures"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument(
        "--horizon-seeds",
        type=int,
        nargs="+",
        default=[0],
        help="Seed subset for the expensive 7.3 long-horizon check.",
    )
    parser.add_argument(
        "--suites",
        nargs="+",
        choices=["alpha_w", "horizon", "entropy", "advantage", "warm_start"],
        default=["alpha_w", "horizon", "entropy", "advantage", "warm_start"],
        help="Subset of stabilization suites to run.",
    )
    args = parser.parse_args()

    config_7_2 = Paper72Config.from_mapping(load_yaml_config(args.config_7_2))
    config_7_3 = Paper73Config.from_mapping(load_yaml_config(args.config_7_3))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    curve_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []

    suite_fns = []
    if "alpha_w" in args.suites:
        suite_fns.extend(
            [
                ("7.2 selection-rate sweep", lambda: run_selection_rate_sweep_7_2(config_7_2, args.seeds)),
                ("7.3 selection-rate sweep", lambda: run_selection_rate_sweep_7_3(config_7_3, args.seeds)),
            ]
        )
    if "horizon" in args.suites:
        suite_fns.append(
            ("7.3 horizon sweep", lambda: run_horizon_sweep_7_3(config_7_3, args.horizon_seeds))
        )
    if "entropy" in args.suites:
        suite_fns.append(
            ("7.3 entropy sweep", lambda: run_entropy_sweep_7_3(config_7_3, args.seeds))
        )
    if "advantage" in args.suites:
        suite_fns.append(
            ("7.3 advantage comparison", lambda: run_advantage_comparison_7_3(config_7_3, args.seeds))
        )
    if "warm_start" in args.suites:
        suite_fns.extend(
            [
                ("7.2 warm start", lambda: run_warm_start_7_2(config_7_2, args.seeds)),
                ("7.3 warm start", lambda: run_warm_start_7_3(config_7_3, args.seeds)),
            ]
        )
    for label, run_suite in suite_fns:
        print(f"running {label}", flush=True)
        rows, summaries = run_suite()
        curve_rows.extend(rows)
        seed_rows.extend(summaries)

    summary_rows = summarize_seed_rows(seed_rows)

    write_csv(args.output_dir / "stabilization_curves.csv", curve_rows)
    write_csv(args.output_dir / "stabilization_seed_summary.csv", seed_rows)
    write_csv(args.output_dir / "stabilization_summary.csv", summary_rows)

    if "alpha_w" in args.suites:
        write_alpha_w_figure(
            curve_rows,
            args.figure_dir / "paper7_stabilization_selection_rate",
        )
        write_per_actor_figure(
            curve_rows,
            args.figure_dir / "paper7_stabilization_per_actor",
        )
    if any(
        suite in args.suites
        for suite in ("horizon", "entropy", "advantage", "warm_start")
    ):
        write_method_figure(
            curve_rows,
            summary_rows,
            args.figure_dir / "paper7_stabilization_methods",
        )

    for row in summary_rows:
        print(
            "{section} {experiment}: final_expected={final_expected_reward:.3f} "
            "min_p_full={final_min_full_probability:.3f} "
            "collapse_pairs={collapsed_actor_seed_pairs:.0f}".format(**row)
        )


def run_selection_rate_sweep_7_2(
    config: Paper72Config,
    seeds: list[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for alpha_w in [0.0005, 0.001, 0.002, 0.005]:
        varied_config = replace(
            config,
            following_learning_rate=alpha_w,
            observer_learning_rate=0.01,
            num_steps=20_000,
        )
        for seed in seeds:
            trainer = build_section_7_2_trainer(
                varied_config,
                seed=seed,
                selection_update_rule="raw_reward",
            )
            trace = trainer.run(
                num_steps=varied_config.num_steps,
                seed=seed,
                eval_interval=varied_config.eval_interval,
                reward_window=varied_config.reward_window,
            )
            rows.extend(
                rows_from_trace(
                    varied_config,
                    trace,
                    section="7.2",
                    experiment=f"7.2 alpha_w={alpha_w:g}",
                    seed=seed,
                    varied_rate_name="following_learning_rate",
                    varied_rate_value=alpha_w,
                    following_learning_rate=alpha_w,
                    observer_learning_rate=varied_config.observer_learning_rate,
                    actor_learning_rate=None,
                    update_rule="raw_reward",
                    entropy_beta=0.0,
                    warm_start_steps=0,
                    horizon_steps=varied_config.num_steps,
                )
            )
            summaries.append(
                seed_summary_from_trace(
                    varied_config,
                    trace,
                    section="7.2",
                    experiment=f"7.2 alpha_w={alpha_w:g}",
                    seed=seed,
                    following_learning_rate=alpha_w,
                    observer_learning_rate=varied_config.observer_learning_rate,
                    actor_learning_rate=None,
                    update_rule="raw_reward",
                    entropy_beta=0.0,
                    warm_start_steps=0,
                    horizon_steps=varied_config.num_steps,
                )
            )
    return rows, summaries


def run_selection_rate_sweep_7_3(
    config: Paper73Config,
    seeds: list[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for alpha_w in [0.0005, 0.001, 0.002, 0.005]:
        varied_config = replace(
            config,
            following_learning_rate=alpha_w,
            observer_learning_rate=0.05,
            actor_learning_rate=0.1,
            num_steps=20_000,
        )
        for seed in seeds:
            trainer = build_section_7_3_trainer(
                varied_config,
                seed=seed,
                selection_update_rule="raw_reward",
            )
            trace = trainer.run(
                num_steps=varied_config.num_steps,
                seed=seed,
                eval_interval=varied_config.eval_interval,
                reward_window=varied_config.reward_window,
            )
            rows.extend(
                rows_from_trace(
                    varied_config,
                    trace,
                    section="7.3",
                    experiment=f"7.3 alpha_w={alpha_w:g}",
                    seed=seed,
                    varied_rate_name="following_learning_rate",
                    varied_rate_value=alpha_w,
                    following_learning_rate=alpha_w,
                    observer_learning_rate=varied_config.observer_learning_rate,
                    actor_learning_rate=varied_config.actor_learning_rate,
                    update_rule="raw_reward",
                    entropy_beta=0.0,
                    warm_start_steps=0,
                    horizon_steps=varied_config.num_steps,
                )
            )
            summaries.append(
                seed_summary_from_trace(
                    varied_config,
                    trace,
                    section="7.3",
                    experiment=f"7.3 alpha_w={alpha_w:g}",
                    seed=seed,
                    following_learning_rate=alpha_w,
                    observer_learning_rate=varied_config.observer_learning_rate,
                    actor_learning_rate=varied_config.actor_learning_rate,
                    update_rule="raw_reward",
                    entropy_beta=0.0,
                    warm_start_steps=0,
                    horizon_steps=varied_config.num_steps,
                )
            )
    return rows, summaries


def run_horizon_sweep_7_3(
    config: Paper73Config,
    seeds: list[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for steps in [20_000, 50_000, 100_000]:
        varied_config = replace(
            config,
            following_learning_rate=0.001,
            observer_learning_rate=0.05,
            actor_learning_rate=0.1,
            num_steps=steps,
            eval_interval=max(200, steps // 100),
        )
        for seed in seeds:
            trainer = build_section_7_3_trainer(
                varied_config,
                seed=seed,
                selection_update_rule="raw_reward",
            )
            trace = trainer.run(
                num_steps=varied_config.num_steps,
                seed=seed,
                eval_interval=varied_config.eval_interval,
                reward_window=varied_config.reward_window,
            )
            label = f"7.3 horizon={steps // 1000}k"
            rows.extend(
                rows_from_trace(
                    varied_config,
                    trace,
                    section="7.3",
                    experiment=label,
                    seed=seed,
                    varied_rate_name="horizon_steps",
                    varied_rate_value=float(steps),
                    following_learning_rate=varied_config.following_learning_rate,
                    observer_learning_rate=varied_config.observer_learning_rate,
                    actor_learning_rate=varied_config.actor_learning_rate,
                    update_rule="raw_reward",
                    entropy_beta=0.0,
                    warm_start_steps=0,
                    horizon_steps=steps,
                )
            )
            summaries.append(
                seed_summary_from_trace(
                    varied_config,
                    trace,
                    section="7.3",
                    experiment=label,
                    seed=seed,
                    following_learning_rate=varied_config.following_learning_rate,
                    observer_learning_rate=varied_config.observer_learning_rate,
                    actor_learning_rate=varied_config.actor_learning_rate,
                    update_rule="raw_reward",
                    entropy_beta=0.0,
                    warm_start_steps=0,
                    horizon_steps=steps,
                )
            )
    return rows, summaries


def run_entropy_sweep_7_3(
    config: Paper73Config,
    seeds: list[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    varied_config = replace(
        config,
        following_learning_rate=0.005,
        observer_learning_rate=0.05,
        actor_learning_rate=0.1,
        num_steps=20_000,
    )
    for beta in [0.0, 0.001, 0.003, 0.01]:
        for seed in seeds:
            trainer = build_section_7_3_trainer(
                varied_config,
                seed=seed,
                selection_update_rule="raw_reward",
                selection_entropy_beta=beta,
            )
            trace = trainer.run(
                num_steps=varied_config.num_steps,
                seed=seed,
                eval_interval=varied_config.eval_interval,
                reward_window=varied_config.reward_window,
            )
            label = f"7.3 entropy beta={beta:g}"
            rows.extend(
                rows_from_trace(
                    varied_config,
                    trace,
                    section="7.3",
                    experiment=label,
                    seed=seed,
                    varied_rate_name="selection_entropy_beta",
                    varied_rate_value=beta,
                    following_learning_rate=varied_config.following_learning_rate,
                    observer_learning_rate=varied_config.observer_learning_rate,
                    actor_learning_rate=varied_config.actor_learning_rate,
                    update_rule="raw_reward",
                    entropy_beta=beta,
                    warm_start_steps=0,
                    horizon_steps=varied_config.num_steps,
                )
            )
            summaries.append(
                seed_summary_from_trace(
                    varied_config,
                    trace,
                    section="7.3",
                    experiment=label,
                    seed=seed,
                    following_learning_rate=varied_config.following_learning_rate,
                    observer_learning_rate=varied_config.observer_learning_rate,
                    actor_learning_rate=varied_config.actor_learning_rate,
                    update_rule="raw_reward",
                    entropy_beta=beta,
                    warm_start_steps=0,
                    horizon_steps=varied_config.num_steps,
                )
            )
    return rows, summaries


def run_advantage_comparison_7_3(
    config: Paper73Config,
    seeds: list[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    varied_config = replace(
        config,
        following_learning_rate=0.005,
        observer_learning_rate=0.05,
        actor_learning_rate=0.1,
        num_steps=20_000,
    )
    for update_rule in ["raw_reward", "advantage_per_actor"]:
        for seed in seeds:
            trainer = build_section_7_3_trainer(
                varied_config,
                seed=seed,
                selection_update_rule=update_rule,
                selection_baseline_learning_rate=0.05,
            )
            trace = trainer.run(
                num_steps=varied_config.num_steps,
                seed=seed,
                eval_interval=varied_config.eval_interval,
                reward_window=varied_config.reward_window,
            )
            label = "7.3 raw reward" if update_rule == "raw_reward" else "7.3 advantage"
            rows.extend(
                rows_from_trace(
                    varied_config,
                    trace,
                    section="7.3",
                    experiment=label,
                    seed=seed,
                    varied_rate_name="selection_update_rule",
                    varied_rate_value=0.0 if update_rule == "raw_reward" else 1.0,
                    following_learning_rate=varied_config.following_learning_rate,
                    observer_learning_rate=varied_config.observer_learning_rate,
                    actor_learning_rate=varied_config.actor_learning_rate,
                    update_rule=update_rule,
                    entropy_beta=0.0,
                    warm_start_steps=0,
                    horizon_steps=varied_config.num_steps,
                )
            )
            summaries.append(
                seed_summary_from_trace(
                    varied_config,
                    trace,
                    section="7.3",
                    experiment=label,
                    seed=seed,
                    following_learning_rate=varied_config.following_learning_rate,
                    observer_learning_rate=varied_config.observer_learning_rate,
                    actor_learning_rate=varied_config.actor_learning_rate,
                    update_rule=update_rule,
                    entropy_beta=0.0,
                    warm_start_steps=0,
                    horizon_steps=varied_config.num_steps,
                )
            )
    return rows, summaries


def run_warm_start_7_2(
    config: Paper72Config,
    seeds: list[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    warm_steps = 5_000
    train_steps = 20_000
    varied_config = replace(
        config,
        following_learning_rate=0.005,
        observer_learning_rate=0.01,
        num_steps=train_steps,
    )
    for seed in seeds:
        trainer = build_section_7_2_trainer(
            varied_config,
            seed=seed,
            selection_update_rule="raw_reward",
            learn_selection=False,
        )
        trainer.run(
            num_steps=warm_steps,
            seed=seed,
            eval_interval=warm_steps,
            reward_window=varied_config.reward_window,
        )
        trainer.following_optimizer = torch.optim.SGD(
            following_parameters(trainer.following_rates_by_actor),
            lr=varied_config.following_learning_rate,
        )
        trace = trainer.run(
            num_steps=train_steps,
            seed=seed + 10_000,
            eval_interval=varied_config.eval_interval,
            reward_window=varied_config.reward_window,
        )
        label = "7.2 warm-start compression"
        rows.extend(
            rows_from_trace(
                varied_config,
                trace,
                section="7.2",
                experiment=label,
                seed=seed,
                varied_rate_name="warm_start_steps",
                varied_rate_value=float(warm_steps),
                following_learning_rate=varied_config.following_learning_rate,
                observer_learning_rate=varied_config.observer_learning_rate,
                actor_learning_rate=None,
                update_rule="raw_reward",
                entropy_beta=0.0,
                warm_start_steps=warm_steps,
                horizon_steps=train_steps,
            )
        )
        summaries.append(
            seed_summary_from_trace(
                varied_config,
                trace,
                section="7.2",
                experiment=label,
                seed=seed,
                following_learning_rate=varied_config.following_learning_rate,
                observer_learning_rate=varied_config.observer_learning_rate,
                actor_learning_rate=None,
                update_rule="raw_reward",
                entropy_beta=0.0,
                warm_start_steps=warm_steps,
                horizon_steps=train_steps,
            )
        )
    return rows, summaries


def run_warm_start_7_3(
    config: Paper73Config,
    seeds: list[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    warm_steps = 5_000
    train_steps = 20_000
    varied_config = replace(
        config,
        following_learning_rate=0.005,
        observer_learning_rate=0.05,
        actor_learning_rate=0.1,
        num_steps=train_steps,
    )
    warm_config = replace(
        Paper72Config(
            seed=config.seed,
            num_steps=warm_steps,
            eval_interval=warm_steps,
            reward_window=config.reward_window,
            num_actors=config.num_actors,
            num_observers=config.num_observers,
            num_states=config.num_states,
            state_dim=config.state_dim,
            observation_dim=config.observation_dim,
            num_actions=config.num_actions,
            actor_selection_probs=config.actor_selection_probs,
            state_features=config.state_features,
            state_distribution=config.state_distribution,
            reward_table=config.reward_table,
            observer_names=config.observer_names,
            observer_input_masks=config.observer_input_masks,
            following_learning_rate=config.following_learning_rate,
            observer_learning_rate=0.05,
            decoder_sharpness=4.0,
        ),
        num_steps=warm_steps,
    )
    for seed in seeds:
        warm_trainer = build_section_7_2_trainer(
            warm_config,
            seed=seed,
            selection_update_rule="raw_reward",
            learn_selection=False,
        )
        warm_trainer.run(
            num_steps=warm_steps,
            seed=seed,
            eval_interval=warm_steps,
            reward_window=warm_config.reward_window,
        )
        trainer = build_section_7_3_trainer(
            varied_config,
            seed=seed,
            selection_update_rule="raw_reward",
        )
        for warm_observer, joint_observer in zip(
            warm_trainer.observers,
            trainer.observers,
            strict=True,
        ):
            joint_observer.message_module.load_state_dict(
                warm_observer.message_module.state_dict()
            )
        trace = trainer.run(
            num_steps=train_steps,
            seed=seed + 20_000,
            eval_interval=varied_config.eval_interval,
            reward_window=varied_config.reward_window,
        )
        label = "7.3 warm-start compression"
        rows.extend(
            rows_from_trace(
                varied_config,
                trace,
                section="7.3",
                experiment=label,
                seed=seed,
                varied_rate_name="warm_start_steps",
                varied_rate_value=float(warm_steps),
                following_learning_rate=varied_config.following_learning_rate,
                observer_learning_rate=varied_config.observer_learning_rate,
                actor_learning_rate=varied_config.actor_learning_rate,
                update_rule="raw_reward",
                entropy_beta=0.0,
                warm_start_steps=warm_steps,
                horizon_steps=train_steps,
            )
        )
        summaries.append(
            seed_summary_from_trace(
                varied_config,
                trace,
                section="7.3",
                experiment=label,
                seed=seed,
                following_learning_rate=varied_config.following_learning_rate,
                observer_learning_rate=varied_config.observer_learning_rate,
                actor_learning_rate=varied_config.actor_learning_rate,
                update_rule="raw_reward",
                entropy_beta=0.0,
                warm_start_steps=warm_steps,
                horizon_steps=train_steps,
            )
        )
    return rows, summaries


def build_section_7_2_trainer(
    config: Paper72Config,
    *,
    seed: int,
    selection_update_rule: str,
    selection_entropy_beta: float = 0.0,
    selection_baseline_learning_rate: float = 0.05,
    learn_selection: bool = True,
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
    following_optimizer = None
    if learn_selection:
        following_optimizer = torch.optim.SGD(
            following_parameters(following_rates_by_actor),
            lr=config.following_learning_rate,
        )
    return GroupImperfectStateTrainer(
        env=env,
        actors=actors,
        observers=observers,
        following_rates_by_actor=following_rates_by_actor,
        actor_selection_probs=config.actor_selection_probs,
        following_optimizer=following_optimizer,
        observer_optimizer=torch.optim.SGD(
            observer_parameters(observers),
            lr=config.observer_learning_rate,
        ),
        optimal_actions=optimal_actions(config.reward_table),
        selection_update_rule=selection_update_rule,
        selection_baseline_learning_rate=selection_baseline_learning_rate,
        selection_entropy_beta=selection_entropy_beta,
    )


def build_section_7_3_trainer(
    config: Paper73Config,
    *,
    seed: int,
    selection_update_rule: str,
    selection_entropy_beta: float = 0.0,
    selection_baseline_learning_rate: float = 0.05,
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
            torch.optim.SGD(actor.parameters(), lr=config.actor_learning_rate)
            for actor in actors
        ],
        optimal_actions=optimal_actions(config.reward_table),
        selection_update_rule=selection_update_rule,
        selection_baseline_learning_rate=selection_baseline_learning_rate,
        selection_entropy_beta=selection_entropy_beta,
    )


def following_parameters(
    following_rates_by_actor: list[FollowingRates],
) -> list[torch.nn.Parameter]:
    return [
        parameter
        for rates in following_rates_by_actor
        for parameter in rates.parameters()
    ]


def rows_from_trace(
    config: Any,
    trace: Any,
    *,
    section: str,
    experiment: str,
    seed: int,
    varied_rate_name: str,
    varied_rate_value: float,
    following_learning_rate: float,
    observer_learning_rate: float | None,
    actor_learning_rate: float | None,
    update_rule: str,
    entropy_beta: float,
    warm_start_steps: int,
    horizon_steps: int,
) -> list[dict[str, Any]]:
    rows = trace_to_rows(
        config=config,
        trace=trace,
        experiment=experiment,
        seed=seed,
        varied_rate_name=varied_rate_name,
        varied_rate_value=varied_rate_value,
        following_learning_rate=following_learning_rate,
        observer_learning_rate=observer_learning_rate,
        actor_learning_rate=actor_learning_rate,
        target_values=[sum(features) for features in config.state_features],
    )
    for row in rows:
        row["section"] = section
        row["selection_update_rule"] = update_rule
        row["selection_entropy_beta"] = float(entropy_beta)
        row["warm_start_steps"] = int(warm_start_steps)
        row["horizon_steps"] = int(horizon_steps)
    return rows


def seed_summary_from_trace(
    config: Any,
    trace: Any,
    *,
    section: str,
    experiment: str,
    seed: int,
    following_learning_rate: float,
    observer_learning_rate: float | None,
    actor_learning_rate: float | None,
    update_rule: str,
    entropy_beta: float,
    warm_start_steps: int,
    horizon_steps: int,
) -> dict[str, Any]:
    final_selection = trace.final_selection_weights_by_actor
    final_messages = trace.final_observer_messages_by_state
    if final_selection is None or final_messages is None:
        raise ValueError("trace is missing final evaluation data")
    full_index = config.observer_names.index(FULL_OBSERVER_NAME)
    partial_index = config.observer_names.index(PARTIAL_OBSERVER_NAME)
    target_values = [sum(features) for features in config.state_features]
    full_messages = [message[0] for message in final_messages[full_index]]
    full_r2 = r_squared(full_messages, target_values)
    full_probs = [selection[full_index] for selection in final_selection]
    partial_probs = [selection[partial_index] for selection in final_selection]
    collapsed_actors = [
        actor_index
        for actor_index, selection in enumerate(final_selection)
        if selection[full_index] < 0.5
        or selection[partial_index] > selection[full_index]
    ]
    counts = selected_counts(trace, config.num_actors, config.num_observers)
    ceiling = ideal_reward_ceiling(config)
    return {
        "section": section,
        "experiment": experiment,
        "seed": int(seed),
        "following_learning_rate": float(following_learning_rate),
        "observer_learning_rate": (
            "" if observer_learning_rate is None else float(observer_learning_rate)
        ),
        "actor_learning_rate": (
            "" if actor_learning_rate is None else float(actor_learning_rate)
        ),
        "selection_update_rule": update_rule,
        "selection_entropy_beta": float(entropy_beta),
        "warm_start_steps": int(warm_start_steps),
        "horizon_steps": int(horizon_steps),
        "final_expected_reward": float(trace.final_expected_reward),
        "final_sampled_reward": float(trace.final_average_reward),
        "final_optimal_action_probability": float(trace.final_mean_optimal_action_prob),
        "success": float(trace.final_expected_reward >= 0.95 * ceiling),
        "mean_full_probability": float(fmean(full_probs)),
        "min_full_probability": float(min(full_probs)),
        "max_partial_probability": float(max(partial_probs)),
        "full_observer_r2": float(full_r2),
        "collapsed_actor_count": len(collapsed_actors),
        "collapsed_actors": format_float_sequence([float(i) for i in collapsed_actors]),
        "selected_counts_by_actor": counts,
    }


def selected_counts(trace: Any, num_actors: int, num_observers: int) -> str:
    counts = Counter(
        zip(
            trace.selected_actor_indices,
            trace.selected_observer_indices,
            strict=True,
        )
    )
    parts: list[str] = []
    for actor_index in range(num_actors):
        actor_counts = [
            counts[(actor_index, observer_index)]
            for observer_index in range(num_observers)
        ]
        parts.append(
            f"actor{actor_index}:"
            + "/".join(str(int(count)) for count in actor_counts)
        )
    return ";".join(parts)


def summarize_seed_rows(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in seed_rows:
        grouped[(str(row["section"]), str(row["experiment"]))].append(row)
    summaries: list[dict[str, Any]] = []
    for (section, experiment), rows in sorted(grouped.items()):
        summaries.append(
            {
                "section": section,
                "experiment": experiment,
                "following_learning_rate": rows[0]["following_learning_rate"],
                "observer_learning_rate": rows[0]["observer_learning_rate"],
                "actor_learning_rate": rows[0]["actor_learning_rate"],
                "selection_update_rule": rows[0]["selection_update_rule"],
                "selection_entropy_beta": rows[0]["selection_entropy_beta"],
                "warm_start_steps": rows[0]["warm_start_steps"],
                "horizon_steps": rows[0]["horizon_steps"],
                "final_expected_reward": fmean(
                    float(row["final_expected_reward"]) for row in rows
                ),
                "final_sampled_reward": fmean(
                    float(row["final_sampled_reward"]) for row in rows
                ),
                "final_optimal_action_probability": fmean(
                    float(row["final_optimal_action_probability"]) for row in rows
                ),
                "success_rate": fmean(float(row["success"]) for row in rows),
                "final_mean_full_probability": fmean(
                    float(row["mean_full_probability"]) for row in rows
                ),
                "final_min_full_probability": min(
                    float(row["min_full_probability"]) for row in rows
                ),
                "final_max_partial_probability": max(
                    float(row["max_partial_probability"]) for row in rows
                ),
                "final_full_observer_r2": fmean(
                    float(row["full_observer_r2"]) for row in rows
                ),
                "collapsed_actor_seed_pairs": sum(
                    int(row["collapsed_actor_count"]) for row in rows
                ),
            }
        )
    return summaries


def write_alpha_w_figure(rows: list[dict[str, Any]], output_base: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.0))
    plot_sweep_metric(
        axes[0, 0],
        rows,
        section="7.2",
        experiment_prefix="7.2 alpha_w=",
        metric="expected_reward",
        title="7.2 expected reward",
        ylabel="reward",
    )
    plot_sweep_metric(
        axes[0, 1],
        rows,
        section="7.2",
        experiment_prefix="7.2 alpha_w=",
        metric="min_full_probability",
        title=r"7.2 min $P(\mathrm{full})$",
        ylabel="probability",
    )
    plot_sweep_metric(
        axes[1, 0],
        rows,
        section="7.3",
        experiment_prefix="7.3 alpha_w=",
        metric="expected_reward",
        title="7.3 expected reward",
        ylabel="reward",
    )
    plot_sweep_metric(
        axes[1, 1],
        rows,
        section="7.3",
        experiment_prefix="7.3 alpha_w=",
        metric="min_full_probability",
        title=r"7.3 min $P(\mathrm{full})$",
        ylabel="probability",
    )
    fig.suptitle("Section 7 Stabilization: Slower Selection Updates", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, output_base)


def write_method_figure(
    rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    output_base: Path,
) -> None:
    selected_experiments = [
        "7.3 alpha_w=0.005",
        "7.3 alpha_w=0.001",
        "7.3 horizon=100k",
        "7.3 entropy beta=0.003",
        "7.3 advantage",
        "7.3 warm-start compression",
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.2))
    plot_experiment_curves(
        axes[0],
        rows,
        selected_experiments,
        "expected_reward",
        "Expected reward",
        "reward",
    )
    plot_experiment_curves(
        axes[1],
        rows,
        selected_experiments,
        "min_full_probability",
        r"Min $P(\mathrm{full})$",
        "probability",
    )
    summary_lookup = {
        str(row["experiment"]): row
        for row in summary_rows
        if str(row["experiment"]) in selected_experiments
    }
    labels = [label.replace("7.3 ", "") for label in selected_experiments]
    collapse_counts = [
        float(summary_lookup[label]["collapsed_actor_seed_pairs"])
        if label in summary_lookup
        else 0.0
        for label in selected_experiments
    ]
    axes[2].bar(range(len(labels)), collapse_counts, color="#4c72b0")
    axes[2].set_title("Collapsed Actor/seed pairs")
    axes[2].set_ylabel("count")
    axes[2].set_xticks(range(len(labels)))
    axes[2].set_xticklabels(labels, rotation=35, ha="right")
    axes[2].grid(True, axis="y", alpha=0.25)
    fig.suptitle("Section 7.3 Stabilization Variants", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_figure(fig, output_base)


def write_per_actor_figure(rows: list[dict[str, Any]], output_base: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.0))
    plot_per_actor_full(
        axes[0],
        rows,
        experiment="7.3 alpha_w=0.005",
        title=r"Baseline $P_j(\mathrm{full})$",
    )
    plot_per_actor_full(
        axes[1],
        rows,
        experiment="7.3 alpha_w=0.001",
        title=r"Slower selection $P_j(\mathrm{full})$",
    )
    fig.suptitle("Per-Actor Full-Observer Selection", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_figure(fig, output_base)


def plot_sweep_metric(
    axis: plt.Axes,
    rows: list[dict[str, Any]],
    *,
    section: str,
    experiment_prefix: str,
    metric: str,
    title: str,
    ylabel: str,
) -> None:
    filtered = [
        row
        for row in rows
        if str(row["section"]) == section
        and str(row["experiment"]).startswith(experiment_prefix)
    ]
    experiments = sorted(
        {str(row["experiment"]) for row in filtered},
        key=lambda name: float(name.split("=")[-1]),
    )
    for experiment in experiments:
        series = metric_series(filtered, experiment, metric)
        label = r"$\alpha_w$=" + experiment.split("=")[-1]
        axis.plot(
            [point[0] for point in series],
            [point[1] for point in series],
            linewidth=1.7,
            label=label,
        )
    reference = 1.0
    axis.axhline(reference, color="black", linewidth=1.0, alpha=0.5)
    axis.axhline(0.95 * reference, color="black", linestyle="--", linewidth=0.9, alpha=0.4)
    axis.set_title(title)
    axis.set_xlabel("training step")
    axis.set_ylabel(ylabel)
    axis.set_ylim(-0.05, 1.05)
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend(frameon=False, fontsize=8)


def plot_experiment_curves(
    axis: plt.Axes,
    rows: list[dict[str, Any]],
    experiments: list[str],
    metric: str,
    title: str,
    ylabel: str,
) -> None:
    for experiment in experiments:
        series = metric_series(rows, experiment, metric)
        if not series:
            continue
        label = experiment.replace("7.3 ", "")
        axis.plot(
            [point[0] for point in series],
            [point[1] for point in series],
            linewidth=1.7,
            label=label,
        )
    axis.axhline(1.0, color="black", linewidth=1.0, alpha=0.5)
    axis.axhline(0.95, color="black", linestyle="--", linewidth=0.9, alpha=0.4)
    axis.set_title(title)
    axis.set_xlabel("training step")
    axis.set_ylabel(ylabel)
    axis.set_ylim(-0.05, 1.05)
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend(frameon=False, fontsize=8)


def plot_per_actor_full(
    axis: plt.Axes,
    rows: list[dict[str, Any]],
    *,
    experiment: str,
    title: str,
) -> None:
    full_rows = [
        row
        for row in rows
        if str(row["experiment"]) == experiment
        and str(row["observer_name"]) == FULL_OBSERVER_NAME
    ]
    actor_indices = sorted({int(row["actor_index"]) for row in full_rows})
    for actor_index in actor_indices:
        grouped: dict[int, list[float]] = defaultdict(list)
        for row in full_rows:
            if int(row["actor_index"]) == actor_index:
                grouped[int(row["step"])].append(float(row["selection_probability"]))
        series = [(step, fmean(values)) for step, values in sorted(grouped.items())]
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


def metric_series(
    rows: list[dict[str, Any]],
    experiment: str,
    metric: str,
) -> list[tuple[int, float]]:
    if metric == "min_full_probability":
        return min_full_probability_series(rows, experiment)
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        if (
            str(row["experiment"]) == experiment
            and int(row["actor_index"]) == 0
            and int(row["observer_index"]) == 0
        ):
            grouped[int(row["step"])].append(float(row[metric]))
    return [(step, fmean(values)) for step, values in sorted(grouped.items())]


def min_full_probability_series(
    rows: list[dict[str, Any]],
    experiment: str,
) -> list[tuple[int, float]]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        if (
            str(row["experiment"]) == experiment
            and str(row["observer_name"]) == FULL_OBSERVER_NAME
        ):
            grouped[int(row["step"])].append(float(row["selection_probability"]))
    return [(step, min(values)) for step, values in sorted(grouped.items())]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
