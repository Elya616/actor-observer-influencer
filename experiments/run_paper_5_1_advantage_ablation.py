"""Compare raw-reward and advantage updates for paper Section 5.1."""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, pstdev
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
    MultiObserver,
    Observer,
)
from actor_observer_rl.envs import PartialObservableStateEnv, StepResult
from actor_observer_rl.policies import SumDecoderPolicy
from actor_observer_rl.training import StagedMultiObserverTrainer
from actor_observer_rl.utils.seeding import set_global_seed


OBSERVER_NAMES = ["zero", "first_feature", "sum"]
OBSERVER_MESSAGES = [
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 1.0],
    [0.0, 1.0, 1.0, 2.0],
]
STATE_DISTRIBUTION = [0.25, 0.25, 0.25, 0.25]
POSITIVE_REWARD_TABLE = [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
]
CENTERED_REWARD_TABLE = [
    [0.5, -0.5, -0.5],
    [-0.5, 0.5, -0.5],
    [-0.5, 0.5, -0.5],
    [-0.5, -0.5, 0.5],
]


@dataclass(frozen=True)
class AblationCondition:
    """One controlled Section 5.1 comparison condition."""

    name: str
    family: str
    reward_table: list[list[float]]
    learning_rate: float = 0.05
    decoder_sharpness: float = 4.0
    reward_noise_std: float = 0.0


class NoisyRewardEnv:
    """Partial-observation env wrapper with zero-mean reward noise."""

    def __init__(
        self,
        base_env: PartialObservableStateEnv,
        reward_noise_std: float,
    ) -> None:
        self.base_env = base_env
        self.reward_noise_std = float(reward_noise_std)
        self._generator = torch.Generator()

    @property
    def reward_table(self) -> list[list[float]]:
        return self.base_env.reward_table

    @property
    def state_distribution(self) -> list[float]:
        return self.base_env.state_distribution

    @property
    def num_states(self) -> int:
        return self.base_env.num_states

    @property
    def num_actions(self) -> int:
        return self.base_env.num_actions

    def reset(self, seed: int | None = None):
        if seed is not None:
            self._generator.manual_seed(seed + 10_000)
        return self.base_env.reset(seed=seed)

    def step(self, action: int) -> StepResult:
        result = self.base_env.step(action)
        reward = float(result.reward)
        if self.reward_noise_std > 0.0:
            noise = torch.normal(
                mean=0.0,
                std=self.reward_noise_std,
                size=(),
                generator=self._generator,
            )
            reward += float(noise.item())
        return StepResult(
            state=result.state,
            observation=result.observation,
            reward=reward,
            done=result.done,
            info=result.info,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_env, name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/paper_5_1_advantage"))
    parser.add_argument("--figure-dir", type=Path, default=Path("docs/figures"))
    parser.add_argument("--num-steps", type=int, default=5000)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--reward-window", type=int, default=100)
    parser.add_argument("--baseline-beta", type=float, default=0.1)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = parser.parse_args()

    conditions = build_conditions()
    rows = run_conditions(
        conditions=conditions,
        seeds=args.seeds,
        num_steps=args.num_steps,
        eval_interval=args.eval_interval,
        log_interval=args.log_interval,
        reward_window=args.reward_window,
        baseline_beta=args.baseline_beta,
    )
    summaries = summarize(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    curve_path = args.output_dir / "section_5_1_advantage_curves.csv"
    summary_path = args.output_dir / "section_5_1_advantage_summary.csv"
    write_csv(curve_path, rows)
    write_csv(summary_path, summaries)
    write_main_figure(
        rows,
        args.figure_dir / "paper5_1_advantage_positive_comparison",
        condition_name="positive_easy",
    )
    write_summary_figure(
        summaries,
        args.figure_dir / "paper5_1_advantage_summary",
    )

    print(f"wrote_curves={curve_path}")
    print(f"wrote_summary={summary_path}")
    print(
        "wrote_figures="
        f"{args.figure_dir / 'paper5_1_advantage_positive_comparison'}.[pdf|png|svg], "
        f"{args.figure_dir / 'paper5_1_advantage_summary'}.[pdf|png|svg]"
    )
    print_summary(summaries)


def build_conditions() -> list[AblationCondition]:
    conditions = [
        AblationCondition(
            name="positive_easy",
            family="positive_reward",
            reward_table=POSITIVE_REWARD_TABLE,
            learning_rate=0.05,
            decoder_sharpness=4.0,
        ),
        AblationCondition(
            name="centered_easy",
            family="centered_reward",
            reward_table=CENTERED_REWARD_TABLE,
            learning_rate=0.05,
            decoder_sharpness=4.0,
        ),
        AblationCondition(
            name="positive_noise_0.2",
            family="noise",
            reward_table=POSITIVE_REWARD_TABLE,
            learning_rate=0.05,
            decoder_sharpness=4.0,
            reward_noise_std=0.2,
        ),
        AblationCondition(
            name="positive_noise_0.5",
            family="noise",
            reward_table=POSITIVE_REWARD_TABLE,
            learning_rate=0.05,
            decoder_sharpness=4.0,
            reward_noise_std=0.5,
        ),
        AblationCondition(
            name="offset_5_gap_1",
            family="common_offset",
            reward_table=offset_reward_table(offset=5.0, gap=1.0),
            learning_rate=0.05,
            decoder_sharpness=4.0,
        ),
        AblationCondition(
            name="offset_20_gap_1",
            family="common_offset",
            reward_table=offset_reward_table(offset=20.0, gap=1.0),
            learning_rate=0.05,
            decoder_sharpness=4.0,
        ),
        AblationCondition(
            name="offset_5_gap_0.1",
            family="small_gap_offset",
            reward_table=offset_reward_table(offset=5.0, gap=0.1),
            learning_rate=0.05,
            decoder_sharpness=4.0,
        ),
        AblationCondition(
            name="offset_20_gap_0.1",
            family="small_gap_offset",
            reward_table=offset_reward_table(offset=20.0, gap=0.1),
            learning_rate=0.05,
            decoder_sharpness=4.0,
        ),
        AblationCondition(
            name="offset_5_gap_1_noise_2",
            family="offset_noise",
            reward_table=offset_reward_table(offset=5.0, gap=1.0),
            learning_rate=0.05,
            decoder_sharpness=4.0,
            reward_noise_std=2.0,
        ),
        AblationCondition(
            name="quality_medium",
            family="quality_separation",
            reward_table=POSITIVE_REWARD_TABLE,
            learning_rate=0.05,
            decoder_sharpness=1.2,
        ),
        AblationCondition(
            name="quality_hard",
            family="quality_separation",
            reward_table=POSITIVE_REWARD_TABLE,
            learning_rate=0.05,
            decoder_sharpness=0.4,
        ),
    ]
    for learning_rate in [0.001, 0.005, 0.01, 0.1]:
        conditions.append(
            AblationCondition(
                name=f"lr_{learning_rate:g}",
                family="learning_rate",
                reward_table=POSITIVE_REWARD_TABLE,
                learning_rate=learning_rate,
                decoder_sharpness=4.0,
            )
        )
    return conditions


def offset_reward_table(offset: float, gap: float) -> list[list[float]]:
    return [
        [offset + gap, offset, offset],
        [offset, offset + gap, offset],
        [offset, offset + gap, offset],
        [offset, offset, offset + gap],
    ]


def run_conditions(
    *,
    conditions: list[AblationCondition],
    seeds: list[int],
    num_steps: int,
    eval_interval: int,
    log_interval: int,
    reward_window: int,
    baseline_beta: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for condition in conditions:
        for update_rule in ["raw_reward", "advantage_global"]:
            for seed in seeds:
                trainer = build_trainer(
                    condition=condition,
                    update_rule=update_rule,
                    baseline_beta=baseline_beta,
                    seed=seed,
                )
                trace = trainer.run(
                    num_steps=num_steps,
                    seed=seed,
                    eval_interval=eval_interval,
                    reward_window=reward_window,
                )
                rows.extend(
                    trace_to_rows(
                        condition,
                        update_rule,
                        trace,
                        seed,
                        log_interval=log_interval,
                    )
                )
    return rows


def build_trainer(
    *,
    condition: AblationCondition,
    update_rule: str,
    baseline_beta: float,
    seed: int,
) -> StagedMultiObserverTrainer:
    set_global_seed(seed)
    base_env = PartialObservableStateEnv(
        reward_table=condition.reward_table,
        state_distribution=STATE_DISTRIBUTION,
        observation_dim=1,
    )
    env = NoisyRewardEnv(base_env, reward_noise_std=condition.reward_noise_std)
    observers = [
        Observer(message_module=FixedTableMessageModule(messages))
        for messages in OBSERVER_MESSAGES
    ]
    multi_observer = MultiObserver(
        observers=observers,
        following_rates=FollowingRates(num_observers=len(observers)),
    )
    actor = Actor(
        SumDecoderPolicy(
            num_actions=3,
            sharpness=condition.decoder_sharpness,
        )
    )
    following_optimizer = torch.optim.SGD(
        list(multi_observer.following_parameters()),
        lr=condition.learning_rate,
    )
    return StagedMultiObserverTrainer(
        env=env,
        actor=actor,
        multi_observer=multi_observer,
        following_optimizer=following_optimizer,
        optimal_actions=optimal_actions(condition.reward_table),
        selection_update_rule=update_rule,
        selection_baseline_beta=baseline_beta,
    )


def trace_to_rows(
    condition: AblationCondition,
    update_rule: str,
    trace: Any,
    seed: int,
    *,
    log_interval: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    best_observer_index = best_fixed_observer(trace)
    best_fixed_reward = float(trace.per_observer_expected_rewards[0][best_observer_index])
    threshold_steps = threshold_crossings(
        trace.following_weights,
        best_observer_index=best_observer_index,
    )
    for step_index, weights in enumerate(trace.following_weights):
        step = step_index + 1
        if step != 1 and step != len(trace.following_weights) and step % log_interval != 0:
            continue
        cumulative_reward = trace.cumulative_rewards[step_index]
        cumulative_regret = best_fixed_reward * step - cumulative_reward
        for observer_index, observer_name in enumerate(OBSERVER_NAMES):
            rows.append(
                {
                    "condition": condition.name,
                    "family": condition.family,
                    "update_rule": update_rule,
                    "seed": int(seed),
                    "step": int(step),
                    "learning_rate": float(condition.learning_rate),
                    "decoder_sharpness": float(condition.decoder_sharpness),
                    "reward_noise_std": float(condition.reward_noise_std),
                    "observer_index": int(observer_index),
                    "observer_name": observer_name,
                    "best_observer_index": int(best_observer_index),
                    "is_best_observer": observer_index == best_observer_index,
                    "selection_probability": float(weights[observer_index]),
                    "logit": float(trace.following_logits[step_index][observer_index]),
                    "selection_entropy": float(trace.selection_entropies[step_index]),
                    "reward": float(trace.rewards[step_index]),
                    "average_reward": float(recent_average(trace.rewards, step, window=100)),
                    "cumulative_reward": float(cumulative_reward),
                    "cumulative_regret": float(cumulative_regret),
                    "best_observer_probability": float(weights[best_observer_index]),
                    "update_multiplier": float(
                        trace.selection_update_multipliers[step_index]
                    ),
                    "baseline": float(trace.selection_baselines[step_index]),
                    "threshold_0_7_step": threshold_steps[0.7],
                    "threshold_0_8_step": threshold_steps[0.8],
                    "threshold_0_9_step": threshold_steps[0.9],
                    "best_fixed_reward": best_fixed_reward,
                }
            )
    return rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    final_step_by_condition: dict[tuple[str, str], int] = {}
    for row in rows:
        key = (str(row["condition"]), str(row["update_rule"]))
        final_step_by_condition[key] = max(
            final_step_by_condition.get(key, 0),
            int(row["step"]),
        )

    seed_summaries: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
    for (condition, update_rule), final_step in final_step_by_condition.items():
        final_rows = [
            row
            for row in rows
            if row["condition"] == condition
            and row["update_rule"] == update_rule
            and int(row["step"]) == final_step
            and bool(row["is_best_observer"])
        ]
        for row in final_rows:
            seed_summaries[(condition, update_rule)].append(
                {
                    "final_average_reward": float(row["average_reward"]),
                    "final_best_observer_probability": float(
                        row["best_observer_probability"]
                    ),
                    "cumulative_reward": float(row["cumulative_reward"]),
                    "cumulative_regret": float(row["cumulative_regret"]),
                    "threshold_0_7_step": float(row["threshold_0_7_step"]),
                    "threshold_0_8_step": float(row["threshold_0_8_step"]),
                    "threshold_0_9_step": float(row["threshold_0_9_step"]),
                    "final_entropy": float(row["selection_entropy"]),
                    "learning_rate": float(row["learning_rate"]),
                    "decoder_sharpness": float(row["decoder_sharpness"]),
                    "reward_noise_std": float(row["reward_noise_std"]),
                }
            )

    summaries: list[dict[str, Any]] = []
    for (condition, update_rule), seed_rows in sorted(seed_summaries.items()):
        metrics = {
            metric: [float(row[metric]) for row in seed_rows]
            for metric in [
                "final_average_reward",
                "final_best_observer_probability",
                "cumulative_reward",
                "cumulative_regret",
                "threshold_0_7_step",
                "threshold_0_8_step",
                "threshold_0_9_step",
                "final_entropy",
            ]
        }
        summaries.append(
            {
                "condition": condition,
                "family": condition_family(condition),
                "update_rule": update_rule,
                "learning_rate": seed_rows[0]["learning_rate"],
                "decoder_sharpness": seed_rows[0]["decoder_sharpness"],
                "reward_noise_std": seed_rows[0]["reward_noise_std"],
                **{
                    f"{metric}_mean": fmean(values)
                    for metric, values in metrics.items()
                },
                **{
                    f"{metric}_std": pstdev(values)
                    for metric, values in metrics.items()
                },
                "num_seeds": len(seed_rows),
            }
        )
    return summaries


def condition_family(condition: str) -> str:
    if condition.startswith("offset_") and "noise" in condition:
        return "offset_noise"
    if condition.startswith("offset_") and "gap_0.1" in condition:
        return "small_gap_offset"
    if condition.startswith("offset_"):
        return "common_offset"
    if condition.startswith("positive_noise"):
        return "noise"
    if condition.startswith("quality"):
        return "quality_separation"
    if condition.startswith("lr_"):
        return "learning_rate"
    if condition.startswith("centered"):
        return "centered_reward"
    return "positive_reward"


def best_fixed_observer(trace: Any) -> int:
    per_observer = trace.per_observer_expected_rewards[0]
    return max(range(len(per_observer)), key=lambda index: per_observer[index])


def threshold_crossings(
    following_weights: list[list[float]],
    *,
    best_observer_index: int,
) -> dict[float, int]:
    crossings = {0.7: -1, 0.8: -1, 0.9: -1}
    for step_index, weights in enumerate(following_weights):
        probability = float(weights[best_observer_index])
        for threshold in crossings:
            if crossings[threshold] == -1 and probability >= threshold:
                crossings[threshold] = step_index + 1
    final_step = len(following_weights)
    return {
        threshold: final_step + 1 if step == -1 else step
        for threshold, step in crossings.items()
    }


def recent_average(values: list[float], step: int, window: int) -> float:
    start = max(0, step - window)
    return fmean(values[start:step])


def write_main_figure(
    rows: list[dict[str, Any]],
    output_base: Path,
    condition_name: str,
) -> None:
    condition_rows = [row for row in rows if row["condition"] == condition_name]
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 7.0))
    plot_update_rule_series(
        axes[0, 0],
        condition_rows,
        "average_reward",
        "Average reward",
        "reward",
    )
    plot_update_rule_series(
        axes[0, 1],
        condition_rows,
        "best_observer_probability",
        "Best-Observer selection probability",
        "probability",
    )
    plot_update_rule_series(
        axes[0, 2],
        condition_rows,
        "selection_entropy",
        "Selection entropy",
        "entropy",
    )
    plot_logits(axes[1, 0], condition_rows, update_rule="raw_reward")
    plot_logits(axes[1, 1], condition_rows, update_rule="advantage_global")
    plot_final_lambda_bars(axes[1, 2], condition_rows)
    fig.suptitle("Section 5.1 Raw Reward vs Global Advantage", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, output_base)


def write_summary_figure(rows: list[dict[str, Any]], output_base: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.0))
    plot_summary_bars(
        axes[0],
        rows,
        "final_best_observer_probability_mean",
        "Final best-Observer probability",
    )
    plot_summary_bars(
        axes[1],
        rows,
        "cumulative_regret_mean",
        "Cumulative regret",
    )
    plot_summary_bars(
        axes[2],
        rows,
        "threshold_0_8_step_mean",
        r"Step to $P(\mathrm{best})>0.8$",
    )
    fig.suptitle("Section 5.1 Advantage Ablation Summary", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    save_figure(fig, output_base)


def plot_update_rule_series(
    axis: plt.Axes,
    rows: list[dict[str, Any]],
    metric: str,
    title: str,
    ylabel: str,
) -> None:
    compact_rows = [
        row
        for row in rows
        if int(row["observer_index"]) == int(row["best_observer_index"])
    ]
    for update_rule in ["raw_reward", "advantage_global"]:
        series = mean_series(
            compact_rows,
            lambda row, update_rule=update_rule: row["update_rule"] == update_rule,
            metric,
        )
        axis.plot(
            [point[0] for point in series],
            [point[1] for point in series],
            linewidth=1.8,
            label=update_rule,
        )
    axis.set_title(title)
    axis.set_xlabel("training step")
    axis.set_ylabel(ylabel)
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend(frameon=False, fontsize=8)


def plot_logits(
    axis: plt.Axes,
    rows: list[dict[str, Any]],
    *,
    update_rule: str,
) -> None:
    rule_rows = [row for row in rows if row["update_rule"] == update_rule]
    for observer_name in OBSERVER_NAMES:
        series = mean_series(
            rule_rows,
            lambda row, observer_name=observer_name: row["observer_name"] == observer_name,
            "logit",
        )
        axis.plot(
            [point[0] for point in series],
            [point[1] for point in series],
            linewidth=1.6,
            label=observer_name,
        )
    axis.set_title(f"Logits: {update_rule}")
    axis.set_xlabel("training step")
    axis.set_ylabel("logit")
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend(frameon=False, fontsize=8)


def plot_final_lambda_bars(axis: plt.Axes, rows: list[dict[str, Any]]) -> None:
    final_step = max(int(row["step"]) for row in rows)
    width = 0.35
    positions = list(range(len(OBSERVER_NAMES)))
    for offset, update_rule in [(-width / 2, "raw_reward"), (width / 2, "advantage_global")]:
        values = []
        for observer_name in OBSERVER_NAMES:
            final_rows = [
                row
                for row in rows
                if row["update_rule"] == update_rule
                and row["observer_name"] == observer_name
                and int(row["step"]) == final_step
            ]
            values.append(fmean(float(row["selection_probability"]) for row in final_rows))
        axis.bar(
            [position + offset for position in positions],
            values,
            width=width,
            label=update_rule,
        )
    axis.set_title(r"Final $\lambda_i$")
    axis.set_xticks(positions)
    axis.set_xticklabels(OBSERVER_NAMES, rotation=20)
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("probability")
    axis.legend(frameon=False, fontsize=8)


def plot_summary_bars(
    axis: plt.Axes,
    rows: list[dict[str, Any]],
    metric: str,
    title: str,
) -> None:
    selected_conditions = [
        "positive_easy",
        "offset_20_gap_1",
        "offset_20_gap_0.1",
        "offset_5_gap_1_noise_2",
        "quality_hard",
    ]
    width = 0.36
    positions = list(range(len(selected_conditions)))
    for offset, update_rule in [(-width / 2, "raw_reward"), (width / 2, "advantage_global")]:
        values = []
        for condition in selected_conditions:
            match = [
                row
                for row in rows
                if row["condition"] == condition and row["update_rule"] == update_rule
            ]
            values.append(float(match[0][metric]))
        axis.bar(
            [position + offset for position in positions],
            values,
            width=width,
            label=update_rule,
        )
    axis.set_title(title)
    axis.set_xticks(positions)
    axis.set_xticklabels(selected_conditions, rotation=35, ha="right")
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend(frameon=False, fontsize=8)


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


def optimal_actions(reward_table: list[list[float]]) -> list[int]:
    return [
        max(range(len(rewards)), key=lambda action: rewards[action])
        for rewards in reward_table
    ]


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


def print_summary(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        print(
            "condition={condition} rule={update_rule} "
            "final_reward={final_average_reward_mean:.3f} "
            "p_best={final_best_observer_probability_mean:.3f} "
            "regret={cumulative_regret_mean:.1f} "
            "step80={threshold_0_8_step_mean:.0f} "
            "p_best_std={final_best_observer_probability_std:.3f}".format(**row)
        )


if __name__ == "__main__":
    main()
