"""1 Actor x 1 Observer joint phi+theta diagnosis (meeting 2026-07-22).

The meeting asks for a systematic single-actor single-observer decomposition of
the "joint compression + policy learning fails at K>=32" claim:

* ``policy_only``            fixed ideal compression, learn theta only (M1)
* ``compression_only``       fixed sum-decoder policy, learn phi only (M2)
* ``joint``                  learn phi and theta together (M3)
* ``freeze_phi_train_theta`` hard-code the phi learned by ``joint``, re-learn a
  fresh theta from scratch (M4, the professor's key contradiction test)
* ``freeze_theta_train_phi`` freeze the theta learned by ``joint``, continue
  training phi from its joint endpoint (M5)
* ``phi_then_theta``         train phi alone (fixed sum decoder), freeze it,
  then train a fresh RBF theta (curriculum control)
* ``phi_then_joint``         train phi alone, then continue with both phi and
  theta updating (tests whether joint dynamics destroy a good phi)
* ``theta_then_joint``       train theta on ideal phi, then switch to learnable
  phi and update both (tests whether joint dynamics destroy a good theta)

Every run records phi weight-trajectory statistics so that "the weights
converged" is a measurable statement (end-window movement), not an eyeball
judgement (M6).

Example:

    python experiments/run_joint_phi_theta_1x1_diag.py \
        --modes policy_only,compression_only,joint \
        --state-counts 8 --steps 20000 --seeds 0 1 \
        --out outputs/diag_1x1_k8_smoke.json
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any

import torch

from actor_observer_rl.envs import FeaturePartialObservableStateEnv
from actor_observer_rl.policies.rbf_policy import RBFPolicy
from actor_observer_rl.policies.sum_decoder import LearnableSumDecoderPolicy
from actor_observer_rl.training import (
    ActorCompressor,
    UpdatedImperfectStateTrace,
    UpdatedImperfectStateTrainer,
    build_scalar_sum_actor_compressors,
    cumulative_binary_state_features,
    diagonal_reward_table,
    fixed_sum_observer_policy,
)
from actor_observer_rl.utils.seeding import set_global_seed

ALL_MODES = (
    "policy_only",
    "compression_only",
    "joint",
    "freeze_phi_train_theta",
    "freeze_theta_train_phi",
    "phi_then_theta",
    "phi_then_joint",
    "theta_then_joint",
)
JOINT_DEPENDENT_MODES = {"freeze_phi_train_theta", "freeze_theta_train_phi"}
STAGED_MODES = {"phi_then_theta", "phi_then_joint", "theta_then_joint"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--modes",
        type=str,
        default=",".join(ALL_MODES),
        help="Comma-separated subset of: " + ", ".join(ALL_MODES),
    )
    parser.add_argument("--state-counts", nargs="+", type=int, default=[8, 16, 32])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=300000)
    parser.add_argument(
        "--freeze-steps",
        type=int,
        default=None,
        help="Steps for the freeze phases. Defaults to --steps.",
    )
    parser.add_argument("--eval-interval", type=int, default=1000)
    parser.add_argument("--lr-phi", type=float, default=0.01)
    parser.add_argument("--lr-theta", type=float, default=0.01)
    parser.add_argument("--rbf-sigma", type=float, default=0.5)
    parser.add_argument("--sharpness", type=float, default=4.0)
    parser.add_argument(
        "--policy-head",
        choices=("rbf", "learnable_sum"),
        default="rbf",
        help="Learnable observer policy head for modes that train theta.",
    )
    parser.add_argument("--epsilon-start", type=float, default=0.1)
    parser.add_argument("--epsilon-end", type=float, default=0.0)
    parser.add_argument(
        "--anneal-segments",
        type=int,
        default=4,
        help="Number of constant-epsilon segments for the linear anneal.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/diag_1x1_phi_theta.json"),
    )
    return parser.parse_args()


def make_env(num_states: int) -> FeaturePartialObservableStateEnv:
    return FeaturePartialObservableStateEnv(
        state_features=cumulative_binary_state_features(num_states),
        reward_table=diagonal_reward_table(num_states),
        state_distribution=[1.0 / num_states for _ in range(num_states)],
        observation_dim=1,
    )


def build_full_compressor(num_states: int, *, learnable: bool) -> ActorCompressor:
    return build_scalar_sum_actor_compressors(num_states, learnable=learnable)[2]


def make_learnable_policy(num_states: int, args: argparse.Namespace) -> torch.nn.Module:
    if args.policy_head == "rbf":
        return RBFPolicy(num_states, num_states, sigma=args.rbf_sigma)
    if args.policy_head == "learnable_sum":
        return LearnableSumDecoderPolicy(
            num_actions=num_states,
            init_sharpness=args.sharpness,
        )
    raise ValueError(f"unknown policy head: {args.policy_head}")


def freeze_module(module: torch.nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def unfreeze_module(module: torch.nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(True)


def epsilon_schedule(start: float, end: float, segments: int) -> list[float]:
    if segments <= 1:
        return [end]
    return [
        start + (end - start) * index / (segments - 1) for index in range(segments)
    ]


def run_trainer_with_anneal(
    trainer: UpdatedImperfectStateTrainer,
    *,
    steps: int,
    seed: int,
    eval_interval: int,
    epsilons: list[float],
) -> list[UpdatedImperfectStateTrace]:
    """Run in constant-epsilon segments; modules and optimizers persist."""
    segments = len(epsilons)
    base = steps // segments
    segment_steps = [base] * segments
    segment_steps[-1] += steps - base * segments
    traces: list[UpdatedImperfectStateTrace] = []
    for segment_index, (epsilon, num_steps) in enumerate(
        zip(epsilons, segment_steps, strict=True)
    ):
        if num_steps <= 0:
            continue
        trainer.action_exploration_probability = float(epsilon)
        traces.append(
            trainer.run(
                num_steps=num_steps,
                seed=seed if segment_index == 0 else None,
                eval_interval=eval_interval,
            )
        )
    return traces


def l2(left: list[float], right: list[float]) -> float:
    return math.sqrt(
        sum((a - b) ** 2 for a, b in zip(left, right, strict=True))
    )


def phi_trajectory_stats(traces: list[UpdatedImperfectStateTrace]) -> dict[str, Any]:
    """Convergence evidence for phi: end-window movement, not eyeballing."""
    snapshots: list[list[float]] = []
    for trace in traces:
        for weights, biases in zip(
            trace.actor_compressor_linear_weights,
            trace.actor_compressor_linear_biases,
            strict=True,
        ):
            snapshots.append(list(weights[0]) + list(biases[0]))
    if len(snapshots) < 2:
        return {}
    movements = [
        l2(snapshots[index], snapshots[index + 1])
        for index in range(len(snapshots) - 1)
    ]
    tail = max(1, len(movements) // 10)
    final = snapshots[-1]
    return {
        "phi_total_movement": sum(movements),
        "phi_mean_movement_last10pct": fmean(movements[-tail:]),
        "phi_max_movement_last10pct": max(movements[-tail:]),
        "phi_final_max_abs": max(abs(value) for value in final),
        "phi_moved_from_start": l2(snapshots[0], final),
    }


def message_stats(trace: UpdatedImperfectStateTrace) -> dict[str, Any]:
    messages = [
        float(entry[0]) for entry in trace.actor_messages_by_state[-1][0]
    ]
    num_states = len(messages)
    rounded_matches = sum(
        int(max(0, min(num_states - 1, round(value)))) == index
        for index, value in enumerate(messages)
    )
    gaps = [
        abs(messages[index + 1] - messages[index])
        for index in range(num_states - 1)
    ]
    return {
        "final_messages": messages,
        "message_round_accuracy": rounded_matches / num_states,
        "message_min_gap": min(gaps) if gaps else 0.0,
    }


def summarize_traces(traces: list[UpdatedImperfectStateTrace]) -> dict[str, Any]:
    last = traces[-1]
    record: dict[str, Any] = {
        "final_expected_reward": last.expected_rewards[-1],
        "final_mean_correct_action_prob": last.mean_correct_action_probs[-1],
        "expected_reward_curve": [
            reward for trace in traces for reward in trace.expected_rewards
        ],
        "correct_action_prob_curve": [
            prob for trace in traces for prob in trace.mean_correct_action_probs
        ],
    }
    record.update(message_stats(last))
    record.update(phi_trajectory_stats(traces))
    return record


def run_staged_mode(
    mode: str,
    *,
    num_states: int,
    seed: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Curriculum: stage-1 single component, then stage-2 joint or freeze."""
    set_global_seed(seed)
    env = make_env(num_states)
    epsilons = epsilon_schedule(
        args.epsilon_start, args.epsilon_end, args.anneal_segments
    )
    stage1_steps = args.steps
    stage2_steps = args.freeze_steps or args.steps

    if mode in {"phi_then_theta", "phi_then_joint"}:
        compressor = build_full_compressor(num_states, learnable=True)
        stage1_policy = fixed_sum_observer_policy(
            num_actions=num_states, sharpness=args.sharpness
        )
        stage1 = UpdatedImperfectStateTrainer(
            env=env,
            actor_compressors=[compressor],
            observer_policies=[stage1_policy],
            compression_optimizer=torch.optim.SGD(
                compressor.parameters(), lr=args.lr_phi
            ),
        )
        stage1_traces = run_trainer_with_anneal(
            stage1,
            steps=stage1_steps,
            seed=seed,
            eval_interval=args.eval_interval,
            epsilons=epsilons,
        )
        stage1_summary = summarize_traces(stage1_traces)

        policy = make_learnable_policy(num_states, args)
        if mode == "phi_then_theta":
            freeze_module(compressor)
            stage2 = UpdatedImperfectStateTrainer(
                env=env,
                actor_compressors=[compressor],
                observer_policies=[policy],
                policy_optimizers=[
                    torch.optim.SGD(policy.parameters(), lr=args.lr_theta)
                ],
            )
        else:
            unfreeze_module(compressor)
            stage2 = UpdatedImperfectStateTrainer(
                env=env,
                actor_compressors=[compressor],
                observer_policies=[policy],
                compression_optimizer=torch.optim.SGD(
                    compressor.parameters(), lr=args.lr_phi
                ),
                policy_optimizers=[
                    torch.optim.SGD(policy.parameters(), lr=args.lr_theta)
                ],
            )
    elif mode == "theta_then_joint":
        compressor = build_full_compressor(num_states, learnable=False)
        policy = make_learnable_policy(num_states, args)
        stage1 = UpdatedImperfectStateTrainer(
            env=env,
            actor_compressors=[compressor],
            observer_policies=[policy],
            policy_optimizers=[torch.optim.SGD(policy.parameters(), lr=args.lr_theta)],
        )
        stage1_traces = run_trainer_with_anneal(
            stage1,
            steps=stage1_steps,
            seed=seed,
            eval_interval=args.eval_interval,
            epsilons=epsilons,
        )
        stage1_summary = summarize_traces(stage1_traces)

        # Replace ideal phi with a fresh learnable compressor; keep trained theta.
        learnable = build_full_compressor(num_states, learnable=True)
        stage2 = UpdatedImperfectStateTrainer(
            env=env,
            actor_compressors=[learnable],
            observer_policies=[policy],
            compression_optimizer=torch.optim.SGD(
                learnable.parameters(), lr=args.lr_phi
            ),
            policy_optimizers=[torch.optim.SGD(policy.parameters(), lr=args.lr_theta)],
        )
        compressor = learnable
    else:
        raise ValueError(f"unknown staged mode: {mode}")

    stage2_traces = run_trainer_with_anneal(
        stage2,
        steps=stage2_steps,
        seed=None,
        eval_interval=args.eval_interval,
        epsilons=epsilons,
    )
    stage2_summary = summarize_traces(stage2_traces)
    record = {
        "mode": mode,
        "num_states": num_states,
        "seed": seed,
        "steps": stage1_steps,
        "stage2_steps": stage2_steps,
        "lr_phi": args.lr_phi,
        "lr_theta": args.lr_theta,
        "epsilons": epsilons,
        "stage1_expected_reward": stage1_summary["final_expected_reward"],
        "stage1_mean_correct_action_prob": stage1_summary[
            "final_mean_correct_action_prob"
        ],
        "stage1_message_round_accuracy": stage1_summary["message_round_accuracy"],
        "final_expected_reward": stage2_summary["final_expected_reward"],
        "final_mean_correct_action_prob": stage2_summary[
            "final_mean_correct_action_prob"
        ],
        "message_round_accuracy": stage2_summary["message_round_accuracy"],
        "final_messages": stage2_summary["final_messages"],
        "message_min_gap": stage2_summary["message_min_gap"],
        "reward_drop": (
            stage1_summary["final_expected_reward"]
            - stage2_summary["final_expected_reward"]
        ),
    }
    for key in (
        "phi_total_movement",
        "phi_mean_movement_last10pct",
        "phi_max_movement_last10pct",
        "phi_final_max_abs",
        "phi_moved_from_start",
    ):
        if key in stage2_summary:
            record[key] = stage2_summary[key]
        elif key in stage1_summary:
            record[key] = stage1_summary[key]
    return record


def run_mode(
    mode: str,
    *,
    num_states: int,
    seed: int,
    args: argparse.Namespace,
    joint_artifacts: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Run one diagnosis mode. Returns (record, joint_artifacts_or_None)."""
    if mode in STAGED_MODES:
        return run_staged_mode(
            mode, num_states=num_states, seed=seed, args=args
        ), None

    set_global_seed(seed)
    env = make_env(num_states)
    epsilons = epsilon_schedule(
        args.epsilon_start, args.epsilon_end, args.anneal_segments
    )
    freeze_steps = args.freeze_steps or args.steps

    if mode == "policy_only":
        compressor = build_full_compressor(num_states, learnable=False)
        policy = make_learnable_policy(num_states, args)
        trainer = UpdatedImperfectStateTrainer(
            env=env,
            actor_compressors=[compressor],
            observer_policies=[policy],
            policy_optimizers=[torch.optim.SGD(policy.parameters(), lr=args.lr_theta)],
        )
        steps = args.steps
    elif mode == "compression_only":
        compressor = build_full_compressor(num_states, learnable=True)
        policy = fixed_sum_observer_policy(
            num_actions=num_states, sharpness=args.sharpness
        )
        trainer = UpdatedImperfectStateTrainer(
            env=env,
            actor_compressors=[compressor],
            observer_policies=[policy],
            compression_optimizer=torch.optim.SGD(
                compressor.parameters(), lr=args.lr_phi
            ),
        )
        steps = args.steps
    elif mode == "joint":
        compressor = build_full_compressor(num_states, learnable=True)
        policy = make_learnable_policy(num_states, args)
        trainer = UpdatedImperfectStateTrainer(
            env=env,
            actor_compressors=[compressor],
            observer_policies=[policy],
            compression_optimizer=torch.optim.SGD(
                compressor.parameters(), lr=args.lr_phi
            ),
            policy_optimizers=[torch.optim.SGD(policy.parameters(), lr=args.lr_theta)],
        )
        steps = args.steps
    elif mode == "freeze_phi_train_theta":
        if joint_artifacts is None:
            raise ValueError(f"{mode} requires a prior joint run")
        compressor = build_full_compressor(num_states, learnable=True)
        compressor.load_state_dict(joint_artifacts["compressor_state"])
        freeze_module(compressor)
        policy = make_learnable_policy(num_states, args)
        trainer = UpdatedImperfectStateTrainer(
            env=env,
            actor_compressors=[compressor],
            observer_policies=[policy],
            policy_optimizers=[torch.optim.SGD(policy.parameters(), lr=args.lr_theta)],
        )
        steps = freeze_steps
    elif mode == "freeze_theta_train_phi":
        if joint_artifacts is None:
            raise ValueError(f"{mode} requires a prior joint run")
        compressor = build_full_compressor(num_states, learnable=True)
        compressor.load_state_dict(joint_artifacts["compressor_state"])
        policy = make_learnable_policy(num_states, args)
        policy.load_state_dict(joint_artifacts["policy_state"])
        freeze_module(policy)
        trainer = UpdatedImperfectStateTrainer(
            env=env,
            actor_compressors=[compressor],
            observer_policies=[policy],
            compression_optimizer=torch.optim.SGD(
                compressor.parameters(), lr=args.lr_phi
            ),
        )
        steps = freeze_steps
    else:
        raise ValueError(f"unknown mode: {mode}")

    traces = run_trainer_with_anneal(
        trainer,
        steps=steps,
        seed=seed,
        eval_interval=args.eval_interval,
        epsilons=epsilons,
    )
    record = {
        "mode": mode,
        "num_states": num_states,
        "seed": seed,
        "steps": steps,
        "lr_phi": args.lr_phi,
        "lr_theta": args.lr_theta,
        "epsilons": epsilons,
    }
    record.update(summarize_traces(traces))
    if mode == "freeze_theta_train_phi":
        record["phi_improved_reward"] = (
            record["final_expected_reward"]
            - float(joint_artifacts["final_expected_reward"])
        )

    artifacts = None
    if mode == "joint":
        artifacts = {
            "compressor_state": copy.deepcopy(compressor.state_dict()),
            "policy_state": copy.deepcopy(policy.state_dict()),
            "final_expected_reward": record["final_expected_reward"],
        }
    return record, artifacts


def run_diagnosis(args: argparse.Namespace) -> list[dict[str, Any]]:
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    for mode in modes:
        if mode not in ALL_MODES:
            raise ValueError(f"unknown mode: {mode}")
    needs_joint = bool(JOINT_DEPENDENT_MODES.intersection(modes))
    if needs_joint and "joint" not in modes:
        modes = ["joint", *modes]
    ordered = [mode for mode in ALL_MODES if mode in modes]

    records: list[dict[str, Any]] = []
    for num_states in args.state_counts:
        for seed in args.seeds:
            joint_artifacts: dict[str, Any] | None = None
            for mode in ordered:
                print(
                    f"[K={num_states} seed={seed}] mode={mode} ...",
                    flush=True,
                )
                record, artifacts = run_mode(
                    mode,
                    num_states=num_states,
                    seed=seed,
                    args=args,
                    joint_artifacts=joint_artifacts,
                )
                if artifacts is not None:
                    joint_artifacts = artifacts
                records.append(record)
                stage1 = ""
                if "stage1_expected_reward" in record:
                    stage1 = (
                        f" stage1_r={record['stage1_expected_reward']:.3f}"
                        f" drop={record.get('reward_drop', 0.0):+.3f}"
                    )
                print(
                    f"    reward={record['final_expected_reward']:.3f} "
                    f"p_opt={record['final_mean_correct_action_prob']:.3f} "
                    f"round_acc={record['message_round_accuracy']:.2f}"
                    f"{stage1}",
                    flush=True,
                )
    return records


def print_summary(records: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault((record["mode"], record["num_states"]), []).append(record)
    print("\nSummary (mean over seeds):")
    header = (
        f"{'mode':<24} {'K':>4} {'reward':>8} {'p_opt':>8} "
        f"{'round_acc':>9} {'phi_tail_move':>13}"
    )
    print(header)
    for mode in ALL_MODES:
        for (group_mode, num_states), rows in sorted(grouped.items()):
            if group_mode != mode:
                continue
            tail_moves = [
                row["phi_mean_movement_last10pct"]
                for row in rows
                if "phi_mean_movement_last10pct" in row
            ]
            tail = f"{fmean(tail_moves):.4f}" if tail_moves else "-"
            print(
                f"{mode:<24} {num_states:>4} "
                f"{fmean(row['final_expected_reward'] for row in rows):>8.3f} "
                f"{fmean(row['final_mean_correct_action_prob'] for row in rows):>8.3f} "
                f"{fmean(row['message_round_accuracy'] for row in rows):>9.2f} "
                f"{tail:>13}"
            )


def main() -> None:
    args = parse_args()
    records = run_diagnosis(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
        },
        "records": records,
    }
    with args.out.open("w") as handle:
        json.dump(payload, handle, indent=2)
    print_summary(records)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
