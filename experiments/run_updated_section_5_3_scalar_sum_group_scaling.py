"""Updated-paper Section 5.3 scalar-sum group scaling.

Section 5.3 has multiple Actor-compressors and multiple Observers. Each
Observer independently selects an Actor-compressor, receives its scalar message,
acts, and sends reward-gradient feedback to the selected compressor. The learned
objects are:

* Observer policies ``theta_i``.
* Observer selection logits ``w_i``.
* Actor-compressor message maps ``phi_j``.

The experiments here use one timescale:

``alpha_phi = alpha_w = alpha_theta = alpha``.

The only optional stabilizers are uniform sampling floors for selection and
action sampling.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

import matplotlib.pyplot as plt
import torch

from actor_observer_rl.envs import FeaturePartialObservableStateEnv
from actor_observer_rl.policies import LearnableSumDecoderPolicy, SoftmaxPolicy
from actor_observer_rl.training import (
    SelectionLogits,
    UpdatedImperfectStateTrainer,
    build_scalar_sum_actor_compressors,
    cumulative_binary_state_features,
    diagonal_reward_table,
    fixed_sum_observer_policy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-counts", nargs="+", type=int, default=[3, 5, 8])
    parser.add_argument("--observer-counts", nargs="+", type=int, default=[3])
    parser.add_argument("--actor-compressor-counts", nargs="+", type=int, default=[3])
    parser.add_argument(
        "--learning-rates",
        nargs="+",
        type=float,
        default=[0.003, 0.005, 0.01],
        help="Learning rate for phi and w (and for theta unless --lr-theta is set).",
    )
    parser.add_argument(
        "--lr-theta",
        type=float,
        default=None,
        help="Override Observer policy lr. Prefer ~1e-4 with learnable_sum.",
    )
    parser.add_argument(
        "--policy-head",
        choices=("softmax", "learnable_sum", "fixed_sum"),
        default="softmax",
        help="Observer policy head. learnable_sum matches the working 1x1 joint "
        "recipe; fixed_sum freezes theta to the Section 5.2 decoder, which "
        "isolates how much of the 5.3 gap comes from learning theta.",
    )
    parser.add_argument(
        "--policy-init",
        choices=("default", "sum_decoder"),
        default="default",
        help="Observer policy initialization. 'sum_decoder' warms SoftmaxPolicy "
        "to the Section 5.2 quadratic decoder (up to an additive constant). "
        "Ignored for fixed_sum / learnable_sum (already sum-decoder shaped).",
    )
    parser.add_argument(
        "--sharpness",
        type=float,
        default=4.0,
        help="Decoder sharpness for --policy-head fixed_sum and for "
        "--policy-init sum_decoder (matches Section 5.2).",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=None,
        help="Clip the summed multi-Observer gradient. Unset reproduces the "
        "original behaviour, where a seed can diverge into huge messages.",
    )
    parser.add_argument(
        "--loss-reduction",
        choices=("sum", "mean"),
        default="sum",
        help="'sum' reproduces the original behaviour, where the usable alpha "
        "shrinks as the Observer count grows. 'mean' divides by the Observer "
        "count so alpha transfers across M.",
    )
    parser.add_argument(
        "--selection-warmup-steps",
        type=int,
        default=0,
        help="If >0, first train only selection with oracle (fixed-mask) "
        "compressors, then (optionally) reset phi and train phi+w.",
    )
    parser.add_argument(
        "--warmup-reset-phi",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After selection warmup, re-apply --phi-init before training phi "
        "(default). --no-warmup-reset-phi keeps the exact oracle init used "
        "during warmup.",
    )
    parser.add_argument(
        "--phi-init",
        choices=("zero", "oracle", "oracle-noise"),
        default="zero",
        help="Compressor initialization, independent of any warmup. 'zero' is "
        "the paper default. 'oracle' copies the availability mask into the "
        "weights. 'oracle-noise' adds N(0, sigma^2) on top of oracle, which "
        "tests whether the stable recipe needs the exact solution or only a "
        "starting region.",
    )
    parser.add_argument(
        "--phi-init-noise",
        type=float,
        default=0.1,
        help="Standard deviation sigma for --phi-init oracle-noise.",
    )
    parser.add_argument(
        "--lr-phi",
        type=float,
        default=None,
        help="Compression learning rate. Defaults to --learning-rates / alpha.",
    )
    parser.add_argument(
        "--phi-lr-schedule",
        choices=("constant", "cosine"),
        default="constant",
        help="Anneal the compression learning rate while phi trains. At a "
        "constant alpha the messages keep drifting, so the slope inflates until "
        "mid-states cross the decoder's decision boundary.",
    )
    parser.add_argument(
        "--phi-lr-final-fraction",
        type=float,
        default=0.1,
        help="Final compression lr as a fraction of --lr-phi for "
        "--phi-lr-schedule cosine.",
    )
    parser.add_argument(
        "--message-l2-coef",
        type=float,
        default=0.0,
        help="Add coef * ||m||^2 to the loss while learning phi, to stop "
        "REINFORCE from walking messages off to infinity.",
    )
    parser.add_argument("--selection-exploration", type=float, default=0.2)
    parser.add_argument("--action-exploration", type=float, default=0.05)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--steps", type=int, default=120000)
    parser.add_argument("--eval-interval", type=int, default=4000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/updated_section_5_3_scalar_sum_group_scaling"),
    )
    parser.add_argument("--figure-dir", type=Path, default=Path("docs/figures"))
    parser.add_argument(
        "--figure-prefix",
        type=str,
        default=None,
        help="Prefix for generated figure filenames. Defaults to output directory name.",
    )
    parser.add_argument(
        "--doc-path",
        type=Path,
        default=Path("docs/updated_section_5_3_scalar_sum_group_results.md"),
    )
    return parser.parse_args()


def make_env(num_states: int) -> FeaturePartialObservableStateEnv:
    return FeaturePartialObservableStateEnv(
        state_features=cumulative_binary_state_features(num_states),
        reward_table=diagonal_reward_table(num_states),
        state_distribution=[1.0 / num_states for _ in range(num_states)],
        observation_dim=1,
    )


def build_actor_compressors(num_states: int, count: int) -> list[torch.nn.Module]:
    if count < 3:
        raise ValueError("actor-compressor count must be at least 3")
    compressors = build_scalar_sum_actor_compressors(num_states, learnable=True)
    for index in range(3, count):
        extra = build_scalar_sum_actor_compressors(num_states, learnable=True)[1]
        extra.name = f"partial_{index}"
        compressors.append(extra)
    return compressors


def make_observer_policy(
    policy_head: str,
    num_actions: int,
    sharpness: float = 4.0,
    policy_init: str = "default",
) -> torch.nn.Module:
    if policy_head == "softmax":
        policy = SoftmaxPolicy(input_dim=1, num_actions=num_actions)
        if policy_init == "sum_decoder":
            policy.init_as_sum_decoder(sharpness=sharpness)
        elif policy_init != "default":
            raise ValueError(f"unknown policy_init: {policy_init}")
        return policy
    if policy_head == "learnable_sum":
        return LearnableSumDecoderPolicy(
            num_actions=num_actions, init_sharpness=sharpness
        )
    if policy_head == "fixed_sum":
        return fixed_sum_observer_policy(
            num_actions=num_actions, sharpness=sharpness
        )
    raise ValueError(f"unknown policy head: {policy_head}")


def decoder_ceiling(num_states: int, sharpness: float = 4.0) -> float:
    """Expected reward of the fixed decoder when messages are exactly the state ID.

    The quadratic decoder never puts all mass on the correct action, so this is
    the highest reward any compressor can buy. Report rewards relative to it
    instead of against a flat 0.95.
    """
    policy = fixed_sum_observer_policy(num_actions=num_states, sharpness=sharpness)
    rewards = torch.tensor(diagonal_reward_table(num_states), dtype=torch.float32)
    total = 0.0
    with torch.no_grad():
        for state_index in range(num_states):
            probs = policy.action_distribution(
                torch.tensor([float(state_index)])
            ).probs
            total += float(torch.dot(probs, rewards[state_index]).item())
    return total / num_states


def phi_lr_at(
    step: int,
    total_steps: int,
    initial_lr: float,
    schedule: str,
    final_fraction: float,
) -> float:
    """Compression lr at ``step`` of the phi-training phase."""
    if schedule == "constant":
        return float(initial_lr)
    if schedule != "cosine":
        raise ValueError(f"unknown phi lr schedule: {schedule}")
    if not 0.0 < final_fraction <= 1.0:
        raise ValueError("phi_lr_final_fraction must be in (0, 1]")
    progress = 0.0 if total_steps <= 0 else min(1.0, max(0.0, step / total_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(initial_lr) * (
        final_fraction + (1.0 - final_fraction) * cosine
    )


def _run_phi_phase(
    *,
    trainer: UpdatedImperfectStateTrainer,
    optimizer: torch.optim.Optimizer,
    num_steps: int,
    seed: int | None,
    eval_interval: int,
    initial_lr: float,
    schedule: str,
    final_fraction: float,
    step_offset: int,
) -> list[tuple[Any, int, bool]]:
    """Run the phi-training phase, re-setting the lr once per evaluation block.

    With ``final_fraction=1`` the chunked path is bit-identical to one long
    ``run`` call: chunk boundaries consume no randomness.
    """
    if schedule == "constant":
        trace = trainer.run(
            num_steps=num_steps, seed=seed, eval_interval=eval_interval
        )
        return [(trace, step_offset, False)]

    traces: list[tuple[Any, int, bool]] = []
    completed = 0
    while completed < num_steps:
        chunk = min(eval_interval, num_steps - completed)
        lr = phi_lr_at(
            completed, num_steps, initial_lr, schedule, final_fraction
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        trace = trainer.run(
            num_steps=chunk,
            seed=seed if completed == 0 else None,
            eval_interval=eval_interval,
        )
        traces.append((trace, step_offset + completed, completed > 0))
        completed += chunk
    return traces


def _set_compressor_weights(
    actor_compressors: list[torch.nn.Module],
    mode: str,
    *,
    noise_std: float = 0.0,
    generator: torch.Generator | None = None,
) -> None:
    """Set ``phi``: ``oracle`` copies the availability mask, ``zero`` clears it.

    ``oracle-noise`` perturbs the oracle weights and bias with ``N(0,
    noise_std^2)``, so the run starts in a neighbourhood of the solution rather
    than at it.
    """
    if mode not in {"oracle", "zero", "oracle-noise"}:
        raise ValueError(f"unknown compressor weight mode: {mode}")
    if mode == "oracle-noise" and noise_std < 0.0:
        raise ValueError("noise_std must be non-negative")
    for actor in actor_compressors:
        module = actor.message_module
        mask = module.input_mask.detach().cpu().reshape(-1).tolist()
        with torch.no_grad():
            if mode == "zero":
                module.linear.weight.zero_()
                module.linear.bias.zero_()
                continue
            module.linear.weight.copy_(
                torch.tensor([mask], dtype=module.linear.weight.dtype)
            )
            module.linear.bias.zero_()
            if mode == "oracle-noise" and noise_std > 0.0:
                for parameter in (module.linear.weight, module.linear.bias):
                    parameter.add_(
                        torch.randn(
                            parameter.shape,
                            dtype=parameter.dtype,
                            generator=generator,
                        )
                        * float(noise_std)
                    )


def run_one(
    *,
    num_states: int,
    num_observers: int,
    num_actor_compressors: int,
    alpha: float,
    lr_theta: float,
    policy_head: str,
    selection_exploration: float,
    action_exploration: float,
    seed: int,
    steps: int,
    eval_interval: int,
    sharpness: float = 4.0,
    policy_init: str = "default",
    max_grad_norm: float | None = None,
    loss_reduction: str = "sum",
    selection_warmup_steps: int = 0,
    warmup_reset_phi: bool = True,
    lr_phi: float | None = None,
    message_l2_coef: float = 0.0,
    phi_init: str = "zero",
    phi_init_noise: float = 0.1,
    phi_lr_schedule: str = "constant",
    phi_lr_final_fraction: float = 0.1,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if selection_warmup_steps < 0:
        raise ValueError("selection_warmup_steps must be non-negative")
    if selection_warmup_steps >= steps and selection_warmup_steps > 0:
        raise ValueError("selection_warmup_steps must be smaller than steps")
    if phi_init not in {"zero", "oracle", "oracle-noise"}:
        raise ValueError(f"unknown phi_init: {phi_init}")
    phi_lr = float(alpha if lr_phi is None else lr_phi)
    # Own generator so the perturbation is reproducible from the seed alone and
    # does not depend on how much global RNG earlier configs consumed.
    phi_init_generator = torch.Generator().manual_seed(int(seed) + 987654321)

    env = make_env(num_states)
    actor_compressors = build_actor_compressors(num_states, num_actor_compressors)
    _set_compressor_weights(
        actor_compressors,
        phi_init,
        noise_std=phi_init_noise,
        generator=phi_init_generator,
    )
    policies = [
        make_observer_policy(
            policy_head, num_states, sharpness, policy_init=policy_init
        )
        for _ in range(num_observers)
    ]
    selections = [SelectionLogits(num_actor_compressors) for _ in range(num_observers)]

    compression_parameters = [
        parameter for actor in actor_compressors for parameter in actor.parameters()
    ]
    selection_parameters = [
        parameter for selection in selections for parameter in selection.parameters()
    ]
    # fixed_sum has no trainable theta; passing None keeps the Section 5.2 setup
    # (message still gets gradient through the action log-prob).
    policy_optimizers = (
        None
        if policy_head == "fixed_sum"
        else [torch.optim.SGD(policy.parameters(), lr=lr_theta) for policy in policies]
    )
    selection_optimizer = torch.optim.SGD(selection_parameters, lr=alpha)
    compression_optimizer = torch.optim.SGD(compression_parameters, lr=phi_lr)

    trainer = UpdatedImperfectStateTrainer(
        env=env,
        actor_compressors=actor_compressors,
        observer_policies=policies,
        selection_logits_by_observer=selections,
        compression_optimizer=None,
        selection_optimizer=selection_optimizer,
        policy_optimizers=policy_optimizers,
        selection_exploration_probability=selection_exploration,
        selection_log_prob_source="learned",
        action_exploration_probability=action_exploration,
        action_log_prob_source="learned",
        sample_selection=True,
        sample_actions=True,
        max_grad_norm=max_grad_norm,
        loss_reduction=loss_reduction,
        message_l2_coef=message_l2_coef,
    )

    traces_with_offsets: list[tuple[Any, int, bool]] = []
    if selection_warmup_steps > 0:
        # Phase 1: oracle compressors make the three Actors distinguishable, so
        # selection can lock onto the full compressor before phi starts moving.
        _set_compressor_weights(actor_compressors, "oracle")
        trainer.compression_optimizer = None
        warmup_trace = trainer.run(
            num_steps=selection_warmup_steps,
            seed=seed,
            eval_interval=eval_interval,
        )
        traces_with_offsets.append((warmup_trace, 0, False))
        if warmup_reset_phi:
            # Phase 2a: restore the requested phi init under warm-started w.
            _set_compressor_weights(
                actor_compressors,
                phi_init,
                noise_std=phi_init_noise,
                generator=phi_init_generator,
            )
        # Phase 2b: keep the exact oracle init (warmup_reset_phi=False) and see
        # whether REINFORCE walks a good phi off to infinity.
        trainer.compression_optimizer = compression_optimizer
        traces_with_offsets.extend(
            _run_phi_phase(
                trainer=trainer,
                optimizer=compression_optimizer,
                num_steps=steps - selection_warmup_steps,
                seed=None,
                eval_interval=eval_interval,
                initial_lr=phi_lr,
                schedule=phi_lr_schedule,
                final_fraction=phi_lr_final_fraction,
                step_offset=selection_warmup_steps,
            )
        )
    else:
        trainer.compression_optimizer = compression_optimizer
        traces_with_offsets.extend(
            _run_phi_phase(
                trainer=trainer,
                optimizer=compression_optimizer,
                num_steps=steps,
                seed=seed,
                eval_interval=eval_interval,
                initial_lr=phi_lr,
                schedule=phi_lr_schedule,
                final_fraction=phi_lr_final_fraction,
                step_offset=0,
            )
        )

    curve_rows = _curves_from_traces(
        traces_with_offsets=traces_with_offsets,
        trainer=trainer,
        policies=policies,
        num_states=num_states,
        num_observers=num_observers,
        num_actor_compressors=num_actor_compressors,
        alpha=alpha,
        lr_theta=lr_theta,
        lr_phi=phi_lr,
        policy_head=policy_head,
        policy_init=policy_init,
        selection_exploration=selection_exploration,
        action_exploration=action_exploration,
        seed=seed,
        selection_warmup_steps=selection_warmup_steps,
        warmup_reset_phi=warmup_reset_phi,
        message_l2_coef=message_l2_coef,
        phi_init=phi_init,
        phi_init_noise=phi_init_noise,
        phi_lr_schedule=phi_lr_schedule,
        phi_lr_final_fraction=phi_lr_final_fraction,
    )
    final = curve_rows[-1]
    ceiling = decoder_ceiling(num_states, sharpness)
    run_row = dict(final)
    run_row.update(
        {
            "decoder_ceiling": ceiling,
            "reward_over_ceiling": float(final["expected_reward"]) / ceiling,
            "success_reward": float(final["expected_reward"] >= 0.95),
            "success_ceiling": float(
                float(final["expected_reward"]) >= 0.98 * ceiling
            ),
            "success_internal": float(
                final["expected_reward"] >= 0.95
                and final["min_p_full"] >= 0.90
                and final["min_correct_action_prob"] >= 0.90
                and final["full_message_round_accuracy"] >= 0.95
            ),
        }
    )
    return curve_rows, run_row


def _curves_from_traces(
    *,
    traces_with_offsets: list[tuple[Any, int, bool]],
    trainer: UpdatedImperfectStateTrainer,
    policies: list[torch.nn.Module],
    num_states: int,
    num_observers: int,
    num_actor_compressors: int,
    alpha: float,
    lr_theta: float,
    lr_phi: float,
    policy_head: str,
    policy_init: str,
    selection_exploration: float,
    action_exploration: float,
    seed: int,
    selection_warmup_steps: int,
    warmup_reset_phi: bool,
    message_l2_coef: float,
    phi_init: str,
    phi_init_noise: float,
    phi_lr_schedule: str,
    phi_lr_final_fraction: float,
) -> list[dict[str, Any]]:
    curve_rows: list[dict[str, Any]] = []
    previous_full_messages: list[float] | None = None
    previous_selection: list[float] | None = None
    for trace, step_offset, skip_first_eval in traces_with_offsets:
        for eval_index, step in enumerate(trace.evaluation_steps):
            if skip_first_eval and eval_index == 0:
                # Chunk boundary: this evaluation is the previous chunk's last.
                continue
            messages_by_actor = trace.actor_messages_by_state[eval_index]
            full_actor_index = 2
            full_messages = [
                float(message[0]) for message in messages_by_actor[full_actor_index]
            ]
            selection_probs_by_observer = (
                trace.selection_probabilities_by_observer[eval_index]
            )
            p_full_values = [
                float(probs[full_actor_index]) for probs in selection_probs_by_observer
            ]
            state_correct_by_observer = observer_state_correct_probs(
                policies=trainer.observer_policies,
                messages_by_actor=messages_by_actor,
                selection_probs_by_observer=selection_probs_by_observer,
            )
            full_state_correct_by_observer = observer_full_correct_probs(
                policies=trainer.observer_policies,
                full_messages=full_messages,
            )
            flat_correct = [
                value
                for observer_values in state_correct_by_observer
                for value in observer_values
            ]
            flat_full_correct = [
                value
                for observer_values in full_state_correct_by_observer
                for value in observer_values
            ]
            metrics = message_metrics(full_messages)
            mean_selection = [
                fmean(probs[index] for probs in selection_probs_by_observer)
                for index in range(num_actor_compressors)
            ]
            full_message_movement = (
                0.0
                if previous_full_messages is None
                else l2_distance(full_messages, previous_full_messages)
            )
            selection_movement = (
                0.0
                if previous_selection is None
                else l2_distance(mean_selection, previous_selection)
            )
            previous_full_messages = full_messages
            previous_selection = mean_selection
            curve_rows.append(
                {
                    "state_count": num_states,
                    "observer_count": num_observers,
                    "actor_compressor_count": num_actor_compressors,
                    "alpha": alpha,
                    "lr_theta": lr_theta,
                    "lr_phi": lr_phi,
                    "policy_head": policy_head,
                    "policy_init": policy_init,
                    "selection_exploration": selection_exploration,
                    "action_exploration": action_exploration,
                    "selection_warmup_steps": selection_warmup_steps,
                    "warmup_reset_phi": warmup_reset_phi,
                    "message_l2_coef": message_l2_coef,
                    "phi_init": phi_init,
                    "phi_init_noise": phi_init_noise,
                    "phi_lr_schedule": phi_lr_schedule,
                    "phi_lr_final_fraction": phi_lr_final_fraction,
                    "seed": seed,
                    "step": step + step_offset,
                    "expected_reward": trace.expected_rewards[eval_index],
                    "mean_correct_action_prob": fmean(flat_correct),
                    "min_correct_action_prob": min(flat_correct),
                    "missed_regions_095": sum(prob < 0.95 for prob in flat_correct),
                    "full_mean_correct_action_prob": fmean(flat_full_correct),
                    "full_min_correct_action_prob": min(flat_full_correct),
                    "full_missed_regions_095": sum(
                        prob < 0.95 for prob in flat_full_correct
                    ),
                    "mean_p_full": fmean(p_full_values),
                    "min_p_full": min(p_full_values),
                    "mean_p_partial": (
                        mean_selection[1] if len(mean_selection) > 1 else 0.0
                    ),
                    "mean_p_zero": mean_selection[0],
                    "selection_movement": selection_movement,
                    "full_message_mae": metrics["mae"],
                    "full_message_max_abs_error": metrics["max_abs_error"],
                    "full_message_r2": metrics["r2"],
                    "full_message_round_accuracy": metrics["round_accuracy"],
                    "full_message_min_gap": metrics["min_gap"],
                    "full_message_movement": full_message_movement,
                    "selection_probs_by_observer": "|".join(
                        ";".join(f"{value:.6g}" for value in probs)
                        for probs in selection_probs_by_observer
                    ),
                    "full_messages": ";".join(
                        f"{value:.6g}" for value in full_messages
                    ),
                    "observer_correct_action_probs": "|".join(
                        ";".join(f"{value:.6g}" for value in probs)
                        for probs in state_correct_by_observer
                    ),
                }
            )
    return curve_rows


def observer_state_correct_probs(
    *,
    policies: list[torch.nn.Module],
    messages_by_actor: list[list[list[float]]],
    selection_probs_by_observer: list[list[float]],
) -> list[list[float]]:
    values: list[list[float]] = []
    with torch.no_grad():
        for observer_index, policy in enumerate(policies):
            observer_values: list[float] = []
            selection_probs = selection_probs_by_observer[observer_index]
            for state_index in range(len(messages_by_actor[0])):
                total = 0.0
                for actor_index, actor_messages in enumerate(messages_by_actor):
                    message = torch.tensor(actor_messages[state_index], dtype=torch.float32)
                    distribution = policy.action_distribution(message)
                    total += float(selection_probs[actor_index]) * float(
                        distribution.probs[state_index].item()
                    )
                observer_values.append(total)
            values.append(observer_values)
    return values


def observer_full_correct_probs(
    *,
    policies: list[torch.nn.Module],
    full_messages: list[float],
) -> list[list[float]]:
    values: list[list[float]] = []
    with torch.no_grad():
        for policy in policies:
            observer_values: list[float] = []
            for action, message in enumerate(full_messages):
                distribution = policy.action_distribution(torch.tensor([message]))
                observer_values.append(float(distribution.probs[action].item()))
            values.append(observer_values)
    return values


def message_metrics(messages: list[float]) -> dict[str, float]:
    targets = [float(index) for index in range(len(messages))]
    errors = [value - target for value, target in zip(messages, targets, strict=True)]
    rounded_matches = 0
    for value, target in zip(messages, targets, strict=True):
        rounded = int(round(value))
        rounded = max(0, min(len(messages) - 1, rounded))
        rounded_matches += int(rounded == int(target))
    gaps = [
        abs(messages[index + 1] - messages[index])
        for index in range(len(messages) - 1)
    ]
    return {
        "mae": fmean(abs(error) for error in errors),
        "max_abs_error": max(abs(error) for error in errors),
        "r2": r_squared(messages, targets),
        "round_accuracy": rounded_matches / len(messages),
        "min_gap": min(gaps) if gaps else 0.0,
    }


def r_squared(values: list[float], targets: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    x = torch.tensor(targets, dtype=torch.float32)
    y = torch.tensor(values, dtype=torch.float32)
    design = torch.stack([x, torch.ones_like(x)], dim=1)
    solution = torch.linalg.lstsq(design, y.reshape(-1, 1)).solution.reshape(-1)
    predicted = design @ solution
    total = torch.sum((y - y.mean()) ** 2)
    if float(total.item()) <= 1e-12:
        return 0.0
    residual = torch.sum((y - predicted) ** 2)
    return float((1.0 - residual / total).item())


def l2_distance(left: list[float], right: list[float]) -> float:
    return math.sqrt(
        sum((lval - rval) ** 2 for lval, rval in zip(left, right, strict=True))
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"no rows to write for {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def mean(rows: list[dict[str, Any]], key: str) -> float:
    return fmean(float(row[key]) for row in rows)


def std(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    return pstdev(values) if len(values) > 1 else 0.0


def summarize_runs(run_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[int, int, int, float, float, float], list[dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        grouped[
            (
                int(row["state_count"]),
                int(row["observer_count"]),
                int(row["actor_compressor_count"]),
                float(row["alpha"]),
                float(row["selection_exploration"]),
                float(row["action_exploration"]),
            )
        ].append(row)

    summary_rows: list[dict[str, Any]] = []
    for (state_count, observer_count, actor_count, alpha, eps_w, eps_a), rows in sorted(grouped.items()):
        summary_rows.append(
            {
                "state_count": state_count,
                "observer_count": observer_count,
                "actor_compressor_count": actor_count,
                "alpha": alpha,
                "selection_exploration": eps_w,
                "action_exploration": eps_a,
                "num_seeds": len(rows),
                "mean_final_expected_reward": mean(rows, "expected_reward"),
                "std_final_expected_reward": std(rows, "expected_reward"),
                "mean_min_correct_action_prob": mean(rows, "min_correct_action_prob"),
                "mean_missed_regions_095": mean(rows, "missed_regions_095"),
                "mean_p_full": mean(rows, "mean_p_full"),
                "mean_min_p_full": mean(rows, "min_p_full"),
                "mean_full_message_mae": mean(rows, "full_message_mae"),
                "mean_full_message_r2": mean(rows, "full_message_r2"),
                "decoder_ceiling": mean(rows, "decoder_ceiling"),
                "mean_reward_over_ceiling": mean(rows, "reward_over_ceiling"),
                "reward_success_rate": mean(rows, "success_reward"),
                "ceiling_success_rate": mean(rows, "success_ceiling"),
                "internal_success_rate": mean(rows, "success_internal"),
            }
        )
    best_rows: list[dict[str, Any]] = []
    keys = sorted(
        {
            (
                int(row["state_count"]),
                int(row["observer_count"]),
                int(row["actor_compressor_count"]),
            )
            for row in summary_rows
        }
    )
    for key in keys:
        candidates = [
            row
            for row in summary_rows
            if (
                int(row["state_count"]),
                int(row["observer_count"]),
                int(row["actor_compressor_count"]),
            )
            == key
        ]
        best_rows.append(
            max(
                candidates,
                key=lambda row: (
                    float(row["internal_success_rate"]),
                    float(row["reward_success_rate"]),
                    float(row["mean_final_expected_reward"]),
                    float(row["mean_p_full"]),
                    -float(row["mean_missed_regions_095"]),
                ),
            )
        )
    return summary_rows, best_rows


def plot_curves(
    curve_rows: list[dict[str, Any]],
    best_rows: list[dict[str, Any]],
    figure_dir: Path,
    figure_prefix: str,
) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    best_by_config = {
        (
            int(row["state_count"]),
            int(row["observer_count"]),
            int(row["actor_compressor_count"]),
        ): (
            float(row["alpha"]),
            float(row["selection_exploration"]),
            float(row["action_exploration"]),
        )
        for row in best_rows
    }
    rows = [
        row
        for row in curve_rows
        if (
            float(row["alpha"]),
            float(row["selection_exploration"]),
            float(row["action_exploration"]),
        )
        == best_by_config[
            (
                int(row["state_count"]),
                int(row["observer_count"]),
                int(row["actor_compressor_count"]),
            )
        ]
    ]
    grouped: dict[tuple[int, int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                int(row["state_count"]),
                int(row["observer_count"]),
                int(row["actor_compressor_count"]),
                int(row["step"]),
            )
        ].append(row)

    by_config: dict[tuple[int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for (state_count, observer_count, actor_count, step), group_rows in grouped.items():
        by_config[(state_count, observer_count, actor_count)].append(
            {
                "step": step,
                "expected_reward": mean(group_rows, "expected_reward"),
                "mean_p_full": mean(group_rows, "mean_p_full"),
                "full_message_mae": mean(group_rows, "full_message_mae"),
                "min_correct_action_prob": mean(group_rows, "min_correct_action_prob"),
            }
        )
    for config in by_config:
        by_config[config].sort(key=lambda row: row["step"])

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    panels = [
        ("expected_reward", "expected reward"),
        ("mean_p_full", "mean P(full)"),
        ("full_message_mae", "full-message MAE"),
        ("min_correct_action_prob", "min correct-action prob."),
    ]
    for axis, (metric, label) in zip(axes.reshape(-1), panels, strict=True):
        for (state_count, observer_count, actor_count), state_rows in sorted(by_config.items()):
            alpha, eps_w, eps_a = best_by_config[(state_count, observer_count, actor_count)]
            axis.plot(
                [row["step"] for row in state_rows],
                [row[metric] for row in state_rows],
                label=(
                    f"K={state_count}, O={observer_count}, C={actor_count}, "
                    f"a={alpha:g}, ew={eps_w:g}, ea={eps_a:g}"
                ),
            )
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    axes[1, 0].set_xlabel("training step")
    axes[1, 1].set_xlabel("training step")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Updated Section 5.3: group scalar-sum learning")
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(figure_dir / f"{figure_prefix}_curves.{suffix}", dpi=180)
    plt.close(fig)


def plot_final_messages(
    run_rows: list[dict[str, Any]],
    best_rows: list[dict[str, Any]],
    figure_dir: Path,
    figure_prefix: str,
) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    largest = max(
        best_rows,
        key=lambda row: (
            int(row["state_count"]),
            int(row["observer_count"]),
            int(row["actor_compressor_count"]),
        ),
    )
    selected = [
        row
        for row in run_rows
        if int(row["state_count"]) == int(largest["state_count"])
        and int(row["observer_count"]) == int(largest["observer_count"])
        and int(row["actor_compressor_count"]) == int(largest["actor_compressor_count"])
        and float(row["alpha"]) == float(largest["alpha"])
        and float(row["selection_exploration"]) == float(largest["selection_exploration"])
        and float(row["action_exploration"]) == float(largest["action_exploration"])
    ]
    state_count = int(largest["state_count"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for row in selected:
        messages = [float(value) for value in str(row["full_messages"]).split(";")]
        axes[0].plot(range(state_count), messages, alpha=0.55, marker="o", label=f"seed {row['seed']}")
        probs = [
            [float(value) for value in observer.split(";")]
            for observer in str(row["selection_probs_by_observer"]).split("|")
        ]
        axes[1].plot(
            range(len(probs)),
            [observer_probs[2] for observer_probs in probs],
            marker="o",
            alpha=0.65,
            label=f"seed {row['seed']}",
        )
    axes[0].plot(range(state_count), range(state_count), color="black", linestyle="--", linewidth=1)
    axes[0].set_title(f"K={state_count} full-compressor messages")
    axes[0].set_xlabel("state ID")
    axes[0].set_ylabel("message")
    axes[1].set_title("P(full) by Observer")
    axes[1].set_xlabel("Observer")
    axes[1].set_ylabel("probability")
    axes[1].set_ylim(-0.02, 1.02)
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(figure_dir / f"{figure_prefix}_final_messages_selection.{suffix}", dpi=180)
    plt.close(fig)


def write_report(
    path: Path,
    args: argparse.Namespace,
    summary_rows: list[dict[str, Any]],
    best_rows: list[dict[str, Any]],
    figure_prefix: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Updated Section 5.3 Scalar-Sum Group Results",
        "",
        "This is the updated-paper multi-Observer, multi-Actor-compressor imperfect-state case.",
        "",
        "Setup:",
        "",
        "```text",
        "multiple Observers",
        "multiple Actor-compressors",
        "learn Observer policies theta",
        "learn Observer selection weights w",
        "learn selected Actor-compressor compression phi",
        "one timescale: alpha_phi = alpha_w = alpha_theta = alpha",
        "scalar message m = f_phi(s)",
        "cumulative binary state vector, where sum(s) is the state ID",
        "```",
        "",
        "## Best Setting By Configuration",
        "",
        "| K | Observers | Compressors | alpha | eps_w | eps_a | reward | ceiling | reward/ceiling | min P(full) | msg MAE | success >=0.95 | success >=98% ceiling |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    row_template = (
        "| {state_count} | {observer_count} | {actor_compressor_count} | "
        "{alpha:.4g} | {selection_exploration:.3g} | {action_exploration:.3g} | "
        "{mean_final_expected_reward:.3f} | {decoder_ceiling:.3f} | "
        "{mean_reward_over_ceiling:.3f} | {mean_min_p_full:.3f} | "
        "{mean_full_message_mae:.3f} | {reward_success_rate:.2f} | "
        "{ceiling_success_rate:.2f} |"
    )
    for row in best_rows:
        lines.append(row_template.format(**row))
    lines.extend(
        [
            "",
            "## All Settings",
            "",
            "| K | Observers | Compressors | alpha | eps_w | eps_a | reward | ceiling | reward/ceiling | min P(full) | msg MAE | success >=0.95 | success >=98% ceiling |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary_rows:
        lines.append(row_template.format(**row))
    lines.extend(
        [
            "",
            "## Files",
            "",
            "```text",
            f"outputs: {args.output_dir}",
            "figures:",
            f"  {args.figure_dir / (figure_prefix + '_curves.png')}",
            f"  {args.figure_dir / (figure_prefix + '_final_messages_selection.png')}",
            "```",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    figure_prefix = args.figure_prefix or args.output_dir.name
    with (args.output_dir / "config.json").open("w") as handle:
        json.dump(
            vars(args)
            | {
                "output_dir": str(args.output_dir),
                "figure_dir": str(args.figure_dir),
                "doc_path": str(args.doc_path),
                "figure_prefix": figure_prefix,
            },
            handle,
            indent=2,
        )

    curve_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    configs = [
        (state_count, observer_count, actor_count, alpha, seed)
        for state_count in args.state_counts
        for observer_count in args.observer_counts
        for actor_count in args.actor_compressor_counts
        for alpha in args.learning_rates
        for seed in args.seeds
    ]
    for index, (state_count, observer_count, actor_count, alpha, seed) in enumerate(configs, start=1):
        lr_theta = float(args.lr_theta) if args.lr_theta is not None else float(alpha)
        lr_phi = float(args.lr_phi) if args.lr_phi is not None else float(alpha)
        print(
            f"[{index}/{len(configs)}] K={state_count} O={observer_count} "
            f"C={actor_count} alpha={alpha:g} lr_phi={lr_phi:g} "
            f"lr_theta={lr_theta:g} policy={args.policy_head} "
            f"policy_init={args.policy_init} "
            f"eps_w={args.selection_exploration:g} "
            f"eps_a={args.action_exploration:g} "
            f"warmup={args.selection_warmup_steps} "
            f"reset_phi={args.warmup_reset_phi} "
            f"phi_init={args.phi_init} "
            f"phi_noise={args.phi_init_noise:g} "
            f"phi_lr_sched={args.phi_lr_schedule}"
            f"({args.phi_lr_final_fraction:g}) "
            f"msg_l2={args.message_l2_coef:g} seed={seed}",
            flush=True,
        )
        curves, run = run_one(
            num_states=state_count,
            num_observers=observer_count,
            num_actor_compressors=actor_count,
            alpha=alpha,
            lr_theta=lr_theta,
            policy_head=args.policy_head,
            selection_exploration=args.selection_exploration,
            action_exploration=args.action_exploration,
            seed=seed,
            steps=args.steps,
            eval_interval=args.eval_interval,
            sharpness=args.sharpness,
            policy_init=args.policy_init,
            max_grad_norm=args.max_grad_norm,
            loss_reduction=args.loss_reduction,
            selection_warmup_steps=args.selection_warmup_steps,
            warmup_reset_phi=args.warmup_reset_phi,
            lr_phi=args.lr_phi,
            message_l2_coef=args.message_l2_coef,
            phi_init=args.phi_init,
            phi_init_noise=args.phi_init_noise,
            phi_lr_schedule=args.phi_lr_schedule,
            phi_lr_final_fraction=args.phi_lr_final_fraction,
        )
        curve_rows.extend(curves)
        run_rows.append(run)
        print(
            f"    -> reward={run['expected_reward']:.4f} "
            f"({run['reward_over_ceiling'] * 100:.1f}% of ceiling "
            f"{run['decoder_ceiling']:.4f}) "
            f"min_p_full={run['min_p_full']:.3f} "
            f"mae={run['full_message_mae']:.3f}",
            flush=True,
        )

    summary_rows, best_rows = summarize_runs(run_rows)
    write_csv(args.output_dir / "curves.csv", curve_rows)
    write_csv(args.output_dir / "runs.csv", run_rows)
    write_csv(args.output_dir / "summary.csv", summary_rows)
    write_csv(args.output_dir / "best_by_config.csv", best_rows)
    plot_curves(curve_rows, best_rows, args.figure_dir, figure_prefix)
    plot_final_messages(run_rows, best_rows, args.figure_dir, figure_prefix)
    write_report(args.doc_path, args, summary_rows, best_rows, figure_prefix)

    print("\nBest by configuration:")
    for row in best_rows:
        print(
            f"K={row['state_count']:>2} O={row['observer_count']:>2} "
            f"C={row['actor_compressor_count']:>2} alpha={row['alpha']:<6g} "
            f"reward={row['mean_final_expected_reward']:.3f} "
            f"({row['mean_reward_over_ceiling'] * 100:.1f}% of ceiling) "
            f"p_full={row['mean_p_full']:.3f} "
            f"success>=0.95={row['reward_success_rate']:.2f} "
            f"success>=98%ceil={row['ceiling_success_rate']:.2f}"
        )


if __name__ == "__main__":
    main()
