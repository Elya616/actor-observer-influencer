"""Combined imperfect-reward + imperfect-state ladder (toward Section 6).

Climbs the integration ladder on the fused loop one knob at a time and prints a
pass/fail summary against the no-learning PA baseline. Small scale (K=3, 3
Observers, 3 Actor-compressors: zero / partial / full) so the run is fast and the
expected behavior is known.
"""

from __future__ import annotations

import argparse

import torch

from actor_observer_rl.envs.feature_partial_observable import (
    FeaturePartialObservableStateEnv,
)
from actor_observer_rl.policies import SoftmaxPolicy
from actor_observer_rl.training.combined_group import CombinedGroupTrainer
from actor_observer_rl.training.informational_state import (
    SelectionLogits,
    build_scalar_sum_actor_compressors,
    cumulative_binary_state_features,
    diagonal_reward_table,
)
from actor_observer_rl.utils.seeding import set_global_seed


def make_env(num_states: int) -> FeaturePartialObservableStateEnv:
    return FeaturePartialObservableStateEnv(
        state_features=cumulative_binary_state_features(num_states),
        reward_table=diagonal_reward_table(num_states),
        state_distribution=[1.0 / num_states for _ in range(num_states)],
        observation_dim=1,
    )


def build(num_states: int, num_observers: int, learnable_compression: bool):
    env = make_env(num_states)
    compressors = build_scalar_sum_actor_compressors(
        num_states, learnable=learnable_compression
    )
    policies = [SoftmaxPolicy(input_dim=1, num_actions=num_states) for _ in range(num_observers)]
    selection = [SelectionLogits(len(compressors)) for _ in range(num_observers)]
    return env, compressors, policies, selection


def run_rung(
    name: str,
    *,
    num_states: int,
    num_observers: int,
    learn_compression: bool,
    learn_selection: bool,
    learn_policy: bool,
    learn_following: bool,
    force_report: bool,
    num_steps: int,
    seed: int,
    policy_lr: float,
    compression_lr: float,
    selection_lr: float,
    budget: float,
    epsilon: float,
) -> dict:
    set_global_seed(seed)
    # Compression is only "learnable" (zero-init, trainable) when we learn it;
    # otherwise use the fixed ground-truth masks so the message equals the sum.
    env, compressors, policies, selection = build(
        num_states, num_observers, learnable_compression=learn_compression
    )

    # Two-timescale (Section 6): the policy theta adapts fast to the current
    # message, while the compression phi and selection w move on a slower
    # timescale so the policy is not chasing a representation that keeps
    # drifting. Concretely compression_lr, selection_lr <= policy_lr.
    compression_optimizer = (
        torch.optim.SGD(
            [p for c in compressors for p in c.parameters() if p.requires_grad],
            lr=compression_lr,
        )
        if learn_compression
        else None
    )
    selection_optimizer = (
        torch.optim.SGD([p for s in selection for p in s.parameters()], lr=selection_lr)
        if learn_selection
        else None
    )
    policy_optimizers = [
        torch.optim.SGD(p.parameters(), lr=policy_lr) if learn_policy else None
        for p in policies
    ]

    trainer = CombinedGroupTrainer(
        env=env,
        actor_compressors=compressors,
        observer_policies=policies,
        selection_logits_by_observer=selection,
        budget=budget,
        c=2.0,
        compression_optimizer=compression_optimizer,
        selection_optimizer=selection_optimizer,
        policy_optimizers=policy_optimizers,
        learn_following=learn_following,
        force_report=force_report,
        selection_exploration_probability=epsilon,
    )
    trace = trainer.run(num_steps=num_steps, seed=seed, eval_interval=max(1, num_steps // 10))
    # Mean selection probability on the full compressor (index 2) across observers.
    final_sel = trace.selection_probabilities_by_observer[-1]
    p_full = sum(row[2] for row in final_sel) / len(final_sel)
    return {
        "name": name,
        "reward": trace.final_expected_reward,
        "correct": trace.final_mean_correct_action_prob,
        "p_full": p_full,
        "report_rate": trace.report_rate[-1],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", type=int, default=3)
    parser.add_argument("--observers", type=int, default=3)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--lr", type=float, default=0.05,
        help="Base (policy theta) learning rate.",
    )
    parser.add_argument(
        "--compression-lr", type=float, default=None,
        help="Learning rate for compression phi (default: --lr, i.e. single "
        "timescale). Set below --lr for the Section 6 two-timescale (phi slow).",
    )
    parser.add_argument(
        "--selection-lr", type=float, default=None,
        help="Learning rate for selection w (default: --lr).",
    )
    parser.add_argument(
        "--two-timescale", action="store_true",
        help="Shortcut: put phi and w on a slower timescale (--lr / 5) than the "
        "policy, matching the Section 6 two-timescale discipline.",
    )
    parser.add_argument("--budget", type=float, default=2.0)
    parser.add_argument("--epsilon", type=float, default=0.2)
    args = parser.parse_args()

    slow_lr = args.lr / 5.0 if args.two_timescale else args.lr
    compression_lr = args.compression_lr if args.compression_lr is not None else slow_lr
    selection_lr = args.selection_lr if args.selection_lr is not None else slow_lr

    common = dict(
        num_states=args.states,
        num_observers=args.observers,
        num_steps=args.steps,
        seed=args.seed,
        policy_lr=args.lr,
        compression_lr=compression_lr,
        selection_lr=selection_lr,
        budget=args.budget,
        epsilon=args.epsilon,
    )

    rungs = [
        # PA baseline: nothing learned, fixed-full compression, report gated by
        # frozen uniform mu.
        dict(name="PA (all frozen)", learn_compression=False, learn_selection=False,
             learn_policy=False, learn_following=False, force_report=False),
        # Route A rung: policy fixed... here policy is the learnable object we add
        # first on top of fixed compression + fixed selection is not meaningful,
        # so we climb the meaningful combined rungs:
        dict(name="learn selection only", learn_compression=False, learn_selection=True,
             learn_policy=False, learn_following=False, force_report=True),
        dict(name="learn policy only", learn_compression=False, learn_selection=False,
             learn_policy=True, learn_following=False, force_report=True),
        dict(name="learn selection+policy", learn_compression=False, learn_selection=True,
             learn_policy=True, learn_following=False, force_report=True),
        dict(name="learn sel+policy + mu gate", learn_compression=False, learn_selection=True,
             learn_policy=True, learn_following=True, force_report=False),
        dict(name="learn sel+policy+compression", learn_compression=True, learn_selection=True,
             learn_policy=True, learn_following=False, force_report=True),
    ]

    print(
        f"states={args.states} observers={args.observers} steps={args.steps} "
        f"policy_lr={args.lr} compression_lr={compression_lr} "
        f"selection_lr={selection_lr} budget={args.budget} eps={args.epsilon}"
    )
    print(f"{'rung':<32} {'reward':>8} {'correct':>8} {'P(full)':>8} {'report':>7}")
    baseline_correct = None
    for rung in rungs:
        result = run_rung(**rung, **common)
        if baseline_correct is None:
            baseline_correct = result["correct"]
        print(
            f"{result['name']:<32} {result['reward']:>8.3f} {result['correct']:>8.3f} "
            f"{result['p_full']:>8.3f} {result['report_rate']:>7.3f}"
        )


if __name__ == "__main__":
    main()
