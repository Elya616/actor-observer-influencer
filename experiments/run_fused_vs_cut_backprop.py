"""Fused single-network vs split (cut) backprop control (meeting 2026-07-22).

The professor's biggest bet is that the failure of joint phi+theta learning
comes from the explicit gradient hand-off between the compression module and
the policy module. The control he asked for: build the compression layer and
the policy head as ONE network, train it with ordinary backprop, and compare.

Arms:

* ``fused_ce``          fused net, supervised cross-entropy state -> optimal
                        action. Verifies the architecture + optimizer can even
                        represent/learn the mapping before RL enters.
* ``fused_reinforce``   fused net, REINFORCE, one optimizer, one backward.
* ``cut_reinforce``     the repo's joint path: separate phi and theta modules
                        with separate optimizers (same lrs, same epsilon).

If ``fused_reinforce`` succeeds where ``cut_reinforce`` fails, the split is
implicated. If both fail while ``fused_ce`` succeeds, the REINFORCE signal is
implicated, not the split.

Example:

    python experiments/run_fused_vs_cut_backprop.py \
        --state-counts 16 32 --steps 50000 --seeds 0 1 \
        --out outputs/diag_fused_vs_cut.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean
from typing import Any

import torch

from actor_observer_rl.envs import FeaturePartialObservableStateEnv
from actor_observer_rl.policies.rbf_policy import RBFPolicy
from actor_observer_rl.training import (
    UpdatedImperfectStateTrainer,
    build_scalar_sum_actor_compressors,
    cumulative_binary_state_features,
    diagonal_reward_table,
)
from actor_observer_rl.utils.seeding import set_global_seed

ALL_ARMS = ("fused_ce", "fused_reinforce", "cut_reinforce")


class FusedCompressionPolicyNet(torch.nn.Module):
    """Compression layer + RBF policy head as one network.

    Mirrors the split architecture exactly: a linear map from the cumulative
    binary state vector to a scalar message, then Gaussian RBF features around
    the integer state IDs, then a linear softmax head. The only difference
    from the split path is that this is a single module trained by a single
    optimizer with one ordinary ``backward()`` call.
    """

    def __init__(self, num_states: int, num_actions: int, sigma: float = 0.5) -> None:
        super().__init__()
        self.compression = torch.nn.Linear(num_states, 1)
        with torch.no_grad():
            self.compression.weight.zero_()
            self.compression.bias.zero_()
        self.register_buffer(
            "centers", torch.arange(num_states, dtype=torch.float32)
        )
        self.sigma = float(sigma)
        self.head = torch.nn.Linear(num_states, num_actions)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        message = self.compression(state.float())  # (..., 1)
        features = torch.exp(
            -((message - self.centers) ** 2) / (2.0 * self.sigma**2)
        )
        return self.head(features)

    def message(self, state: torch.Tensor) -> torch.Tensor:
        return self.compression(state.float())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arms",
        type=str,
        default=",".join(ALL_ARMS),
        help="Comma-separated subset of: " + ", ".join(ALL_ARMS),
    )
    parser.add_argument("--state-counts", nargs="+", type=int, default=[16, 32])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--eval-interval", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--rbf-sigma", type=float, default=0.5)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/diag_fused_vs_cut.json"),
    )
    return parser.parse_args()


def make_env(num_states: int) -> FeaturePartialObservableStateEnv:
    return FeaturePartialObservableStateEnv(
        state_features=cumulative_binary_state_features(num_states),
        reward_table=diagonal_reward_table(num_states),
        state_distribution=[1.0 / num_states for _ in range(num_states)],
        observation_dim=1,
    )


def evaluate_fused(
    net: FusedCompressionPolicyNet,
    states: torch.Tensor,
    reward_table: torch.Tensor,
) -> tuple[float, float]:
    """Return (expected reward, mean correct-action probability)."""
    with torch.no_grad():
        probs = torch.softmax(net(states), dim=-1)
        expected = float((probs * reward_table).sum(dim=-1).mean().item())
        correct = float(probs.diagonal().mean().item())
    return expected, correct


def run_fused(
    *,
    objective: str,
    num_states: int,
    seed: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    set_global_seed(seed)
    net = FusedCompressionPolicyNet(num_states, num_states, sigma=args.rbf_sigma)
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr)
    states = torch.tensor(
        cumulative_binary_state_features(num_states), dtype=torch.float32
    )
    reward_table = torch.tensor(
        diagonal_reward_table(num_states), dtype=torch.float32
    )
    generator = torch.Generator().manual_seed(seed)

    reward_curve: list[float] = []
    correct_curve: list[float] = []
    for step in range(args.steps):
        state_index = int(
            torch.randint(num_states, (1,), generator=generator).item()
        )
        state = states[state_index]
        logits = net(state)
        if objective == "ce":
            loss = torch.nn.functional.cross_entropy(
                logits.unsqueeze(0),
                torch.tensor([state_index]),
            )
        else:
            probs = torch.softmax(logits, dim=-1)
            uniform = torch.full_like(probs, 1.0 / num_states)
            mixed = (1.0 - args.epsilon) * probs + args.epsilon * uniform
            action = int(
                torch.multinomial(mixed, 1, generator=generator).item()
            )
            reward = float(reward_table[state_index, action].item())
            log_prob = torch.log(probs[action].clamp_min(1e-12))
            loss = -reward * log_prob
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if (step + 1) % args.eval_interval == 0 or step + 1 == args.steps:
            expected, correct = evaluate_fused(net, states, reward_table)
            reward_curve.append(expected)
            correct_curve.append(correct)

    return {
        "final_expected_reward": reward_curve[-1],
        "final_mean_correct_action_prob": correct_curve[-1],
        "expected_reward_curve": reward_curve,
        "correct_action_prob_curve": correct_curve,
    }


def run_cut_reinforce(
    *,
    num_states: int,
    seed: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    set_global_seed(seed)
    env = make_env(num_states)
    compressor = build_scalar_sum_actor_compressors(num_states, learnable=True)[2]
    policy = RBFPolicy(num_states, num_states, sigma=args.rbf_sigma)
    trainer = UpdatedImperfectStateTrainer(
        env=env,
        actor_compressors=[compressor],
        observer_policies=[policy],
        compression_optimizer=torch.optim.SGD(compressor.parameters(), lr=args.lr),
        policy_optimizers=[torch.optim.SGD(policy.parameters(), lr=args.lr)],
        action_exploration_probability=args.epsilon,
    )
    trace = trainer.run(
        num_steps=args.steps, seed=seed, eval_interval=args.eval_interval
    )
    return {
        "final_expected_reward": trace.expected_rewards[-1],
        "final_mean_correct_action_prob": trace.mean_correct_action_probs[-1],
        "expected_reward_curve": trace.expected_rewards,
        "correct_action_prob_curve": trace.mean_correct_action_probs,
    }


def run_arm(
    arm: str, *, num_states: int, seed: int, args: argparse.Namespace
) -> dict[str, Any]:
    if arm == "fused_ce":
        result = run_fused(
            objective="ce", num_states=num_states, seed=seed, args=args
        )
    elif arm == "fused_reinforce":
        result = run_fused(
            objective="reinforce", num_states=num_states, seed=seed, args=args
        )
    elif arm == "cut_reinforce":
        result = run_cut_reinforce(num_states=num_states, seed=seed, args=args)
    else:
        raise ValueError(f"unknown arm: {arm}")
    return {
        "arm": arm,
        "num_states": num_states,
        "seed": seed,
        "steps": args.steps,
        "lr": args.lr,
        "epsilon": args.epsilon,
        **result,
    }


def main() -> None:
    args = parse_args()
    arms = [arm.strip() for arm in args.arms.split(",") if arm.strip()]
    for arm in arms:
        if arm not in ALL_ARMS:
            raise ValueError(f"unknown arm: {arm}")

    records: list[dict[str, Any]] = []
    for num_states in args.state_counts:
        for seed in args.seeds:
            for arm in arms:
                print(f"[K={num_states} seed={seed}] arm={arm} ...", flush=True)
                record = run_arm(
                    arm, num_states=num_states, seed=seed, args=args
                )
                records.append(record)
                print(
                    f"    reward={record['final_expected_reward']:.3f} "
                    f"p_opt={record['final_mean_correct_action_prob']:.3f}",
                    flush=True,
                )

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

    print("\nSummary (mean over seeds):")
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault((record["arm"], record["num_states"]), []).append(record)
    print(f"{'arm':<16} {'K':>4} {'reward':>8} {'p_opt':>8}")
    for (arm, num_states), rows in sorted(grouped.items()):
        print(
            f"{arm:<16} {num_states:>4} "
            f"{fmean(row['final_expected_reward'] for row in rows):>8.3f} "
            f"{fmean(row['final_mean_correct_action_prob'] for row in rows):>8.3f}"
        )
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
