"""Scale multi-actor / multi-observer under scalar + embedding (K fixed).

Peter's guidance: keep the number of states and actions fixed so compression /
policy complexity does not grow, and only scale the number of Actors /
Observers. This script does that for the imperfect-reward joint-learning
settings:

Following Peter's rule of changing only one variable per stress test, the
default sweep is two SINGLE-variable axes:

* actors    -- 1 Observer, vary N Actors (learn mu + policy)
* observers -- fixed Actors (``--fixed-actors``), vary M Observers

The joint ``6_2`` mode (N Observers x N Actors growing together) is kept only
for reference; it changes two variables at once and so is not a clean stress.

Observation is the raw scalar state ID; the policy is an Embedding lookup
(the architecture Peter asked us to try after the Gemini note). Budget B is
set equal to N so the report rate does not artificially collapse as N grows
(Peter noted the rate budget may need to increase with scale).
"""

from __future__ import annotations

import argparse
from statistics import fmean

import torch

from actor_observer_rl.agents import Actor, BudgetedObserver
from actor_observer_rl.envs import IIDStateBanditEnv
from actor_observer_rl.policies import EmbeddingPolicy, SoftmaxPolicy
from actor_observer_rl.training import GroupMultiActorTrainer, MultiActorTrainer
from actor_observer_rl.utils.seeding import set_global_seed


def diagonal_reward(num_states: int = 3) -> list[list[float]]:
    """Diagonal bandit: correct=1.0, wrong=0.0.

    Negative wrong-rewards hurt REINFORCE at large K; a zero floor matched the
    best K=64 force_report ablation (p_opt≈0.94). Soft neighbor rewards (0.1)
    are slightly harder (~0.89–0.91).
    """
    return [
        [1.0 if a == s else 0.0 for a in range(num_states)]
        for s in range(num_states)
    ]


def build_policy(kind: str, num_states: int, num_actions: int, observation_dim: int):
    if kind == "embed":
        return EmbeddingPolicy(
            num_states=num_states, num_actions=num_actions, embedding_dim=16
        )
    if kind == "linear":
        return SoftmaxPolicy(input_dim=observation_dim, num_actions=num_actions)
    raise ValueError(f"unknown policy: {kind}")


def run_one_observer(
    *,
    num_actors: int,
    num_states: int,
    num_steps: int,
    policy_kind: str,
    observation_mode: str,
    seed: int,
    actor_lr: float,
    explore_eps: float = 0.0,
    explore_eps_end: float | None = None,
) -> list[float]:
    set_global_seed(seed)
    reward_table = diagonal_reward(num_states)
    env = IIDStateBanditEnv(
        reward_table=reward_table,
        state_distribution=[1.0 / num_states] * num_states,
        observation_mode=observation_mode,
    )
    actors = []
    optimizers = []
    for _ in range(num_actors):
        policy = build_policy(
            policy_kind, num_states, num_states, env.observation_dim
        )
        actor = Actor(policy=policy)
        actors.append(actor)
        optimizers.append(torch.optim.SGD(actor.parameters(), lr=actor_lr))

    observer = BudgetedObserver(
        num_actors=num_actors,
        budget=float(num_actors),
        utility_learning_rate=0.2,
        following_learning_rate=0.01,
        c=2.0,
        seed=seed,
    )
    trainer = MultiActorTrainer(
        env=env,
        actors=actors,
        observer=observer,
        actor_selection_probs=[1.0 / num_actors] * num_actors,
        actor_optimizers=optimizers,
        optimal_actions_per_state=list(range(num_states)),
        snapshot_interval=max(200, num_steps // 20),
        action_exploration_probability=explore_eps,
        action_exploration_final=explore_eps_end,
    )
    trace = trainer.run(num_steps=num_steps, seed=seed)
    history = trace.per_actor_optimal_action_probs
    return list(history[-1]) if history else []


def run_group(
    *,
    num_actors: int,
    num_observers: int,
    num_states: int,
    num_steps: int,
    policy_kind: str,
    observation_mode: str,
    seed: int,
    actor_lr: float,
    explore_eps: float = 0.0,
    explore_eps_end: float | None = None,
) -> list[float]:
    set_global_seed(seed)
    reward_table = diagonal_reward(num_states)
    env = IIDStateBanditEnv(
        reward_table=reward_table,
        state_distribution=[1.0 / num_states] * num_states,
        observation_mode=observation_mode,
    )
    actors = []
    optimizers = []
    for _ in range(num_actors):
        policy = build_policy(
            policy_kind, num_states, num_states, env.observation_dim
        )
        actor = Actor(policy=policy)
        actors.append(actor)
        optimizers.append(torch.optim.SGD(actor.parameters(), lr=actor_lr))

    observers = [
        BudgetedObserver(
            num_actors=num_actors,
            budget=float(num_actors),
            utility_learning_rate=0.2,
            following_learning_rate=0.01,
            c=2.0,
            seed=seed + i,
        )
        for i in range(num_observers)
    ]
    trainer = GroupMultiActorTrainer(
        env=env,
        actors=actors,
        observers=observers,
        observer_reward_tables=[reward_table for _ in range(num_observers)],
        actor_selection_probs=[1.0 / num_actors] * num_actors,
        actor_optimizers=optimizers,
        optimal_actions_per_state=list(range(num_states)),
        snapshot_interval=max(200, num_steps // 20),
        action_exploration_probability=explore_eps,
        action_exploration_final=explore_eps_end,
    )
    trace = trainer.run(num_steps=num_steps, seed=seed)
    history = trace.per_actor_optimal_action_probs
    return list(history[-1]) if history else []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("actors", "observers", "6_2", "both"),
        default="both",
        help="Single-variable stress axes (Peter's one-variable-at-a-time rule): "
        "'actors' = 1 Observer x N Actors (vary actors only); "
        "'observers' = --fixed-actors Actors x M Observers (vary observers only). "
        "'both' runs those two single-variable axes. '6_2' = N Observers x N "
        "Actors is the joint paper target where TWO axes grow together -- kept "
        "for reference, not a single-variable stress.",
    )
    parser.add_argument(
        "--ns",
        type=int,
        nargs="+",
        default=[3, 8, 16],
        help="Counts to sweep for the swept axis (actors or observers). States "
        "stay fixed at --states.",
    )
    parser.add_argument(
        "--fixed-actors",
        type=int,
        default=8,
        help="Actor count held constant while sweeping observers ('observers' "
        "mode). Keeps only one variable changing.",
    )
    parser.add_argument("--states", type=int, default=3)
    parser.add_argument("--steps", type=int, default=40000)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument(
        "--policy",
        choices=("embed", "linear"),
        default="embed",
        help="Policy head under scalar_id (default: embedding).",
    )
    parser.add_argument(
        "--observation-mode",
        choices=("scalar_id", "one_hot"),
        default="scalar_id",
    )
    parser.add_argument(
        "--actor-lr",
        type=float,
        default=None,
        help="Override Actor SGD lr for BOTH modes. When unset, each mode uses "
        "its stress-tested default (--actor-lr-4-2 / --actor-lr-6-2).",
    )
    parser.add_argument(
        "--actor-lr-4-2",
        type=float,
        default=0.05,
        help="§4.2 (1 Obs x N Act) Actor SGD lr; embed is stable at 0.05.",
    )
    parser.add_argument(
        "--actor-lr-6-2",
        type=float,
        default=0.03,
        help="§6.2 (N Obs x N Act) Actor SGD lr; N×N scale stress needs ~0.03 "
        "(0.1 left some actors stuck).",
    )
    parser.add_argument(
        "--explore-eps",
        type=float,
        default=0.0,
        help="Uniform action-exploration weight ε: π_mix=(1-ε)π+ε/K. Default 0 "
        "keeps the original (no-exploration) behavior.",
    )
    parser.add_argument(
        "--explore-eps-end",
        type=float,
        default=None,
        help="Anneal ε linearly to this value by the last step (None=constant). "
        "Use 0.0 to explore early then sharpen.",
    )
    args = parser.parse_args()

    lr_4_2 = args.actor_lr if args.actor_lr is not None else args.actor_lr_4_2
    lr_6_2 = args.actor_lr if args.actor_lr is not None else args.actor_lr_6_2

    print(
        f"K={args.states} steps={args.steps} seeds={args.seeds} "
        f"policy={args.policy} obs={args.observation_mode} "
        f"lr_4_2={lr_4_2} lr_6_2={lr_6_2} eps={args.explore_eps}->{args.explore_eps_end}"
    )
    print(f"{'setting':<28} {'N':>4} {'mean_p_opt':>10} {'min_p_opt':>9} detail")

    # Axis 1 (single variable): vary Actors, hold Observers = 1.
    if args.mode in ("actors", "both"):
        for n in args.ns:
            seed_means = []
            last_detail = []
            for seed in range(args.seeds):
                probs = run_one_observer(
                    num_actors=n,
                    num_states=args.states,
                    num_steps=args.steps,
                    policy_kind=args.policy,
                    observation_mode=args.observation_mode,
                    seed=seed,
                    actor_lr=lr_4_2,
                    explore_eps=args.explore_eps,
                    explore_eps_end=args.explore_eps_end,
                )
                last_detail = probs
                seed_means.append(fmean(probs) if probs else 0.0)
            mean = fmean(seed_means)
            detail = "|".join(f"{p:.3f}" for p in last_detail)
            print(
                f"{'actors: 1Obs x NAct':<28} {n:>4} {mean:>10.3f} "
                f"{min(last_detail) if last_detail else 0:>9.3f} {detail}"
            )

    # Axis 2 (single variable): vary Observers, hold Actors = --fixed-actors.
    if args.mode in ("observers", "both"):
        for m in args.ns:
            seed_means = []
            last_detail = []
            for seed in range(args.seeds):
                probs = run_group(
                    num_actors=args.fixed_actors,
                    num_observers=m,
                    num_states=args.states,
                    num_steps=args.steps,
                    policy_kind=args.policy,
                    observation_mode=args.observation_mode,
                    seed=seed,
                    actor_lr=lr_6_2,
                    explore_eps=args.explore_eps,
                    explore_eps_end=args.explore_eps_end,
                )
                last_detail = probs
                seed_means.append(fmean(probs) if probs else 0.0)
            mean = fmean(seed_means)
            detail = "|".join(f"{p:.3f}" for p in last_detail)
            label = f"observers: {args.fixed_actors}Act x MObs"
            print(
                f"{label:<28} {m:>4} {mean:>10.3f} "
                f"{min(last_detail) if last_detail else 0:>9.3f} {detail}"
            )

    # Reference only (NOT single-variable): joint N Observers x N Actors (paper 6.2).
    if args.mode == "6_2":
        for n in args.ns:
            seed_means = []
            last_detail = []
            for seed in range(args.seeds):
                probs = run_group(
                    num_actors=n,
                    num_observers=n,
                    num_states=args.states,
                    num_steps=args.steps,
                    policy_kind=args.policy,
                    observation_mode=args.observation_mode,
                    seed=seed,
                    actor_lr=lr_6_2,
                    explore_eps=args.explore_eps,
                    explore_eps_end=args.explore_eps_end,
                )
                last_detail = probs
                seed_means.append(fmean(probs) if probs else 0.0)
            mean = fmean(seed_means)
            detail = "|".join(f"{p:.3f}" for p in last_detail)
            print(
                f"{'6.2 NObs x NAct (joint)':<28} {n:>4} {mean:>10.3f} "
                f"{min(last_detail) if last_detail else 0:>9.3f} {detail}"
            )


if __name__ == "__main__":
    main()
