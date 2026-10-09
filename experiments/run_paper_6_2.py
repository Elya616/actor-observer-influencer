"""Run paper Section 6.2 group-level joint-learning experiment.

``N`` Observers (each with following rates ``mu_i``) and ``N`` Actors (each with
a learnable softmax policy) learn simultaneously. Per Eq. (29), Actor ``j``'s
effective reward aggregates over Observers,
``r_eff(j) = sum_i theta(mu_{i,j}) R_i(s, a)``, before the REINFORCE update. The
two-timescale discipline (``alpha_U > alpha_mu``) keeps utility estimates ahead
of the following-rate and policy updates.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from actor_observer_rl.agents import Actor, BudgetedObserver
from actor_observer_rl.envs import IIDStateBanditEnv
from actor_observer_rl.policies import EmbeddingPolicy, RBFPolicy, SoftmaxPolicy
from actor_observer_rl.training import GroupMultiActorTrainer
from actor_observer_rl.utils.config import Paper62Config, load_yaml_config
from actor_observer_rl.utils.seeding import set_global_seed


def build_actor_policy(policy_kind: str, env, config):
    if policy_kind == "embed":
        return EmbeddingPolicy(
            num_states=config.num_states, num_actions=config.num_actions
        )
    if policy_kind == "rbf":
        return RBFPolicy(num_states=config.num_states, num_actions=config.num_actions)
    return SoftmaxPolicy(input_dim=env.observation_dim, num_actions=config.num_actions)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/paper_6_2_group.yaml"),
        help="Path to paper Section 6.2 YAML config.",
    )
    parser.add_argument(
        "--observation-mode",
        choices=("scalar_id", "one_hot"),
        default="scalar_id",
        help="Actor state representation (default scalar_id).",
    )
    parser.add_argument(
        "--policy",
        choices=("embed", "rbf", "linear"),
        default="embed",
        help="Actor policy head (default embed, as suggested for scalar IDs).",
    )
    args = parser.parse_args()

    config = Paper62Config.from_mapping(load_yaml_config(args.config))
    set_global_seed(config.seed)

    env = IIDStateBanditEnv(
        reward_table=config.observer_reward_tables[0],
        state_distribution=config.state_distribution,
        observation_mode=args.observation_mode,
    )
    actors: list[Actor] = []
    actor_optimizers: list[torch.optim.Optimizer | None] = []
    for _ in range(config.num_actors):
        policy = build_actor_policy(args.policy, env, config)
        actor = Actor(policy=policy)
        actors.append(actor)
        actor_optimizers.append(
            torch.optim.SGD(actor.parameters(), lr=config.actor_learning_rate)
        )

    observers = [
        BudgetedObserver(
            num_actors=config.num_actors,
            budget=config.budget,
            utility_learning_rate=config.utility_learning_rate,
            following_learning_rate=config.following_learning_rate,
            c=config.c,
            seed=config.seed + i,
        )
        for i in range(config.num_observers)
    ]

    trainer = GroupMultiActorTrainer(
        env=env,
        actors=actors,
        observers=observers,
        observer_reward_tables=config.observer_reward_tables,
        actor_selection_probs=config.actor_selection_probs,
        actor_optimizers=actor_optimizers,
        optimal_actions_per_state=config.optimal_actions,
        snapshot_interval=config.snapshot_interval,
    )
    trace = trainer.run(num_steps=config.num_steps, seed=config.seed)

    final_mu = trace.final_following_rates or []
    final_u = trace.final_utilities or []
    optimal_prob_history = trace.per_actor_optimal_action_probs
    final_optimal_probs = optimal_prob_history[-1] if optimal_prob_history else []

    print(f"num_steps={trace.num_steps}")
    print(
        "actor_selection_frequency="
        + "|".join(f"{value:.3f}" for value in trace.actor_selection_frequencies())
    )
    if final_optimal_probs:
        print(
            "final_optimal_action_prob_per_actor="
            + "|".join(f"{value:.3f}" for value in final_optimal_probs)
        )
    for i in range(config.num_observers):
        print(f"--- Observer {i} ---")
        print("  final_utility_est=" + "|".join(f"{v:.3f}" for v in final_u[i]))
        print("  final_mu=" + "|".join(f"{v:.3f}" for v in final_mu[i]))
        print(f"  sum_mu={sum(final_mu[i]):.3f}/{config.budget}")


if __name__ == "__main__":
    main()
