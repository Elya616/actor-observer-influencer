"""Run Case 2 i.i.d. state experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from actor_observer_rl.agents import Actor, Observer
from actor_observer_rl.envs import IIDStateBanditEnv
from actor_observer_rl.policies import SoftmaxPolicy
from actor_observer_rl.training import Trainer
from actor_observer_rl.utils.config import Case2Config, load_yaml_config
from actor_observer_rl.utils.seeding import set_global_seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/case2_iid_states.yaml"),
        help="Path to Case 2 YAML config.",
    )
    args = parser.parse_args()

    config = Case2Config.from_mapping(load_yaml_config(args.config))
    set_global_seed(config.seed)

    env = IIDStateBanditEnv(
        reward_table=config.reward_table,
        state_distribution=config.state_distribution,
        observation_mode="one_hot",
    )
    policy = SoftmaxPolicy(
        input_dim=config.num_states,
        num_actions=config.num_actions,
    )
    actor = Actor(policy)
    observer = Observer()
    optimizer = torch.optim.SGD(actor.parameters(), lr=config.learning_rate)
    trainer = Trainer(
        env=env,
        actor=actor,
        actor_optimizer=optimizer,
        observer=observer,
        optimal_actions=config.optimal_actions,
    )
    trace = trainer.run(num_steps=config.num_steps, seed=config.seed)

    print(f"final_reward={trace.final_reward}")
    print(f"final_optimal_action_prob={trace.final_optimal_action_prob:.4f}")
    print(f"average_reward_last_100={sum(trace.rewards[-100:]) / len(trace.rewards[-100:]):.4f}")


if __name__ == "__main__":
    main()
