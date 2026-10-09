"""Run Case 1 single-state experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from actor_observer_rl.agents import Actor, Observer
from actor_observer_rl.envs import SingleStateBanditEnv
from actor_observer_rl.policies import SoftmaxPolicy
from actor_observer_rl.training import Trainer
from actor_observer_rl.utils.config import Case1Config, load_yaml_config
from actor_observer_rl.utils.seeding import set_global_seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/case1_single_state.yaml"),
        help="Path to Case 1 YAML config.",
    )
    args = parser.parse_args()

    config = Case1Config.from_mapping(load_yaml_config(args.config))
    set_global_seed(config.seed)

    env = SingleStateBanditEnv(reward_table=config.reward_table)
    if env.state_dim != config.state_dim:
        raise ValueError("environment state_dim does not match config state_dim")

    policy = SoftmaxPolicy(
        input_dim=config.state_dim,
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
        optimal_action=config.optimal_action,
    )
    trace = trainer.run(num_steps=config.num_steps, seed=config.seed)

    print(f"final_reward={trace.final_reward}")
    print(f"final_optimal_action_prob={trace.final_optimal_action_prob:.4f}")


if __name__ == "__main__":
    main()
