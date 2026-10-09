"""Run Case 3 partial-observability experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from actor_observer_rl.agents import Actor, Observer, ObserverMessageModule
from actor_observer_rl.envs import PartialObservableStateEnv
from actor_observer_rl.policies import SoftmaxPolicy
from actor_observer_rl.training import PartialObservableTrainer
from actor_observer_rl.utils.config import Case3Config, load_yaml_config
from actor_observer_rl.utils.seeding import set_global_seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/case3_partial_observable.yaml"),
        help="Path to Case 3 YAML config.",
    )
    args = parser.parse_args()

    config = Case3Config.from_mapping(load_yaml_config(args.config))
    set_global_seed(config.seed)

    env = PartialObservableStateEnv(
        reward_table=config.reward_table,
        state_distribution=config.state_distribution,
        observation_dim=config.observation_dim,
    )
    message_module = ObserverMessageModule(
        input_dim=env.state_dim,
        message_dim=config.message_dim,
    )
    observer = Observer(message_module=message_module)
    policy = SoftmaxPolicy(
        input_dim=config.observation_dim + config.message_dim,
        num_actions=config.num_actions,
    )
    actor = Actor(policy)
    actor_optimizer = torch.optim.SGD(actor.parameters(), lr=config.actor_learning_rate)
    observer_optimizer = torch.optim.SGD(
        observer.parameters(),
        lr=config.observer_learning_rate,
    )
    trainer = PartialObservableTrainer(
        env=env,
        actor=actor,
        observer=observer,
        actor_optimizer=actor_optimizer,
        observer_optimizer=observer_optimizer,
        optimal_actions=config.optimal_actions,
    )
    trace = trainer.run(num_steps=config.num_steps, seed=config.seed, eval_interval=10)

    print(f"final_reward={trace.final_reward}")
    print(f"final_expected_reward={trace.final_expected_reward:.4f}")
    print(f"final_mean_optimal_action_prob={trace.final_mean_optimal_action_prob:.4f}")
    print(
        "average_reward_last_100="
        f"{sum(trace.rewards[-100:]) / len(trace.rewards[-100:]):.4f}"
    )

if __name__ == "__main__":
    main()
