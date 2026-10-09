"""Run Case 4 multi-observer experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from actor_observer_rl.agents import (
    Actor,
    FollowingRates,
    MultiObserver,
    Observer,
    ObserverMessageModule,
)
from actor_observer_rl.envs import PartialObservableStateEnv
from actor_observer_rl.policies import SoftmaxPolicy
from actor_observer_rl.training import MultiObserverTrainer
from actor_observer_rl.utils.config import Case4Config, load_yaml_config
from actor_observer_rl.utils.seeding import set_global_seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/case4_multi_observer.yaml"),
        help="Path to Case 4 YAML config.",
    )
    args = parser.parse_args()

    config = Case4Config.from_mapping(load_yaml_config(args.config))
    set_global_seed(config.seed)

    env = PartialObservableStateEnv(
        reward_table=config.reward_table,
        state_distribution=config.state_distribution,
        observation_dim=config.observation_dim,
    )
    observers = [
        Observer(
            message_module=ObserverMessageModule(
                input_dim=env.state_dim,
                message_dim=config.message_dim,
            )
        )
        for _ in range(config.num_observers)
    ]
    following_rates = FollowingRates(num_observers=config.num_observers)
    multi_observer = MultiObserver(
        observers=observers,
        following_rates=following_rates,
    )
    policy = SoftmaxPolicy(
        input_dim=config.observation_dim + config.message_dim,
        num_actions=config.num_actions,
    )
    actor = Actor(policy)
    actor_optimizer = torch.optim.SGD(actor.parameters(), lr=config.actor_learning_rate)
    observer_optimizer = torch.optim.SGD(
        list(multi_observer.observer_parameters()),
        lr=config.observer_learning_rate,
    )
    following_optimizer = torch.optim.SGD(
        list(multi_observer.following_parameters()),
        lr=config.following_learning_rate,
    )
    trainer = MultiObserverTrainer(
        env=env,
        actor=actor,
        multi_observer=multi_observer,
        actor_optimizer=actor_optimizer,
        observer_optimizer=observer_optimizer,
        following_optimizer=following_optimizer,
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
    final_weights = trace.final_following_weights or []
    formatted_weights = "|".join(f"{weight:.3f}" for weight in final_weights)
    print(f"final_following_weights={formatted_weights}")
    counts = [0] * config.num_observers
    for selected in trace.selected_observer_indices:
        counts[selected] += 1
    total = max(1, len(trace.selected_observer_indices))
    selection_histogram = "|".join(f"{count / total:.3f}" for count in counts)
    print(f"selection_histogram={selection_histogram}")


if __name__ == "__main__":
    main()
