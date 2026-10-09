"""Run paper Section 4.1 multi-actor experiment.

Single ``BudgetedObserver`` watches ``N`` fixed-policy actors. The
experiment demonstrates that the observer's projected gradient ascent on
``mu`` reallocates its budget toward high-utility actors over time.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from actor_observer_rl.agents import BudgetedObserver
from actor_observer_rl.envs import IIDStateBanditEnv
from actor_observer_rl.training import MultiActorTrainer, make_fixed_softmax_actor
from actor_observer_rl.utils.config import Paper41Config, load_yaml_config
from actor_observer_rl.utils.seeding import set_global_seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/paper_4_1_multi_actor.yaml"),
        help="Path to paper Section 4.1 YAML config.",
    )
    parser.add_argument(
        "--observation-mode",
        choices=("scalar_id", "one_hot"),
        default="scalar_id",
        help="Actor state representation (default scalar_id).",
    )
    args = parser.parse_args()

    config = Paper41Config.from_mapping(load_yaml_config(args.config))
    set_global_seed(config.seed)

    env = IIDStateBanditEnv(
        reward_table=config.reward_table,
        state_distribution=config.state_distribution,
        observation_mode=args.observation_mode,
    )
    actors = [
        make_fixed_softmax_actor(
            num_states=config.num_states,
            num_actions=config.num_actions,
            target_action=target,
            sharpness=config.actor_sharpness,
            input_dim=env.observation_dim,
        )
        for target in config.fixed_actor_targets
    ]
    observer = BudgetedObserver(
        num_actors=config.num_actors,
        budget=config.budget,
        utility_learning_rate=config.utility_learning_rate,
        following_learning_rate=config.following_learning_rate,
        c=config.c,
        seed=config.seed,
    )
    trainer = MultiActorTrainer(
        env=env,
        actors=actors,
        observer=observer,
        actor_selection_probs=config.actor_selection_probs,
        actor_optimizers=None,
        snapshot_interval=config.snapshot_interval,
    )
    trace = trainer.run(num_steps=config.num_steps, seed=config.seed)

    expected_per_actor = _expected_rewards_per_actor(
        reward_table=config.reward_table,
        state_distribution=config.state_distribution,
        targets=config.fixed_actor_targets,
    )

    final_mu = trace.final_following_rates or []
    final_u = trace.final_utilities or []
    selection_freq = trace.actor_selection_frequencies()

    print(f"num_steps={trace.num_steps}")
    print(
        "expected_per_actor=" + "|".join(f"{value:.3f}" for value in expected_per_actor)
    )
    print(
        "selection_frequency=" + "|".join(f"{value:.3f}" for value in selection_freq)
    )
    print(
        "final_utility_estimates=" + "|".join(f"{value:.3f}" for value in final_u)
    )
    print(
        "final_following_rates=" + "|".join(f"{value:.3f}" for value in final_mu)
    )
    print(f"sum_following_rates={sum(final_mu):.3f}/{config.budget}")
    best_actor = max(range(config.num_actors), key=lambda j: expected_per_actor[j])
    print(
        f"best_actor_index={best_actor} (expected={expected_per_actor[best_actor]:.3f}) "
        f"mu_best={final_mu[best_actor]:.3f}"
    )


def _expected_rewards_per_actor(
    *,
    reward_table: list[list[float]],
    state_distribution: list[float],
    targets: list[int],
) -> list[float]:
    expected: list[float] = []
    for action in targets:
        value = 0.0
        for state_index, probability in enumerate(state_distribution):
            value += probability * reward_table[state_index][action]
        expected.append(value)
    return expected


if __name__ == "__main__":
    main()
