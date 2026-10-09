"""Run paper Section 6.1 group-level experiment.

``N`` Observers each watch the same ``N`` fixed-policy Actors, but every
Observer has its own reward function ``R_i``. The experiment shows that each
Observer's projected gradient ascent on ``mu_i`` concentrates its budget on the
Actor it values most (high estimated utility), so different Observers
specialize toward different Actors.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from actor_observer_rl.agents import BudgetedObserver
from actor_observer_rl.envs import IIDStateBanditEnv
from actor_observer_rl.training import GroupMultiActorTrainer, make_fixed_softmax_actor
from actor_observer_rl.utils.config import Paper61Config, load_yaml_config
from actor_observer_rl.utils.seeding import set_global_seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/paper_6_1_group.yaml"),
        help="Path to paper Section 6.1 YAML config.",
    )
    parser.add_argument(
        "--observation-mode",
        choices=("scalar_id", "one_hot"),
        default="scalar_id",
        help="Actor state representation (default scalar_id).",
    )
    args = parser.parse_args()

    config = Paper61Config.from_mapping(load_yaml_config(args.config))
    set_global_seed(config.seed)

    env = IIDStateBanditEnv(
        reward_table=config.observer_reward_tables[0],
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
        actor_optimizers=None,
        snapshot_interval=config.snapshot_interval,
    )
    trace = trainer.run(num_steps=config.num_steps, seed=config.seed)

    expected = _expected_utilities(config)
    final_mu = trace.final_following_rates or []
    final_u = trace.final_utilities or []

    print(f"num_steps={trace.num_steps}")
    for i in range(config.num_observers):
        best_actor = max(range(config.num_actors), key=lambda j: expected[i][j])
        argmax_mu = max(range(config.num_actors), key=lambda j: final_mu[i][j])
        print(f"--- Observer {i} ---")
        print("  expected_utility=" + "|".join(f"{v:.3f}" for v in expected[i]))
        print("  final_utility_est=" + "|".join(f"{v:.3f}" for v in final_u[i]))
        print("  final_mu=" + "|".join(f"{v:.3f}" for v in final_mu[i]))
        print(
            f"  highest-utility actor={best_actor}, argmax(mu)={argmax_mu}, "
            f"match={'YES' if best_actor == argmax_mu else 'NO'}"
        )


def _expected_utilities(config: Paper61Config) -> list[list[float]]:
    """Analytic U_{i,j} = E_s[ R_i(s, target_j) ] for each observer/actor."""

    utilities: list[list[float]] = []
    for table in config.observer_reward_tables:
        per_actor: list[float] = []
        for target in config.fixed_actor_targets:
            value = 0.0
            for state_index, probability in enumerate(config.state_distribution):
                value += probability * table[state_index][target]
            per_actor.append(value)
        utilities.append(per_actor)
    return utilities


if __name__ == "__main__":
    main()
