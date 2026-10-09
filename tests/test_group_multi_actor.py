"""End-to-end tests for the paper Section 6 group-level trainer."""

from __future__ import annotations

import pytest
import torch

from actor_observer_rl.agents import Actor, BudgetedObserver
from actor_observer_rl.envs import IIDStateBanditEnv
from actor_observer_rl.policies import SoftmaxPolicy
from actor_observer_rl.training import GroupMultiActorTrainer, make_fixed_softmax_actor


def _assortative_setup(
    seed: int = 0,
    budget: float = 2.0,
    c: float = 1.0,
    following_lr: float = 0.05,
) -> tuple[IIDStateBanditEnv, list[Actor], list[BudgetedObserver], list[list[list[float]]]]:
    # Single state; Actor j is pinned to action j. Each Observer most values a
    # different Actor (diagonal high reward), so each mu_i should concentrate on
    # the matching Actor i.
    observer_reward_tables = [
        [[0.9, 0.4, 0.1]],
        [[0.1, 0.9, 0.4]],
        [[0.4, 0.1, 0.9]],
    ]
    env = IIDStateBanditEnv(
        reward_table=observer_reward_tables[0],
        state_distribution=[1.0],
        observation_mode="one_hot",
    )
    actors = [
        make_fixed_softmax_actor(num_states=1, num_actions=3, target_action=target)
        for target in (0, 1, 2)
    ]
    observers = [
        BudgetedObserver(
            num_actors=3,
            budget=budget,
            utility_learning_rate=0.1,
            following_learning_rate=following_lr,
            c=c,
            seed=seed + i,
        )
        for i in range(3)
    ]
    return env, actors, observers, observer_reward_tables


def test_section_6_1_each_observer_follows_its_high_utility_actor() -> None:
    env, actors, observers, tables = _assortative_setup(seed=0)
    trainer = GroupMultiActorTrainer(
        env=env,
        actors=actors,
        observers=observers,
        observer_reward_tables=tables,
        actor_selection_probs=[1 / 3, 1 / 3, 1 / 3],
        actor_optimizers=None,
        snapshot_interval=200,
    )
    trace = trainer.run(num_steps=8000, seed=0)

    final_mu = trace.final_following_rates or []
    final_u = trace.final_utilities or []
    assert len(final_mu) == 3 and len(final_u) == 3

    for i in range(3):
        # Observer i's utility estimate should peak at Actor i.
        assert max(range(3), key=lambda j: final_u[i][j]) == i
        # And its following rate should concentrate on Actor i.
        assert max(range(3), key=lambda j: final_mu[i][j]) == i
        assert sum(final_mu[i]) <= observers[i].budget + 1e-5
        assert all(value >= -1e-7 for value in final_mu[i])


def test_section_6_1_no_actor_updates_when_optimizers_none() -> None:
    env, actors, observers, tables = _assortative_setup(seed=1)
    initial_bias = [actor.policy.linear.bias.detach().clone() for actor in actors]
    trainer = GroupMultiActorTrainer(
        env=env,
        actors=actors,
        observers=observers,
        observer_reward_tables=tables,
        actor_selection_probs=[1 / 3, 1 / 3, 1 / 3],
        actor_optimizers=None,
        snapshot_interval=100,
    )
    trainer.run(num_steps=400, seed=1)
    for actor, before in zip(actors, initial_bias, strict=True):
        assert torch.allclose(actor.policy.linear.bias.detach(), before)


def test_section_6_2_actors_learn_with_aggregated_reward() -> None:
    reward_table = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    observer_reward_tables = [reward_table, reward_table, reward_table]
    optimal_actions = [0, 1, 2]
    env = IIDStateBanditEnv(
        reward_table=reward_table,
        state_distribution=[1 / 3, 1 / 3, 1 / 3],
        observation_mode="one_hot",
    )
    torch.manual_seed(7)
    actors = [
        Actor(policy=SoftmaxPolicy(input_dim=3, num_actions=3)) for _ in range(3)
    ]
    optimizers = [torch.optim.SGD(actor.parameters(), lr=0.3) for actor in actors]
    observers = [
        BudgetedObserver(
            num_actors=3,
            budget=3.0,
            utility_learning_rate=0.2,
            following_learning_rate=0.01,
            c=2.0,
            init_mu=[1.0, 1.0, 1.0],
            seed=7 + i,
        )
        for i in range(3)
    ]

    initial_probs = _per_actor_optimal_prob(actors, optimal_actions)
    trainer = GroupMultiActorTrainer(
        env=env,
        actors=actors,
        observers=observers,
        observer_reward_tables=observer_reward_tables,
        actor_selection_probs=[1 / 3, 1 / 3, 1 / 3],
        actor_optimizers=optimizers,
        optimal_actions_per_state=optimal_actions,
        snapshot_interval=200,
    )
    trace = trainer.run(num_steps=8000, seed=7)

    final_probs = trace.per_actor_optimal_action_probs[-1]
    initial_mean = sum(initial_probs) / len(initial_probs)
    final_mean = sum(final_probs) / len(final_probs)
    assert final_mean > initial_mean + 0.30
    assert all(p > 0.6 for p in final_probs)

    final_mu = trace.final_following_rates or []
    for mu_i in final_mu:
        assert sum(mu_i) <= 3.0 + 1e-5
        assert all(value >= -1e-7 for value in mu_i)


def test_trainer_rejects_observer_reward_table_count_mismatch() -> None:
    env, actors, observers, tables = _assortative_setup(seed=0)
    with pytest.raises(ValueError):
        GroupMultiActorTrainer(
            env=env,
            actors=actors,
            observers=observers,
            observer_reward_tables=tables[:2],
            actor_selection_probs=[1 / 3, 1 / 3, 1 / 3],
        )


def test_trainer_rejects_unnormalized_selection_probs() -> None:
    env, actors, observers, tables = _assortative_setup(seed=0)
    with pytest.raises(ValueError):
        GroupMultiActorTrainer(
            env=env,
            actors=actors,
            observers=observers,
            observer_reward_tables=tables,
            actor_selection_probs=[0.4, 0.4, 0.4],
        )


def _per_actor_optimal_prob(
    actors: list[Actor], optimal_actions: list[int]
) -> list[float]:
    num_states = len(optimal_actions)
    probabilities: list[float] = []
    with torch.no_grad():
        for actor in actors:
            average = 0.0
            for state_index in range(num_states):
                one_hot = torch.nn.functional.one_hot(
                    torch.tensor(state_index), num_classes=num_states
                ).to(dtype=torch.float32)
                distribution = actor.policy.action_distribution(one_hot)
                average += float(
                    distribution.probs[optimal_actions[state_index]].item()
                )
            probabilities.append(average / num_states)
    return probabilities
