"""End-to-end tests for the paper Section 4 multi-actor trainer."""

from __future__ import annotations

import math

import pytest
import torch

from actor_observer_rl.agents import Actor, BudgetedObserver
from actor_observer_rl.envs import IIDStateBanditEnv
from actor_observer_rl.policies import SoftmaxPolicy
from actor_observer_rl.training import MultiActorTrainer, make_fixed_softmax_actor


def _make_section_4_1_setup(
    seed: int = 0,
    budget: float = 2.0,
    c: float = 1.0,
    following_lr: float = 0.05,
) -> tuple[IIDStateBanditEnv, list[Actor], BudgetedObserver]:
    reward_table = [
        [1.0, 0.5, 0.0],
        [0.5, 1.0, 0.0],
        [0.0, 0.5, 1.0],
    ]
    state_distribution = [1 / 3, 1 / 3, 1 / 3]
    env = IIDStateBanditEnv(
        reward_table=reward_table,
        state_distribution=state_distribution,
        observation_mode="one_hot",
    )
    actors = [
        make_fixed_softmax_actor(num_states=3, num_actions=3, target_action=target)
        for target in (0, 1, 2)
    ]
    observer = BudgetedObserver(
        num_actors=3,
        budget=budget,
        utility_learning_rate=0.1,
        following_learning_rate=following_lr,
        c=c,
        seed=seed,
    )
    return env, actors, observer


def test_section_4_1_observer_follows_high_utility_actor() -> None:
    env, actors, observer = _make_section_4_1_setup(seed=0)
    trainer = MultiActorTrainer(
        env=env,
        actors=actors,
        observer=observer,
        actor_selection_probs=[1 / 3, 1 / 3, 1 / 3],
        actor_optimizers=None,
        snapshot_interval=200,
    )
    trace = trainer.run(num_steps=6000, seed=0)

    utilities = trace.final_utilities or []
    following = trace.final_following_rates or []
    assert len(utilities) == 3 and len(following) == 3
    expected = [0.5, 2 / 3, 1 / 3]
    for estimate, target in zip(utilities, expected, strict=True):
        assert math.isclose(estimate, target, abs_tol=0.18)
    assert utilities[1] > utilities[0]
    assert utilities[0] > utilities[2]
    assert sum(following) <= observer.budget + 1e-5
    assert all(value >= -1e-7 for value in following)
    assert following[1] > following[0]
    assert following[1] > following[2]


def test_section_4_1_no_actor_updates_when_optimizers_none() -> None:
    env, actors, observer = _make_section_4_1_setup(seed=1)
    initial_logits = [actor.policy.linear.bias.detach().clone() for actor in actors]
    trainer = MultiActorTrainer(
        env=env,
        actors=actors,
        observer=observer,
        actor_selection_probs=[1 / 3, 1 / 3, 1 / 3],
        actor_optimizers=None,
        snapshot_interval=100,
    )
    trainer.run(num_steps=500, seed=1)
    for actor, before in zip(actors, initial_logits, strict=True):
        assert torch.allclose(actor.policy.linear.bias.detach(), before)


def test_section_4_2_actor_policies_learn_when_optimizers_present() -> None:
    reward_table = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    state_distribution = [1 / 3, 1 / 3, 1 / 3]
    optimal_actions = [0, 1, 2]
    env = IIDStateBanditEnv(
        reward_table=reward_table,
        state_distribution=state_distribution,
        observation_mode="one_hot",
    )
    torch.manual_seed(7)
    actors = [
        Actor(policy=SoftmaxPolicy(input_dim=3, num_actions=3)) for _ in range(3)
    ]
    optimizers = [torch.optim.SGD(actor.parameters(), lr=0.3) for actor in actors]
    observer = BudgetedObserver(
        num_actors=3,
        budget=3.0,
        utility_learning_rate=0.2,
        following_learning_rate=0.01,
        c=2.0,
        init_mu=[1.0, 1.0, 1.0],
        seed=7,
    )

    initial_probs = _per_actor_optimal_prob(actors, optimal_actions)

    trainer = MultiActorTrainer(
        env=env,
        actors=actors,
        observer=observer,
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
    assert sum(final_mu) <= observer.budget + 1e-5
    assert all(value >= -1e-7 for value in final_mu)


def test_trainer_rejects_inconsistent_actor_count() -> None:
    env, actors, observer = _make_section_4_1_setup(seed=0)
    smaller_observer = BudgetedObserver(
        num_actors=2,
        budget=1.0,
        utility_learning_rate=0.1,
        following_learning_rate=0.01,
    )
    with pytest.raises(ValueError):
        MultiActorTrainer(
            env=env,
            actors=actors,
            observer=smaller_observer,
            actor_selection_probs=[0.5, 0.5, 0.0],
        )


def test_trainer_rejects_unnormalized_selection_probs() -> None:
    env, actors, observer = _make_section_4_1_setup(seed=0)
    with pytest.raises(ValueError):
        MultiActorTrainer(
            env=env,
            actors=actors,
            observer=observer,
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
