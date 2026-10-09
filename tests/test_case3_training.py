"""Training tests for Case 3 partial observability."""

from __future__ import annotations

import torch

from actor_observer_rl.agents import Actor, Observer, ObserverMessageModule
from actor_observer_rl.envs import PartialObservableStateEnv
from actor_observer_rl.policies import SoftmaxPolicy
from actor_observer_rl.training import PartialObservableTrainer
from actor_observer_rl.utils.seeding import set_global_seed


def _make_case3_trainer(seed: int = 0) -> PartialObservableTrainer:
    set_global_seed(seed)
    env = PartialObservableStateEnv(
        reward_table=[[1.0, 0.0], [0.0, 1.0]],
        state_distribution=[0.5, 0.5],
        observation_dim=1,
    )
    observer = Observer(
        message_module=ObserverMessageModule(input_dim=env.state_dim, message_dim=2)
    )
    actor = Actor(SoftmaxPolicy(input_dim=3, num_actions=2))
    return PartialObservableTrainer(
        env=env,
        actor=actor,
        observer=observer,
        actor_optimizer=torch.optim.SGD(actor.parameters(), lr=0.1),
        observer_optimizer=torch.optim.SGD(observer.parameters(), lr=0.1),
        optimal_actions=[0, 1],
    )


def test_case3_training_improves_expected_reward_with_zero_actor_observation() -> None:
    trainer = _make_case3_trainer(seed=0)
    initial_expected_reward, initial_mean_prob = trainer.evaluate_policy()

    trace = trainer.run(num_steps=1200, seed=11, eval_interval=100)

    assert trace.final_expected_reward is not None
    assert trace.final_mean_optimal_action_prob is not None
    assert initial_mean_prob is not None
    assert trace.final_expected_reward > initial_expected_reward
    assert trace.final_expected_reward > 0.8
    assert trace.final_mean_optimal_action_prob > initial_mean_prob
    assert trace.final_mean_optimal_action_prob > 0.8
    assert all(torch.isfinite(torch.tensor(trace.losses)))


def test_case3_training_is_reproducible_with_fixed_seed() -> None:
    def run_once() -> tuple[list[int], list[int], list[float]]:
        trainer = _make_case3_trainer(seed=3)
        trace = trainer.run(num_steps=40, seed=17, eval_interval=10)
        return trace.state_indices, trace.actions, trace.rewards

    states_a, actions_a, rewards_a = run_once()
    states_b, actions_b, rewards_b = run_once()

    assert states_a == states_b
    assert actions_a == actions_b
    assert rewards_a == rewards_b
