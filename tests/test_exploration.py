"""Uniform action-exploration (epsilon-mix) for the imperfect-reward trainers.

Covers the shared helper (``mixed_action_sample`` / ``linear_anneal``) and its
wiring into ``MultiActorTrainer`` (Section 4.2) and ``GroupMultiActorTrainer``
(Section 6.2): default eps=0 stays byte-for-byte on the original REINFORCE
path, while eps>0 samples from ``pi_mix = (1-eps) pi_theta + eps / K`` and
updates with ``log pi_mix``.
"""

from __future__ import annotations

import torch

from actor_observer_rl.agents import Actor, BudgetedObserver
from actor_observer_rl.envs import IIDStateBanditEnv
from actor_observer_rl.policies import SoftmaxPolicy
from actor_observer_rl.training import GroupMultiActorTrainer, MultiActorTrainer
from actor_observer_rl.training.updates import linear_anneal, mixed_action_sample


def _one_hot(state_index: int, num_states: int = 3) -> torch.Tensor:
    return torch.nn.functional.one_hot(
        torch.tensor(state_index), num_classes=num_states
    ).to(dtype=torch.float32)


def test_linear_anneal_endpoints() -> None:
    assert linear_anneal(0.1, None, 5, 100) == 0.1
    assert linear_anneal(0.2, 0.0, 0, 100) == 0.2
    assert linear_anneal(0.2, 0.0, 99, 100) == 0.0
    assert abs(linear_anneal(1.0, 0.0, 50, 101) - 0.5) < 1e-9
    assert linear_anneal(0.3, 0.1, 0, 1) == 0.1  # degenerate horizon -> end


def test_mixed_sample_eps_zero_matches_policy_act() -> None:
    policy = SoftmaxPolicy(input_dim=3, num_actions=3)
    inp = _one_hot(0)

    torch.manual_seed(0)
    action, log_prob, probs = mixed_action_sample(policy, inp, 0.0)
    torch.manual_seed(0)
    reference = policy.act(inp, sample=True)

    assert int(action.item()) == int(reference.action.item())
    assert torch.allclose(log_prob, reference.log_prob)
    assert torch.allclose(probs, reference.probs)


def test_mixed_sample_uses_mixed_distribution_and_logprob() -> None:
    policy = SoftmaxPolicy(input_dim=3, num_actions=3)
    inp = _one_hot(0)
    eps = 0.3

    dist = policy.action_distribution(inp)
    expected_mixed = (1.0 - eps) * dist.probs + eps / 3.0

    torch.manual_seed(1)
    action, log_prob, probs = mixed_action_sample(policy, inp, eps, log_prob_source="mixed")

    assert torch.allclose(probs, expected_mixed)
    assert torch.allclose(log_prob, torch.log(expected_mixed[action].clamp_min(1e-12)))
    # Uniform floor: no action falls below eps / K.
    assert bool((probs >= eps / 3.0 - 1e-6).all())


def test_mixed_sample_learned_logprob_source() -> None:
    policy = SoftmaxPolicy(input_dim=3, num_actions=3)
    inp = _one_hot(0)
    dist = policy.action_distribution(inp)

    torch.manual_seed(2)
    action, log_prob, _ = mixed_action_sample(policy, inp, 0.3, log_prob_source="learned")
    assert torch.allclose(log_prob, dist.log_prob(action))


def _make_learnable_setup(seed: int, exploration: float, exploration_final=None):
    reward_table = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    env = IIDStateBanditEnv(
        reward_table=reward_table,
        state_distribution=[1 / 3, 1 / 3, 1 / 3],
        observation_mode="one_hot",
    )
    torch.manual_seed(seed)
    actors = [Actor(policy=SoftmaxPolicy(input_dim=3, num_actions=3)) for _ in range(3)]
    optimizers = [torch.optim.SGD(actor.parameters(), lr=0.3) for actor in actors]
    observer = BudgetedObserver(
        num_actors=3,
        budget=3.0,
        utility_learning_rate=0.2,
        following_learning_rate=0.01,
        c=2.0,
        init_mu=[1.0, 1.0, 1.0],
        seed=seed,
    )
    trainer = MultiActorTrainer(
        env=env,
        actors=actors,
        observer=observer,
        actor_selection_probs=[1 / 3, 1 / 3, 1 / 3],
        actor_optimizers=optimizers,
        optimal_actions_per_state=[0, 1, 2],
        snapshot_interval=200,
        action_exploration_probability=exploration,
        action_exploration_final=exploration_final,
    )
    return env, actors, observer, trainer


def test_multi_actor_default_is_off_and_deterministic() -> None:
    _, _, _, trainer_a = _make_learnable_setup(seed=7, exploration=0.0)
    _, _, _, trainer_b = _make_learnable_setup(seed=7, exploration=0.0)
    trace_a = trainer_a.run(num_steps=400, seed=7)
    trace_b = trainer_b.run(num_steps=400, seed=7)

    assert trace_a.metadata["action_exploration_probability"] == 0.0
    assert trace_a.metadata["action_log_prob_source"] == "mixed"
    assert trace_a.actions == trace_b.actions
    assert trace_a.rewards == trace_b.rewards


def test_multi_actor_full_exploration_covers_all_actions() -> None:
    # eps=1.0 makes pi_mix uniform, so every action must appear regardless of pi.
    _, _, _, trainer = _make_learnable_setup(seed=3, exploration=1.0)
    trace = trainer.run(num_steps=600, seed=3)
    assert set(trace.actions) == {0, 1, 2}
    assert trace.metadata["action_exploration_probability"] == 1.0


def test_multi_actor_annealed_exploration_still_learns() -> None:
    # Explore early, sharpen late: should still improve optimal-action prob.
    _, _, _, trainer = _make_learnable_setup(
        seed=5, exploration=0.3, exploration_final=0.0
    )
    trace = trainer.run(num_steps=8000, seed=5)
    final_probs = trace.per_actor_optimal_action_probs[-1]
    assert sum(final_probs) / len(final_probs) > 0.6
    assert trace.metadata["action_exploration_final"] == 0.0


def test_group_multi_actor_exploration_records_metadata_and_covers_actions() -> None:
    reward_table = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    env = IIDStateBanditEnv(
        reward_table=reward_table,
        state_distribution=[1 / 3, 1 / 3, 1 / 3],
        observation_mode="one_hot",
    )
    torch.manual_seed(11)
    actors = [Actor(policy=SoftmaxPolicy(input_dim=3, num_actions=3)) for _ in range(3)]
    optimizers = [torch.optim.SGD(actor.parameters(), lr=0.3) for actor in actors]
    observers = [
        BudgetedObserver(
            num_actors=3,
            budget=3.0,
            utility_learning_rate=0.2,
            following_learning_rate=0.01,
            c=2.0,
            seed=11 + i,
        )
        for i in range(2)
    ]
    trainer = GroupMultiActorTrainer(
        env=env,
        actors=actors,
        observers=observers,
        observer_reward_tables=[reward_table, reward_table],
        actor_selection_probs=[1 / 3, 1 / 3, 1 / 3],
        actor_optimizers=optimizers,
        optimal_actions_per_state=[0, 1, 2],
        snapshot_interval=200,
        action_exploration_probability=1.0,
    )
    trace = trainer.run(num_steps=600, seed=11)
    assert set(trace.actions) == {0, 1, 2}
    assert trace.metadata["action_exploration_probability"] == 1.0
    assert trace.metadata["action_log_prob_source"] == "mixed"
