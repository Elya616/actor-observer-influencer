"""Checks for the updated-paper imperfect-state architecture."""

from __future__ import annotations

import torch

from actor_observer_rl.envs import FeaturePartialObservableStateEnv
from actor_observer_rl.policies import SoftmaxPolicy
from actor_observer_rl.training import (
    SelectionLogits,
    UpdatedImperfectStateTrainer,
    build_scalar_sum_actor_compressors,
    cumulative_binary_state_features,
    diagonal_reward_table,
    fixed_sum_observer_policy,
)


def _env(num_states: int = 3) -> FeaturePartialObservableStateEnv:
    return FeaturePartialObservableStateEnv(
        state_features=cumulative_binary_state_features(num_states),
        reward_table=diagonal_reward_table(num_states),
        state_distribution=[1.0 / num_states for _ in range(num_states)],
        observation_dim=1,
    )


def _clone_parameters(module: torch.nn.Module) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in module.parameters()]


def _parameters_moved(
    before: list[torch.Tensor],
    module: torch.nn.Module,
) -> bool:
    return any(
        not torch.allclose(old, new.detach())
        for old, new in zip(before, module.parameters(), strict=True)
    )


def test_section_51_dyadic_learns_only_actor_compressor_phi() -> None:
    env = _env()
    actor_compressors = build_scalar_sum_actor_compressors(3, learnable=True)
    actor_compressor = actor_compressors[2]
    observer_policy = fixed_sum_observer_policy(num_actions=3)
    before_phi = _clone_parameters(actor_compressor)

    trainer = UpdatedImperfectStateTrainer(
        env=env,
        actor_compressors=[actor_compressor],
        observer_policies=[observer_policy],
        compression_optimizer=torch.optim.SGD(actor_compressor.parameters(), lr=0.05),
        sample_actions=True,
    )

    trace = trainer.run(num_steps=10, seed=2, eval_interval=5)

    assert trace.metadata["learn_compression"] is True
    assert trace.metadata["learn_selection"] is False
    assert trace.metadata["learn_policies"] is False
    assert _parameters_moved(before_phi, actor_compressor)
    assert trace.selected_actor_indices[0] == [0]


def test_section_52_one_observer_updates_selection_and_selected_compressor() -> None:
    env = _env()
    actor_compressors = build_scalar_sum_actor_compressors(3, learnable=True)
    selection = SelectionLogits(
        num_actor_compressors=3,
        init_logits=[-1.0, -1.0, 1.0],
    )
    before_phi = [_clone_parameters(actor) for actor in actor_compressors]
    before_logits = selection.logits.detach().clone()

    trainer = UpdatedImperfectStateTrainer(
        env=env,
        actor_compressors=actor_compressors,
        observer_policies=[fixed_sum_observer_policy(num_actions=3)],
        selection_logits_by_observer=[selection],
        compression_optimizer=torch.optim.SGD(
            [parameter for actor in actor_compressors for parameter in actor.parameters()],
            lr=0.05,
        ),
        selection_optimizer=torch.optim.SGD(selection.parameters(), lr=0.05),
        sample_selection=False,
        sample_actions=True,
    )

    trace = trainer.run(num_steps=1, seed=3, eval_interval=1)

    selected = trace.selected_actor_indices[0][0]
    assert selected == 2
    for actor_index, actor in enumerate(actor_compressors):
        moved = _parameters_moved(before_phi[actor_index], actor)
        assert moved is (actor_index == selected)
    assert not torch.allclose(before_logits, selection.logits.detach())
    assert trace.compression_feedback_counts[0] == [0, 0, 1]


def test_section_53_all_observers_select_and_compressor_aggregates_feedback() -> None:
    env = _env()
    actor_compressors = build_scalar_sum_actor_compressors(3, learnable=True)
    selections = [
        SelectionLogits(3, init_logits=[-1.0, -1.0, 1.0]),
        SelectionLogits(3, init_logits=[-1.0, -1.0, 1.0]),
    ]
    policies = [SoftmaxPolicy(input_dim=1, num_actions=3) for _ in range(2)]
    before_phi = [_clone_parameters(actor) for actor in actor_compressors]

    trainer = UpdatedImperfectStateTrainer(
        env=env,
        actor_compressors=actor_compressors,
        observer_policies=policies,
        selection_logits_by_observer=selections,
        compression_optimizer=torch.optim.SGD(
            [parameter for actor in actor_compressors for parameter in actor.parameters()],
            lr=0.05,
        ),
        selection_optimizer=torch.optim.SGD(
            [parameter for selection in selections for parameter in selection.parameters()],
            lr=0.05,
        ),
        policy_optimizers=[
            torch.optim.SGD(policy.parameters(), lr=0.05)
            for policy in policies
        ],
        sample_selection=False,
        sample_actions=True,
    )

    trace = trainer.run(num_steps=1, seed=4, eval_interval=1)

    assert len(trace.selected_actor_indices[0]) == 2
    assert trace.selected_actor_indices[0] == [2, 2]
    assert trace.compression_feedback_counts[0] == [0, 0, 2]
    assert _parameters_moved(before_phi[2], actor_compressors[2])
    assert not _parameters_moved(before_phi[0], actor_compressors[0])
    assert not _parameters_moved(before_phi[1], actor_compressors[1])


def test_selection_logit_uniform_floor_keeps_every_actor_available() -> None:
    selection = SelectionLogits(4, init_logits=[20.0, -20.0, -20.0, -20.0])

    mixed = selection.mixed_probabilities(exploration_probability=0.2)

    assert torch.all(mixed >= torch.full_like(mixed, 0.2 / 4))
    assert torch.isclose(mixed.sum(), torch.tensor(1.0))


def test_scalar_sum_builders_use_cumulative_state_representation() -> None:
    states = cumulative_binary_state_features(5)
    compressors = build_scalar_sum_actor_compressors(5, learnable=False)
    full = compressors[2]

    assert states[0] == [0.0, 0.0, 0.0, 0.0, 0.0]
    assert states[3] == [1.0, 1.0, 1.0, 0.0, 0.0]
    messages = [
        float(full.message(torch.tensor(state)).detach().reshape(-1)[0].item())
        for state in states
    ]
    assert messages == [0.0, 1.0, 2.0, 3.0, 4.0]
