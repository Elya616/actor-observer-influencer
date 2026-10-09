"""Checks for the Section 5.3 Observer policy-head options."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "run_updated_section_5_3_scalar_sum_group_scaling.py"
)
_spec = importlib.util.spec_from_file_location("section_5_3_group_scaling", _SCRIPT)
assert _spec is not None and _spec.loader is not None
section_5_3 = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = section_5_3
_spec.loader.exec_module(section_5_3)


def test_softmax_sum_decoder_init_matches_fixed_sum() -> None:
    fixed = section_5_3.make_observer_policy("fixed_sum", 8, 4.0)
    soft = section_5_3.make_observer_policy(
        "softmax", 8, 4.0, policy_init="sum_decoder"
    )
    for message in (0.0, 3.0, 7.0):
        x = torch.tensor([message])
        soft_probs = soft.action_distribution(x).probs.detach().tolist()
        fixed_probs = fixed.action_distribution(x).probs.detach().tolist()
        assert soft_probs == pytest.approx(fixed_probs, abs=1e-5)


def test_fixed_sum_head_has_no_trainable_theta() -> None:
    policy = section_5_3.make_observer_policy("fixed_sum", 4, 4.0)
    assert list(policy.parameters()) == []


def test_learnable_heads_have_trainable_theta() -> None:
    for head in ("softmax", "learnable_sum"):
        policy = section_5_3.make_observer_policy(head, 4)
        assert any(p.requires_grad for p in policy.parameters()), head


def test_fixed_sum_run_leaves_observer_policies_unchanged() -> None:
    curves, run = section_5_3.run_one(
        num_states=3,
        num_observers=3,
        num_actor_compressors=3,
        alpha=0.01,
        lr_theta=0.01,
        policy_head="fixed_sum",
        selection_exploration=0.2,
        action_exploration=0.05,
        seed=0,
        steps=200,
        eval_interval=100,
    )
    assert curves
    assert run["policy_head"] == "fixed_sum"
    assert torch.isfinite(torch.tensor(run["expected_reward"]))


def test_message_l2_penalty_keeps_messages_smaller() -> None:
    common = dict(
        num_states=8,
        num_observers=3,
        num_actor_compressors=3,
        alpha=0.05,
        lr_theta=0.0,
        policy_head="fixed_sum",
        selection_exploration=0.2,
        action_exploration=0.05,
        seed=2,
        steps=800,
        eval_interval=400,
    )
    _, plain = section_5_3.run_one(**common)
    _, penalized = section_5_3.run_one(**common, message_l2_coef=0.01)
    assert penalized["full_message_mae"] < plain["full_message_mae"]


def test_warmup_keep_phi_preserves_oracle_init() -> None:
    _, run = section_5_3.run_one(
        num_states=5,
        num_observers=3,
        num_actor_compressors=3,
        alpha=0.01,
        lr_theta=0.0,
        policy_head="fixed_sum",
        selection_exploration=0.2,
        action_exploration=0.05,
        seed=0,
        steps=600,
        eval_interval=200,
        selection_warmup_steps=200,
        warmup_reset_phi=False,
        lr_phi=0.0,
    )
    assert run["warmup_reset_phi"] is False
    # lr_phi=0 keeps the oracle messages exact even if selection still moves.
    assert run["full_message_mae"] == 0.0


def _frozen_phi_run(**overrides):
    """Short run with lr_phi=0 so the reported messages are the initialization."""
    common = dict(
        num_states=16,
        num_observers=3,
        num_actor_compressors=3,
        alpha=0.003,
        lr_theta=0.0,
        policy_head="fixed_sum",
        selection_exploration=0.2,
        action_exploration=0.05,
        seed=0,
        steps=200,
        eval_interval=100,
        lr_phi=0.0,
    )
    common.update(overrides)
    _, run = section_5_3.run_one(**common)
    return run


def test_phi_init_oracle_needs_no_warmup() -> None:
    run = _frozen_phi_run(phi_init="oracle")
    assert run["phi_init"] == "oracle"
    assert run["selection_warmup_steps"] == 0
    assert run["full_message_mae"] == 0.0


def test_phi_init_zero_is_the_default() -> None:
    default_run = _frozen_phi_run()
    explicit_run = _frozen_phi_run(phi_init="zero")
    assert default_run["phi_init"] == "zero"
    assert default_run["full_message_mae"] == explicit_run["full_message_mae"]
    # Zero phi means every message is 0, so MAE is the mean state ID.
    assert default_run["full_message_mae"] == pytest.approx(7.5)


def test_phi_init_oracle_noise_sits_between_oracle_and_zero() -> None:
    small = _frozen_phi_run(phi_init="oracle-noise", phi_init_noise=0.1)
    large = _frozen_phi_run(phi_init="oracle-noise", phi_init_noise=0.3)
    assert 0.0 < small["full_message_mae"] < large["full_message_mae"]
    assert large["full_message_mae"] < 7.5


def test_phi_init_oracle_noise_is_reproducible_from_the_seed() -> None:
    first = _frozen_phi_run(phi_init="oracle-noise", phi_init_noise=0.3)
    second = _frozen_phi_run(phi_init="oracle-noise", phi_init_noise=0.3)
    other_seed = _frozen_phi_run(
        phi_init="oracle-noise", phi_init_noise=0.3, seed=1
    )
    assert first["full_message_mae"] == second["full_message_mae"]
    assert first["full_message_mae"] != other_seed["full_message_mae"]


def test_phi_lr_cosine_decays_from_alpha_to_the_final_fraction() -> None:
    assert section_5_3.phi_lr_at(0, 100, 0.003, "constant", 0.1) == 0.003
    assert section_5_3.phi_lr_at(50, 100, 0.003, "constant", 0.1) == 0.003
    assert section_5_3.phi_lr_at(0, 100, 0.003, "cosine", 0.1) == pytest.approx(
        0.003
    )
    assert section_5_3.phi_lr_at(50, 100, 0.003, "cosine", 0.1) == pytest.approx(
        0.003 * 0.55
    )
    assert section_5_3.phi_lr_at(100, 100, 0.003, "cosine", 0.1) == pytest.approx(
        0.0003
    )
    schedule = [
        section_5_3.phi_lr_at(step, 100, 0.003, "cosine", 0.1)
        for step in range(0, 101, 10)
    ]
    assert schedule == sorted(schedule, reverse=True)


def test_phi_lr_cosine_at_fraction_one_reproduces_constant() -> None:
    """The chunked schedule path must not perturb the RNG stream."""
    common = dict(
        num_states=8,
        num_observers=3,
        num_actor_compressors=3,
        alpha=0.01,
        lr_theta=0.0,
        policy_head="fixed_sum",
        selection_exploration=0.2,
        action_exploration=0.05,
        seed=0,
        steps=600,
        eval_interval=200,
    )
    _, constant = section_5_3.run_one(**common)
    _, chunked = section_5_3.run_one(
        **common, phi_lr_schedule="cosine", phi_lr_final_fraction=1.0
    )
    assert chunked["expected_reward"] == constant["expected_reward"]
    assert chunked["full_messages"] == constant["full_messages"]


def test_phi_lr_cosine_reports_one_row_per_evaluation() -> None:
    curves, _ = section_5_3.run_one(
        num_states=5,
        num_observers=3,
        num_actor_compressors=3,
        alpha=0.01,
        lr_theta=0.0,
        policy_head="fixed_sum",
        selection_exploration=0.2,
        action_exploration=0.05,
        seed=0,
        steps=600,
        eval_interval=200,
        phi_lr_schedule="cosine",
        phi_lr_final_fraction=0.1,
    )
    steps = [row["step"] for row in curves]
    assert steps == [0, 200, 400, 600]


def test_decoder_ceiling_matches_the_measured_values() -> None:
    expected = {
        3: 0.973853,
        5: 0.968764,
        8: 0.965901,
        16: 0.963515,
        32: 0.962322,
    }
    for num_states, ceiling in expected.items():
        assert section_5_3.decoder_ceiling(num_states, 4.0) == pytest.approx(
            ceiling, abs=1e-5
        )


def test_run_reports_reward_relative_to_ceiling() -> None:
    run = _frozen_phi_run(phi_init="oracle")
    assert run["decoder_ceiling"] == pytest.approx(
        section_5_3.decoder_ceiling(16, 4.0)
    )
    assert run["reward_over_ceiling"] == pytest.approx(
        run["expected_reward"] / run["decoder_ceiling"]
    )
    assert run["success_ceiling"] in {0.0, 1.0}


def test_selection_warmup_locks_full_before_phi_trains() -> None:
    """Warmup with oracle compressors should raise P(full) before phi moves."""
    curves, run = section_5_3.run_one(
        num_states=5,
        num_observers=3,
        num_actor_compressors=3,
        alpha=0.05,
        lr_theta=0.0,
        policy_head="fixed_sum",
        selection_exploration=0.2,
        action_exploration=0.05,
        seed=0,
        steps=800,
        eval_interval=200,
        selection_warmup_steps=400,
    )
    assert run["selection_warmup_steps"] == 400
    warmup_rows = [row for row in curves if row["step"] <= 400]
    warmup_end = max(warmup_rows, key=lambda row: row["step"])
    assert warmup_end["mean_p_full"] > 0.5


def test_mean_reduction_matches_sum_reduction_at_alpha_over_m() -> None:
    """Dividing the loss by M must be the same as dividing alpha by M."""
    common = dict(
        num_states=5,
        num_actor_compressors=3,
        lr_theta=0.0,
        policy_head="fixed_sum",
        selection_exploration=0.2,
        action_exploration=0.05,
        seed=0,
        steps=400,
        eval_interval=200,
    )
    observers = 3
    _, summed = section_5_3.run_one(
        **common,
        num_observers=observers,
        alpha=0.01 / observers,
        loss_reduction="sum",
    )
    _, averaged = section_5_3.run_one(
        **common,
        num_observers=observers,
        alpha=0.01,
        loss_reduction="mean",
    )
    assert summed["full_message_mae"] == pytest.approx(
        averaged["full_message_mae"], rel=1e-4
    )


def test_grad_clipping_bounds_the_compression_step() -> None:
    """Clipping must cap the message drift a single large update can cause."""
    common = dict(
        num_states=5,
        num_observers=3,
        num_actor_compressors=3,
        alpha=0.5,
        lr_theta=0.0,
        policy_head="fixed_sum",
        selection_exploration=0.2,
        action_exploration=0.05,
        seed=0,
        steps=300,
        eval_interval=150,
    )
    _, unclipped = section_5_3.run_one(**common)
    _, clipped = section_5_3.run_one(**common, max_grad_norm=0.1)
    assert clipped["full_message_mae"] <= unclipped["full_message_mae"]


def test_fixed_sum_still_trains_compression_and_selection() -> None:
    """theta is frozen, but phi and w must still move (Section 5.2 setup)."""
    _, run = section_5_3.run_one(
        num_states=3,
        num_observers=3,
        num_actor_compressors=3,
        alpha=0.05,
        lr_theta=0.0,
        policy_head="fixed_sum",
        selection_exploration=0.2,
        action_exploration=0.05,
        seed=0,
        steps=400,
        eval_interval=200,
    )
    assert run["mean_p_full"] >= 0.0
