"""Smoke tests for the 1x1 joint phi+theta diagnosis scripts."""

from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

from run_fused_vs_cut_backprop import (  # noqa: E402
    FusedCompressionPolicyNet,
    run_arm,
)
from run_joint_phi_theta_1x1_diag import (  # noqa: E402
    ALL_MODES,
    run_mode,
)


def _tiny_args(**overrides) -> Namespace:
    args = Namespace(
        state_counts=[3],
        seeds=[0],
        steps=8,
        freeze_steps=4,
        eval_interval=2,
        anneal_segments=2,
        epsilon_start=0.1,
        epsilon_end=0.0,
        lr_phi=0.01,
        lr_theta=0.01,
        rbf_sigma=0.5,
        sharpness=4.0,
        policy_head="rbf",
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_all_modes_run_and_freeze_phi_does_not_update() -> None:
    args = _tiny_args()
    joint_artifacts = None
    records = []
    for mode in ALL_MODES:
        record, artifacts = run_mode(
            mode,
            num_states=3,
            seed=0,
            args=args,
            joint_artifacts=joint_artifacts,
        )
        if artifacts is not None:
            joint_artifacts = artifacts
        records.append(record)
        assert "final_expected_reward" in record
        assert "final_mean_correct_action_prob" in record
        assert "message_round_accuracy" in record

    assert joint_artifacts is not None
    freeze = next(row for row in records if row["mode"] == "freeze_phi_train_theta")
    # Frozen phi should not move after the freeze phase starts.
    assert freeze.get("phi_moved_from_start", 0.0) < 1e-8


def test_freeze_modes_require_joint_artifacts() -> None:
    args = _tiny_args()
    try:
        run_mode(
            "freeze_phi_train_theta",
            num_states=3,
            seed=0,
            args=args,
            joint_artifacts=None,
        )
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "requires a prior joint run" in str(exc)


def test_fused_net_forward_and_ce_arm_smoke() -> None:
    net = FusedCompressionPolicyNet(num_states=3, num_actions=3, sigma=0.5)
    logits = net(torch.tensor([1.0, 1.0, 0.0]))
    assert logits.shape == torch.Size([3])
    assert torch.isfinite(logits).all()

    # Tiny argparse-like namespace for run_arm.
    class _Args:
        steps = 20
        eval_interval = 10
        lr = 0.05
        rbf_sigma = 0.5
        epsilon = 0.1

    for arm in ("fused_ce", "fused_reinforce", "cut_reinforce"):
        record = run_arm(arm, num_states=3, seed=0, args=_Args())
        assert torch.isfinite(torch.tensor(record["final_expected_reward"]))
        assert 0.0 <= record["final_mean_correct_action_prob"] <= 1.0


def test_staged_modes_smoke() -> None:
    args = _tiny_args(steps=6, freeze_steps=4)
    for mode in ("phi_then_theta", "phi_then_joint", "theta_then_joint"):
        record, artifacts = run_mode(
            mode,
            num_states=3,
            seed=0,
            args=args,
            joint_artifacts=None,
        )
        assert artifacts is None
        assert record["mode"] == mode
        assert "stage1_expected_reward" in record
        assert "reward_drop" in record
        assert "final_mean_correct_action_prob" in record


def test_cli_writes_json(tmp_path: Path) -> None:
    from run_joint_phi_theta_1x1_diag import main as diag_main

    out = tmp_path / "diag.json"
    previous_argv = sys.argv
    sys.argv = [
        "run_joint_phi_theta_1x1_diag.py",
        "--modes",
        "policy_only,compression_only,joint",
        "--state-counts",
        "3",
        "--seeds",
        "0",
        "--steps",
        "6",
        "--eval-interval",
        "2",
        "--anneal-segments",
        "2",
        "--out",
        str(out),
    ]
    try:
        diag_main()
    finally:
        sys.argv = previous_argv
    payload = json.loads(out.read_text())
    assert len(payload["records"]) == 3
    modes = {row["mode"] for row in payload["records"]}
    assert modes == {"policy_only", "compression_only", "joint"}
