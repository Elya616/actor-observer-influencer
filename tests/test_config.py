"""Tests for config loading and validation."""

from __future__ import annotations

import pytest

from actor_observer_rl.utils.config import (
    Case1Config,
    Case2Config,
    Case3Config,
    Case4Config,
    Paper41Config,
    Paper42Config,
    Paper51Config,
    Paper52Config,
    Paper53Config,
    Paper61Config,
    Paper62Config,
    Paper71Config,
    Paper72Config,
    Paper73Config,
    load_yaml_config,
)


def test_load_yaml_config_reads_mapping(tmp_path) -> None:
    config_path = tmp_path / "case1.yaml"
    config_path.write_text(
        "\n".join(
            [
                "case: case1_single_state",
                "seed: 0",
                "num_steps: 10",
                "learning_rate: 0.1",
                "state_dim: 1",
                "num_actions: 2",
                "reward_table: [0.0, 1.0]",
                "optimal_action: 1",
            ]
        ),
        encoding="utf-8",
    )

    config = Case1Config.from_mapping(load_yaml_config(config_path))

    assert config.num_steps == 10
    assert config.learning_rate == 0.1
    assert config.reward_table == [0.0, 1.0]
    assert config.optimal_action == 1


def test_case1_config_rejects_reward_action_mismatch() -> None:
    raw_config = {
        "case": "case1_single_state",
        "seed": 0,
        "num_steps": 10,
        "learning_rate": 0.1,
        "state_dim": 1,
        "num_actions": 3,
        "reward_table": [0.0, 1.0],
        "optimal_action": 1,
    }

    with pytest.raises(ValueError, match="reward_table length"):
        Case1Config.from_mapping(raw_config)


def test_load_yaml_config_rejects_non_mapping(tmp_path) -> None:
    config_path = tmp_path / "bad.yaml"
    config_path.write_text("- not\n- a\n- mapping\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must be a YAML mapping"):
        load_yaml_config(config_path)


def test_case2_config_validates_contextual_bandit_fields() -> None:
    raw_config = {
        "case": "case2_iid_states",
        "seed": 0,
        "num_steps": 100,
        "learning_rate": 0.05,
        "num_states": 2,
        "num_actions": 2,
        "state_distribution": [0.4, 0.6],
        "reward_table": [[1.0, 0.0], [0.0, 1.0]],
        "optimal_actions": [0, 1],
    }

    config = Case2Config.from_mapping(raw_config)

    assert config.num_states == 2
    assert config.num_actions == 2
    assert config.optimal_actions == [0, 1]


def test_case2_config_rejects_wrong_optimal_action() -> None:
    raw_config = {
        "case": "case2_iid_states",
        "seed": 0,
        "num_steps": 100,
        "learning_rate": 0.05,
        "num_states": 2,
        "num_actions": 2,
        "state_distribution": [0.4, 0.6],
        "reward_table": [[1.0, 0.0], [0.0, 1.0]],
        "optimal_actions": [1, 1],
    }

    with pytest.raises(ValueError, match="maximum-reward"):
        Case2Config.from_mapping(raw_config)


def test_case3_config_validates_partial_observable_fields() -> None:
    raw_config = {
        "case": "case3_partial_observable",
        "seed": 0,
        "num_steps": 100,
        "actor_learning_rate": 0.05,
        "observer_learning_rate": 0.05,
        "num_states": 2,
        "observation_dim": 1,
        "message_dim": 2,
        "num_actions": 2,
        "state_distribution": [0.5, 0.5],
        "reward_table": [[1.0, 0.0], [0.0, 1.0]],
        "optimal_actions": [0, 1],
    }

    config = Case3Config.from_mapping(raw_config)

    assert config.observation_dim == 1
    assert config.message_dim == 2
    assert config.optimal_actions == [0, 1]


def test_case3_config_rejects_wrong_optimal_action() -> None:
    raw_config = {
        "case": "case3_partial_observable",
        "seed": 0,
        "num_steps": 100,
        "actor_learning_rate": 0.05,
        "observer_learning_rate": 0.05,
        "num_states": 2,
        "observation_dim": 1,
        "message_dim": 2,
        "num_actions": 2,
        "state_distribution": [0.5, 0.5],
        "reward_table": [[1.0, 0.0], [0.0, 1.0]],
        "optimal_actions": [1, 1],
    }

    with pytest.raises(ValueError, match="maximum-reward"):
        Case3Config.from_mapping(raw_config)


def _case4_raw_config(**overrides) -> dict[str, object]:
    raw_config = {
        "case": "case4_multi_observer",
        "seed": 0,
        "num_steps": 100,
        "actor_learning_rate": 0.05,
        "observer_learning_rate": 0.05,
        "following_learning_rate": 0.05,
        "num_observers": 2,
        "num_states": 2,
        "observation_dim": 1,
        "message_dim": 2,
        "num_actions": 2,
        "state_distribution": [0.5, 0.5],
        "reward_table": [[1.0, 0.0], [0.0, 1.0]],
        "optimal_actions": [0, 1],
    }
    raw_config.update(overrides)
    return raw_config


def test_case4_config_validates_multi_observer_fields() -> None:
    config = Case4Config.from_mapping(_case4_raw_config())

    assert config.num_observers == 2
    assert config.observation_dim == 1
    assert config.message_dim == 2
    assert config.optimal_actions == [0, 1]


def test_case4_config_rejects_single_observer() -> None:
    with pytest.raises(ValueError, match="num_observers"):
        Case4Config.from_mapping(_case4_raw_config(num_observers=1))


def test_case4_config_rejects_wrong_optimal_action() -> None:
    with pytest.raises(ValueError, match="maximum-reward"):
        Case4Config.from_mapping(_case4_raw_config(optimal_actions=[1, 1]))


def test_case4_config_rejects_invalid_following_learning_rate() -> None:
    with pytest.raises(ValueError, match="following_learning_rate"):
        Case4Config.from_mapping(_case4_raw_config(following_learning_rate=0.0))


def _paper41_raw_config(**overrides) -> dict[str, object]:
    raw_config = {
        "case": "paper_4_1_multi_actor",
        "seed": 0,
        "num_steps": 100,
        "num_actors": 3,
        "num_states": 1,
        "num_actions": 3,
        "state_distribution": [1.0],
        "actor_selection_probs": [1 / 3, 1 / 3, 1 / 3],
        "reward_table": [[0.9, 0.5, 0.1]],
        "fixed_actor_targets": [0, 1, 2],
        "budget": 1.0,
        "c": 2.0,
        "utility_learning_rate": 0.1,
        "following_learning_rate": 0.02,
        "snapshot_interval": 10,
        "actor_sharpness": 8.0,
    }
    raw_config.update(overrides)
    return raw_config


def test_paper41_config_allows_single_state_controlled_case() -> None:
    config = Paper41Config.from_mapping(_paper41_raw_config())

    assert config.num_states == 1
    assert config.fixed_actor_targets == [0, 1, 2]


def _paper42_raw_config(**overrides) -> dict[str, object]:
    raw_config = {
        "case": "paper_4_2_multi_actor",
        "seed": 0,
        "num_steps": 100,
        "num_actors": 3,
        "num_states": 3,
        "num_actions": 3,
        "state_distribution": [1 / 3, 1 / 3, 1 / 3],
        "actor_selection_probs": [1 / 3, 1 / 3, 1 / 3],
        "reward_table": [[1.0, 0.1, 0.2], [0.1, 1.0, 0.0], [0.0, 0.2, 1.0]],
        "budget": 3.0,
        "c": 2.0,
        "utility_learning_rate": 0.2,
        "following_learning_rate": 0.01,
        "actor_learning_rate": 0.2,
        "snapshot_interval": 10,
        "optimal_actions": [0, 1, 2],
    }
    raw_config.update(overrides)
    return raw_config


def test_paper42_config_validates_joint_learning_fields() -> None:
    config = Paper42Config.from_mapping(_paper42_raw_config())

    assert config.num_actors == 3
    assert config.optimal_actions == [0, 1, 2]


def test_paper42_config_rejects_wrong_optimal_action() -> None:
    with pytest.raises(ValueError, match="maximum-reward"):
        Paper42Config.from_mapping(_paper42_raw_config(optimal_actions=[1, 1, 2]))


def _paper61_raw_config(**overrides) -> dict[str, object]:
    raw_config = {
        "case": "paper_6_1_group",
        "seed": 0,
        "num_steps": 100,
        "num_actors": 3,
        "num_observers": 3,
        "num_states": 1,
        "num_actions": 3,
        "state_distribution": [1.0],
        "actor_selection_probs": [1 / 3, 1 / 3, 1 / 3],
        "observer_reward_tables": [
            [[0.9, 0.4, 0.1]],
            [[0.1, 0.9, 0.4]],
            [[0.4, 0.1, 0.9]],
        ],
        "fixed_actor_targets": [0, 1, 2],
        "budget": 2.0,
        "c": 1.0,
        "utility_learning_rate": 0.1,
        "following_learning_rate": 0.05,
        "snapshot_interval": 10,
        "actor_sharpness": 8.0,
    }
    raw_config.update(overrides)
    return raw_config


def test_paper61_config_validates_group_fields() -> None:
    config = Paper61Config.from_mapping(_paper61_raw_config())

    assert config.num_observers == 3
    assert len(config.observer_reward_tables) == 3
    assert config.fixed_actor_targets == [0, 1, 2]


def test_paper61_config_rejects_reward_table_count_mismatch() -> None:
    with pytest.raises(ValueError, match="observer_reward_tables length"):
        Paper61Config.from_mapping(
            _paper61_raw_config(observer_reward_tables=[[[0.9, 0.4, 0.1]]])
        )


def _paper62_raw_config(**overrides) -> dict[str, object]:
    diagonal = [[1.0, 0.1, 0.2], [0.1, 1.0, 0.0], [0.0, 0.2, 1.0]]
    raw_config = {
        "case": "paper_6_2_group",
        "seed": 0,
        "num_steps": 100,
        "num_actors": 3,
        "num_observers": 3,
        "num_states": 3,
        "num_actions": 3,
        "state_distribution": [1 / 3, 1 / 3, 1 / 3],
        "actor_selection_probs": [1 / 3, 1 / 3, 1 / 3],
        "observer_reward_tables": [diagonal, diagonal, diagonal],
        "budget": 3.0,
        "c": 2.0,
        "utility_learning_rate": 0.2,
        "following_learning_rate": 0.01,
        "actor_learning_rate": 0.1,
        "snapshot_interval": 10,
        "optimal_actions": [0, 1, 2],
    }
    raw_config.update(overrides)
    return raw_config


def test_paper62_config_validates_joint_group_fields() -> None:
    config = Paper62Config.from_mapping(_paper62_raw_config())

    assert config.num_observers == 3
    assert config.optimal_actions == [0, 1, 2]


def test_paper62_config_rejects_actor_count_mismatch() -> None:
    with pytest.raises(ValueError, match="actor_selection_probs length"):
        Paper62Config.from_mapping(
            _paper62_raw_config(actor_selection_probs=[0.5, 0.5])
        )


def _paper51_raw_config(**overrides) -> dict[str, object]:
    raw_config = {
        "case": "paper_5_1_observer_selection",
        "seed": 0,
        "num_steps": 100,
        "eval_interval": 10,
        "reward_window": 20,
        "num_observers": 3,
        "num_states": 4,
        "observation_dim": 1,
        "num_actions": 3,
        "state_distribution": [0.25, 0.25, 0.25, 0.25],
        "reward_table": [
            [1.0, -0.1, -0.1],
            [-0.1, 1.0, -0.1],
            [-0.1, 1.0, -0.1],
            [-0.1, -0.1, 1.0],
        ],
        "observer_names": ["zero", "first_feature", "sum"],
        "observer_messages": [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0, 2.0],
        ],
        "following_learning_rate": 0.05,
        "decoder_sharpness": 4.0,
    }
    raw_config.update(overrides)
    return raw_config


def test_paper51_config_validates_observer_selection_fields() -> None:
    config = Paper51Config.from_mapping(_paper51_raw_config())

    assert config.num_observers == 3
    assert config.observer_names == ["zero", "first_feature", "sum"]
    assert config.observer_messages[2] == [0.0, 1.0, 1.0, 2.0]


def test_paper51_config_rejects_message_state_mismatch() -> None:
    with pytest.raises(ValueError, match="observer_messages row"):
        Paper51Config.from_mapping(
            _paper51_raw_config(observer_messages=[[0.0], [0.0], [0.0]])
        )


def _paper52_raw_config(**overrides) -> dict[str, object]:
    raw_config = {
        "case": "paper_5_2_selection_compression",
        "seed": 0,
        "num_steps": 100,
        "eval_interval": 10,
        "reward_window": 20,
        "num_observers": 3,
        "num_states": 4,
        "state_dim": 2,
        "observation_dim": 1,
        "num_actions": 3,
        "state_features": [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        "state_distribution": [0.25, 0.25, 0.25, 0.25],
        "reward_table": [
            [1.0, -0.1, -0.1],
            [-0.1, 1.0, -0.1],
            [-0.1, 1.0, -0.1],
            [-0.1, -0.1, 1.0],
        ],
        "observer_names": ["zero_input", "first_feature", "full_sum"],
        "observer_input_masks": [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ],
        "following_learning_rate": 0.005,
        "observer_learning_rate": 0.01,
        "decoder_sharpness": 4.0,
    }
    raw_config.update(overrides)
    return raw_config


def test_paper52_config_validates_selection_compression_fields() -> None:
    config = Paper52Config.from_mapping(_paper52_raw_config())

    assert config.state_dim == 2
    assert config.observer_names == ["zero_input", "first_feature", "full_sum"]
    assert config.observer_input_masks[2] == [1.0, 1.0]


def test_paper52_config_rejects_mask_feature_mismatch() -> None:
    with pytest.raises(ValueError, match="observer_input_masks row"):
        Paper52Config.from_mapping(
            _paper52_raw_config(observer_input_masks=[[0.0], [1.0], [1.0]])
        )


def _paper53_raw_config(**overrides) -> dict[str, object]:
    raw_config = {
        "case": "paper_5_3_joint_imperfect_state",
        "seed": 0,
        "num_steps": 100,
        "eval_interval": 10,
        "reward_window": 20,
        "num_observers": 3,
        "num_states": 3,
        "state_dim": 2,
        "observation_dim": 1,
        "num_actions": 3,
        "state_features": [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ],
        "state_distribution": [1 / 3, 1 / 3, 1 / 3],
        "reward_table": [
            [1.0, -0.1, -0.1],
            [-0.1, 1.0, -0.1],
            [-0.1, -0.1, 1.0],
        ],
        "observer_names": ["zero_input", "first_feature", "full_state"],
        "observer_input_masks": [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ],
        "following_learning_rate": 0.005,
        "observer_learning_rate": 0.05,
        "actor_learning_rate": 0.05,
    }
    raw_config.update(overrides)
    return raw_config


def test_paper53_config_validates_joint_imperfect_state_fields() -> None:
    config = Paper53Config.from_mapping(_paper53_raw_config())

    assert config.num_states == 3
    assert config.observer_names == ["zero_input", "first_feature", "full_state"]
    assert config.actor_learning_rate == 0.05


def test_paper53_config_rejects_invalid_actor_learning_rate() -> None:
    with pytest.raises(ValueError, match="actor_learning_rate"):
        Paper53Config.from_mapping(_paper53_raw_config(actor_learning_rate=0.0))


def test_paper53_config_rejects_mask_feature_mismatch() -> None:
    with pytest.raises(ValueError, match="observer_input_masks row"):
        Paper53Config.from_mapping(
            _paper53_raw_config(observer_input_masks=[[0.0], [1.0], [1.0]])
        )


def _paper71_raw_config(**overrides) -> dict[str, object]:
    raw_config = {
        "case": "paper_7_1_group_observer_selection",
        "seed": 0,
        "num_steps": 100,
        "eval_interval": 10,
        "reward_window": 20,
        "num_actors": 3,
        "num_observers": 3,
        "num_states": 3,
        "observation_dim": 1,
        "num_actions": 3,
        "actor_selection_probs": [1 / 3, 1 / 3, 1 / 3],
        "state_distribution": [1 / 3, 1 / 3, 1 / 3],
        "reward_table": [
            [1.0, -0.1, -0.1],
            [-0.1, 1.0, -0.1],
            [-0.1, -0.1, 1.0],
        ],
        "observer_names": ["zero_input", "first_feature", "full_state"],
        "observer_messages": [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0],
            [0.0, 1.0, 2.0],
        ],
        "following_learning_rate": 0.02,
        "decoder_sharpness": 4.0,
    }
    raw_config.update(overrides)
    return raw_config


def test_paper71_config_validates_group_selection_fields() -> None:
    config = Paper71Config.from_mapping(_paper71_raw_config())

    assert config.num_actors == 3
    assert config.observer_names == ["zero_input", "first_feature", "full_state"]
    assert config.observer_messages[2] == [0.0, 1.0, 2.0]


def test_paper71_config_rejects_actor_probability_mismatch() -> None:
    with pytest.raises(ValueError, match="actor_selection_probs"):
        Paper71Config.from_mapping(
            _paper71_raw_config(actor_selection_probs=[0.5, 0.5])
        )


def _paper72_raw_config(**overrides) -> dict[str, object]:
    raw_config = {
        "case": "paper_7_2_group_selection_compression",
        "seed": 0,
        "num_steps": 100,
        "eval_interval": 10,
        "reward_window": 20,
        "num_actors": 3,
        "num_observers": 3,
        "num_states": 3,
        "state_dim": 2,
        "observation_dim": 1,
        "num_actions": 3,
        "actor_selection_probs": [1 / 3, 1 / 3, 1 / 3],
        "state_features": [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ],
        "state_distribution": [1 / 3, 1 / 3, 1 / 3],
        "reward_table": [
            [1.0, -0.1, -0.1],
            [-0.1, 1.0, -0.1],
            [-0.1, -0.1, 1.0],
        ],
        "observer_names": ["zero_input", "first_feature", "full_state"],
        "observer_input_masks": [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ],
        "following_learning_rate": 0.005,
        "observer_learning_rate": 0.01,
        "decoder_sharpness": 4.0,
    }
    raw_config.update(overrides)
    return raw_config


def test_paper72_config_validates_group_selection_compression_fields() -> None:
    config = Paper72Config.from_mapping(_paper72_raw_config())

    assert config.num_actors == 3
    assert config.state_dim == 2
    assert config.observer_input_masks[2] == [1.0, 1.0]


def test_paper72_config_rejects_mask_feature_mismatch() -> None:
    with pytest.raises(ValueError, match="observer_input_masks row"):
        Paper72Config.from_mapping(
            _paper72_raw_config(observer_input_masks=[[0.0], [1.0], [1.0]])
        )


def _paper73_raw_config(**overrides) -> dict[str, object]:
    raw_config = dict(_paper72_raw_config())
    raw_config.update(
        {
            "case": "paper_7_3_group_joint_imperfect_state",
            "observer_learning_rate": 0.05,
            "actor_learning_rate": 0.05,
        }
    )
    raw_config.update(overrides)
    return raw_config


def test_paper73_config_validates_group_joint_learning_fields() -> None:
    config = Paper73Config.from_mapping(_paper73_raw_config())

    assert config.num_actors == 3
    assert config.actor_learning_rate == 0.05
    assert config.observer_names == ["zero_input", "first_feature", "full_state"]


def test_paper73_config_rejects_invalid_actor_learning_rate() -> None:
    with pytest.raises(ValueError, match="actor_learning_rate"):
        Paper73Config.from_mapping(_paper73_raw_config(actor_learning_rate=0.0))
