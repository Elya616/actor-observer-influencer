"""Config loading helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Case1Config:
    """Typed config for the single-state bandit experiment."""

    seed: int
    num_steps: int
    learning_rate: float
    state_dim: int
    num_actions: int
    reward_table: list[float]
    optimal_action: int

    @classmethod
    def from_mapping(cls, config: dict[str, Any]) -> "Case1Config":
        case_name = config.get("case")
        if case_name != "case1_single_state":
            raise ValueError(f"expected case1_single_state config, got {case_name!r}")

        reward_table = [float(reward) for reward in config["reward_table"]]
        num_actions = int(config["num_actions"])
        optimal_action = int(config["optimal_action"])

        if len(reward_table) != num_actions:
            raise ValueError("reward_table length must match num_actions")
        if not 0 <= optimal_action < num_actions:
            raise ValueError("optimal_action must be a valid action index")

        parsed = cls(
            seed=int(config["seed"]),
            num_steps=int(config["num_steps"]),
            learning_rate=float(config["learning_rate"]),
            state_dim=int(config["state_dim"]),
            num_actions=num_actions,
            reward_table=reward_table,
            optimal_action=optimal_action,
        )
        parsed.validate()
        return parsed

    def validate(self) -> None:
        if self.num_steps <= 0:
            raise ValueError("num_steps must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.state_dim <= 0:
            raise ValueError("state_dim must be positive")
        if self.num_actions <= 1:
            raise ValueError("num_actions must be greater than one")


@dataclass(frozen=True)
class Case2Config:
    """Typed config for the i.i.d. states contextual-bandit experiment."""

    seed: int
    num_steps: int
    learning_rate: float
    num_states: int
    num_actions: int
    state_distribution: list[float]
    reward_table: list[list[float]]
    optimal_actions: list[int]

    @classmethod
    def from_mapping(cls, config: dict[str, Any]) -> "Case2Config":
        case_name = config.get("case")
        if case_name != "case2_iid_states":
            raise ValueError(f"expected case2_iid_states config, got {case_name!r}")

        num_states = int(config["num_states"])
        num_actions = int(config["num_actions"])
        state_distribution = [
            float(probability) for probability in config["state_distribution"]
        ]
        reward_table = [
            [float(reward) for reward in row] for row in config["reward_table"]
        ]
        optimal_actions = [int(action) for action in config["optimal_actions"]]

        parsed = cls(
            seed=int(config["seed"]),
            num_steps=int(config["num_steps"]),
            learning_rate=float(config["learning_rate"]),
            num_states=num_states,
            num_actions=num_actions,
            state_distribution=state_distribution,
            reward_table=reward_table,
            optimal_actions=optimal_actions,
        )
        parsed.validate()
        return parsed

    def validate(self) -> None:
        if self.num_steps <= 0:
            raise ValueError("num_steps must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.num_states <= 1:
            raise ValueError("num_states must be greater than one")
        if self.num_actions <= 1:
            raise ValueError("num_actions must be greater than one")
        if len(self.state_distribution) != self.num_states:
            raise ValueError("state_distribution length must match num_states")
        if any(probability < 0.0 for probability in self.state_distribution):
            raise ValueError("state_distribution cannot contain negative probabilities")
        if not math.isclose(
            sum(self.state_distribution),
            1.0,
            rel_tol=1e-6,
            abs_tol=1e-6,
        ):
            raise ValueError("state_distribution must sum to 1")
        if len(self.reward_table) != self.num_states:
            raise ValueError("reward_table row count must match num_states")
        if any(len(row) != self.num_actions for row in self.reward_table):
            raise ValueError("each reward_table row length must match num_actions")
        if len(self.optimal_actions) != self.num_states:
            raise ValueError("optimal_actions length must match num_states")
        for state_index, optimal_action in enumerate(self.optimal_actions):
            if not 0 <= optimal_action < self.num_actions:
                raise ValueError("optimal_actions must be valid action indices")
            best_reward = max(self.reward_table[state_index])
            if self.reward_table[state_index][optimal_action] != best_reward:
                raise ValueError("optimal_actions must identify maximum-reward actions")


@dataclass(frozen=True)
class Case3Config:
    """Typed config for partial-observability Actor/Observer experiments."""

    seed: int
    num_steps: int
    actor_learning_rate: float
    observer_learning_rate: float
    num_states: int
    observation_dim: int
    message_dim: int
    num_actions: int
    state_distribution: list[float]
    reward_table: list[list[float]]
    optimal_actions: list[int]

    @classmethod
    def from_mapping(cls, config: dict[str, Any]) -> "Case3Config":
        case_name = config.get("case")
        if case_name != "case3_partial_observable":
            raise ValueError(f"expected case3_partial_observable config, got {case_name!r}")

        num_states = int(config["num_states"])
        num_actions = int(config["num_actions"])
        state_distribution = [
            float(probability) for probability in config["state_distribution"]
        ]
        reward_table = [
            [float(reward) for reward in row] for row in config["reward_table"]
        ]
        optimal_actions = [int(action) for action in config["optimal_actions"]]

        parsed = cls(
            seed=int(config["seed"]),
            num_steps=int(config["num_steps"]),
            actor_learning_rate=float(config["actor_learning_rate"]),
            observer_learning_rate=float(config["observer_learning_rate"]),
            num_states=num_states,
            observation_dim=int(config["observation_dim"]),
            message_dim=int(config["message_dim"]),
            num_actions=num_actions,
            state_distribution=state_distribution,
            reward_table=reward_table,
            optimal_actions=optimal_actions,
        )
        parsed.validate()
        return parsed

    def validate(self) -> None:
        if self.num_steps <= 0:
            raise ValueError("num_steps must be positive")
        if self.actor_learning_rate <= 0:
            raise ValueError("actor_learning_rate must be positive")
        if self.observer_learning_rate <= 0:
            raise ValueError("observer_learning_rate must be positive")
        if self.num_states <= 1:
            raise ValueError("num_states must be greater than one")
        if self.observation_dim <= 0:
            raise ValueError("observation_dim must be positive")
        if self.message_dim <= 0:
            raise ValueError("message_dim must be positive")
        if self.num_actions <= 1:
            raise ValueError("num_actions must be greater than one")
        if len(self.state_distribution) != self.num_states:
            raise ValueError("state_distribution length must match num_states")
        if any(probability < 0.0 for probability in self.state_distribution):
            raise ValueError("state_distribution cannot contain negative probabilities")
        if not math.isclose(
            sum(self.state_distribution),
            1.0,
            rel_tol=1e-6,
            abs_tol=1e-6,
        ):
            raise ValueError("state_distribution must sum to 1")
        if len(self.reward_table) != self.num_states:
            raise ValueError("reward_table row count must match num_states")
        if any(len(row) != self.num_actions for row in self.reward_table):
            raise ValueError("each reward_table row length must match num_actions")
        if len(self.optimal_actions) != self.num_states:
            raise ValueError("optimal_actions length must match num_states")
        for state_index, optimal_action in enumerate(self.optimal_actions):
            if not 0 <= optimal_action < self.num_actions:
                raise ValueError("optimal_actions must be valid action indices")
            best_reward = max(self.reward_table[state_index])
            if self.reward_table[state_index][optimal_action] != best_reward:
                raise ValueError("optimal_actions must identify maximum-reward actions")


@dataclass(frozen=True)
class Case4Config:
    """Typed config for multi-observer Actor/Observers/following-rate experiments."""

    seed: int
    num_steps: int
    actor_learning_rate: float
    observer_learning_rate: float
    following_learning_rate: float
    num_observers: int
    num_states: int
    observation_dim: int
    message_dim: int
    num_actions: int
    state_distribution: list[float]
    reward_table: list[list[float]]
    optimal_actions: list[int]

    @classmethod
    def from_mapping(cls, config: dict[str, Any]) -> "Case4Config":
        case_name = config.get("case")
        if case_name != "case4_multi_observer":
            raise ValueError(f"expected case4_multi_observer config, got {case_name!r}")

        num_states = int(config["num_states"])
        num_actions = int(config["num_actions"])
        state_distribution = [
            float(probability) for probability in config["state_distribution"]
        ]
        reward_table = [
            [float(reward) for reward in row] for row in config["reward_table"]
        ]
        optimal_actions = [int(action) for action in config["optimal_actions"]]

        parsed = cls(
            seed=int(config["seed"]),
            num_steps=int(config["num_steps"]),
            actor_learning_rate=float(config["actor_learning_rate"]),
            observer_learning_rate=float(config["observer_learning_rate"]),
            following_learning_rate=float(config["following_learning_rate"]),
            num_observers=int(config["num_observers"]),
            num_states=num_states,
            observation_dim=int(config["observation_dim"]),
            message_dim=int(config["message_dim"]),
            num_actions=num_actions,
            state_distribution=state_distribution,
            reward_table=reward_table,
            optimal_actions=optimal_actions,
        )
        parsed.validate()
        return parsed

    def validate(self) -> None:
        if self.num_steps <= 0:
            raise ValueError("num_steps must be positive")
        if self.actor_learning_rate <= 0:
            raise ValueError("actor_learning_rate must be positive")
        if self.observer_learning_rate <= 0:
            raise ValueError("observer_learning_rate must be positive")
        if self.following_learning_rate <= 0:
            raise ValueError("following_learning_rate must be positive")
        if self.num_observers <= 1:
            raise ValueError("num_observers must be greater than one")
        if self.num_states <= 1:
            raise ValueError("num_states must be greater than one")
        if self.observation_dim <= 0:
            raise ValueError("observation_dim must be positive")
        if self.message_dim <= 0:
            raise ValueError("message_dim must be positive")
        if self.num_actions <= 1:
            raise ValueError("num_actions must be greater than one")
        if len(self.state_distribution) != self.num_states:
            raise ValueError("state_distribution length must match num_states")
        if any(probability < 0.0 for probability in self.state_distribution):
            raise ValueError("state_distribution cannot contain negative probabilities")
        if not math.isclose(
            sum(self.state_distribution),
            1.0,
            rel_tol=1e-6,
            abs_tol=1e-6,
        ):
            raise ValueError("state_distribution must sum to 1")
        if len(self.reward_table) != self.num_states:
            raise ValueError("reward_table row count must match num_states")
        if any(len(row) != self.num_actions for row in self.reward_table):
            raise ValueError("each reward_table row length must match num_actions")
        if len(self.optimal_actions) != self.num_states:
            raise ValueError("optimal_actions length must match num_states")
        for state_index, optimal_action in enumerate(self.optimal_actions):
            if not 0 <= optimal_action < self.num_actions:
                raise ValueError("optimal_actions must be valid action indices")
            best_reward = max(self.reward_table[state_index])
            if self.reward_table[state_index][optimal_action] != best_reward:
                raise ValueError("optimal_actions must identify maximum-reward actions")


@dataclass(frozen=True)
class Paper41Config:
    """Typed config for paper Section 4.1 (fixed-policy multi-actor)."""

    seed: int
    num_steps: int
    num_actors: int
    num_states: int
    num_actions: int
    state_distribution: list[float]
    actor_selection_probs: list[float]
    reward_table: list[list[float]]
    fixed_actor_targets: list[int]
    budget: float
    c: float
    utility_learning_rate: float
    following_learning_rate: float
    snapshot_interval: int
    actor_sharpness: float = 5.0

    @classmethod
    def from_mapping(cls, config: dict[str, Any]) -> "Paper41Config":
        case_name = config.get("case")
        if case_name != "paper_4_1_multi_actor":
            raise ValueError(f"expected paper_4_1_multi_actor config, got {case_name!r}")

        parsed = cls(
            seed=int(config["seed"]),
            num_steps=int(config["num_steps"]),
            num_actors=int(config["num_actors"]),
            num_states=int(config["num_states"]),
            num_actions=int(config["num_actions"]),
            state_distribution=[float(p) for p in config["state_distribution"]],
            actor_selection_probs=[float(p) for p in config["actor_selection_probs"]],
            reward_table=[[float(r) for r in row] for row in config["reward_table"]],
            fixed_actor_targets=[int(t) for t in config["fixed_actor_targets"]],
            budget=float(config["budget"]),
            c=float(config["c"]),
            utility_learning_rate=float(config["utility_learning_rate"]),
            following_learning_rate=float(config["following_learning_rate"]),
            snapshot_interval=int(config.get("snapshot_interval", 1)),
            actor_sharpness=float(config.get("actor_sharpness", 5.0)),
        )
        parsed.validate()
        return parsed

    def validate(self) -> None:
        _validate_paper4_common(
            num_steps=self.num_steps,
            num_actors=self.num_actors,
            num_states=self.num_states,
            num_actions=self.num_actions,
            state_distribution=self.state_distribution,
            actor_selection_probs=self.actor_selection_probs,
            reward_table=self.reward_table,
            budget=self.budget,
            c=self.c,
            utility_learning_rate=self.utility_learning_rate,
            following_learning_rate=self.following_learning_rate,
            snapshot_interval=self.snapshot_interval,
        )
        if len(self.fixed_actor_targets) != self.num_actors:
            raise ValueError("fixed_actor_targets length must match num_actors")
        for target in self.fixed_actor_targets:
            if not 0 <= target < self.num_actions:
                raise ValueError("fixed_actor_targets entries must be valid action indices")
        if self.actor_sharpness <= 0.0:
            raise ValueError("actor_sharpness must be positive")


@dataclass(frozen=True)
class Paper42Config:
    """Typed config for paper Section 4.2 (learnable-policy multi-actor)."""

    seed: int
    num_steps: int
    num_actors: int
    num_states: int
    num_actions: int
    state_distribution: list[float]
    actor_selection_probs: list[float]
    reward_table: list[list[float]]
    budget: float
    c: float
    utility_learning_rate: float
    following_learning_rate: float
    actor_learning_rate: float
    snapshot_interval: int
    optimal_actions: list[int]

    @classmethod
    def from_mapping(cls, config: dict[str, Any]) -> "Paper42Config":
        case_name = config.get("case")
        if case_name != "paper_4_2_multi_actor":
            raise ValueError(f"expected paper_4_2_multi_actor config, got {case_name!r}")

        parsed = cls(
            seed=int(config["seed"]),
            num_steps=int(config["num_steps"]),
            num_actors=int(config["num_actors"]),
            num_states=int(config["num_states"]),
            num_actions=int(config["num_actions"]),
            state_distribution=[float(p) for p in config["state_distribution"]],
            actor_selection_probs=[float(p) for p in config["actor_selection_probs"]],
            reward_table=[[float(r) for r in row] for row in config["reward_table"]],
            budget=float(config["budget"]),
            c=float(config["c"]),
            utility_learning_rate=float(config["utility_learning_rate"]),
            following_learning_rate=float(config["following_learning_rate"]),
            actor_learning_rate=float(config["actor_learning_rate"]),
            snapshot_interval=int(config.get("snapshot_interval", 1)),
            optimal_actions=[int(a) for a in config["optimal_actions"]],
        )
        parsed.validate()
        return parsed

    def validate(self) -> None:
        _validate_paper4_common(
            num_steps=self.num_steps,
            num_actors=self.num_actors,
            num_states=self.num_states,
            num_actions=self.num_actions,
            state_distribution=self.state_distribution,
            actor_selection_probs=self.actor_selection_probs,
            reward_table=self.reward_table,
            budget=self.budget,
            c=self.c,
            utility_learning_rate=self.utility_learning_rate,
            following_learning_rate=self.following_learning_rate,
            snapshot_interval=self.snapshot_interval,
        )
        if self.actor_learning_rate <= 0.0:
            raise ValueError("actor_learning_rate must be positive")
        if len(self.optimal_actions) != self.num_states:
            raise ValueError("optimal_actions length must match num_states")
        for state_index, optimal_action in enumerate(self.optimal_actions):
            if not 0 <= optimal_action < self.num_actions:
                raise ValueError("optimal_actions must be valid action indices")
            best_reward = max(self.reward_table[state_index])
            if self.reward_table[state_index][optimal_action] != best_reward:
                raise ValueError("optimal_actions must identify maximum-reward actions")


@dataclass(frozen=True)
class Paper51Config:
    """Typed config for paper Section 5.1 Observer-selection experiments."""

    seed: int
    num_steps: int
    eval_interval: int
    reward_window: int
    num_observers: int
    num_states: int
    observation_dim: int
    num_actions: int
    state_distribution: list[float]
    reward_table: list[list[float]]
    observer_names: list[str]
    observer_messages: list[list[float]]
    following_learning_rate: float
    decoder_sharpness: float

    @classmethod
    def from_mapping(cls, config: dict[str, Any]) -> "Paper51Config":
        case_name = config.get("case")
        if case_name != "paper_5_1_observer_selection":
            raise ValueError(
                f"expected paper_5_1_observer_selection config, got {case_name!r}"
            )

        parsed = cls(
            seed=int(config["seed"]),
            num_steps=int(config["num_steps"]),
            eval_interval=int(config.get("eval_interval", 1)),
            reward_window=int(config.get("reward_window", 100)),
            num_observers=int(config["num_observers"]),
            num_states=int(config["num_states"]),
            observation_dim=int(config["observation_dim"]),
            num_actions=int(config["num_actions"]),
            state_distribution=[float(p) for p in config["state_distribution"]],
            reward_table=[[float(r) for r in row] for row in config["reward_table"]],
            observer_names=[str(name) for name in config["observer_names"]],
            observer_messages=[
                [float(value) for value in row] for row in config["observer_messages"]
            ],
            following_learning_rate=float(config["following_learning_rate"]),
            decoder_sharpness=float(config["decoder_sharpness"]),
        )
        parsed.validate()
        return parsed

    def validate(self) -> None:
        if self.num_steps <= 0:
            raise ValueError("num_steps must be positive")
        if self.eval_interval <= 0:
            raise ValueError("eval_interval must be positive")
        if self.reward_window <= 0:
            raise ValueError("reward_window must be positive")
        if self.num_observers < 2:
            raise ValueError("num_observers must be at least two")
        if self.num_states <= 0:
            raise ValueError("num_states must be positive")
        if self.observation_dim <= 0:
            raise ValueError("observation_dim must be positive")
        if self.num_actions <= 1:
            raise ValueError("num_actions must be greater than one")
        if self.following_learning_rate <= 0.0:
            raise ValueError("following_learning_rate must be positive")
        if self.decoder_sharpness <= 0.0:
            raise ValueError("decoder_sharpness must be positive")
        if len(self.state_distribution) != self.num_states:
            raise ValueError("state_distribution length must match num_states")
        if any(p < 0.0 for p in self.state_distribution):
            raise ValueError("state_distribution cannot contain negative probabilities")
        if not math.isclose(sum(self.state_distribution), 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("state_distribution must sum to 1")
        if len(self.reward_table) != self.num_states:
            raise ValueError("reward_table row count must match num_states")
        if any(len(row) != self.num_actions for row in self.reward_table):
            raise ValueError("each reward_table row length must match num_actions")
        if len(self.observer_names) != self.num_observers:
            raise ValueError("observer_names length must match num_observers")
        if len(set(self.observer_names)) != len(self.observer_names):
            raise ValueError("observer_names must be unique")
        if len(self.observer_messages) != self.num_observers:
            raise ValueError("observer_messages length must match num_observers")
        for row in self.observer_messages:
            if len(row) != self.num_states:
                raise ValueError("each observer_messages row must match num_states")


@dataclass(frozen=True)
class Paper52Config:
    """Typed config for paper Section 5.2 selection/compression experiments."""

    seed: int
    num_steps: int
    eval_interval: int
    reward_window: int
    num_observers: int
    num_states: int
    state_dim: int
    observation_dim: int
    num_actions: int
    state_features: list[list[float]]
    state_distribution: list[float]
    reward_table: list[list[float]]
    observer_names: list[str]
    observer_input_masks: list[list[float]]
    following_learning_rate: float
    observer_learning_rate: float
    decoder_sharpness: float

    @classmethod
    def from_mapping(cls, config: dict[str, Any]) -> "Paper52Config":
        case_name = config.get("case")
        if case_name != "paper_5_2_selection_compression":
            raise ValueError(
                f"expected paper_5_2_selection_compression config, got {case_name!r}"
            )

        parsed = cls(
            seed=int(config["seed"]),
            num_steps=int(config["num_steps"]),
            eval_interval=int(config.get("eval_interval", 1)),
            reward_window=int(config.get("reward_window", 100)),
            num_observers=int(config["num_observers"]),
            num_states=int(config["num_states"]),
            state_dim=int(config["state_dim"]),
            observation_dim=int(config["observation_dim"]),
            num_actions=int(config["num_actions"]),
            state_features=[
                [float(value) for value in row] for row in config["state_features"]
            ],
            state_distribution=[float(p) for p in config["state_distribution"]],
            reward_table=[[float(r) for r in row] for row in config["reward_table"]],
            observer_names=[str(name) for name in config["observer_names"]],
            observer_input_masks=[
                [float(value) for value in row]
                for row in config["observer_input_masks"]
            ],
            following_learning_rate=float(config["following_learning_rate"]),
            observer_learning_rate=float(config["observer_learning_rate"]),
            decoder_sharpness=float(config["decoder_sharpness"]),
        )
        parsed.validate()
        return parsed

    def validate(self) -> None:
        if self.num_steps <= 0:
            raise ValueError("num_steps must be positive")
        if self.eval_interval <= 0:
            raise ValueError("eval_interval must be positive")
        if self.reward_window <= 0:
            raise ValueError("reward_window must be positive")
        if self.num_observers < 2:
            raise ValueError("num_observers must be at least two")
        if self.num_states <= 0:
            raise ValueError("num_states must be positive")
        if self.state_dim <= 0:
            raise ValueError("state_dim must be positive")
        if self.observation_dim <= 0:
            raise ValueError("observation_dim must be positive")
        if self.num_actions <= 1:
            raise ValueError("num_actions must be greater than one")
        if self.following_learning_rate <= 0.0:
            raise ValueError("following_learning_rate must be positive")
        if self.observer_learning_rate <= 0.0:
            raise ValueError("observer_learning_rate must be positive")
        if self.decoder_sharpness <= 0.0:
            raise ValueError("decoder_sharpness must be positive")
        if len(self.state_features) != self.num_states:
            raise ValueError("state_features row count must match num_states")
        if any(len(row) != self.state_dim for row in self.state_features):
            raise ValueError("each state_features row length must match state_dim")
        if len(self.state_distribution) != self.num_states:
            raise ValueError("state_distribution length must match num_states")
        if any(p < 0.0 for p in self.state_distribution):
            raise ValueError("state_distribution cannot contain negative probabilities")
        if not math.isclose(sum(self.state_distribution), 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("state_distribution must sum to 1")
        if len(self.reward_table) != self.num_states:
            raise ValueError("reward_table row count must match num_states")
        if any(len(row) != self.num_actions for row in self.reward_table):
            raise ValueError("each reward_table row length must match num_actions")
        if len(self.observer_names) != self.num_observers:
            raise ValueError("observer_names length must match num_observers")
        if len(set(self.observer_names)) != len(self.observer_names):
            raise ValueError("observer_names must be unique")
        if len(self.observer_input_masks) != self.num_observers:
            raise ValueError("observer_input_masks length must match num_observers")
        if any(len(row) != self.state_dim for row in self.observer_input_masks):
            raise ValueError("each observer_input_masks row length must match state_dim")


@dataclass(frozen=True)
class Paper53Config:
    """Typed config for paper Section 5.3 joint imperfect-state experiments."""

    seed: int
    num_steps: int
    eval_interval: int
    reward_window: int
    num_observers: int
    num_states: int
    state_dim: int
    observation_dim: int
    num_actions: int
    state_features: list[list[float]]
    state_distribution: list[float]
    reward_table: list[list[float]]
    observer_names: list[str]
    observer_input_masks: list[list[float]]
    following_learning_rate: float
    observer_learning_rate: float
    actor_learning_rate: float

    @classmethod
    def from_mapping(cls, config: dict[str, Any]) -> "Paper53Config":
        case_name = config.get("case")
        if case_name != "paper_5_3_joint_imperfect_state":
            raise ValueError(
                f"expected paper_5_3_joint_imperfect_state config, got {case_name!r}"
            )

        parsed = cls(
            seed=int(config["seed"]),
            num_steps=int(config["num_steps"]),
            eval_interval=int(config.get("eval_interval", 1)),
            reward_window=int(config.get("reward_window", 100)),
            num_observers=int(config["num_observers"]),
            num_states=int(config["num_states"]),
            state_dim=int(config["state_dim"]),
            observation_dim=int(config["observation_dim"]),
            num_actions=int(config["num_actions"]),
            state_features=[
                [float(value) for value in row] for row in config["state_features"]
            ],
            state_distribution=[float(p) for p in config["state_distribution"]],
            reward_table=[[float(r) for r in row] for row in config["reward_table"]],
            observer_names=[str(name) for name in config["observer_names"]],
            observer_input_masks=[
                [float(value) for value in row]
                for row in config["observer_input_masks"]
            ],
            following_learning_rate=float(config["following_learning_rate"]),
            observer_learning_rate=float(config["observer_learning_rate"]),
            actor_learning_rate=float(config["actor_learning_rate"]),
        )
        parsed.validate()
        return parsed

    def validate(self) -> None:
        if self.num_steps <= 0:
            raise ValueError("num_steps must be positive")
        if self.eval_interval <= 0:
            raise ValueError("eval_interval must be positive")
        if self.reward_window <= 0:
            raise ValueError("reward_window must be positive")
        if self.num_observers < 2:
            raise ValueError("num_observers must be at least two")
        if self.num_states <= 0:
            raise ValueError("num_states must be positive")
        if self.state_dim <= 0:
            raise ValueError("state_dim must be positive")
        if self.observation_dim <= 0:
            raise ValueError("observation_dim must be positive")
        if self.num_actions <= 1:
            raise ValueError("num_actions must be greater than one")
        if self.following_learning_rate <= 0.0:
            raise ValueError("following_learning_rate must be positive")
        if self.observer_learning_rate <= 0.0:
            raise ValueError("observer_learning_rate must be positive")
        if self.actor_learning_rate <= 0.0:
            raise ValueError("actor_learning_rate must be positive")
        if len(self.state_features) != self.num_states:
            raise ValueError("state_features row count must match num_states")
        if any(len(row) != self.state_dim for row in self.state_features):
            raise ValueError("each state_features row length must match state_dim")
        if len(self.state_distribution) != self.num_states:
            raise ValueError("state_distribution length must match num_states")
        if any(p < 0.0 for p in self.state_distribution):
            raise ValueError("state_distribution cannot contain negative probabilities")
        if not math.isclose(sum(self.state_distribution), 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("state_distribution must sum to 1")
        if len(self.reward_table) != self.num_states:
            raise ValueError("reward_table row count must match num_states")
        if any(len(row) != self.num_actions for row in self.reward_table):
            raise ValueError("each reward_table row length must match num_actions")
        if len(self.observer_names) != self.num_observers:
            raise ValueError("observer_names length must match num_observers")
        if len(set(self.observer_names)) != len(self.observer_names):
            raise ValueError("observer_names must be unique")
        if len(self.observer_input_masks) != self.num_observers:
            raise ValueError("observer_input_masks length must match num_observers")
        if any(len(row) != self.state_dim for row in self.observer_input_masks):
            raise ValueError("each observer_input_masks row length must match state_dim")


@dataclass(frozen=True)
class Paper61Config:
    """Typed config for paper Section 6.1 (group-level, fixed Actor policies)."""

    seed: int
    num_steps: int
    num_actors: int
    num_observers: int
    num_states: int
    num_actions: int
    state_distribution: list[float]
    actor_selection_probs: list[float]
    observer_reward_tables: list[list[list[float]]]
    fixed_actor_targets: list[int]
    budget: float
    c: float
    utility_learning_rate: float
    following_learning_rate: float
    snapshot_interval: int
    actor_sharpness: float = 5.0

    @classmethod
    def from_mapping(cls, config: dict[str, Any]) -> "Paper61Config":
        case_name = config.get("case")
        if case_name != "paper_6_1_group":
            raise ValueError(f"expected paper_6_1_group config, got {case_name!r}")

        parsed = cls(
            seed=int(config["seed"]),
            num_steps=int(config["num_steps"]),
            num_actors=int(config["num_actors"]),
            num_observers=int(config["num_observers"]),
            num_states=int(config["num_states"]),
            num_actions=int(config["num_actions"]),
            state_distribution=[float(p) for p in config["state_distribution"]],
            actor_selection_probs=[float(p) for p in config["actor_selection_probs"]],
            observer_reward_tables=[
                [[float(r) for r in row] for row in table]
                for table in config["observer_reward_tables"]
            ],
            fixed_actor_targets=[int(t) for t in config["fixed_actor_targets"]],
            budget=float(config["budget"]),
            c=float(config["c"]),
            utility_learning_rate=float(config["utility_learning_rate"]),
            following_learning_rate=float(config["following_learning_rate"]),
            snapshot_interval=int(config.get("snapshot_interval", 1)),
            actor_sharpness=float(config.get("actor_sharpness", 5.0)),
        )
        parsed.validate()
        return parsed

    def validate(self) -> None:
        _validate_paper6_common(
            num_steps=self.num_steps,
            num_actors=self.num_actors,
            num_observers=self.num_observers,
            num_states=self.num_states,
            num_actions=self.num_actions,
            state_distribution=self.state_distribution,
            actor_selection_probs=self.actor_selection_probs,
            observer_reward_tables=self.observer_reward_tables,
            budget=self.budget,
            c=self.c,
            utility_learning_rate=self.utility_learning_rate,
            following_learning_rate=self.following_learning_rate,
            snapshot_interval=self.snapshot_interval,
        )
        if len(self.fixed_actor_targets) != self.num_actors:
            raise ValueError("fixed_actor_targets length must match num_actors")
        for target in self.fixed_actor_targets:
            if not 0 <= target < self.num_actions:
                raise ValueError("fixed_actor_targets entries must be valid action indices")
        if self.actor_sharpness <= 0.0:
            raise ValueError("actor_sharpness must be positive")


@dataclass(frozen=True)
class Paper62Config:
    """Typed config for paper Section 6.2 (group-level joint learning)."""

    seed: int
    num_steps: int
    num_actors: int
    num_observers: int
    num_states: int
    num_actions: int
    state_distribution: list[float]
    actor_selection_probs: list[float]
    observer_reward_tables: list[list[list[float]]]
    budget: float
    c: float
    utility_learning_rate: float
    following_learning_rate: float
    actor_learning_rate: float
    snapshot_interval: int
    optimal_actions: list[int]

    @classmethod
    def from_mapping(cls, config: dict[str, Any]) -> "Paper62Config":
        case_name = config.get("case")
        if case_name != "paper_6_2_group":
            raise ValueError(f"expected paper_6_2_group config, got {case_name!r}")

        parsed = cls(
            seed=int(config["seed"]),
            num_steps=int(config["num_steps"]),
            num_actors=int(config["num_actors"]),
            num_observers=int(config["num_observers"]),
            num_states=int(config["num_states"]),
            num_actions=int(config["num_actions"]),
            state_distribution=[float(p) for p in config["state_distribution"]],
            actor_selection_probs=[float(p) for p in config["actor_selection_probs"]],
            observer_reward_tables=[
                [[float(r) for r in row] for row in table]
                for table in config["observer_reward_tables"]
            ],
            budget=float(config["budget"]),
            c=float(config["c"]),
            utility_learning_rate=float(config["utility_learning_rate"]),
            following_learning_rate=float(config["following_learning_rate"]),
            actor_learning_rate=float(config["actor_learning_rate"]),
            snapshot_interval=int(config.get("snapshot_interval", 1)),
            optimal_actions=[int(a) for a in config["optimal_actions"]],
        )
        parsed.validate()
        return parsed

    def validate(self) -> None:
        _validate_paper6_common(
            num_steps=self.num_steps,
            num_actors=self.num_actors,
            num_observers=self.num_observers,
            num_states=self.num_states,
            num_actions=self.num_actions,
            state_distribution=self.state_distribution,
            actor_selection_probs=self.actor_selection_probs,
            observer_reward_tables=self.observer_reward_tables,
            budget=self.budget,
            c=self.c,
            utility_learning_rate=self.utility_learning_rate,
            following_learning_rate=self.following_learning_rate,
            snapshot_interval=self.snapshot_interval,
        )
        if self.actor_learning_rate <= 0.0:
            raise ValueError("actor_learning_rate must be positive")
        if len(self.optimal_actions) != self.num_states:
            raise ValueError("optimal_actions length must match num_states")
        for optimal_action in self.optimal_actions:
            if not 0 <= optimal_action < self.num_actions:
                raise ValueError("optimal_actions must be valid action indices")


def _validate_paper6_common(
    *,
    num_steps: int,
    num_actors: int,
    num_observers: int,
    num_states: int,
    num_actions: int,
    state_distribution: list[float],
    actor_selection_probs: list[float],
    observer_reward_tables: list[list[list[float]]],
    budget: float,
    c: float,
    utility_learning_rate: float,
    following_learning_rate: float,
    snapshot_interval: int,
) -> None:
    if num_observers < 1:
        raise ValueError("num_observers must be at least one")
    if len(observer_reward_tables) != num_observers:
        raise ValueError("observer_reward_tables length must match num_observers")
    for table in observer_reward_tables:
        _validate_paper4_common(
            num_steps=num_steps,
            num_actors=num_actors,
            num_states=num_states,
            num_actions=num_actions,
            state_distribution=state_distribution,
            actor_selection_probs=actor_selection_probs,
            reward_table=table,
            budget=budget,
            c=c,
            utility_learning_rate=utility_learning_rate,
            following_learning_rate=following_learning_rate,
            snapshot_interval=snapshot_interval,
        )


@dataclass(frozen=True)
class Paper71Config:
    """Typed config for paper Section 7.1 group Observer selection."""

    seed: int
    num_steps: int
    eval_interval: int
    reward_window: int
    num_actors: int
    num_observers: int
    num_states: int
    observation_dim: int
    num_actions: int
    actor_selection_probs: list[float]
    state_distribution: list[float]
    reward_table: list[list[float]]
    observer_names: list[str]
    observer_messages: list[list[float]]
    following_learning_rate: float
    decoder_sharpness: float

    @classmethod
    def from_mapping(cls, config: dict[str, Any]) -> "Paper71Config":
        case_name = config.get("case")
        if case_name != "paper_7_1_group_observer_selection":
            raise ValueError(
                f"expected paper_7_1_group_observer_selection config, got {case_name!r}"
            )

        parsed = cls(
            seed=int(config["seed"]),
            num_steps=int(config["num_steps"]),
            eval_interval=int(config.get("eval_interval", 1)),
            reward_window=int(config.get("reward_window", 100)),
            num_actors=int(config["num_actors"]),
            num_observers=int(config["num_observers"]),
            num_states=int(config["num_states"]),
            observation_dim=int(config["observation_dim"]),
            num_actions=int(config["num_actions"]),
            actor_selection_probs=[float(p) for p in config["actor_selection_probs"]],
            state_distribution=[float(p) for p in config["state_distribution"]],
            reward_table=[[float(r) for r in row] for row in config["reward_table"]],
            observer_names=[str(name) for name in config["observer_names"]],
            observer_messages=[
                [float(value) for value in row] for row in config["observer_messages"]
            ],
            following_learning_rate=float(config["following_learning_rate"]),
            decoder_sharpness=float(config["decoder_sharpness"]),
        )
        parsed.validate()
        return parsed

    def validate(self) -> None:
        _validate_paper7_common(
            num_steps=self.num_steps,
            eval_interval=self.eval_interval,
            reward_window=self.reward_window,
            num_actors=self.num_actors,
            num_observers=self.num_observers,
            num_states=self.num_states,
            observation_dim=self.observation_dim,
            num_actions=self.num_actions,
            actor_selection_probs=self.actor_selection_probs,
            state_distribution=self.state_distribution,
            reward_table=self.reward_table,
            observer_names=self.observer_names,
            following_learning_rate=self.following_learning_rate,
        )
        if self.decoder_sharpness <= 0.0:
            raise ValueError("decoder_sharpness must be positive")
        if len(self.observer_messages) != self.num_observers:
            raise ValueError("observer_messages length must match num_observers")
        for row in self.observer_messages:
            if len(row) != self.num_states:
                raise ValueError("each observer_messages row must match num_states")


@dataclass(frozen=True)
class Paper72Config:
    """Typed config for paper Section 7.2 group selection/compression."""

    seed: int
    num_steps: int
    eval_interval: int
    reward_window: int
    num_actors: int
    num_observers: int
    num_states: int
    state_dim: int
    observation_dim: int
    num_actions: int
    actor_selection_probs: list[float]
    state_features: list[list[float]]
    state_distribution: list[float]
    reward_table: list[list[float]]
    observer_names: list[str]
    observer_input_masks: list[list[float]]
    following_learning_rate: float
    observer_learning_rate: float
    decoder_sharpness: float

    @classmethod
    def from_mapping(cls, config: dict[str, Any]) -> "Paper72Config":
        case_name = config.get("case")
        if case_name != "paper_7_2_group_selection_compression":
            raise ValueError(
                f"expected paper_7_2_group_selection_compression config, got {case_name!r}"
            )

        parsed = cls(
            seed=int(config["seed"]),
            num_steps=int(config["num_steps"]),
            eval_interval=int(config.get("eval_interval", 1)),
            reward_window=int(config.get("reward_window", 100)),
            num_actors=int(config["num_actors"]),
            num_observers=int(config["num_observers"]),
            num_states=int(config["num_states"]),
            state_dim=int(config["state_dim"]),
            observation_dim=int(config["observation_dim"]),
            num_actions=int(config["num_actions"]),
            actor_selection_probs=[float(p) for p in config["actor_selection_probs"]],
            state_features=[
                [float(value) for value in row] for row in config["state_features"]
            ],
            state_distribution=[float(p) for p in config["state_distribution"]],
            reward_table=[[float(r) for r in row] for row in config["reward_table"]],
            observer_names=[str(name) for name in config["observer_names"]],
            observer_input_masks=[
                [float(value) for value in row]
                for row in config["observer_input_masks"]
            ],
            following_learning_rate=float(config["following_learning_rate"]),
            observer_learning_rate=float(config["observer_learning_rate"]),
            decoder_sharpness=float(config["decoder_sharpness"]),
        )
        parsed.validate()
        return parsed

    def validate(self) -> None:
        _validate_paper7_common(
            num_steps=self.num_steps,
            eval_interval=self.eval_interval,
            reward_window=self.reward_window,
            num_actors=self.num_actors,
            num_observers=self.num_observers,
            num_states=self.num_states,
            observation_dim=self.observation_dim,
            num_actions=self.num_actions,
            actor_selection_probs=self.actor_selection_probs,
            state_distribution=self.state_distribution,
            reward_table=self.reward_table,
            observer_names=self.observer_names,
            following_learning_rate=self.following_learning_rate,
        )
        _validate_paper7_feature_fields(
            num_states=self.num_states,
            state_dim=self.state_dim,
            state_features=self.state_features,
            observer_input_masks=self.observer_input_masks,
            num_observers=self.num_observers,
        )
        if self.observer_learning_rate <= 0.0:
            raise ValueError("observer_learning_rate must be positive")
        if self.decoder_sharpness <= 0.0:
            raise ValueError("decoder_sharpness must be positive")


@dataclass(frozen=True)
class Paper73Config:
    """Typed config for paper Section 7.3 group joint imperfect-state learning."""

    seed: int
    num_steps: int
    eval_interval: int
    reward_window: int
    num_actors: int
    num_observers: int
    num_states: int
    state_dim: int
    observation_dim: int
    num_actions: int
    actor_selection_probs: list[float]
    state_features: list[list[float]]
    state_distribution: list[float]
    reward_table: list[list[float]]
    observer_names: list[str]
    observer_input_masks: list[list[float]]
    following_learning_rate: float
    observer_learning_rate: float
    actor_learning_rate: float

    @classmethod
    def from_mapping(cls, config: dict[str, Any]) -> "Paper73Config":
        case_name = config.get("case")
        if case_name != "paper_7_3_group_joint_imperfect_state":
            raise ValueError(
                f"expected paper_7_3_group_joint_imperfect_state config, got {case_name!r}"
            )

        parsed = cls(
            seed=int(config["seed"]),
            num_steps=int(config["num_steps"]),
            eval_interval=int(config.get("eval_interval", 1)),
            reward_window=int(config.get("reward_window", 100)),
            num_actors=int(config["num_actors"]),
            num_observers=int(config["num_observers"]),
            num_states=int(config["num_states"]),
            state_dim=int(config["state_dim"]),
            observation_dim=int(config["observation_dim"]),
            num_actions=int(config["num_actions"]),
            actor_selection_probs=[float(p) for p in config["actor_selection_probs"]],
            state_features=[
                [float(value) for value in row] for row in config["state_features"]
            ],
            state_distribution=[float(p) for p in config["state_distribution"]],
            reward_table=[[float(r) for r in row] for row in config["reward_table"]],
            observer_names=[str(name) for name in config["observer_names"]],
            observer_input_masks=[
                [float(value) for value in row]
                for row in config["observer_input_masks"]
            ],
            following_learning_rate=float(config["following_learning_rate"]),
            observer_learning_rate=float(config["observer_learning_rate"]),
            actor_learning_rate=float(config["actor_learning_rate"]),
        )
        parsed.validate()
        return parsed

    def validate(self) -> None:
        _validate_paper7_common(
            num_steps=self.num_steps,
            eval_interval=self.eval_interval,
            reward_window=self.reward_window,
            num_actors=self.num_actors,
            num_observers=self.num_observers,
            num_states=self.num_states,
            observation_dim=self.observation_dim,
            num_actions=self.num_actions,
            actor_selection_probs=self.actor_selection_probs,
            state_distribution=self.state_distribution,
            reward_table=self.reward_table,
            observer_names=self.observer_names,
            following_learning_rate=self.following_learning_rate,
        )
        _validate_paper7_feature_fields(
            num_states=self.num_states,
            state_dim=self.state_dim,
            state_features=self.state_features,
            observer_input_masks=self.observer_input_masks,
            num_observers=self.num_observers,
        )
        if self.observer_learning_rate <= 0.0:
            raise ValueError("observer_learning_rate must be positive")
        if self.actor_learning_rate <= 0.0:
            raise ValueError("actor_learning_rate must be positive")


def _validate_paper7_common(
    *,
    num_steps: int,
    eval_interval: int,
    reward_window: int,
    num_actors: int,
    num_observers: int,
    num_states: int,
    observation_dim: int,
    num_actions: int,
    actor_selection_probs: list[float],
    state_distribution: list[float],
    reward_table: list[list[float]],
    observer_names: list[str],
    following_learning_rate: float,
) -> None:
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    if eval_interval <= 0:
        raise ValueError("eval_interval must be positive")
    if reward_window <= 0:
        raise ValueError("reward_window must be positive")
    if num_actors < 2:
        raise ValueError("num_actors must be at least two")
    if num_observers < 2:
        raise ValueError("num_observers must be at least two")
    if num_states <= 0:
        raise ValueError("num_states must be positive")
    if observation_dim <= 0:
        raise ValueError("observation_dim must be positive")
    if num_actions <= 1:
        raise ValueError("num_actions must be greater than one")
    if following_learning_rate <= 0.0:
        raise ValueError("following_learning_rate must be positive")
    if len(actor_selection_probs) != num_actors:
        raise ValueError("actor_selection_probs length must match num_actors")
    if any(p < 0.0 for p in actor_selection_probs):
        raise ValueError("actor_selection_probs cannot contain negative probabilities")
    if not math.isclose(
        sum(actor_selection_probs),
        1.0,
        rel_tol=1e-6,
        abs_tol=1e-6,
    ):
        raise ValueError("actor_selection_probs must sum to 1")
    if len(state_distribution) != num_states:
        raise ValueError("state_distribution length must match num_states")
    if any(p < 0.0 for p in state_distribution):
        raise ValueError("state_distribution cannot contain negative probabilities")
    if not math.isclose(sum(state_distribution), 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("state_distribution must sum to 1")
    if len(reward_table) != num_states:
        raise ValueError("reward_table row count must match num_states")
    if any(len(row) != num_actions for row in reward_table):
        raise ValueError("each reward_table row length must match num_actions")
    if len(observer_names) != num_observers:
        raise ValueError("observer_names length must match num_observers")
    if len(set(observer_names)) != len(observer_names):
        raise ValueError("observer_names must be unique")


def _validate_paper7_feature_fields(
    *,
    num_states: int,
    state_dim: int,
    state_features: list[list[float]],
    observer_input_masks: list[list[float]],
    num_observers: int,
) -> None:
    if state_dim <= 0:
        raise ValueError("state_dim must be positive")
    if len(state_features) != num_states:
        raise ValueError("state_features row count must match num_states")
    if any(len(row) != state_dim for row in state_features):
        raise ValueError("each state_features row length must match state_dim")
    if len(observer_input_masks) != num_observers:
        raise ValueError("observer_input_masks length must match num_observers")
    if any(len(row) != state_dim for row in observer_input_masks):
        raise ValueError("each observer_input_masks row length must match state_dim")

def _validate_paper4_common(
    *,
    num_steps: int,
    num_actors: int,
    num_states: int,
    num_actions: int,
    state_distribution: list[float],
    actor_selection_probs: list[float],
    reward_table: list[list[float]],
    budget: float,
    c: float,
    utility_learning_rate: float,
    following_learning_rate: float,
    snapshot_interval: int,
) -> None:
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    if num_actors < 2:
        raise ValueError("num_actors must be at least two")
    if num_states <= 0:
        raise ValueError("num_states must be positive")
    if num_actions <= 1:
        raise ValueError("num_actions must be greater than one")
    if budget <= 0.0:
        raise ValueError("budget must be positive")
    if c <= 0.0:
        raise ValueError("c must be positive")
    if utility_learning_rate <= 0.0:
        raise ValueError("utility_learning_rate must be positive")
    if following_learning_rate <= 0.0:
        raise ValueError("following_learning_rate must be positive")
    if snapshot_interval <= 0:
        raise ValueError("snapshot_interval must be positive")
    if len(state_distribution) != num_states:
        raise ValueError("state_distribution length must match num_states")
    if any(p < 0.0 for p in state_distribution):
        raise ValueError("state_distribution cannot contain negative probabilities")
    if not math.isclose(sum(state_distribution), 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("state_distribution must sum to 1")
    if len(actor_selection_probs) != num_actors:
        raise ValueError("actor_selection_probs length must match num_actors")
    if any(p < 0.0 for p in actor_selection_probs):
        raise ValueError("actor_selection_probs cannot contain negative probabilities")
    if not math.isclose(sum(actor_selection_probs), 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("actor_selection_probs must sum to 1")
    if len(reward_table) != num_states:
        raise ValueError("reward_table row count must match num_states")
    if any(len(row) != num_actions for row in reward_table):
        raise ValueError("each reward_table row length must match num_actions")


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        raise ValueError(f"config at {config_path} must be a YAML mapping")
    return config
