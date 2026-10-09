"""Run controlled scaling checks for pre-Section-8 mechanisms.

The meeting notes ask for scaling to be done gradually and with structure in
the state/reward design. This script keeps that discipline: each check scales
one already-tested mechanism while preserving an interpretable target.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict, deque
from pathlib import Path
from statistics import fmean
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path("/private/tmp/actor_observer_rl_mplcache").resolve()),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from actor_observer_rl.agents import (
    Actor,
    BudgetedObserver,
    FixedTableMessageModule,
    FollowingRates,
    MaskedObserverMessageModule,
    MultiObserver,
    Observer,
    ObserverMessageModule,
)
from actor_observer_rl.envs import FeaturePartialObservableStateEnv, IIDStateBanditEnv, PartialObservableStateEnv
from actor_observer_rl.policies import NeuralPolicy, SumDecoderPolicy
from actor_observer_rl.training import GroupImperfectStateTrainer, GroupMultiActorTrainer, StagedMultiObserverTrainer, make_fixed_softmax_actor
from actor_observer_rl.training.updates import reinforce_loss
from actor_observer_rl.utils.seeding import set_global_seed
from experiments.run_paper_7 import save_figure


class QuadrantDecoderPolicy(torch.nn.Module):
    """Fixed decoder mapping 2-D messages near quadrant centers to actions."""

    def __init__(self, sharpness: float = 12.0) -> None:
        super().__init__()
        self.sharpness = float(sharpness)
        self.register_buffer(
            "centers",
            torch.tensor(
                [
                    [0.25, 0.25],
                    [0.75, 0.25],
                    [0.25, 0.75],
                    [0.75, 0.75],
                ],
                dtype=torch.float32,
            ),
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        message = input_tensor.float().reshape(-1)
        if message.numel() != 2:
            raise ValueError("QuadrantDecoderPolicy expects a 2-D message")
        distances = ((self.centers - message) ** 2).sum(dim=-1)
        return -self.sharpness * distances

    def action_distribution(
        self,
        input_tensor: torch.Tensor,
    ) -> torch.distributions.Categorical:
        return torch.distributions.Categorical(logits=self(input_tensor))

    def act(self, input_tensor: torch.Tensor, sample: bool = True):
        distribution = self.action_distribution(input_tensor)
        action = distribution.sample() if sample else distribution.probs.argmax(dim=-1)
        from actor_observer_rl.agents.actor import ActionResult

        return ActionResult(
            action=action,
            log_prob=distribution.log_prob(action),
            probs=distribution.probs,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--state-grid-sizes", type=int, nargs="+", default=[8, 16])
    parser.add_argument("--sum-feature-counts", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--agent-sizes", type=int, nargs="+", default=[3, 5, 8])
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/previous_cases_scaling"))
    parser.add_argument("--figure-dir", type=Path, default=Path("docs/figures"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    rows.extend(run_policy_grid_scaling(args.state_grid_sizes, args.seeds))
    rows.extend(run_single_observer_compression_scaling(args.sum_feature_counts, args.seeds))
    rows.extend(run_multi_observer_compression_scaling(args.sum_feature_counts, args.seeds))
    rows.extend(run_reward_following_agent_scaling(args.agent_sizes, args.seeds))
    rows.extend(run_group_state_selection_actor_scaling(args.agent_sizes, args.seeds))

    summaries = summarize(rows)
    write_csv(args.output_dir / "previous_cases_scaling_curves.csv", rows)
    write_csv(args.output_dir / "previous_cases_scaling_summary.csv", summaries)
    write_state_figure(rows, args.figure_dir / "previous_scaling_state_ladder")
    write_agent_figure(rows, args.figure_dir / "previous_scaling_agent_ladder")

    for row in summaries:
        print(
            "{stage} size={size}: final_expected={final_expected_reward:.3f} "
            "mechanism={final_mechanism_score:.3f}".format(**row)
        )


def run_policy_grid_scaling(grid_sizes: list[int], seeds: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for grid_size in grid_sizes:
        for seed in seeds:
            rows.extend(
                run_policy_grid_one(
                    grid_size=grid_size,
                    seed=seed,
                    num_steps=60_000,
                    eval_interval=600,
                    learning_rate=0.003,
                    hidden_dim=48,
                )
            )
    return rows


def run_policy_grid_one(
    *,
    grid_size: int,
    seed: int,
    num_steps: int,
    eval_interval: int,
    learning_rate: float,
    hidden_dim: int,
) -> list[dict[str, Any]]:
    set_global_seed(seed)
    states, optimal_actions = build_grid_task(grid_size)
    reward_table = build_reward_table(optimal_actions, num_actions=4)
    actor = Actor(NeuralPolicy(input_dim=2, hidden_dim=hidden_dim, num_actions=4))
    optimizer = torch.optim.Adam(actor.parameters(), lr=learning_rate)
    generator = seeded_generator(seed, "policy_grid")
    reward_window: deque[float] = deque(maxlen=100)
    representative = representative_state_indices(grid_size)
    rows: list[dict[str, Any]] = []

    def record(step: int) -> None:
        expected, optimal_prob, min_rep = evaluate_actor_on_grid(
            actor,
            states,
            reward_table,
            optimal_actions,
            representative,
        )
        rows.append(
            base_row(
                stage="fully_observed_policy",
                size=grid_size * grid_size,
                seed=seed,
                step=step,
                expected_reward=expected,
                sampled_reward=fmean(reward_window) if reward_window else 0.0,
                mechanism_score=optimal_prob,
                secondary_score=min_rep,
            )
        )

    record(0)
    for step_index in range(num_steps):
        state_index = sample_index(len(states), generator)
        action_result = actor.act(states[state_index], sample=True)
        action = int(action_result.action.detach().cpu().item())
        reward = reward_table[state_index][action]
        loss = reinforce_loss(reward, action_result.log_prob)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        reward_window.append(float(reward))
        step = step_index + 1
        if step % eval_interval == 0 or step == num_steps:
            record(step)
    return rows


def run_single_observer_compression_scaling(
    feature_counts: list[int],
    seeds: list[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for feature_count in feature_counts:
        for seed in seeds:
            rows.extend(
                run_single_observer_compression_one(
                    feature_count=feature_count,
                    seed=seed,
                    num_steps=50_000,
                    eval_interval=500,
                    learning_rate=0.005,
                )
            )
    return rows


def run_single_observer_compression_one(
    *,
    feature_count: int,
    seed: int,
    num_steps: int,
    eval_interval: int,
    learning_rate: float,
) -> list[dict[str, Any]]:
    set_global_seed(seed)
    states, optimal_actions = build_binary_sum_task(feature_count)
    reward_table = build_reward_table(
        optimal_actions,
        num_actions=feature_count + 1,
    )
    observer = Observer(ObserverMessageModule(input_dim=feature_count, message_dim=1))
    initialize_message_module(observer, scale=0.05)
    actor = Actor(SumDecoderPolicy(num_actions=feature_count + 1, sharpness=4.0))
    optimizer = torch.optim.Adam(observer.parameters(), lr=learning_rate)
    generator = seeded_generator(seed, "single_observer_compression")
    reward_window: deque[float] = deque(maxlen=100)
    representative = representative_binary_sum_indices(feature_count)
    rows: list[dict[str, Any]] = []

    def record(step: int) -> None:
        expected, optimal_prob, min_rep = evaluate_observer_decoder_on_grid(
            observer,
            actor,
            states,
            reward_table,
            optimal_actions,
            representative,
        )
        sum_score = sum_r2(observer, states, optimal_actions)
        rows.append(
            base_row(
                stage="single_observer_sum_compression",
                size=2**feature_count,
                seed=seed,
                step=step,
                expected_reward=expected,
                sampled_reward=fmean(reward_window) if reward_window else 0.0,
                mechanism_score=sum_score,
                secondary_score=min_rep,
                optimal_action_probability=optimal_prob,
            )
        )

    record(0)
    for step_index in range(num_steps):
        state_index = sample_index(len(states), generator)
        message = observer.message(states[state_index])
        action_result = actor.act(message, sample=True)
        action = int(action_result.action.detach().cpu().item())
        reward = reward_table[state_index][action]
        loss = reinforce_loss(reward, action_result.log_prob)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        reward_window.append(float(reward))
        step = step_index + 1
        if step % eval_interval == 0 or step == num_steps:
            record(step)
    return rows


def run_multi_observer_compression_scaling(
    feature_counts: list[int],
    seeds: list[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for feature_count in feature_counts:
        for seed in seeds:
            rows.extend(
                run_multi_observer_compression_one(
                    feature_count=feature_count,
                    seed=seed,
                    num_steps=50_000,
                    eval_interval=500,
                )
            )
    return rows


def run_multi_observer_compression_one(
    *,
    feature_count: int,
    seed: int,
    num_steps: int,
    eval_interval: int,
) -> list[dict[str, Any]]:
    set_global_seed(seed)
    state_features_tensor, optimal_actions = build_binary_sum_task(feature_count)
    state_features = state_features_tensor.tolist()
    reward_table = build_reward_table(
        optimal_actions,
        num_actions=feature_count + 1,
    )
    env = FeaturePartialObservableStateEnv(
        state_features=state_features,
        reward_table=reward_table,
        state_distribution=[1.0 / len(state_features)] * len(state_features),
        observation_dim=1,
    )
    partial_mask = [1.0] + [0.0] * (feature_count - 1)
    observers = [
        masked_observer(input_dim=feature_count, message_dim=1, mask=[0.0] * feature_count),
        masked_observer(input_dim=feature_count, message_dim=1, mask=partial_mask),
        masked_observer(input_dim=feature_count, message_dim=1, mask=[1.0] * feature_count),
    ]
    actor = Actor(SumDecoderPolicy(num_actions=feature_count + 1, sharpness=4.0))
    multi_observer = MultiObserver(
        observers=observers,
        following_rates=FollowingRates(num_observers=3),
    )
    trainer = StagedMultiObserverTrainer(
        env=env,
        actor=actor,
        multi_observer=multi_observer,
        following_optimizer=torch.optim.SGD(
            list(multi_observer.following_parameters()),
            lr=0.002,
        ),
        observer_optimizer=torch.optim.Adam(
            list(multi_observer.observer_parameters()),
            lr=0.01,
        ),
        optimal_actions=optimal_actions,
    )
    trace = trainer.run(
        num_steps=num_steps,
        seed=seed,
        eval_interval=eval_interval,
        reward_window=100,
    )
    rows: list[dict[str, Any]] = []
    for eval_index, step in enumerate(trace.evaluation_steps):
        full_probability = (
            1.0 / 3.0
            if step == 0
            else trace.following_weights[min(step - 1, len(trace.following_weights) - 1)][2]
        )
        full_messages = trace.observer_messages_by_state[eval_index][2]
        full_score = sum_r2_from_messages(full_messages, optimal_actions)
        rows.append(
            base_row(
                stage="multi_observer_sum_state",
                size=2**feature_count,
                seed=seed,
                step=step,
                expected_reward=trace.expected_rewards[eval_index],
                sampled_reward=trace.average_rewards[eval_index],
                mechanism_score=full_probability,
                secondary_score=full_score,
                optimal_action_probability=trace.mean_optimal_action_probs[eval_index],
            )
        )
    return rows


def run_reward_following_agent_scaling(
    agent_sizes: list[int],
    seeds: list[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for agent_count in agent_sizes:
        for seed in seeds:
            rows.extend(
                run_reward_following_one(
                    agent_count=agent_count,
                    seed=seed,
                    num_steps=18_000,
                    snapshot_interval=180,
                )
            )
    return rows


def run_reward_following_one(
    *,
    agent_count: int,
    seed: int,
    num_steps: int,
    snapshot_interval: int,
) -> list[dict[str, Any]]:
    set_global_seed(seed)
    observer_reward_tables = reward_following_tables(agent_count)
    env = IIDStateBanditEnv(
        reward_table=observer_reward_tables[0],
        state_distribution=[1.0],
        observation_mode="one_hot",
    )
    actors = [
        make_fixed_softmax_actor(
            num_states=1,
            num_actions=agent_count,
            target_action=actor_index,
            sharpness=8.0,
        )
        for actor_index in range(agent_count)
    ]
    observers = [
        BudgetedObserver(
            num_actors=agent_count,
            budget=2.0,
            utility_learning_rate=0.2,
            following_learning_rate=0.02,
            c=2.0,
            init_mu=[2.0 / agent_count] * agent_count,
            seed=seed + observer_index,
        )
        for observer_index in range(agent_count)
    ]
    trainer = GroupMultiActorTrainer(
        env=env,
        actors=actors,
        observers=observers,
        observer_reward_tables=observer_reward_tables,
        actor_selection_probs=[1.0 / agent_count] * agent_count,
        actor_optimizers=None,
        snapshot_interval=snapshot_interval,
    )
    trace = trainer.run(num_steps=num_steps, seed=seed)
    rows: list[dict[str, Any]] = []
    for snapshot_index, following_rates in enumerate(trace.following_rates):
        step = (snapshot_index + 1) * snapshot_interval
        preferred_top = 0
        preferred_share_sum = 0.0
        utility_error_sum = 0.0
        for observer_index, rates in enumerate(following_rates):
            top_actor = max(range(agent_count), key=lambda index: rates[index])
            if top_actor == observer_index:
                preferred_top += 1
            preferred_share_sum += rates[observer_index] / max(sum(rates), 1e-12)
            true_utilities = observer_reward_tables[observer_index][0]
            estimates = trace.utility_estimates[snapshot_index][observer_index]
            utility_error_sum += max(
                abs(estimates[index] - true_utilities[index])
                for index in range(agent_count)
            )
        rows.append(
            base_row(
                stage="reward_following_agents",
                size=agent_count,
                seed=seed,
                step=step,
                expected_reward=preferred_top / agent_count,
                sampled_reward=preferred_share_sum / agent_count,
                mechanism_score=preferred_top / agent_count,
                secondary_score=1.0 - min(1.0, utility_error_sum / agent_count),
            )
        )
    return rows


def run_group_state_selection_actor_scaling(
    agent_sizes: list[int],
    seeds: list[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for actor_count in agent_sizes:
        for seed in seeds:
            rows.extend(
                run_group_state_selection_one(
                    actor_count=actor_count,
                    seed=seed,
                    num_steps=16_000,
                    eval_interval=160,
                )
            )
    return rows


def run_group_state_selection_one(
    *,
    actor_count: int,
    seed: int,
    num_steps: int,
    eval_interval: int,
) -> list[dict[str, Any]]:
    set_global_seed(seed)
    reward_table = [
        [1.0, -0.1, -0.1],
        [-0.1, 1.0, -0.1],
        [-0.1, -0.1, 1.0],
    ]
    env = PartialObservableStateEnv(
        reward_table=reward_table,
        state_distribution=[1 / 3, 1 / 3, 1 / 3],
        observation_dim=1,
    )
    observers = [
        Observer(message_module=FixedTableMessageModule([0.0, 0.0, 0.0])),
        Observer(message_module=FixedTableMessageModule([0.0, 1.0, 1.0])),
        Observer(message_module=FixedTableMessageModule([0.0, 1.0, 2.0])),
    ]
    actors = [
        Actor(SumDecoderPolicy(num_actions=3, sharpness=4.0))
        for _ in range(actor_count)
    ]
    rates_by_actor = [FollowingRates(num_observers=3) for _ in range(actor_count)]
    trainer = GroupImperfectStateTrainer(
        env=env,
        actors=actors,
        observers=observers,
        following_rates_by_actor=rates_by_actor,
        actor_selection_probs=[1.0 / actor_count] * actor_count,
        following_optimizer=torch.optim.SGD(
            [parameter for rates in rates_by_actor for parameter in rates.parameters()],
            lr=0.02,
        ),
        optimal_actions=[0, 1, 2],
    )
    trace = trainer.run(
        num_steps=num_steps,
        seed=seed,
        eval_interval=eval_interval,
        reward_window=100,
    )
    rows: list[dict[str, Any]] = []
    for eval_index, step in enumerate(trace.evaluation_steps):
        full_probs = [
            actor_weights[2]
            for actor_weights in trace.selection_weights_by_actor[eval_index]
        ]
        rows.append(
            base_row(
                stage="group_state_selectors",
                size=actor_count,
                seed=seed,
                step=step,
                expected_reward=trace.expected_rewards[eval_index],
                sampled_reward=trace.average_rewards[eval_index],
                mechanism_score=fmean(full_probs),
                secondary_score=min(full_probs),
                optimal_action_probability=trace.mean_optimal_action_probs[eval_index],
            )
        )
    return rows


def base_row(
    *,
    stage: str,
    size: int,
    seed: int,
    step: int,
    expected_reward: float,
    sampled_reward: float,
    mechanism_score: float,
    secondary_score: float,
    optimal_action_probability: float | None = None,
) -> dict[str, Any]:
    return {
        "stage": stage,
        "size": int(size),
        "seed": int(seed),
        "step": int(step),
        "expected_reward": float(expected_reward),
        "sampled_reward": float(sampled_reward),
        "mechanism_score": float(mechanism_score),
        "secondary_score": float(secondary_score),
        "optimal_action_probability": (
            "" if optimal_action_probability is None else float(optimal_action_probability)
        ),
    }


def build_grid_task(grid_size: int) -> tuple[torch.Tensor, list[int]]:
    state_features, actions = build_grid_task_lists(grid_size)
    return torch.tensor(state_features, dtype=torch.float32), actions


def build_binary_sum_task(feature_count: int) -> tuple[torch.Tensor, list[int]]:
    if feature_count <= 0:
        raise ValueError("feature_count must be positive")
    states: list[list[float]] = []
    actions: list[int] = []
    for state_index in range(2**feature_count):
        bits = [
            float((state_index >> bit_index) & 1)
            for bit_index in range(feature_count)
        ]
        states.append(bits)
        actions.append(int(sum(bits)))
    return torch.tensor(states, dtype=torch.float32), actions


def build_grid_task_lists(grid_size: int) -> tuple[list[list[float]], list[int]]:
    states: list[list[float]] = []
    actions: list[int] = []
    denom = float(grid_size - 1)
    for y_index in range(grid_size):
        for x_index in range(grid_size):
            x_value = x_index / denom
            y_value = y_index / denom
            states.append([x_value, y_value])
            actions.append(region_action(x_value, y_value))
    return states, actions


def region_action(x_value: float, y_value: float) -> int:
    return (1 if x_value >= 0.5 else 0) + 2 * (1 if y_value >= 0.5 else 0)


def build_reward_table(optimal_actions: list[int], num_actions: int) -> list[list[float]]:
    table: list[list[float]] = []
    for action in optimal_actions:
        rewards = [-0.1] * num_actions
        rewards[action] = 1.0
        table.append(rewards)
    return table


def representative_state_indices(grid_size: int) -> dict[str, int]:
    def index(x_index: int, y_index: int) -> int:
        return y_index * grid_size + x_index

    last = grid_size - 1
    mid_low = max(0, grid_size // 2 - 1)
    mid_high = min(last, grid_size // 2)
    return {
        "lower_left": index(0, 0),
        "lower_right": index(last, 0),
        "upper_left": index(0, last),
        "upper_right": index(last, last),
        "boundary_low": index(mid_low, mid_low),
        "boundary_high": index(mid_high, mid_high),
    }


def representative_binary_sum_indices(feature_count: int) -> dict[str, int]:
    return {
        "all_zero": 0,
        "first_feature": 1,
        "all_one": 2**feature_count - 1,
    }


def evaluate_actor_on_grid(
    actor: Actor,
    states: torch.Tensor,
    reward_table: list[list[float]],
    optimal_actions: list[int],
    representatives: dict[str, int],
) -> tuple[float, float, float]:
    expected = 0.0
    optimal_prob = 0.0
    representative_probs = []
    with torch.no_grad():
        for state_index, state in enumerate(states):
            probs = actor.policy.action_distribution(state).probs.detach().cpu()
            rewards = torch.tensor(reward_table[state_index], dtype=torch.float32)
            expected += float((probs * rewards).sum().item())
            optimal_prob += float(probs[optimal_actions[state_index]].item())
        for state_index in representatives.values():
            probs = actor.policy.action_distribution(states[state_index]).probs
            representative_probs.append(float(probs[optimal_actions[state_index]].item()))
    count = len(states)
    return expected / count, optimal_prob / count, min(representative_probs)


def evaluate_observer_decoder_on_grid(
    observer: Observer,
    actor: Actor,
    states: torch.Tensor,
    reward_table: list[list[float]],
    optimal_actions: list[int],
    representatives: dict[str, int],
) -> tuple[float, float, float]:
    expected = 0.0
    optimal_prob = 0.0
    representative_probs = []
    with torch.no_grad():
        for state_index, state in enumerate(states):
            message = observer.message(state)
            probs = actor.policy.action_distribution(message).probs.detach().cpu()
            rewards = torch.tensor(reward_table[state_index], dtype=torch.float32)
            expected += float((probs * rewards).sum().item())
            optimal_prob += float(probs[optimal_actions[state_index]].item())
        for state_index in representatives.values():
            message = observer.message(states[state_index])
            probs = actor.policy.action_distribution(message).probs
            representative_probs.append(float(probs[optimal_actions[state_index]].item()))
    count = len(states)
    return expected / count, optimal_prob / count, min(representative_probs)


def coordinate_r2(observer: Observer, states: torch.Tensor) -> float:
    with torch.no_grad():
        messages = [
            observer.message(state).detach().cpu().reshape(-1).tolist()
            for state in states
        ]
    return coordinate_r2_from_messages(messages, states.detach().cpu().tolist())


def sum_r2(
    observer: Observer,
    states: torch.Tensor,
    target_sums: list[int],
) -> float:
    with torch.no_grad():
        messages = [
            observer.message(state).detach().cpu().reshape(-1).tolist()
            for state in states
        ]
    return sum_r2_from_messages(messages, target_sums)


def sum_r2_from_messages(
    messages: list[list[float]],
    target_sums: list[int],
) -> float:
    values = [float(message[0]) for message in messages]
    return r_squared(values, [float(value) for value in target_sums])


def coordinate_r2_from_messages(
    messages: list[list[float]],
    state_features: list[list[float]],
) -> float:
    if not messages:
        return 0.0
    message_columns = list(zip(*messages, strict=True))
    state_columns = list(zip(*state_features, strict=True))
    if len(message_columns) < 2 or len(state_columns) < 2:
        return 0.0
    direct = 0.5 * (
        r_squared(list(message_columns[0]), list(state_columns[0]))
        + r_squared(list(message_columns[1]), list(state_columns[1]))
    )
    swapped = 0.5 * (
        r_squared(list(message_columns[0]), list(state_columns[1]))
        + r_squared(list(message_columns[1]), list(state_columns[0]))
    )
    return max(direct, swapped)


def r_squared(values: list[float], targets: list[float]) -> float:
    if len(values) != len(targets):
        raise ValueError("values and targets must have the same length")
    target_mean = fmean(targets)
    value_mean = fmean(values)
    target_ss = sum((target - target_mean) ** 2 for target in targets)
    value_ss = sum((value - value_mean) ** 2 for value in values)
    if target_ss <= 1e-12 or value_ss <= 1e-12:
        return 0.0
    covariance = sum(
        (value - value_mean) * (target - target_mean)
        for value, target in zip(values, targets, strict=True)
    )
    return max(0.0, min(1.0, covariance**2 / (target_ss * value_ss)))


def masked_observer(
    *,
    input_dim: int,
    message_dim: int,
    mask: list[float],
) -> Observer:
    module = MaskedObserverMessageModule(
        input_dim=input_dim,
        message_dim=message_dim,
        input_mask=mask,
    )
    with torch.no_grad():
        module.linear.weight.normal_(mean=0.0, std=0.05)
        module.linear.bias.zero_()
    return Observer(message_module=module)


def initialize_message_module(observer: Observer, scale: float) -> None:
    module = observer.message_module
    if module is None or not hasattr(module, "linear"):
        return
    with torch.no_grad():
        module.linear.weight.normal_(mean=0.0, std=scale)
        module.linear.bias.zero_()


def reward_following_tables(agent_count: int) -> list[list[list[float]]]:
    tables: list[list[list[float]]] = []
    for observer_index in range(agent_count):
        utilities = [0.15] * agent_count
        utilities[observer_index] = 1.0
        utilities[(observer_index + 1) % agent_count] = 0.45
        tables.append([utilities])
    return tables


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    final_by_stage = {}
    for row in rows:
        key = (str(row["stage"]), int(row["size"]))
        if key not in final_by_stage or int(row["step"]) > final_by_stage[key]:
            final_by_stage[key] = int(row["step"])
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (str(row["stage"]), int(row["size"]))
        if int(row["step"]) == final_by_stage[key]:
            grouped[key].append(row)

    summaries: list[dict[str, Any]] = []
    for (stage, size), final_rows in sorted(grouped.items()):
        summaries.append(
            {
                "stage": stage,
                "size": size,
                "num_seeds": len(final_rows),
                "final_expected_reward": fmean(float(row["expected_reward"]) for row in final_rows),
                "final_sampled_reward": fmean(float(row["sampled_reward"]) for row in final_rows),
                "final_mechanism_score": fmean(float(row["mechanism_score"]) for row in final_rows),
                "final_min_secondary_score": min(float(row["secondary_score"]) for row in final_rows),
                "success_rate": fmean(
                    1.0 if float(row["expected_reward"]) >= 0.95 else 0.0
                    for row in final_rows
                ),
            }
        )
    return summaries


def write_state_figure(rows: list[dict[str, Any]], output_base: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.0))
    state_stages = [
        ("fully_observed_policy", "Fully observed policy", "mechanism_score"),
        ("single_observer_sum_compression", "Single Observer sum compression", "mechanism_score"),
        ("multi_observer_sum_state", "Multi-Observer sum channel", "mechanism_score"),
    ]
    for axis, (stage, title, metric) in zip(axes, state_stages, strict=True):
        plot_stage_metric(axis, rows, stage, "expected_reward", "expected")
        plot_stage_metric(axis, rows, stage, metric, "mechanism")
        axis.set_title(title)
        axis.set_xlabel("training step")
        axis.set_ylim(-0.05, 1.05)
        axis.grid(True, axis="y", alpha=0.25)
        axis.legend(frameon=False, fontsize=8)
    fig.suptitle("Previous-Case Scaling: Structured State Checks")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_figure(fig, output_base)


def write_agent_figure(rows: list[dict[str, Any]], output_base: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    plot_stage_metric(
        axes[0],
        rows,
        "reward_following_agents",
        "mechanism_score",
        "preferred top-rate fraction",
    )
    plot_stage_metric(
        axes[0],
        rows,
        "reward_following_agents",
        "secondary_score",
        "utility tracking",
    )
    axes[0].set_title("Imperfect-reward agent scaling")
    plot_stage_metric(
        axes[1],
        rows,
        "group_state_selectors",
        "mechanism_score",
        "mean P(full)",
    )
    plot_stage_metric(
        axes[1],
        rows,
        "group_state_selectors",
        "secondary_score",
        "min P(full)",
    )
    axes[1].set_title("Group state-selector scaling")
    for axis in axes:
        axis.set_xlabel("training step")
        axis.set_ylim(-0.05, 1.05)
        axis.grid(True, axis="y", alpha=0.25)
        axis.legend(frameon=False, fontsize=8)
    fig.suptitle("Previous-Case Scaling: Agent Checks")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save_figure(fig, output_base)


def plot_stage_metric(
    axis: plt.Axes,
    rows: list[dict[str, Any]],
    stage: str,
    metric: str,
    label: str,
) -> None:
    stage_rows = [row for row in rows if str(row["stage"]) == stage]
    for size in sorted({int(row["size"]) for row in stage_rows}):
        grouped: dict[int, list[float]] = defaultdict(list)
        for row in stage_rows:
            if int(row["size"]) == size:
                grouped[int(row["step"])].append(float(row[metric]))
        series = [(step, fmean(values)) for step, values in sorted(grouped.items())]
        axis.plot(
            [point[0] for point in series],
            [point[1] for point in series],
            linewidth=1.7,
            label=f"{label}, size {size}",
        )


def sample_index(count: int, generator: torch.Generator) -> int:
    return int(torch.randint(0, count, (1,), generator=generator).item())


def seeded_generator(seed: int, label: str) -> torch.Generator:
    generator = torch.Generator()
    offset = sum(ord(char) for char in label)
    generator.manual_seed(seed + 10_000 + offset)
    return generator


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV to {path}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
