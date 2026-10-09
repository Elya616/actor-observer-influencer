"""Staged trainer for paper Section 5 imperfect-state experiments.

Paper Section 5 introduces one Actor and multiple Observers. The Actor selects
one Observer per step, receives that Observer's compressed state message, and
then acts. Sections 5.1--5.3 differ only in which parameter sets are allowed to
learn:

- 5.1: Observer selection logits only;
- 5.2: selection logits plus selected Observer compression;
- 5.3: selection logits, selected Observer compression, and Actor policy.

This trainer keeps those stages explicit by accepting optional optimizers. The
same interaction loop can therefore be reused without turning the first staged
tests into black-box joint learning.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from statistics import fmean
from typing import Any

import torch

from actor_observer_rl.agents import Actor, MultiObserver
from actor_observer_rl.training.partial_observable import build_actor_message_input
from actor_observer_rl.training.updates import reinforce_loss
from actor_observer_rl.utils.seeding import set_global_seed


@dataclass
class StagedMultiObserverTrainingTrace:
    """Trace for staged single-Actor/multiple-Observer experiments."""

    rewards: list[float] = field(default_factory=list)
    average_rewards: list[float] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    state_indices: list[int] = field(default_factory=list)
    selected_observer_indices: list[int] = field(default_factory=list)
    sampled_messages: list[list[float]] = field(default_factory=list)
    action_log_probs: list[float] = field(default_factory=list)
    selection_log_probs: list[float] = field(default_factory=list)
    following_weights_at_sample: list[list[float]] = field(default_factory=list)
    following_weights: list[list[float]] = field(default_factory=list)
    following_logits: list[list[float]] = field(default_factory=list)
    selection_entropies: list[float] = field(default_factory=list)
    selection_update_multipliers: list[float] = field(default_factory=list)
    selection_baselines: list[float] = field(default_factory=list)
    cumulative_rewards: list[float] = field(default_factory=list)
    expected_rewards: list[float] = field(default_factory=list)
    per_observer_expected_rewards: list[list[float]] = field(default_factory=list)
    mean_optimal_action_probs: list[float] = field(default_factory=list)
    per_observer_optimal_action_probs: list[list[float]] = field(default_factory=list)
    observer_messages_by_state: list[list[list[list[float]]]] = field(default_factory=list)
    evaluation_steps: list[int] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def final_expected_reward(self) -> float | None:
        return self.expected_rewards[-1] if self.expected_rewards else None

    @property
    def final_average_reward(self) -> float | None:
        return self.average_rewards[-1] if self.average_rewards else None

    @property
    def final_following_weights(self) -> list[float] | None:
        return self.following_weights[-1] if self.following_weights else None

    @property
    def final_per_observer_expected_rewards(self) -> list[float] | None:
        if not self.per_observer_expected_rewards:
            return None
        return self.per_observer_expected_rewards[-1]

    @property
    def final_mean_optimal_action_prob(self) -> float | None:
        if not self.mean_optimal_action_probs:
            return None
        return self.mean_optimal_action_probs[-1]

    @property
    def final_per_observer_optimal_action_probs(self) -> list[float] | None:
        if not self.per_observer_optimal_action_probs:
            return None
        return self.per_observer_optimal_action_probs[-1]

    @property
    def final_observer_messages_by_state(
        self,
    ) -> list[list[list[float]]] | None:
        if not self.observer_messages_by_state:
            return None
        return self.observer_messages_by_state[-1]


class StagedMultiObserverTrainer:
    """Single Actor, multiple Observers, staged learning switches.

    The loss includes the Observer-selection log-probability when
    ``following_optimizer`` is provided. It includes the Actor action
    log-probability when either ``actor_optimizer`` or ``observer_optimizer`` is
    provided. This matches the paper stages:

    - 5.1 learns only ``w`` through ``log lambda_{k_t}``;
    - 5.2 learns ``w`` and selected ``phi_{k_t}`` through ``log pi``;
    - 5.3 additionally learns Actor parameters ``theta``.
    """

    def __init__(
        self,
        env: Any,
        actor: Actor,
        multi_observer: MultiObserver,
        following_optimizer: torch.optim.Optimizer | None = None,
        observer_optimizer: torch.optim.Optimizer | None = None,
        actor_optimizer: torch.optim.Optimizer | None = None,
        include_actor_observation: bool = False,
        optimal_actions: Sequence[int] | None = None,
        selection_update_rule: str = "raw_reward",
        selection_baseline_beta: float = 0.01,
        selection_fixed_baseline: float | None = None,
        selection_baseline_initial: float = 0.0,
        device: str | torch.device = "cpu",
    ) -> None:
        if selection_update_rule not in {"raw_reward", "advantage_global"}:
            raise ValueError(
                "selection_update_rule must be 'raw_reward' or 'advantage_global'"
            )
        if not 0.0 < selection_baseline_beta <= 1.0:
            raise ValueError("selection_baseline_beta must be in (0, 1]")
        self.env = env
        self.actor = actor
        self.multi_observer = multi_observer
        self.following_optimizer = following_optimizer
        self.observer_optimizer = observer_optimizer
        self.actor_optimizer = actor_optimizer
        self.include_actor_observation = bool(include_actor_observation)
        self.selection_update_rule = selection_update_rule
        self.selection_baseline_beta = float(selection_baseline_beta)
        self.selection_fixed_baseline = (
            None if selection_fixed_baseline is None else float(selection_fixed_baseline)
        )
        self.selection_baseline = float(selection_baseline_initial)
        self.optimal_actions = (
            None
            if optimal_actions is None
            else tuple(int(action) for action in optimal_actions)
        )
        self.device = torch.device(device)

        if self.optimal_actions is not None and len(self.optimal_actions) != env.num_states:
            raise ValueError("optimal_actions length must match env.num_states")

    def run(
        self,
        num_steps: int,
        seed: int | None = None,
        eval_interval: int = 1,
        reward_window: int = 100,
    ) -> StagedMultiObserverTrainingTrace:
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")
        if eval_interval <= 0:
            raise ValueError("eval_interval must be positive")
        if reward_window <= 0:
            raise ValueError("reward_window must be positive")

        if seed is not None:
            set_global_seed(seed)

        observation, reset_info = self.env.reset(seed=seed)
        state_index = self._state_index_from_info(reset_info)
        rewards_window: deque[float] = deque(maxlen=reward_window)
        trace = StagedMultiObserverTrainingTrace(
            metadata={
                "seed": seed,
                "num_steps": num_steps,
                "eval_interval": eval_interval,
                "reward_window": reward_window,
                "num_observers": self.multi_observer.num_observers,
                "include_actor_observation": self.include_actor_observation,
                "learn_selection": self.following_optimizer is not None,
                "learn_observers": self.observer_optimizer is not None,
                "learn_actor": self.actor_optimizer is not None,
                "has_optimal_actions": self.optimal_actions is not None,
                "selection_update_rule": self.selection_update_rule,
                "selection_baseline_beta": self.selection_baseline_beta,
                "selection_fixed_baseline": self.selection_fixed_baseline,
            }
        )

        self._record_evaluation(trace, step=0, average_reward=0.0)
        cumulative_reward = 0.0

        for step_index in range(num_steps):
            full_state = self._to_tensor(self.env.get_full_state())
            actor_observation = self._to_tensor(observation)
            observer_index, message, selection_log_prob = (
                self.multi_observer.sample_observer_and_message(full_state)
            )
            following_weights_at_sample = (
                self.multi_observer.following_rates.current_weights()
            )
            actor_input = self._actor_input(actor_observation, message)
            action_result = self.actor.act(actor_input, sample=True)
            action = self._action_to_int(action_result.action)

            step_result = self.env.step(action)
            reward = self.multi_observer.report_reward(step_result.reward)
            selection_multiplier = self._selection_update_multiplier(reward)
            loss_terms: list[torch.Tensor] = []
            if self.following_optimizer is not None:
                loss_terms.append(
                    reinforce_loss(selection_multiplier, selection_log_prob)
                )
            if self.actor_optimizer is not None or self.observer_optimizer is not None:
                loss_terms.append(reinforce_loss(reward, action_result.log_prob))
            if loss_terms:
                loss = sum(loss_terms)
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite staged multi-observer loss")
                self._zero_grad()
                loss.backward()
                self._step()
            self._update_selection_baseline(reward)

            trace.rewards.append(reward)
            cumulative_reward += reward
            rewards_window.append(reward)
            trace.actions.append(action)
            trace.state_indices.append(state_index)
            trace.selected_observer_indices.append(observer_index)
            trace.sampled_messages.append(
                [
                    float(value)
                    for value in message.detach().cpu().reshape(-1).tolist()
                ]
            )
            trace.action_log_probs.append(
                float(action_result.log_prob.detach().cpu().item())
            )
            trace.selection_log_probs.append(
                float(selection_log_prob.detach().cpu().item())
            )
            trace.following_weights_at_sample.append(
                following_weights_at_sample.cpu().tolist()
            )
            weights = self.multi_observer.following_rates.current_weights()
            trace.following_weights.append(weights.cpu().tolist())
            trace.following_logits.append(
                self.multi_observer.following_rates.logits.detach().cpu().tolist()
            )
            trace.selection_entropies.append(float(self._entropy(weights).item()))
            trace.selection_update_multipliers.append(float(selection_multiplier))
            trace.selection_baselines.append(float(self.selection_baseline))
            trace.cumulative_rewards.append(float(cumulative_reward))

            current_step = step_index + 1
            if current_step % eval_interval == 0 or current_step == num_steps:
                self._record_evaluation(
                    trace,
                    step=current_step,
                    average_reward=fmean(rewards_window),
                )

            if step_result.done:
                observation, reset_info = self.env.reset(seed=None)
                state_index = self._state_index_from_info(reset_info)
            else:
                observation = step_result.observation
                state_index = self._state_index_from_info(step_result.info)

        return trace

    def evaluate_policy(self) -> tuple[float, list[float]]:
        """Return marginal expected reward and per-Observer expected rewards."""

        expected_reward = 0.0
        per_observer = [0.0] * self.multi_observer.num_observers
        with torch.no_grad():
            weights = self.multi_observer.following_rates.current_weights()
            for state_index in range(self.env.num_states):
                full_state = self._to_tensor(self.env.full_state_for_index(state_index))
                observation = self._to_tensor(
                    self.env.actor_observation_for_index(state_index)
                )
                rewards = torch.tensor(
                    self.env.reward_table[state_index],
                    dtype=torch.float32,
                    device=self.device,
                )
                state_probability = self.env.state_probability(state_index)
                for observer_index, observer in enumerate(self.multi_observer.observers):
                    message = observer.message(full_state)
                    actor_input = self._actor_input(observation, message)
                    probs = self.actor.policy.action_distribution(actor_input).probs
                    observer_reward = float((probs * rewards).sum().item())
                    per_observer[observer_index] += state_probability * observer_reward
                    expected_reward += (
                        state_probability
                        * float(weights[observer_index].item())
                        * observer_reward
                    )
        return expected_reward, per_observer

    def evaluate_optimal_action_probabilities(self) -> tuple[float, list[float]]:
        """Return marginal and per-Observer high-reward action probabilities."""

        if self.optimal_actions is None:
            return 0.0, []

        mean_probability = 0.0
        per_observer = [0.0] * self.multi_observer.num_observers
        with torch.no_grad():
            weights = self.multi_observer.following_rates.current_weights()
            for state_index in range(self.env.num_states):
                full_state = self._to_tensor(self.env.full_state_for_index(state_index))
                observation = self._to_tensor(
                    self.env.actor_observation_for_index(state_index)
                )
                state_probability = self.env.state_probability(state_index)
                optimal_action = self.optimal_actions[state_index]
                for observer_index, observer in enumerate(self.multi_observer.observers):
                    message = observer.message(full_state)
                    actor_input = self._actor_input(observation, message)
                    probs = self.actor.policy.action_distribution(actor_input).probs
                    optimal_probability = float(probs[..., optimal_action].item())
                    per_observer[observer_index] += (
                        state_probability * optimal_probability
                    )
                    mean_probability += (
                        state_probability
                        * float(weights[observer_index].item())
                        * optimal_probability
                    )
        return mean_probability, per_observer

    def _record_evaluation(
        self,
        trace: StagedMultiObserverTrainingTrace,
        step: int,
        average_reward: float,
    ) -> None:
        expected_reward, per_observer = self.evaluate_policy()
        optimal_prob, per_observer_optimal_probs = (
            self.evaluate_optimal_action_probabilities()
        )
        messages_by_state = self.evaluate_observer_messages_by_state()
        trace.evaluation_steps.append(int(step))
        trace.expected_rewards.append(float(expected_reward))
        trace.per_observer_expected_rewards.append([float(value) for value in per_observer])
        if self.optimal_actions is not None:
            trace.mean_optimal_action_probs.append(float(optimal_prob))
            trace.per_observer_optimal_action_probs.append(
                [float(value) for value in per_observer_optimal_probs]
            )
        trace.observer_messages_by_state.append(messages_by_state)
        trace.average_rewards.append(float(average_reward))

    def evaluate_observer_messages_by_state(self) -> list[list[list[float]]]:
        """Return `[observer][state][message_dim]` message snapshots."""

        snapshots: list[list[list[float]]] = []
        with torch.no_grad():
            for observer in self.multi_observer.observers:
                observer_messages: list[list[float]] = []
                for state_index in range(self.env.num_states):
                    full_state = self._to_tensor(
                        self.env.full_state_for_index(state_index)
                    )
                    message = observer.message(full_state)
                    observer_messages.append(
                        [
                            float(value)
                            for value in message.detach().cpu().reshape(-1).tolist()
                        ]
                    )
                snapshots.append(observer_messages)
        return snapshots

    def _actor_input(
        self,
        observation: torch.Tensor,
        message: torch.Tensor,
    ) -> torch.Tensor:
        if self.include_actor_observation:
            return build_actor_message_input(observation, message)
        return message

    def _zero_grad(self) -> None:
        for optimizer in self._optimizers():
            optimizer.zero_grad()

    def _step(self) -> None:
        for optimizer in self._optimizers():
            optimizer.step()

    def _optimizers(self) -> Iterable[torch.optim.Optimizer]:
        for optimizer in (
            self.actor_optimizer,
            self.observer_optimizer,
            self.following_optimizer,
        ):
            if optimizer is not None:
                yield optimizer

    def _to_tensor(self, value: Any) -> torch.Tensor:
        return torch.as_tensor(value, dtype=torch.float32, device=self.device)

    def _selection_update_multiplier(self, reward: float) -> float:
        if self.selection_update_rule == "raw_reward":
            return float(reward)
        baseline = (
            self.selection_baseline
            if self.selection_fixed_baseline is None
            else self.selection_fixed_baseline
        )
        return float(reward) - float(baseline)

    def _update_selection_baseline(self, reward: float) -> None:
        if self.selection_update_rule != "advantage_global":
            return
        if self.selection_fixed_baseline is not None:
            return
        beta = self.selection_baseline_beta
        self.selection_baseline = (
            (1.0 - beta) * self.selection_baseline + beta * float(reward)
        )

    @staticmethod
    def _entropy(weights: torch.Tensor) -> torch.Tensor:
        safe_weights = torch.clamp(weights, min=1e-12)
        return -(safe_weights * torch.log(safe_weights)).sum()

    @staticmethod
    def _action_to_int(action: torch.Tensor) -> int:
        return int(action.detach().cpu().reshape(-1)[0].item())

    @staticmethod
    def _state_index_from_info(info: dict[str, Any]) -> int:
        state_index = info.get("state_index")
        if state_index is None:
            raise KeyError("staged multi-observer training requires info['state_index']")
        return int(state_index)
