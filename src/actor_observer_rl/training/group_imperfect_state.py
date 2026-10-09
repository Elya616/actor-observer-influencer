"""Trainer for paper Section 7 group-level imperfect-state experiments.

Section 7 extends the paper Section 5 imperfect-state setting from one Actor
to many Actors. At each step, one Actor is active, that Actor selects one
Observer for a compressed state message, and the active Actor acts from that
message. The three paper stages differ only in which parameter sets are
learned:

- 7.1: each Actor's Observer-selection logits;
- 7.2: selection logits plus selected Observer compression;
- 7.3: selection logits, selected Observer compression, and active Actor policy.

This trainer keeps the stages explicit through optional optimizers. It uses one
``FollowingRates`` module per Actor, so Actor ``j`` has its own distribution
``lambda_j`` over the shared Observers.
"""

from __future__ import annotations

import hashlib
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from statistics import fmean
from typing import Any

import torch

from actor_observer_rl.agents import Actor, FollowingRates, Observer
from actor_observer_rl.training.partial_observable import build_actor_message_input
from actor_observer_rl.training.updates import reinforce_loss
from actor_observer_rl.utils.seeding import set_global_seed


def _stream_seed(seed: int, stream_name: str) -> int:
    digest = hashlib.sha256(stream_name.encode("utf-8")).digest()
    offset = int.from_bytes(digest[:8], "big", signed=False)
    return (int(seed) + offset) & ((1 << 63) - 1)


@dataclass
class GroupImperfectStateTrainingTrace:
    """Trace for many-Actor/many-Observer imperfect-state experiments."""

    rewards: list[float] = field(default_factory=list)
    average_rewards: list[float] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    state_indices: list[int] = field(default_factory=list)
    selected_actor_indices: list[int] = field(default_factory=list)
    selected_observer_indices: list[int] = field(default_factory=list)
    selection_log_probs: list[float] = field(default_factory=list)
    selection_update_multipliers: list[float] = field(default_factory=list)
    observer_gradient_norms: list[list[float]] = field(default_factory=list)
    evaluation_steps: list[int] = field(default_factory=list)
    expected_rewards: list[float] = field(default_factory=list)
    per_actor_expected_rewards: list[list[float]] = field(default_factory=list)
    per_actor_observer_expected_rewards: list[list[list[float]]] = field(
        default_factory=list
    )
    mean_optimal_action_probs: list[float] = field(default_factory=list)
    per_actor_optimal_action_probs: list[list[float]] = field(default_factory=list)
    per_actor_observer_optimal_action_probs: list[list[list[float]]] = field(
        default_factory=list
    )
    selection_weights_by_actor: list[list[list[float]]] = field(default_factory=list)
    selection_entropies_by_actor: list[list[float]] = field(default_factory=list)
    selection_baselines_by_actor: list[list[float]] = field(default_factory=list)
    observer_messages_by_state: list[list[list[list[float]]]] = field(
        default_factory=list
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def final_expected_reward(self) -> float | None:
        return self.expected_rewards[-1] if self.expected_rewards else None

    @property
    def final_average_reward(self) -> float | None:
        return self.average_rewards[-1] if self.average_rewards else None

    @property
    def final_selection_weights_by_actor(self) -> list[list[float]] | None:
        if not self.selection_weights_by_actor:
            return None
        return self.selection_weights_by_actor[-1]

    @property
    def final_mean_optimal_action_prob(self) -> float | None:
        if not self.mean_optimal_action_probs:
            return None
        return self.mean_optimal_action_probs[-1]

    @property
    def final_per_actor_expected_rewards(self) -> list[float] | None:
        if not self.per_actor_expected_rewards:
            return None
        return self.per_actor_expected_rewards[-1]

    @property
    def final_per_actor_observer_expected_rewards(
        self,
    ) -> list[list[float]] | None:
        if not self.per_actor_observer_expected_rewards:
            return None
        return self.per_actor_observer_expected_rewards[-1]

    @property
    def final_observer_messages_by_state(
        self,
    ) -> list[list[list[float]]] | None:
        if not self.observer_messages_by_state:
            return None
        return self.observer_messages_by_state[-1]


class GroupImperfectStateTrainer:
    """Many Actors, many Observers, imperfect state information.

    Parameters
    ----------
    env:
        One-step state environment. The Observer receives ``env.get_full_state()``
        and the Actor receives either only the selected message or optionally
        ``concat(o_t, m_t)``.
    actors:
        Actor list. One Actor is sampled per step according to
        ``actor_selection_probs``.
    observers:
        Shared Observer list. The active Actor samples one Observer and only the
        selected Observer's message is built, so only that Observer can receive
        a compression gradient on that step.
    following_rates_by_actor:
        One ``FollowingRates`` module per Actor. This is the group-level
        extension of Section 5's single selection distribution.
    following_optimizer:
        Optimizer for all Actor-specific selection logits. If absent, selection
        logits are fixed.
    observer_optimizer:
        Optimizer for shared Observer compression parameters. If absent,
        Observer messages are fixed.
    actor_optimizers:
        Optional per-Actor optimizers. ``None`` entries keep the corresponding
        Actor policy fixed.
    selection_update_rule:
        Multiplier for the selected Observer's log-probability term. The paper
        update is ``"raw_reward"``. ``"advantage_per_actor"`` subtracts a
        per-Actor running reward baseline from the selection term only.
    selection_entropy_beta:
        Optional entropy coefficient for active-Actor selection. A positive
        value discourages early collapse of the active Actor's selection
        distribution while leaving the sampled-action and compression reward
        terms unchanged.
    """

    def __init__(
        self,
        env: Any,
        actors: Sequence[Actor],
        observers: Sequence[Observer],
        following_rates_by_actor: Sequence[FollowingRates],
        actor_selection_probs: Sequence[float],
        following_optimizer: torch.optim.Optimizer | None = None,
        observer_optimizer: torch.optim.Optimizer | None = None,
        actor_optimizers: Sequence[torch.optim.Optimizer | None] | None = None,
        include_actor_observation: bool = False,
        optimal_actions: Sequence[int] | None = None,
        selection_update_rule: str = "raw_reward",
        selection_baseline_learning_rate: float = 0.05,
        selection_baseline_initial: float = 0.0,
        selection_entropy_beta: float = 0.0,
        device: str | torch.device = "cpu",
    ) -> None:
        actors = list(actors)
        observers = list(observers)
        following_rates_by_actor = list(following_rates_by_actor)
        if len(actors) < 2:
            raise ValueError("GroupImperfectStateTrainer requires at least two actors")
        if len(observers) < 2:
            raise ValueError("GroupImperfectStateTrainer requires at least two observers")
        if len(following_rates_by_actor) != len(actors):
            raise ValueError("following_rates_by_actor length must match actors")
        if len(actor_selection_probs) != len(actors):
            raise ValueError("actor_selection_probs length must match actors")
        if abs(sum(float(prob) for prob in actor_selection_probs) - 1.0) > 1e-6:
            raise ValueError("actor_selection_probs must sum to 1")
        if any(float(prob) < 0.0 for prob in actor_selection_probs):
            raise ValueError("actor_selection_probs entries must be non-negative")
        for rates in following_rates_by_actor:
            if rates.num_observers != len(observers):
                raise ValueError(
                    "following_rates_by_actor entries must match observer count"
                )
        for observer_index, observer in enumerate(observers):
            if observer.message_module is None:
                raise ValueError(
                    f"observer at index {observer_index} is missing a message_module"
                )
        if actor_optimizers is not None and len(actor_optimizers) != len(actors):
            raise ValueError("actor_optimizers length must match actors")
        if selection_update_rule not in {"raw_reward", "advantage_per_actor"}:
            raise ValueError(
                "selection_update_rule must be 'raw_reward' or 'advantage_per_actor'"
            )
        if not 0.0 < selection_baseline_learning_rate <= 1.0:
            raise ValueError("selection_baseline_learning_rate must be in (0, 1]")
        if selection_entropy_beta < 0.0:
            raise ValueError("selection_entropy_beta must be non-negative")

        self.env = env
        self.actors = actors
        self.observers = observers
        self.following_rates_by_actor = following_rates_by_actor
        self.actor_selection_probs = torch.tensor(
            [float(prob) for prob in actor_selection_probs],
            dtype=torch.float32,
        )
        self.following_optimizer = following_optimizer
        self.observer_optimizer = observer_optimizer
        if actor_optimizers is None:
            self.actor_optimizers: list[torch.optim.Optimizer | None] = [
                None
            ] * len(actors)
        else:
            self.actor_optimizers = list(actor_optimizers)
        self.include_actor_observation = bool(include_actor_observation)
        self.selection_update_rule = selection_update_rule
        self.selection_baseline_learning_rate = float(selection_baseline_learning_rate)
        self.selection_entropy_beta = float(selection_entropy_beta)
        self.selection_baselines = [
            float(selection_baseline_initial) for _ in range(len(actors))
        ]
        self.optimal_actions = (
            None
            if optimal_actions is None
            else tuple(int(action) for action in optimal_actions)
        )
        self.device = torch.device(device)
        self._actor_selection_generator = torch.Generator()

        if self.optimal_actions is not None and len(self.optimal_actions) != env.num_states:
            raise ValueError("optimal_actions length must match env.num_states")

    def run(
        self,
        num_steps: int,
        seed: int | None = None,
        eval_interval: int = 1,
        reward_window: int = 100,
    ) -> GroupImperfectStateTrainingTrace:
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")
        if eval_interval <= 0:
            raise ValueError("eval_interval must be positive")
        if reward_window <= 0:
            raise ValueError("reward_window must be positive")

        if seed is not None:
            set_global_seed(seed)
            self._actor_selection_generator.manual_seed(
                _stream_seed(seed, "group_state_actor_selection")
            )
            if hasattr(self.env, "_generator"):
                self.env._generator.manual_seed(
                    _stream_seed(seed, "group_state_env")
                )

        observation, reset_info = self.env.reset(seed=None)
        state_index = self._state_index_from_info(reset_info)
        rewards_window: deque[float] = deque(maxlen=reward_window)
        trace = GroupImperfectStateTrainingTrace(
            metadata={
                "seed": seed,
                "num_steps": num_steps,
                "eval_interval": eval_interval,
                "reward_window": reward_window,
                "num_actors": len(self.actors),
                "num_observers": len(self.observers),
                "include_actor_observation": self.include_actor_observation,
                "learn_selection": self.following_optimizer is not None,
                "learn_observers": self.observer_optimizer is not None,
                "learn_actor": any(
                    optimizer is not None for optimizer in self.actor_optimizers
                ),
                "has_optimal_actions": self.optimal_actions is not None,
                "selection_update_rule": self.selection_update_rule,
                "selection_baseline_learning_rate": (
                    self.selection_baseline_learning_rate
                ),
                "selection_entropy_beta": self.selection_entropy_beta,
            }
        )

        self._record_evaluation(trace, step=0, average_reward=0.0)

        for step_index in range(num_steps):
            full_state = self._to_tensor(self.env.get_full_state())
            actor_observation = self._to_tensor(observation)
            actor_index = self._sample_actor_index()
            observer_index, message, selection_log_prob = self._sample_message(
                actor_index,
                full_state,
            )
            actor = self.actors[actor_index]
            actor_input = self._actor_input(actor_observation, message)
            action_result = actor.act(actor_input, sample=True)
            action = self._action_to_int(action_result.action)

            step_result = self.env.step(action)
            reward = float(step_result.reward)
            loss_terms: list[torch.Tensor] = []
            if self.following_optimizer is not None:
                selection_multiplier = self._selection_update_multiplier(
                    actor_index,
                    reward,
                )
                loss_terms.append(
                    reinforce_loss(selection_multiplier, selection_log_prob)
                )
                if self.selection_entropy_beta > 0.0:
                    loss_terms.append(
                        -self.selection_entropy_beta
                        * self._selection_entropy(actor_index)
                    )
            else:
                selection_multiplier = 0.0
            if (
                self.observer_optimizer is not None
                or self.actor_optimizers[actor_index] is not None
            ):
                loss_terms.append(reinforce_loss(reward, action_result.log_prob))
            observer_gradient_norms = [0.0] * len(self.observers)
            if loss_terms:
                loss = sum(loss_terms)
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite group imperfect-state loss")
                self._zero_grad(actor_index)
                loss.backward()
                if self.observer_optimizer is not None:
                    observer_gradient_norms = self._observer_gradient_norms()
                self._step(actor_index)

            trace.rewards.append(reward)
            rewards_window.append(reward)
            trace.actions.append(action)
            trace.state_indices.append(state_index)
            trace.selected_actor_indices.append(actor_index)
            trace.selected_observer_indices.append(observer_index)
            trace.selection_log_probs.append(
                float(selection_log_prob.detach().cpu().item())
            )
            trace.selection_update_multipliers.append(float(selection_multiplier))
            trace.observer_gradient_norms.append(observer_gradient_norms)
            self._update_selection_baseline(actor_index, reward)

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

    def evaluate_policy(
        self,
    ) -> tuple[float, list[float], list[list[float]]]:
        """Return marginal, per-Actor, and per-Actor/per-Observer rewards."""

        expected_reward = 0.0
        per_actor = [0.0] * len(self.actors)
        per_actor_observer = [
            [0.0] * len(self.observers) for _ in range(len(self.actors))
        ]
        with torch.no_grad():
            for actor_index, actor in enumerate(self.actors):
                weights = self.following_rates_by_actor[
                    actor_index
                ].current_weights()
                actor_probability = float(self.actor_selection_probs[actor_index].item())
                for state_index in range(self.env.num_states):
                    full_state = self._to_tensor(
                        self.env.full_state_for_index(state_index)
                    )
                    observation = self._to_tensor(
                        self.env.actor_observation_for_index(state_index)
                    )
                    rewards = torch.tensor(
                        self.env.reward_table[state_index],
                        dtype=torch.float32,
                        device=self.device,
                    )
                    state_probability = self.env.state_probability(state_index)
                    for observer_index, observer in enumerate(self.observers):
                        message = observer.message(full_state)
                        actor_input = self._actor_input(observation, message)
                        probs = actor.policy.action_distribution(actor_input).probs
                        observer_reward = float((probs * rewards).sum().item())
                        weighted_reward = state_probability * observer_reward
                        per_actor_observer[actor_index][observer_index] += (
                            weighted_reward
                        )
                        per_actor[actor_index] += (
                            weighted_reward * float(weights[observer_index].item())
                        )
                expected_reward += actor_probability * per_actor[actor_index]
        return expected_reward, per_actor, per_actor_observer

    def evaluate_optimal_action_probabilities(
        self,
    ) -> tuple[float, list[float], list[list[float]]]:
        """Return marginal and conditional high-reward action probabilities."""

        if self.optimal_actions is None:
            return 0.0, [], []

        mean_probability = 0.0
        per_actor = [0.0] * len(self.actors)
        per_actor_observer = [
            [0.0] * len(self.observers) for _ in range(len(self.actors))
        ]
        with torch.no_grad():
            for actor_index, actor in enumerate(self.actors):
                weights = self.following_rates_by_actor[
                    actor_index
                ].current_weights()
                actor_probability = float(self.actor_selection_probs[actor_index].item())
                for state_index in range(self.env.num_states):
                    full_state = self._to_tensor(
                        self.env.full_state_for_index(state_index)
                    )
                    observation = self._to_tensor(
                        self.env.actor_observation_for_index(state_index)
                    )
                    state_probability = self.env.state_probability(state_index)
                    optimal_action = self.optimal_actions[state_index]
                    for observer_index, observer in enumerate(self.observers):
                        message = observer.message(full_state)
                        actor_input = self._actor_input(observation, message)
                        probs = actor.policy.action_distribution(actor_input).probs
                        optimal_probability = float(probs[..., optimal_action].item())
                        weighted_probability = state_probability * optimal_probability
                        per_actor_observer[actor_index][observer_index] += (
                            weighted_probability
                        )
                        per_actor[actor_index] += (
                            weighted_probability * float(weights[observer_index].item())
                        )
                mean_probability += actor_probability * per_actor[actor_index]
        return mean_probability, per_actor, per_actor_observer

    def evaluate_observer_messages_by_state(self) -> list[list[list[float]]]:
        """Return ``[observer][state][message_dim]`` message snapshots."""

        snapshots: list[list[list[float]]] = []
        with torch.no_grad():
            for observer in self.observers:
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

    def current_selection_weights_by_actor(self) -> list[list[float]]:
        return [
            rates.current_weights().detach().cpu().tolist()
            for rates in self.following_rates_by_actor
        ]

    def current_selection_entropies_by_actor(self) -> list[float]:
        entropies: list[float] = []
        with torch.no_grad():
            for rates in self.following_rates_by_actor:
                weights = rates.current_weights()
                entropy = -(weights * torch.log(weights.clamp_min(1e-12))).sum()
                entropies.append(float(entropy.detach().cpu().item()))
        return entropies

    def _record_evaluation(
        self,
        trace: GroupImperfectStateTrainingTrace,
        step: int,
        average_reward: float,
    ) -> None:
        expected_reward, per_actor, per_actor_observer = self.evaluate_policy()
        optimal_prob, per_actor_optimal, per_actor_observer_optimal = (
            self.evaluate_optimal_action_probabilities()
        )
        trace.evaluation_steps.append(int(step))
        trace.expected_rewards.append(float(expected_reward))
        trace.per_actor_expected_rewards.append(
            [float(value) for value in per_actor]
        )
        trace.per_actor_observer_expected_rewards.append(
            [
                [float(value) for value in observer_values]
                for observer_values in per_actor_observer
            ]
        )
        if self.optimal_actions is not None:
            trace.mean_optimal_action_probs.append(float(optimal_prob))
            trace.per_actor_optimal_action_probs.append(
                [float(value) for value in per_actor_optimal]
            )
            trace.per_actor_observer_optimal_action_probs.append(
                [
                    [float(value) for value in observer_values]
                    for observer_values in per_actor_observer_optimal
                ]
            )
        trace.selection_weights_by_actor.append(
            self.current_selection_weights_by_actor()
        )
        trace.selection_entropies_by_actor.append(
            self.current_selection_entropies_by_actor()
        )
        trace.selection_baselines_by_actor.append(
            [float(value) for value in self.selection_baselines]
        )
        trace.observer_messages_by_state.append(
            self.evaluate_observer_messages_by_state()
        )
        trace.average_rewards.append(float(average_reward))

    def _sample_actor_index(self) -> int:
        sample = torch.multinomial(
            self.actor_selection_probs,
            num_samples=1,
            replacement=True,
            generator=self._actor_selection_generator,
        )
        return int(sample.item())

    def _sample_message(
        self,
        actor_index: int,
        full_state: torch.Tensor,
    ) -> tuple[int, torch.Tensor, torch.Tensor]:
        distribution = self.following_rates_by_actor[actor_index].distribution()
        sampled = distribution.sample()
        observer_index = int(sampled.item())
        log_prob = distribution.log_prob(sampled)
        message = self.observers[observer_index].message(full_state)
        return observer_index, message, log_prob

    def _selection_update_multiplier(self, actor_index: int, reward: float) -> float:
        if self.selection_update_rule == "raw_reward":
            return float(reward)
        if self.selection_update_rule == "advantage_per_actor":
            return float(reward - self.selection_baselines[actor_index])
        raise RuntimeError(
            f"unsupported selection_update_rule {self.selection_update_rule!r}"
        )

    def _update_selection_baseline(self, actor_index: int, reward: float) -> None:
        if self.selection_update_rule != "advantage_per_actor":
            return
        baseline = self.selection_baselines[actor_index]
        rate = self.selection_baseline_learning_rate
        self.selection_baselines[actor_index] = baseline + rate * (reward - baseline)

    def _selection_entropy(self, actor_index: int) -> torch.Tensor:
        weights = self.following_rates_by_actor[actor_index]()
        return -(weights * torch.log(weights.clamp_min(1e-12))).sum()

    def _actor_input(
        self,
        observation: torch.Tensor,
        message: torch.Tensor,
    ) -> torch.Tensor:
        if self.include_actor_observation:
            return build_actor_message_input(observation, message)
        return message

    def _zero_grad(self, actor_index: int) -> None:
        for optimizer in self._optimizers(actor_index):
            optimizer.zero_grad()

    def _step(self, actor_index: int) -> None:
        for optimizer in self._optimizers(actor_index):
            optimizer.step()

    def _observer_gradient_norms(self) -> list[float]:
        norms: list[float] = []
        for observer in self.observers:
            squared_norm = 0.0
            for parameter in observer.parameters():
                if parameter.grad is not None:
                    squared_norm += float(parameter.grad.detach().pow(2).sum().item())
            norms.append(squared_norm**0.5)
        return norms

    def _optimizers(self, actor_index: int) -> Iterable[torch.optim.Optimizer]:
        for optimizer in (
            self.actor_optimizers[actor_index],
            self.observer_optimizer,
            self.following_optimizer,
        ):
            if optimizer is not None:
                yield optimizer

    def _to_tensor(self, value: Any) -> torch.Tensor:
        return torch.as_tensor(value, dtype=torch.float32, device=self.device)

    @staticmethod
    def _action_to_int(action: torch.Tensor) -> int:
        return int(action.detach().cpu().reshape(-1)[0].item())

    @staticmethod
    def _state_index_from_info(info: dict[str, Any]) -> int:
        state_index = info.get("state_index")
        if state_index is None:
            raise KeyError("group imperfect-state training requires info['state_index']")
        return int(state_index)
