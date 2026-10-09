"""Trainer for paper Section 6 (group-level, imperfect reward).

Section 6 extends Section 4 from a single Observer to ``N`` Observers, each
watching the same ``N`` Actors. Every Observer ``O_i`` has its own reward
function ``R_i``: when the active Actor performs an action, Observer ``i``
perceives reward ``R_i(s_t, a_t)``. Because the reward functions differ across
Observers, different Observers may value the same Actor differently -- the
group-level analogue of the SRA notion that perceived utility is private to the
evaluating agent.

Section 6.1 keeps Actor policies fixed (``actor_optimizers`` is ``None``): each
Observer independently runs the Section 4.1 mechanism on its own reward stream,
maintaining a utility estimate ``U_hat_{i,j}`` and a budgeted following-rate
vector ``mu_i`` over Actors.

Section 6.2 adds learnable Actor policies. Following the paper's Eq. (29), Actor
``j``'s effective reward aggregates over Observers, weighted by each Observer's
reporting probability for that Actor:

    r_eff(j) = sum_i theta(mu_{i,j}) * R_i(s_t, a_t).

The Actor then performs a REINFORCE update with ``r_eff``. The two-timescale
discipline (``utility_learning_rate >> following_learning_rate`` and Actor
learning rate) is recovered through the per-component learning rates chosen at
construction time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from actor_observer_rl.agents.actor import Actor
from actor_observer_rl.agents.budgeted_observer import BudgetedObserver
from actor_observer_rl.training.multi_actor import _stream_seed
from actor_observer_rl.training.updates import (
    linear_anneal,
    mixed_action_sample,
    reinforce_loss,
)
from actor_observer_rl.utils.seeding import set_global_seed


@dataclass
class GroupMultiActorTrainingTrace:
    """Per-step trace produced by ``GroupMultiActorTrainer.run``.

    ``following_rates`` and ``utility_estimates`` snapshots are indexed
    ``[snapshot][observer][actor]``.
    """

    selected_actor_indices: list[int] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    observer_rewards: list[list[float]] = field(default_factory=list)
    following_rates: list[list[list[float]]] = field(default_factory=list)
    utility_estimates: list[list[list[float]]] = field(default_factory=list)
    per_actor_optimal_action_probs: list[list[float]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_steps(self) -> int:
        return len(self.actions)

    @property
    def final_following_rates(self) -> list[list[float]] | None:
        return self.following_rates[-1] if self.following_rates else None

    @property
    def final_utilities(self) -> list[list[float]] | None:
        return self.utility_estimates[-1] if self.utility_estimates else None

    def actor_selection_frequencies(self) -> list[float]:
        if not self.selected_actor_indices:
            return []
        counts: dict[int, int] = {}
        for index in self.selected_actor_indices:
            counts[index] = counts.get(index, 0) + 1
        total = len(self.selected_actor_indices)
        num_actors = int(self.metadata.get("num_actors", max(counts) + 1))
        return [counts.get(actor, 0) / total for actor in range(num_actors)]


class GroupMultiActorTrainer:
    """Coordinator for paper Section 6 group-level imperfect-reward experiments.

    Parameters
    ----------
    env:
        I.i.d. state environment exposing ``reset()`` that returns
        ``(observation, info)`` with ``info["state_index"]``. State sampling is
        taken from here; per-Observer rewards are computed from
        ``observer_reward_tables`` rather than from ``env.step``.
    actors:
        Length-``N`` list of Actors. Each policy must expose ``act`` and
        ``action_distribution``.
    observers:
        Length-``N`` list of ``BudgetedObserver``. Each must have
        ``num_actors == len(actors)``.
    observer_reward_tables:
        ``[num_observers][num_states][num_actions]`` reward tables. Entry
        ``[i][s][a]`` is ``R_i(s, a)``, the reward Observer ``i`` perceives when
        action ``a`` is taken in state ``s``.
    actor_selection_probs:
        Length-``N`` probability vector ``p_j``; must sum to 1.
    actor_optimizers:
        ``None`` recovers Section 6.1 (fixed Actor policies). A length-``N`` list
        enables Section 6.2; entry ``j`` updates Actor ``j`` with the
        Observer-aggregated reward.
    optimal_actions_per_state:
        Optional length-``num_states`` list giving the optimal action per state,
        used for the per-Actor optimal-action-probability diagnostic.
    action_exploration_probability, action_exploration_final, action_log_prob_source:
        Uniform action-exploration controls shared with ``MultiActorTrainer``.
        ``pi_mix = (1 - eps) * pi_theta + eps / num_actions``; ``eps`` anneals
        linearly from ``action_exploration_probability`` toward
        ``action_exploration_final`` (``None`` keeps it constant). ``eps = 0``
        (default) keeps the original REINFORCE loop byte-for-byte.
    """

    def __init__(
        self,
        env: Any,
        actors: list[Actor],
        observers: list[BudgetedObserver],
        observer_reward_tables: list[list[list[float]]],
        actor_selection_probs: list[float],
        actor_optimizers: list[torch.optim.Optimizer | None] | None = None,
        optimal_actions_per_state: list[int] | None = None,
        snapshot_interval: int = 1,
        action_exploration_probability: float = 0.0,
        action_exploration_final: float | None = None,
        action_log_prob_source: str = "mixed",
        device: str | torch.device = "cpu",
    ) -> None:
        num_actors = len(actors)
        if num_actors < 2:
            raise ValueError("GroupMultiActorTrainer requires at least two actors")
        if len(observers) < 1:
            raise ValueError("GroupMultiActorTrainer requires at least one observer")
        if any(observer.num_actors != num_actors for observer in observers):
            raise ValueError("every observer.num_actors must match len(actors)")
        if len(observer_reward_tables) != len(observers):
            raise ValueError("observer_reward_tables length must match len(observers)")
        if len(actor_selection_probs) != num_actors:
            raise ValueError("actor_selection_probs length must match len(actors)")
        if abs(sum(actor_selection_probs) - 1.0) > 1e-6:
            raise ValueError("actor_selection_probs must sum to 1")
        if any(probability < 0.0 for probability in actor_selection_probs):
            raise ValueError("actor_selection_probs entries must be non-negative")
        if actor_optimizers is not None and len(actor_optimizers) != num_actors:
            raise ValueError("actor_optimizers length must match len(actors)")
        if snapshot_interval <= 0:
            raise ValueError("snapshot_interval must be positive")
        if not 0.0 <= action_exploration_probability <= 1.0:
            raise ValueError("action_exploration_probability must be in [0, 1]")
        if action_exploration_final is not None and not (
            0.0 <= action_exploration_final <= 1.0
        ):
            raise ValueError("action_exploration_final must be in [0, 1]")
        if action_log_prob_source not in {"mixed", "learned"}:
            raise ValueError("action_log_prob_source must be 'mixed' or 'learned'")

        reward_tensor = torch.tensor(observer_reward_tables, dtype=torch.float32)
        if reward_tensor.dim() != 3:
            raise ValueError("observer_reward_tables must be 3-D [obs][state][action]")
        if reward_tensor.shape[2] != actors[0].policy.num_actions:
            raise ValueError("reward table action count must match policy num_actions")

        self.env = env
        self.actors = list(actors)
        self.observers = list(observers)
        self.observer_reward_tables = reward_tensor
        self.num_actors = num_actors
        self.num_observers = len(observers)
        self.actor_selection_probs = torch.tensor(
            actor_selection_probs, dtype=torch.float32
        )
        if actor_optimizers is None:
            self.actor_optimizers: list[torch.optim.Optimizer | None] = [None] * num_actors
        else:
            self.actor_optimizers = list(actor_optimizers)
        self.optimal_actions_per_state = (
            list(optimal_actions_per_state)
            if optimal_actions_per_state is not None
            else None
        )
        self.snapshot_interval = int(snapshot_interval)
        self.action_exploration_probability = float(action_exploration_probability)
        self.action_exploration_final = (
            None
            if action_exploration_final is None
            else float(action_exploration_final)
        )
        self.action_log_prob_source = action_log_prob_source
        self.device = torch.device(device)
        self._actor_selection_generator = torch.Generator()

    def _current_exploration(self, step: int, num_steps: int) -> float:
        """Annealed exploration weight ``eps`` for ``step`` (0-indexed)."""

        if (
            self.action_exploration_probability <= 0.0
            and (
                self.action_exploration_final is None
                or self.action_exploration_final <= 0.0
            )
        ):
            return 0.0
        return linear_anneal(
            self.action_exploration_probability,
            self.action_exploration_final,
            step,
            num_steps,
        )

    def run(self, num_steps: int, seed: int | None = None) -> GroupMultiActorTrainingTrace:
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")

        if seed is not None:
            set_global_seed(seed)
            self._actor_selection_generator.manual_seed(
                _stream_seed(seed, "actor_selection")
            )
            if hasattr(self.env, "_generator"):
                self.env._generator.manual_seed(_stream_seed(seed, "env_state"))

        trace = GroupMultiActorTrainingTrace(
            metadata={
                "seed": seed,
                "num_steps": num_steps,
                "num_actors": self.num_actors,
                "num_observers": self.num_observers,
                "budget": [observer.budget for observer in self.observers],
                "c": [observer.c for observer in self.observers],
                "snapshot_interval": self.snapshot_interval,
                "action_exploration_probability": self.action_exploration_probability,
                "action_exploration_final": self.action_exploration_final,
                "action_log_prob_source": self.action_log_prob_source,
            }
        )

        for step in range(num_steps):
            observation, info = self.env.reset()
            state_index = int(info["state_index"])
            actor_index = self._sample_actor_index()

            actor = self.actors[actor_index]
            actor_input = torch.as_tensor(
                observation, dtype=torch.float32, device=self.device
            )
            eps = self._current_exploration(step, num_steps)
            action_tensor, log_prob, _ = mixed_action_sample(
                actor,
                actor_input,
                eps,
                log_prob_source=self.action_log_prob_source,
            )
            action = int(action_tensor.detach().cpu().item())

            per_observer_reward = [
                float(self.observer_reward_tables[i, state_index, action].item())
                for i in range(self.num_observers)
            ]

            for i, observer in enumerate(self.observers):
                observer.update_utility(actor_index, per_observer_reward[i])

            optimizer = self.actor_optimizers[actor_index]
            if optimizer is not None:
                aggregated_reward = self._aggregate_reward(
                    actor_index, per_observer_reward
                )
                loss = reinforce_loss(aggregated_reward, log_prob)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            for observer in self.observers:
                observer.update_following_rates()

            trace.selected_actor_indices.append(actor_index)
            trace.actions.append(action)
            trace.observer_rewards.append(per_observer_reward)

            if (step + 1) % self.snapshot_interval == 0 or step + 1 == num_steps:
                trace.following_rates.append(
                    [observer.current_following_rates() for observer in self.observers]
                )
                trace.utility_estimates.append(
                    [observer.current_utilities() for observer in self.observers]
                )
                trace.per_actor_optimal_action_probs.append(
                    self._snapshot_optimal_action_probs()
                )

        return trace

    def _aggregate_reward(
        self, actor_index: int, per_observer_reward: list[float]
    ) -> float:
        """Eq. (29) aggregate: sum_i theta(mu_{i,j}) * R_i(s, a)."""

        total = 0.0
        for i, observer in enumerate(self.observers):
            total += observer.report_probability(actor_index) * per_observer_reward[i]
        return total

    def _sample_actor_index(self) -> int:
        sample = torch.multinomial(
            self.actor_selection_probs,
            num_samples=1,
            replacement=True,
            generator=self._actor_selection_generator,
        )
        return int(sample.item())

    def _snapshot_optimal_action_probs(self) -> list[float]:
        if self.optimal_actions_per_state is None:
            return []
        snapshot: list[float] = []
        num_states = len(self.optimal_actions_per_state)
        with torch.no_grad():
            for actor in self.actors:
                per_state_probs: list[float] = []
                for state_index in range(num_states):
                    observation = self._eval_observation(state_index, num_states)
                    distribution = actor.policy.action_distribution(observation)
                    optimal_action = self.optimal_actions_per_state[state_index]
                    per_state_probs.append(
                        float(distribution.probs[optimal_action].item())
                    )
                snapshot.append(sum(per_state_probs) / len(per_state_probs))
        return snapshot

    def _eval_observation(self, state_index: int, num_states: int) -> torch.Tensor:
        """Observation for deterministic evaluation, matching the env's mode."""
        observe = getattr(self.env, "observation_for_index", None)
        if callable(observe):
            return observe(state_index).to(dtype=torch.float32, device=self.device)
        return torch.nn.functional.one_hot(
            torch.tensor(state_index),
            num_classes=num_states,
        ).to(dtype=torch.float32, device=self.device)
