"""Trainer for paper Sections 4.1 and 4.2.

A single ``BudgetedObserver`` watches ``N`` Actors that share an i.i.d.
state environment. At each step, one Actor is sampled from the selection
distribution ``p``, performs an action, and produces a reward that the
Observer ingests. The Observer maintains a running estimate of each
Actor's utility and projects its budgeted following-rates ``mu`` toward
high-utility actors.

Section 4.1 keeps Actor policies fixed (``actor_optimizers`` are ``None``):
the only learnable quantities are the Observer's ``mu`` and ``utilities``.
Section 4.2 adds learnable Actor policies: when the Observer's Bernoulli
draw ``Bernoulli(theta(mu_j))`` succeeds, the corresponding Actor receives
its reward and performs a REINFORCE update.

The two-timescale dynamic of the paper is recovered by choosing
``utility_learning_rate >> following_learning_rate`` (and Actor learning
rates) at construction time; the trainer itself runs every update on the
same wall-clock step.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import torch

from actor_observer_rl.agents.actor import Actor
from actor_observer_rl.agents.budgeted_observer import BudgetedObserver
from actor_observer_rl.training.updates import (
    linear_anneal,
    mixed_action_sample,
    reinforce_loss,
)
from actor_observer_rl.utils.seeding import set_global_seed


def _stream_seed(seed: int, stream_name: str) -> int:
    """Derive an independent sub-stream seed deterministically from ``seed``.

    Sampling streams that share the same ``torch.Generator`` seed and the same
    categorical distribution produce identical index sequences. We avoid that
    coupling by hashing the stream name into a 64-bit offset.
    """

    digest = hashlib.sha256(stream_name.encode("utf-8")).digest()
    offset = int.from_bytes(digest[:8], "big", signed=False)
    return (int(seed) + offset) & ((1 << 63) - 1)


@dataclass
class MultiActorTrainingTrace:
    """Per-step trace produced by ``MultiActorTrainer.run``."""

    selected_actor_indices: list[int] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    reported_to_actor: list[bool] = field(default_factory=list)
    following_rates: list[list[float]] = field(default_factory=list)
    report_probabilities: list[list[float]] = field(default_factory=list)
    utility_estimates: list[list[float]] = field(default_factory=list)
    per_actor_optimal_action_probs: list[list[float]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_steps(self) -> int:
        return len(self.rewards)

    @property
    def final_following_rates(self) -> list[float] | None:
        return self.following_rates[-1] if self.following_rates else None

    @property
    def final_utilities(self) -> list[float] | None:
        return self.utility_estimates[-1] if self.utility_estimates else None

    def actor_selection_frequencies(self) -> list[float]:
        """Empirical fraction of steps each actor was selected to act."""

        if not self.selected_actor_indices:
            return []
        counts: dict[int, int] = {}
        for index in self.selected_actor_indices:
            counts[index] = counts.get(index, 0) + 1
        total = len(self.selected_actor_indices)
        num_actors = int(self.metadata.get("num_actors", max(counts) + 1))
        return [counts.get(actor, 0) / total for actor in range(num_actors)]


class MultiActorTrainer:
    """Coordinator for paper Section 4 multi-actor experiments.

    Parameters
    ----------
    env:
        I.i.d. state environment whose ``step`` returns the scalar reward
        ``R(s_t, a_t)``. ``IIDStateBanditEnv`` is the canonical choice.
    actors:
        Sequence of length ``N``. Each Actor must expose ``act(input)`` and
        the underlying policy must expose ``action_distribution(input)``.
    observer:
        ``BudgetedObserver`` whose ``num_actors`` equals ``len(actors)``.
    actor_selection_probs:
        Length-``N`` probability vector ``p_j`` over which Actor acts at
        each step. Must sum to 1.
    actor_optimizers:
        Per-actor optimizer. ``None`` entries disable updates for that
        actor and recover Section 4.1; populated entries enable Section
        4.2 REINFORCE updates whenever the Observer's report draw succeeds.
    optimal_actions_per_state:
        Optional list of length ``num_states`` giving the optimal action for
        each state. When provided, the trainer logs each actor's average
        probability of choosing its state-specific optimal action; useful for
        the mechanism-level figures in Section 4.
    reward_baseline:
        ``None`` (default) keeps the exact Section 4 update
        ``loss = -reward * log_prob``. ``"running_mean"`` subtracts a per-actor
        running-mean baseline before the REINFORCE step, which is needed when
        Actors learn on an all-positive reward such as the structured grid used
        by the Phase 3 scaling probe.
    force_report:
        ``False`` (default) gates each Actor's REINFORCE update behind the
        Observer's Bernoulli draw ``Bernoulli(theta(mu_j))``, the imperfect-
        reward mechanism. ``True`` hard-codes         the report probability to 1 so
        every acting Actor is always updated -- the perfect-reward baseline
        Professor Marbach asked for. Taking the budget ``B`` (and hence the
        following rates and report probabilities) to infinity recovers exactly
        this limit, so it is the performance upper bound the imperfect-reward
        runs should approach as ``B`` grows.
    action_exploration_probability:
        Initial weight ``eps`` of the uniform action floor mixed into each
        Actor's policy: ``pi_mix = (1 - eps) * pi_theta + eps / num_actions``.
        ``0.0`` (default) disables exploration and keeps the original REINFORCE
        loop byte-for-byte. Positive values implement the uniform-exploration
        recipe Professor Marbach / Amanda asked for, which fixed the occasional
        stuck Actor at large ``N``/``K``.
    action_exploration_final:
        Target ``eps`` reached by the final step via linear annealing. ``None``
        (default) keeps ``eps`` constant. Set to ``0.0`` to anneal exploration
        away and sharpen the learned policy by the end of training.
    action_log_prob_source:
        ``"mixed"`` (default) updates with ``log pi_mix`` (correct importance
        weight for the mixed sampler); ``"learned"`` updates with ``log
        pi_theta``. Only relevant when exploration is active.
    """

    def __init__(
        self,
        env: Any,
        actors: list[Actor],
        observer: BudgetedObserver,
        actor_selection_probs: list[float],
        actor_optimizers: list[torch.optim.Optimizer | None] | None = None,
        optimal_actions_per_state: list[int] | None = None,
        snapshot_interval: int = 1,
        reward_baseline: str | None = None,
        baseline_momentum: float = 0.01,
        force_report: bool = False,
        action_exploration_probability: float = 0.0,
        action_exploration_final: float | None = None,
        action_log_prob_source: str = "mixed",
        device: str | torch.device = "cpu",
    ) -> None:
        if len(actors) < 2:
            raise ValueError("MultiActorTrainer requires at least two actors")
        if len(actors) != observer.num_actors:
            raise ValueError("observer.num_actors must match len(actors)")
        if len(actor_selection_probs) != len(actors):
            raise ValueError("actor_selection_probs length must match len(actors)")
        if abs(sum(actor_selection_probs) - 1.0) > 1e-6:
            raise ValueError("actor_selection_probs must sum to 1")
        if any(probability < 0.0 for probability in actor_selection_probs):
            raise ValueError("actor_selection_probs entries must be non-negative")
        if actor_optimizers is not None and len(actor_optimizers) != len(actors):
            raise ValueError("actor_optimizers length must match len(actors)")
        if snapshot_interval <= 0:
            raise ValueError("snapshot_interval must be positive")
        if reward_baseline is not None and reward_baseline != "running_mean":
            raise ValueError("reward_baseline must be None or 'running_mean'")
        if not 0.0 < baseline_momentum <= 1.0:
            raise ValueError("baseline_momentum must be in (0, 1]")
        if not 0.0 <= action_exploration_probability <= 1.0:
            raise ValueError("action_exploration_probability must be in [0, 1]")
        if action_exploration_final is not None and not (
            0.0 <= action_exploration_final <= 1.0
        ):
            raise ValueError("action_exploration_final must be in [0, 1]")
        if action_log_prob_source not in {"mixed", "learned"}:
            raise ValueError("action_log_prob_source must be 'mixed' or 'learned'")

        self.env = env
        self.actors = list(actors)
        self.observer = observer
        self.actor_selection_probs = torch.tensor(
            actor_selection_probs, dtype=torch.float32
        )
        if actor_optimizers is None:
            self.actor_optimizers: list[torch.optim.Optimizer | None] = [None] * len(actors)
        else:
            self.actor_optimizers = list(actor_optimizers)
        self.optimal_actions_per_state = (
            list(optimal_actions_per_state)
            if optimal_actions_per_state is not None
            else None
        )
        self.snapshot_interval = int(snapshot_interval)
        self.reward_baseline = reward_baseline
        self.baseline_momentum = float(baseline_momentum)
        self._running_baselines: list[float | None] = [None] * len(actors)
        self.force_report = bool(force_report)
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

    def _advantage(self, actor_index: int, reward: float) -> float:
        """Subtract an optional per-actor running-mean reward baseline.

        With all-positive rewards (the structured grid task), baseline-free
        REINFORCE collapses toward whichever action is sampled first, because
        every update pushes log-probabilities up. A per-actor running mean
        centers each actor's own rewards and recovers a proper advantage sign.
        When ``reward_baseline`` is ``None`` (the default, used by Section 4),
        the reward is returned unchanged so the original update is exact.
        """

        if self.reward_baseline is None:
            return reward
        current = self._running_baselines[actor_index]
        if current is None:
            self._running_baselines[actor_index] = reward
        else:
            current += self.baseline_momentum * (reward - current)
            self._running_baselines[actor_index] = current
        return reward - self._running_baselines[actor_index]

    def run(
        self,
        num_steps: int,
        seed: int | None = None,
    ) -> MultiActorTrainingTrace:
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")

        if seed is not None:
            set_global_seed(seed)
            self._actor_selection_generator.manual_seed(_stream_seed(seed, "actor_selection"))
            if hasattr(self.env, "_generator"):
                self.env._generator.manual_seed(_stream_seed(seed, "env_state"))

        trace = MultiActorTrainingTrace(
            metadata={
                "seed": seed,
                "num_steps": num_steps,
                "num_actors": len(self.actors),
                "budget": self.observer.budget,
                "c": self.observer.c,
                "utility_learning_rate": self.observer.utility_learning_rate,
                "following_learning_rate": self.observer.following_learning_rate,
                "snapshot_interval": self.snapshot_interval,
                "force_report": self.force_report,
                "action_exploration_probability": self.action_exploration_probability,
                "action_exploration_final": self.action_exploration_final,
                "action_log_prob_source": self.action_log_prob_source,
            }
        )

        for step in range(num_steps):
            observation, _ = self.env.reset()
            actor_index = self._sample_actor_index()

            actor = self.actors[actor_index]
            actor_input = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
            eps = self._current_exploration(step, num_steps)
            action_tensor, log_prob, _ = mixed_action_sample(
                actor,
                actor_input,
                eps,
                log_prob_source=self.action_log_prob_source,
            )
            action = int(action_tensor.detach().cpu().item())

            step_result = self.env.step(action)
            reward = float(step_result.reward)

            self.observer.update_utility(actor_index, reward)

            advantage = self._advantage(actor_index, reward)
            reported = True if self.force_report else self.observer.should_report(actor_index)
            optimizer = self.actor_optimizers[actor_index]
            if reported and optimizer is not None:
                loss = reinforce_loss(advantage, log_prob)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            self.observer.update_following_rates()

            trace.selected_actor_indices.append(actor_index)
            trace.actions.append(action)
            trace.rewards.append(reward)
            trace.reported_to_actor.append(reported)

            if (step + 1) % self.snapshot_interval == 0 or step + 1 == num_steps:
                trace.following_rates.append(self.observer.current_following_rates())
                trace.report_probabilities.append(
                    [float(p) for p in self.observer.report_probabilities()]
                )
                trace.utility_estimates.append(self.observer.current_utilities())
                trace.per_actor_optimal_action_probs.append(
                    self._snapshot_optimal_action_probs()
                )

        return trace

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


def make_fixed_softmax_actor(
    num_states: int,
    num_actions: int,
    target_action: int,
    sharpness: float = 5.0,
    device: str | torch.device = "cpu",
    input_dim: int | None = None,
) -> Actor:
    """Build a deterministic-by-default Actor for Section 4.1 experiments.

    Returns an ``Actor`` whose ``SoftmaxPolicy`` always assigns high
    probability to ``target_action`` regardless of the input state. The
    linear weights are zeroed, so ``input_dim`` only sizes the (unused) weight
    matrix and the behavior is identical for one-hot (``input_dim=num_states``)
    or scalar (``input_dim=1``) observations. ``sharpness`` controls how peaked
    the action distribution is. The returned Actor still constructs ``log_prob``
    so it can be reused in Section 4.2 by attaching an optimizer.
    """

    from actor_observer_rl.policies import SoftmaxPolicy  # local import to avoid cycle

    if target_action < 0 or target_action >= num_actions:
        raise ValueError("target_action outside valid range")
    if sharpness <= 0.0:
        raise ValueError("sharpness must be positive")

    policy = SoftmaxPolicy(
        input_dim=num_states if input_dim is None else input_dim,
        num_actions=num_actions,
    )
    with torch.no_grad():
        policy.linear.weight.zero_()
        bias = torch.full((num_actions,), -sharpness, dtype=torch.float32)
        bias[target_action] = sharpness
        policy.linear.bias.copy_(bias)
    policy.to(device)
    return Actor(policy=policy)
