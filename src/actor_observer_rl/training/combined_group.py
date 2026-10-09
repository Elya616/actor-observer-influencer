"""Combined imperfect-reward + imperfect-state group loop (toward Section 6).

This trainer fuses the two tracks into one loop so the integration ladder can be
climbed one knob at a time:

* imperfect state (Section 5): each Observer selects an Actor-compressor via
  softmax selection logits ``w`` (probabilities ``lambda``), receives that
  compressor's scalar message ``m = f_phi(s)``, and acts with policy ``theta``.
* imperfect reward (Section 4): the reward feedback that drives the learning
  update is *reported* only with probability ``theta(mu) = 1 - e^{-c mu}``, where
  each Observer holds a budgeted following-rate vector ``mu`` over the
  compressors (``sum_k mu_{i,k} <= B``). When the report does not fire, the
  policy/compression update for that step is skipped -- exactly the Section 4
  imperfect-reward gate, now applied to the Section 5 loop.

The four learnable knobs (``phi``, ``w``, ``theta``, ``mu``) can each be frozen
or learned independently, which is what the Route A / Route B ladders need. The
PA baseline freezes all four.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from statistics import fmean
from typing import Any

import torch

from actor_observer_rl.agents.influencer import Influencer
from actor_observer_rl.agents.budgeted_observer import project_capped_simplex
from actor_observer_rl.training.informational_state import (
    ActorCompressor,
    SelectionLogits,
)
from actor_observer_rl.utils.seeding import set_global_seed


@dataclass
class CombinedGroupTrace:
    """Per-evaluation trace for the combined group loop."""

    evaluation_steps: list[int] = field(default_factory=list)
    expected_rewards: list[float] = field(default_factory=list)
    mean_correct_action_probs: list[float] = field(default_factory=list)
    selection_probabilities_by_observer: list[list[list[float]]] = field(
        default_factory=list
    )
    following_rates_by_observer: list[list[list[float]]] = field(default_factory=list)
    report_rate: list[float] = field(default_factory=list)
    influencer_actor_probabilities: list[list[float]] = field(default_factory=list)
    debug_rows: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def final_expected_reward(self) -> float | None:
        return self.expected_rewards[-1] if self.expected_rewards else None

    @property
    def final_mean_correct_action_prob(self) -> float | None:
        return (
            self.mean_correct_action_probs[-1]
            if self.mean_correct_action_probs
            else None
        )


class CombinedGroupTrainer:
    """Group loop with compression, selection, policy, and budgeted following."""

    def __init__(
        self,
        env: Any,
        actor_compressors: Sequence[ActorCompressor],
        observer_policies: Sequence[torch.nn.Module],
        selection_logits_by_observer: Sequence[SelectionLogits],
        *,
        observer_reward_tables: Sequence[Sequence[Sequence[float]]] | None = None,
        budget: float = 2.0,
        c: float = 1.0,
        utility_learning_rate: float = 0.1,
        following_learning_rate: float = 0.01,
        compression_optimizer: torch.optim.Optimizer | None = None,
        selection_optimizer: torch.optim.Optimizer | None = None,
        policy_optimizers: Sequence[torch.optim.Optimizer | None] | None = None,
        learn_following: bool = False,
        force_report: bool = False,
        selection_exploration_probability: float = 0.0,
        optimal_actions: Sequence[int] | None = None,
        device: str | torch.device = "cpu",
        influencer: Influencer | None = None,
        influencer_optimizer: torch.optim.Optimizer | None = None,
        debug: bool = False,
    ) -> None:
        actor_compressors = list(actor_compressors)
        observer_policies = list(observer_policies)
        selection_logits = list(selection_logits_by_observer)
        if len(actor_compressors) < 1:
            raise ValueError("at least one Actor-compressor is required")
        if len(selection_logits) != len(observer_policies):
            raise ValueError("selection logits count must match Observer count")
        num_sources = len(actor_compressors) + int(influencer is not None)
        for logits in selection_logits:
            if logits.num_actor_compressors != num_sources:
                raise ValueError("selection size must match Actors plus optional Influencer")
        if influencer is not None and influencer.logits.numel() != len(actor_compressors):
            raise ValueError("Influencer size must match Actor count")
        if influencer is None and influencer_optimizer is not None:
            raise ValueError("influencer_optimizer requires an Influencer")
        if budget <= 0.0:
            raise ValueError("budget must be positive")
        if not 0.0 <= selection_exploration_probability <= 1.0:
            raise ValueError("selection_exploration_probability must be in [0, 1]")

        if policy_optimizers is None:
            policy_optimizers = [None] * len(observer_policies)
        else:
            policy_optimizers = list(policy_optimizers)
            if len(policy_optimizers) != len(observer_policies):
                raise ValueError("policy_optimizers length must match Observer count")

        self.env = env
        self.actor_compressors = actor_compressors
        self.observer_policies = observer_policies
        self.selection_logits_by_observer = selection_logits
        self.observer_reward_tables = self._validate_reward_tables(
            observer_reward_tables
        )
        self.budget = float(budget)
        self.c = float(c)
        self.utility_learning_rate = float(utility_learning_rate)
        self.following_learning_rate = float(following_learning_rate)
        self.compression_optimizer = compression_optimizer
        self.selection_optimizer = selection_optimizer
        self.policy_optimizers = policy_optimizers
        self.influencer = influencer
        self.influencer_optimizer = influencer_optimizer
        self.debug = debug
        self.learn_following = bool(learn_following)
        self.force_report = bool(force_report)
        self.selection_exploration_probability = float(
            selection_exploration_probability
        )
        self.device = torch.device(device)
        self.optimal_actions = (
            tuple(range(env.num_states))
            if optimal_actions is None
            else tuple(int(a) for a in optimal_actions)
        )
        if len(self.optimal_actions) != env.num_states:
            raise ValueError("optimal_actions length must match env.num_states")

        num_observers = len(observer_policies)
        # Budgeted following rates mu and utility estimates U per Observer.
        uniform = self.budget / num_sources
        self.mu = torch.full(
            (num_observers, num_sources), uniform, dtype=torch.float32
        )
        self.utilities = torch.zeros(
            (num_observers, num_sources), dtype=torch.float32
        )

    @property
    def num_actor_compressors(self) -> int:
        return len(self.actor_compressors)

    @property
    def num_observers(self) -> int:
        return len(self.observer_policies)

    def run(
        self,
        num_steps: int,
        seed: int | None = None,
        eval_interval: int = 200,
    ) -> CombinedGroupTrace:
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")
        if seed is not None:
            set_global_seed(seed)

        trace = CombinedGroupTrace(
            metadata={
                "seed": seed,
                "num_steps": num_steps,
                "num_actor_compressors": self.num_actor_compressors,
                "num_observers": self.num_observers,
                "learn_compression": self.compression_optimizer is not None,
                "learn_selection": self.selection_optimizer is not None,
                "learn_policy": any(o is not None for o in self.policy_optimizers),
                "learn_following": self.learn_following,
                "has_influencer": self.influencer is not None,
                "force_report": self.force_report,
                "budget": self.budget,
                "c": self.c,
            }
        )
        self._record(trace, step=0, report_rate=0.0)

        reports = 0
        reportable = 0
        for step_index in range(num_steps):
            _obs, info = self.env.reset(seed=seed if step_index == 0 else None)
            state_index = int(info["state_index"])
            full_state = self._to_tensor(self.env.get_full_state())

            losses: list[torch.Tensor] = []
            for observer_index, policy in enumerate(self.observer_policies):
                source_index, selection_log_prob = self._select(observer_index)
                actor_index = source_index
                influencer_log_prob = None
                via_influencer = source_index == self.num_actor_compressors
                if via_influencer:
                    actor_index, influencer_log_prob = self.influencer.select_actor()
                actor_message = self.actor_compressors[actor_index].message(full_state)
                message = self.influencer.relay(actor_message) if via_influencer else actor_message
                action_result = policy.act(message, sample=True)
                action = int(action_result.action.detach().cpu().item())
                reward = float(self.observer_reward_tables[observer_index][state_index][action])

                # Observer observes the reward; update its utility estimate of
                # the direct or Influencer source it listened to (fast timescale).
                self._update_utility(observer_index, source_index, reward)

                # Selection logits always update from the observed reward.
                if self.selection_optimizer is not None and selection_log_prob is not None:
                    losses.append(-reward * selection_log_prob)
                if self.influencer_optimizer is not None and influencer_log_prob is not None:
                    losses.append(-reward * influencer_log_prob)

                # Imperfect-reward gate: the policy / compression feedback fires
                # only when the report succeeds.
                reportable += 1
                report = self._report(observer_index, source_index)
                if report:
                    reports += 1
                    if (
                        self.compression_optimizer is not None
                        or self.policy_optimizers[observer_index] is not None
                    ):
                        losses.append(-reward * action_result.log_prob)

                if self.debug:
                    assert torch.isfinite(message).all(), "non-finite message"
                    assert torch.equal(message, actor_message), "relay changed message"
                    assert math.isfinite(reward), "non-finite reward"
                    self._check_probabilities(action_result.probs)
                    trace.debug_rows.append({
                        "step": step_index + 1, "state": state_index,
                        "observer": observer_index, "source": source_index,
                        "actor": actor_index, "via_influencer": via_influencer,
                        "message": message.detach().cpu().reshape(-1).tolist(),
                        "action": action, "reward": reward, "reported": report,
                    })

            if losses:
                loss = torch.stack(losses).sum()
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite combined loss")
                self._zero_grad()
                loss.backward()
                if self.debug:
                    for optimizer in self._optimizers():
                        for group in optimizer.param_groups:
                            for parameter in group["params"]:
                                if parameter.grad is not None:
                                    assert torch.isfinite(parameter.grad).all(), "non-finite gradient"
                self._step()

            if self.learn_following:
                self._update_following_rates()
            if self.debug:
                assert torch.isfinite(self.mu).all() and (self.mu >= 0).all(), "invalid following rates"
                assert (self.mu.sum(dim=1) <= self.budget + 1e-5).all(), "budget exceeded"
                for logits in self.selection_logits_by_observer:
                    self._check_probabilities(logits.probabilities())
                if self.influencer is not None:
                    self._check_probabilities(self.influencer.probabilities())

            current = step_index + 1
            if current % eval_interval == 0 or current == num_steps:
                rate = reports / max(1, reportable)
                self._record(trace, step=current, report_rate=rate)

        return trace

    @torch.no_grad()
    def evaluate(self) -> tuple[float, float]:
        """Expected reward and mean correct-action prob under learned selection."""
        total_reward = 0.0
        total_correct = 0.0
        for state_index in range(self.env.num_states):
            state_prob = self.env.state_probability(state_index)
            state = self._to_tensor(self.env.full_state_for_index(state_index))
            for observer_index, policy in enumerate(self.observer_policies):
                selection_probs = self._selection_probabilities(observer_index)
                if self.influencer is not None:
                    # Marginalize the relay path rather than sampling during evaluation.
                    selection_probs = (
                        selection_probs[:-1]
                        + selection_probs[-1] * self.influencer.probabilities()
                    )
                for actor_index, sel_prob in enumerate(selection_probs):
                    message = self.actor_compressors[actor_index].message(state)
                    action_probs = policy.action_distribution(message).probs
                    rewards = torch.tensor(
                        self.observer_reward_tables[observer_index][state_index],
                        dtype=torch.float32,
                        device=action_probs.device,
                    )
                    weight = float(state_prob * sel_prob) / self.num_observers
                    total_reward += weight * float(torch.dot(action_probs, rewards))
                    correct = self.optimal_actions[state_index]
                    total_correct += weight * float(action_probs[correct])
        return total_reward, total_correct

    @staticmethod
    def _check_probabilities(probs: torch.Tensor) -> None:
        assert torch.isfinite(probs).all() and (probs >= 0).all(), "invalid probabilities"
        assert torch.isclose(probs.sum(), probs.new_tensor(1.0), atol=1e-5), "probabilities must sum to 1"

    def _select(self, observer_index: int) -> tuple[int, torch.Tensor | None]:
        return self.selection_logits_by_observer[observer_index].sample(
            exploration_probability=self.selection_exploration_probability,
            log_prob_source="learned",
            sample=True,
        )

    def _report(self, observer_index: int, actor_index: int) -> bool:
        if self.force_report:
            return True
        mu = float(self.mu[observer_index, actor_index])
        prob = 1.0 - math.exp(-self.c * mu)
        if self.debug:
            assert 0.0 <= prob <= 1.0, "invalid report probability"
        return bool(torch.rand(1).item() < prob)

    def _update_utility(self, observer_index: int, actor_index: int, reward: float) -> None:
        current = float(self.utilities[observer_index, actor_index])
        self.utilities[observer_index, actor_index] = current + self.utility_learning_rate * (
            reward - current
        )

    def _update_following_rates(self) -> None:
        # Projected gradient ascent on sum_k theta(mu_k) U_k with budget B.
        grad = self.c * torch.exp(-self.c * self.mu) * self.utilities
        proposed = self.mu + self.following_learning_rate * grad
        for observer_index in range(self.num_observers):
            self.mu[observer_index] = project_capped_simplex(
                proposed[observer_index], budget=self.budget
            )

    def _selection_probabilities(self, observer_index: int) -> torch.Tensor:
        return self.selection_logits_by_observer[observer_index].current_probabilities()

    def _record(self, trace: CombinedGroupTrace, *, step: int, report_rate: float) -> None:
        expected, correct = self.evaluate()
        trace.evaluation_steps.append(step)
        trace.expected_rewards.append(expected)
        trace.mean_correct_action_probs.append(correct)
        trace.selection_probabilities_by_observer.append(
            [
                logits.current_probabilities().detach().cpu().tolist()
                for logits in self.selection_logits_by_observer
            ]
        )
        trace.following_rates_by_observer.append(
            [row.detach().cpu().tolist() for row in self.mu]
        )
        trace.report_rate.append(report_rate)
        if self.influencer is not None:
            trace.influencer_actor_probabilities.append(
                self.influencer.probabilities().detach().cpu().tolist()
            )

    def _validate_reward_tables(
        self,
        observer_reward_tables: Sequence[Sequence[Sequence[float]]] | None,
    ) -> list[list[list[float]]]:
        if observer_reward_tables is None:
            return [
                [[float(v) for v in row] for row in self.env.reward_table]
                for _ in self.observer_policies
            ]
        tables = [
            [[float(v) for v in row] for row in table]
            for table in observer_reward_tables
        ]
        if len(tables) != len(self.observer_policies):
            raise ValueError("observer_reward_tables length must match Observer count")
        return tables

    def _zero_grad(self) -> None:
        for compressor in self.actor_compressors:
            compressor.zero_grad(set_to_none=True)
        for policy in self.observer_policies:
            policy.zero_grad(set_to_none=True)
        for logits in self.selection_logits_by_observer:
            logits.zero_grad(set_to_none=True)
        if self.influencer is not None:
            self.influencer.zero_grad(set_to_none=True)
        for optimizer in self._optimizers():
            optimizer.zero_grad(set_to_none=True)

    def _step(self) -> None:
        for optimizer in self._optimizers():
            optimizer.step()

    def _optimizers(self) -> list[torch.optim.Optimizer]:
        optimizers: list[torch.optim.Optimizer] = []
        if self.compression_optimizer is not None:
            optimizers.append(self.compression_optimizer)
        if self.selection_optimizer is not None:
            optimizers.append(self.selection_optimizer)
        if self.influencer_optimizer is not None:
            optimizers.append(self.influencer_optimizer)
        optimizers.extend(o for o in self.policy_optimizers if o is not None)
        return optimizers

    def _to_tensor(self, value: Any) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.to(self.device)
        return torch.as_tensor(value, dtype=torch.float32, device=self.device)
