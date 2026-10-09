"""Updated-paper imperfect-state interaction loops.

The May 22 paper version reverses the old imperfect-state direction. Actors
produce compressed state messages. Observers choose which Actor-compressor to
listen to, act from the received message, and send the gradient signal that
updates the selected Actor-compressor.

This module keeps that architecture separate from the older trainers:

* Section 5.1: one Actor-compressor, one Observer, learn compression ``phi``.
* Section 5.2: one Observer selects among Actor-compressors, learn ``w`` and
  the selected ``phi``.
* Section 5.3: many Observers each select an Actor-compressor and act; learn
  Observer policies ``theta_i``, selection logits ``w_i``, and Actor-compressor
  messages ``phi_j``.

The controlled imperfect-state experiments use cumulative binary states and
scalar messages. For state ``k``, the first ``k`` coordinates are one, so the
sum of the state vector is the state ID.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from statistics import fmean
from typing import Any

import torch

from actor_observer_rl.agents import ActionResult
from actor_observer_rl.policies import SumDecoderPolicy
from actor_observer_rl.utils.seeding import set_global_seed


def cumulative_binary_state_features(num_states: int) -> list[list[float]]:
    """Return cumulative state vectors whose coordinate sum is the state ID."""
    if num_states <= 1:
        raise ValueError("num_states must be greater than one")
    return [
        [1.0 if coordinate < state_index else 0.0 for coordinate in range(num_states)]
        for state_index in range(num_states)
    ]


def diagonal_reward_table(
    num_states: int,
    *,
    correct_reward: float = 1.0,
    wrong_reward: float = -0.1,
) -> list[list[float]]:
    """Return a table where action ``k`` is best in state ``k``."""
    if num_states <= 1:
        raise ValueError("num_states must be greater than one")
    return [
        [
            float(correct_reward) if action == state_index else float(wrong_reward)
            for action in range(num_states)
        ]
        for state_index in range(num_states)
    ]


class MaskedLinearActorCompressor(torch.nn.Module):
    """Linear scalar message function with a fixed input-availability mask."""

    def __init__(
        self,
        input_dim: int,
        message_dim: int = 1,
        input_mask: Sequence[float] | torch.Tensor | None = None,
    ) -> None:
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if message_dim <= 0:
            raise ValueError("message_dim must be positive")
        super().__init__()
        if input_mask is None:
            mask = torch.ones(input_dim, dtype=torch.float32)
        else:
            mask = torch.as_tensor(input_mask, dtype=torch.float32)
            if mask.shape != torch.Size([input_dim]):
                raise ValueError("input_mask length must match input_dim")
        self.input_dim = int(input_dim)
        self.message_dim = int(message_dim)
        self.register_buffer("input_mask", mask)
        self.linear = torch.nn.Linear(input_dim, message_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        state = state.float()
        if state.shape[-1] != self.input_dim:
            raise ValueError("state length must match input_dim")
        return self.linear(state * self.input_mask)


class ActorCompressor(torch.nn.Module):
    """Message sender in the updated imperfect-state setup."""

    def __init__(self, message_module: torch.nn.Module, name: str | None = None) -> None:
        super().__init__()
        self.message_module = message_module
        self.name = name or "actor_compressor"

    def message(self, state: torch.Tensor) -> torch.Tensor:
        return self.message_module(state)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.message(state)


class SelectionLogits(torch.nn.Module):
    """Observer-specific categorical distribution over Actor-compressors."""

    def __init__(
        self,
        num_actor_compressors: int,
        init_logits: Sequence[float] | None = None,
    ) -> None:
        if num_actor_compressors <= 1:
            raise ValueError("num_actor_compressors must be greater than one")
        super().__init__()
        if init_logits is None:
            initial = torch.zeros(num_actor_compressors, dtype=torch.float32)
        else:
            initial = torch.as_tensor(init_logits, dtype=torch.float32)
            if initial.shape != torch.Size([num_actor_compressors]):
                raise ValueError("init_logits length must match num_actor_compressors")
        self.num_actor_compressors = int(num_actor_compressors)
        self.logits = torch.nn.Parameter(initial)

    def probabilities(self) -> torch.Tensor:
        return torch.softmax(self.logits, dim=-1)

    def mixed_probabilities(self, exploration_probability: float = 0.0) -> torch.Tensor:
        epsilon = float(exploration_probability)
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("exploration_probability must be in [0, 1]")
        probs = self.probabilities()
        if epsilon == 0.0:
            return probs
        uniform = torch.full_like(probs, 1.0 / self.num_actor_compressors)
        return (1.0 - epsilon) * probs + epsilon * uniform

    def sample(
        self,
        *,
        exploration_probability: float = 0.0,
        log_prob_source: str = "learned",
        sample: bool = True,
    ) -> tuple[int, torch.Tensor]:
        """Sample an Actor-compressor index and return its log probability.

        The sampling distribution may include a uniform floor. By default the
        returned log probability is taken from the learned softmax, matching the
        usual "explore while still updating learned logits" convention.
        """
        if log_prob_source not in {"learned", "mixed"}:
            raise ValueError("log_prob_source must be 'learned' or 'mixed'")
        mixed_probs = self.mixed_probabilities(exploration_probability)
        if sample:
            sampled = torch.distributions.Categorical(probs=mixed_probs).sample()
        else:
            sampled = mixed_probs.argmax(dim=-1)
        if log_prob_source == "mixed":
            log_prob = torch.log(mixed_probs[sampled].clamp_min(1e-12))
        else:
            learned_probs = self.probabilities()
            log_prob = torch.log(learned_probs[sampled].clamp_min(1e-12))
        return int(sampled.item()), log_prob

    def current_probabilities(self) -> torch.Tensor:
        with torch.no_grad():
            return self.probabilities()


@dataclass
class UpdatedImperfectStateTrace:
    """Training trace for the updated-paper imperfect-state loops."""

    rewards: list[float] = field(default_factory=list)
    average_rewards: list[float] = field(default_factory=list)
    state_indices: list[int] = field(default_factory=list)
    selected_actor_indices: list[list[int]] = field(default_factory=list)
    actions: list[list[int]] = field(default_factory=list)
    messages: list[list[list[float]]] = field(default_factory=list)
    compression_feedback_counts: list[list[int]] = field(default_factory=list)
    evaluation_steps: list[int] = field(default_factory=list)
    expected_rewards: list[float] = field(default_factory=list)
    per_observer_expected_rewards: list[list[float]] = field(default_factory=list)
    mean_correct_action_probs: list[float] = field(default_factory=list)
    per_observer_correct_action_probs: list[list[float]] = field(default_factory=list)
    selection_probabilities_by_observer: list[list[list[float]]] = field(
        default_factory=list
    )
    actor_messages_by_state: list[list[list[list[float]]]] = field(
        default_factory=list
    )
    actor_compressor_linear_weights: list[list[list[float]]] = field(
        default_factory=list
    )
    actor_compressor_linear_biases: list[list[list[float]]] = field(
        default_factory=list
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def final_expected_reward(self) -> float | None:
        return self.expected_rewards[-1] if self.expected_rewards else None

    @property
    def final_mean_correct_action_prob(self) -> float | None:
        if not self.mean_correct_action_probs:
            return None
        return self.mean_correct_action_probs[-1]

    @property
    def final_selection_probabilities_by_observer(
        self,
    ) -> list[list[float]] | None:
        if not self.selection_probabilities_by_observer:
            return None
        return self.selection_probabilities_by_observer[-1]

    @property
    def final_actor_messages_by_state(self) -> list[list[list[float]]] | None:
        if not self.actor_messages_by_state:
            return None
        return self.actor_messages_by_state[-1]


class UpdatedImperfectStateTrainer:
    """Paper-aligned imperfect-state trainer for Sections 5.1--5.3."""

    def __init__(
        self,
        env: Any,
        actor_compressors: Sequence[ActorCompressor],
        observer_policies: Sequence[torch.nn.Module],
        selection_logits_by_observer: Sequence[SelectionLogits] | None = None,
        observer_reward_tables: Sequence[Sequence[Sequence[float]]] | None = None,
        compression_optimizer: torch.optim.Optimizer | None = None,
        selection_optimizer: torch.optim.Optimizer | None = None,
        policy_optimizers: Sequence[torch.optim.Optimizer | None] | None = None,
        selection_exploration_probability: float = 0.0,
        selection_log_prob_source: str = "learned",
        action_exploration_probability: float = 0.0,
        action_log_prob_source: str = "learned",
        sample_selection: bool = True,
        sample_actions: bool = True,
        optimal_actions: Sequence[int] | None = None,
        max_grad_norm: float | None = None,
        loss_reduction: str = "sum",
        message_l2_coef: float = 0.0,
        device: str | torch.device = "cpu",
    ) -> None:
        actor_compressors = list(actor_compressors)
        observer_policies = list(observer_policies)
        if not actor_compressors:
            raise ValueError("at least one Actor-compressor is required")
        if not observer_policies:
            raise ValueError("at least one Observer policy is required")
        if selection_logits_by_observer is None:
            selection_logits = None
            if len(actor_compressors) != 1:
                raise ValueError(
                    "selection_logits_by_observer is required when more than "
                    "one Actor-compressor is present"
                )
        else:
            selection_logits = list(selection_logits_by_observer)
            if len(selection_logits) != len(observer_policies):
                raise ValueError(
                    "selection_logits_by_observer length must match Observer count"
                )
            for logits in selection_logits:
                if logits.num_actor_compressors != len(actor_compressors):
                    raise ValueError(
                        "each SelectionLogits must match Actor-compressor count"
                    )
        if policy_optimizers is None:
            policy_optimizers = [None] * len(observer_policies)
        else:
            policy_optimizers = list(policy_optimizers)
            if len(policy_optimizers) != len(observer_policies):
                raise ValueError("policy_optimizers length must match Observer count")
        if not 0.0 <= selection_exploration_probability <= 1.0:
            raise ValueError("selection_exploration_probability must be in [0, 1]")
        if selection_log_prob_source not in {"learned", "mixed"}:
            raise ValueError("selection_log_prob_source must be 'learned' or 'mixed'")
        if not 0.0 <= action_exploration_probability <= 1.0:
            raise ValueError("action_exploration_probability must be in [0, 1]")
        if action_log_prob_source not in {"learned", "mixed"}:
            raise ValueError("action_log_prob_source must be 'learned' or 'mixed'")
        if max_grad_norm is not None and max_grad_norm <= 0.0:
            raise ValueError("max_grad_norm must be positive when set")
        if loss_reduction not in {"sum", "mean"}:
            raise ValueError("loss_reduction must be 'sum' or 'mean'")
        if message_l2_coef < 0.0:
            raise ValueError("message_l2_coef must be non-negative")

        self.env = env
        self.actor_compressors = actor_compressors
        self.observer_policies = observer_policies
        self.selection_logits_by_observer = selection_logits
        self.observer_reward_tables = self._validate_reward_tables(
            observer_reward_tables
        )
        self.compression_optimizer = compression_optimizer
        self.selection_optimizer = selection_optimizer
        self.policy_optimizers = list(policy_optimizers)
        self.selection_exploration_probability = float(
            selection_exploration_probability
        )
        self.selection_log_prob_source = selection_log_prob_source
        self.action_exploration_probability = float(action_exploration_probability)
        self.action_log_prob_source = action_log_prob_source
        self.max_grad_norm = None if max_grad_norm is None else float(max_grad_norm)
        self.loss_reduction = loss_reduction
        self.message_l2_coef = float(message_l2_coef)
        self.sample_selection = bool(sample_selection)
        self.sample_actions = bool(sample_actions)
        self.device = torch.device(device)
        self.optimal_actions = (
            tuple(range(env.num_states))
            if optimal_actions is None
            else tuple(int(action) for action in optimal_actions)
        )
        if len(self.optimal_actions) != env.num_states:
            raise ValueError("optimal_actions length must match env.num_states")

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
        eval_interval: int = 1,
        reward_window: int = 100,
        evaluation_callback: Any | None = None,
    ) -> UpdatedImperfectStateTrace:
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")
        if eval_interval <= 0:
            raise ValueError("eval_interval must be positive")
        if reward_window <= 0:
            raise ValueError("reward_window must be positive")
        if seed is not None:
            set_global_seed(seed)

        rewards_window: deque[float] = deque(maxlen=reward_window)
        trace = UpdatedImperfectStateTrace(
            metadata={
                "seed": seed,
                "num_steps": num_steps,
                "num_actor_compressors": self.num_actor_compressors,
                "num_observers": self.num_observers,
                "learn_compression": self.compression_optimizer is not None,
                "learn_selection": self.selection_optimizer is not None,
                "learn_policies": any(
                    optimizer is not None for optimizer in self.policy_optimizers
                ),
                "selection_exploration_probability": (
                    self.selection_exploration_probability
                ),
                "selection_log_prob_source": self.selection_log_prob_source,
                "action_exploration_probability": (
                    self.action_exploration_probability
                ),
                "action_log_prob_source": self.action_log_prob_source,
            }
        )
        self._record_evaluation(trace, step=0, average_reward=0.0)
        if evaluation_callback is not None:
            evaluation_callback(self, trace, len(trace.evaluation_steps) - 1)

        for step_index in range(num_steps):
            _observation, info = self.env.reset(seed=seed if step_index == 0 else None)
            state_index = self._state_index(info)
            full_state = self._to_tensor(self.env.get_full_state())

            selected_indices: list[int] = []
            actions: list[int] = []
            step_messages: list[list[float]] = []
            losses: list[torch.Tensor] = []
            feedback_counts = [0 for _ in self.actor_compressors]
            rewards: list[float] = []

            for observer_index, policy in enumerate(self.observer_policies):
                actor_index, selection_log_prob = self._select_actor_compressor(
                    observer_index
                )
                selected_indices.append(actor_index)
                feedback_counts[actor_index] += 1

                message = self.actor_compressors[actor_index].message(full_state)
                step_messages.append(
                    [
                        float(value)
                        for value in message.detach().cpu().reshape(-1).tolist()
                    ]
                )
                action_result = self._sample_action(policy, message)
                action = int(action_result.action.detach().cpu().item())
                actions.append(action)
                reward = self._reward(observer_index, state_index, action)
                rewards.append(reward)

                if self.selection_optimizer is not None and selection_log_prob is not None:
                    losses.append(-reward * selection_log_prob)
                if (
                    self.compression_optimizer is not None
                    or self.policy_optimizers[observer_index] is not None
                ):
                    losses.append(-reward * action_result.log_prob)
                if self.compression_optimizer is not None and self.message_l2_coef > 0.0:
                    losses.append(self.message_l2_coef * message.pow(2).mean())

            if losses:
                loss = sum(losses)
                if self.loss_reduction == "mean":
                    loss = loss / len(self.observer_policies)
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        "non-finite updated imperfect-state loss"
                    )
                self._zero_grad()
                loss.backward()
                self._step()

            mean_reward = fmean(rewards)
            rewards_window.append(mean_reward)
            trace.rewards.append(mean_reward)
            trace.average_rewards.append(fmean(rewards_window))
            trace.state_indices.append(state_index)
            trace.selected_actor_indices.append(selected_indices)
            trace.actions.append(actions)
            trace.messages.append(step_messages)
            trace.compression_feedback_counts.append(feedback_counts)

            current_step = step_index + 1
            if current_step % eval_interval == 0 or current_step == num_steps:
                self._record_evaluation(
                    trace,
                    step=current_step,
                    average_reward=fmean(rewards_window),
                )
                if evaluation_callback is not None:
                    evaluation_callback(self, trace, len(trace.evaluation_steps) - 1)

        return trace

    def evaluate(self) -> tuple[float, list[float], float, list[float]]:
        """Return reward and correct-action probabilities under learned selection."""
        observer_rewards = [0.0 for _ in self.observer_policies]
        observer_correct_probs = [0.0 for _ in self.observer_policies]
        with torch.no_grad():
            for state_index in range(self.env.num_states):
                state_probability = self.env.state_probability(state_index)
                state = self._to_tensor(self.env.full_state_for_index(state_index))
                for observer_index, policy in enumerate(self.observer_policies):
                    selection_probs = self._selection_probabilities(observer_index)
                    for actor_index, selection_prob in enumerate(selection_probs):
                        message = self.actor_compressors[actor_index].message(state)
                        action_probs = policy.action_distribution(message).probs
                        rewards = torch.tensor(
                            self.observer_reward_tables[observer_index][state_index],
                            dtype=torch.float32,
                            device=action_probs.device,
                        )
                        observer_rewards[observer_index] += float(
                            state_probability
                            * selection_prob
                            * torch.dot(action_probs, rewards).item()
                        )
                        correct_action = self.optimal_actions[state_index]
                        observer_correct_probs[observer_index] += float(
                            state_probability
                            * selection_prob
                            * action_probs[correct_action].item()
                        )
        return (
            fmean(observer_rewards),
            observer_rewards,
            fmean(observer_correct_probs),
            observer_correct_probs,
        )

    def actor_messages_by_state(self) -> list[list[list[float]]]:
        """Return ``[actor][state][message_dimension]`` for current messages."""
        with torch.no_grad():
            snapshots: list[list[list[float]]] = []
            for actor_compressor in self.actor_compressors:
                actor_messages: list[list[float]] = []
                for state_index in range(self.env.num_states):
                    state = self._to_tensor(self.env.full_state_for_index(state_index))
                    message = actor_compressor.message(state)
                    actor_messages.append(
                        [
                            float(value)
                            for value in message.detach().cpu().reshape(-1).tolist()
                        ]
                    )
                snapshots.append(actor_messages)
            return snapshots

    def actor_compressor_linear_parameters(
        self,
    ) -> tuple[list[list[float]], list[list[float]]]:
        """Return linear weights and biases when compressors expose them.

        The controlled scalar-sum runs use ``MaskedLinearActorCompressor``.
        Recording its raw parameters lets us check whether ``phi`` itself
        settles, not only whether the emitted messages look good.
        """
        weights: list[list[float]] = []
        biases: list[list[float]] = []
        with torch.no_grad():
            for actor_compressor in self.actor_compressors:
                linear = getattr(actor_compressor.message_module, "linear", None)
                if linear is None:
                    weights.append([])
                    biases.append([])
                    continue
                weights.append(
                    [
                        float(value)
                        for value in linear.weight.detach().cpu().reshape(-1).tolist()
                    ]
                )
                biases.append(
                    [
                        float(value)
                        for value in linear.bias.detach().cpu().reshape(-1).tolist()
                    ]
                )
        return weights, biases

    def selection_probabilities_by_observer(self) -> list[list[float]]:
        if self.selection_logits_by_observer is None:
            return [[1.0] for _ in self.observer_policies]
        return [
            logits.current_probabilities().detach().cpu().tolist()
            for logits in self.selection_logits_by_observer
        ]

    def _select_actor_compressor(
        self,
        observer_index: int,
    ) -> tuple[int, torch.Tensor | None]:
        if self.selection_logits_by_observer is None:
            return 0, None
        return self.selection_logits_by_observer[observer_index].sample(
            exploration_probability=self.selection_exploration_probability,
            log_prob_source=self.selection_log_prob_source,
            sample=self.sample_selection,
        )

    def _sample_action(
        self,
        policy: torch.nn.Module,
        message: torch.Tensor,
    ) -> ActionResult:
        distribution = policy.action_distribution(message)
        if self.action_exploration_probability > 0.0:
            probs = distribution.probs
            uniform = torch.full_like(probs, 1.0 / probs.shape[-1])
            mixed_probs = (
                (1.0 - self.action_exploration_probability) * probs
                + self.action_exploration_probability * uniform
            )
            if self.sample_actions:
                action = torch.distributions.Categorical(probs=mixed_probs).sample()
            else:
                action = mixed_probs.argmax(dim=-1)
            if self.action_log_prob_source == "mixed":
                log_prob = torch.log(mixed_probs[action].clamp_min(1e-12))
            else:
                log_prob = distribution.log_prob(action)
            return ActionResult(action=action, log_prob=log_prob, probs=mixed_probs)
        return policy.act(message, sample=self.sample_actions)

    def _record_evaluation(
        self,
        trace: UpdatedImperfectStateTrace,
        *,
        step: int,
        average_reward: float,
    ) -> None:
        expected, per_observer, correct, per_observer_correct = self.evaluate()
        trace.evaluation_steps.append(step)
        trace.expected_rewards.append(expected)
        trace.per_observer_expected_rewards.append(per_observer)
        trace.mean_correct_action_probs.append(correct)
        trace.per_observer_correct_action_probs.append(per_observer_correct)
        trace.selection_probabilities_by_observer.append(
            self.selection_probabilities_by_observer()
        )
        trace.actor_messages_by_state.append(self.actor_messages_by_state())
        weights, biases = self.actor_compressor_linear_parameters()
        trace.actor_compressor_linear_weights.append(weights)
        trace.actor_compressor_linear_biases.append(biases)
        if step == 0 and not trace.average_rewards:
            trace.average_rewards.append(float(average_reward))

    def _selection_probabilities(self, observer_index: int) -> torch.Tensor:
        if self.selection_logits_by_observer is None:
            return torch.ones(1, dtype=torch.float32)
        return self.selection_logits_by_observer[observer_index].current_probabilities()

    def _reward(self, observer_index: int, state_index: int, action: int) -> float:
        return float(self.observer_reward_tables[observer_index][state_index][action])

    def _validate_reward_tables(
        self,
        observer_reward_tables: Sequence[Sequence[Sequence[float]]] | None,
    ) -> list[list[list[float]]]:
        if observer_reward_tables is None:
            return [
                [[float(value) for value in row] for row in self.env.reward_table]
                for _ in self.observer_policies
            ]
        tables = [
            [[float(value) for value in row] for row in table]
            for table in observer_reward_tables
        ]
        if len(tables) != len(self.observer_policies):
            raise ValueError("observer_reward_tables length must match Observer count")
        for table in tables:
            if len(table) != self.env.num_states:
                raise ValueError("each reward table must match env.num_states")
            if any(len(row) != self.env.num_actions for row in table):
                raise ValueError("each reward row must match env.num_actions")
        return tables

    def _zero_grad(self) -> None:
        for actor_compressor in self.actor_compressors:
            actor_compressor.zero_grad(set_to_none=True)
        for policy in self.observer_policies:
            policy.zero_grad(set_to_none=True)
        if self.selection_logits_by_observer is not None:
            for logits in self.selection_logits_by_observer:
                logits.zero_grad(set_to_none=True)
        for optimizer in self._optimizers():
            optimizer.zero_grad(set_to_none=True)

    def _step(self) -> None:
        if self.max_grad_norm is not None:
            for optimizer in self._optimizers():
                for group in optimizer.param_groups:
                    torch.nn.utils.clip_grad_norm_(
                        group["params"], self.max_grad_norm
                    )
        for optimizer in self._optimizers():
            optimizer.step()

    def _optimizers(self) -> list[torch.optim.Optimizer]:
        optimizers: list[torch.optim.Optimizer] = []
        if self.compression_optimizer is not None:
            optimizers.append(self.compression_optimizer)
        if self.selection_optimizer is not None:
            optimizers.append(self.selection_optimizer)
        optimizers.extend(
            optimizer
            for optimizer in self.policy_optimizers
            if optimizer is not None
        )
        return optimizers

    def _state_index(self, info: dict[str, Any]) -> int:
        return int(info["state_index"])

    def _to_tensor(self, value: Any) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.to(self.device)
        return torch.as_tensor(value, dtype=torch.float32, device=self.device)


def build_scalar_sum_actor_compressors(
    num_states: int,
    *,
    learnable: bool,
    zero_initialization: bool = True,
) -> list[ActorCompressor]:
    """Build zero, partial, and full Actor-compressors for scalar-sum tests."""
    if num_states <= 1:
        raise ValueError("num_states must be greater than one")
    masks = [
        [0.0 for _ in range(num_states)],
        [1.0] + [0.0 for _ in range(num_states - 1)],
        [1.0 for _ in range(num_states)],
    ]
    names = ["zero", "partial", "full"]
    compressors: list[ActorCompressor] = []
    for name, mask in zip(names, masks, strict=True):
        module = MaskedLinearActorCompressor(
            input_dim=num_states,
            message_dim=1,
            input_mask=mask,
        )
        with torch.no_grad():
            if learnable:
                if zero_initialization:
                    module.linear.weight.zero_()
                    module.linear.bias.zero_()
            else:
                module.linear.weight.copy_(torch.tensor([mask], dtype=torch.float32))
                module.linear.bias.zero_()
                for parameter in module.parameters():
                    parameter.requires_grad_(False)
        compressors.append(ActorCompressor(module, name=name))
    return compressors


def fixed_sum_observer_policy(num_actions: int, sharpness: float = 4.0) -> SumDecoderPolicy:
    """Return the fixed scalar policy used in Sections 5.1 and 5.2."""
    return SumDecoderPolicy(num_actions=num_actions, sharpness=sharpness)
