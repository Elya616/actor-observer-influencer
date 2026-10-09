"""Training loop for Case 4 multi-observer experiments.

The PDF Case 4 samples a single observer index `k_t` at every step from the
following-rate distribution `rho = softmax(lambda)`:

```
k_t ~ Categorical(rho)
m_t = f_phi_{k_t}(s_t)
log_pi = log pi_theta(a_t | o_t, m_t)
log_rho = log rho_{k_t}
loss = -reported_reward * (log_pi + log_rho)
```

A single `loss.backward()` reaches three parameter sets through three
independent paths:

- the actor parameters `theta` through `log_pi`;
- the selected observer's parameters `phi_{k_t}` through `m_t`. Other
  `phi_i` receive no gradient because their messages never entered the
  forward graph;
- the following-rate logits `lambda` through `log_rho` (REINFORCE on the
  discrete selection).

This matches the per-step updates printed in PDF Figures 4 and 5.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from actor_observer_rl.agents.multi_observer import MultiObserver
from actor_observer_rl.training.partial_observable import (
    build_actor_message_input,
)
from actor_observer_rl.training.updates import reinforce_loss
from actor_observer_rl.utils.seeding import set_global_seed


@dataclass
class MultiObserverTrainingTrace:
    """Metrics recorded from joint Actor / Observers / Following training."""

    rewards: list[float] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    state_indices: list[int] = field(default_factory=list)
    selected_observer_indices: list[int] = field(default_factory=list)
    selection_log_probs: list[float] = field(default_factory=list)
    expected_rewards: list[float] = field(default_factory=list)
    mean_optimal_action_probs: list[float] = field(default_factory=list)
    evaluation_steps: list[int] = field(default_factory=list)
    actor_grad_norms: list[float] = field(default_factory=list)
    observer_grad_norms: list[list[float]] = field(default_factory=list)
    following_grad_norms: list[float] = field(default_factory=list)
    following_weights: list[list[float]] = field(default_factory=list)
    message_norms: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def final_reward(self) -> float | None:
        return self.rewards[-1] if self.rewards else None

    @property
    def final_expected_reward(self) -> float | None:
        return self.expected_rewards[-1] if self.expected_rewards else None

    @property
    def final_mean_optimal_action_prob(self) -> float | None:
        if not self.mean_optimal_action_probs:
            return None
        return self.mean_optimal_action_probs[-1]

    @property
    def final_following_weights(self) -> list[float] | None:
        return self.following_weights[-1] if self.following_weights else None


class MultiObserverTrainer:
    """Joint trainer for the PDF Case 4 update rules.

    Each step samples one observer, routes its message to the actor, samples
    an action, and updates `theta`, `phi_{k_t}`, and `lambda` from the same
    reward-weighted loss. Reward routing keeps the single-observer
    convention: the first observer is the primary reward channel and its
    `report_reward` controls what the actor learns from.
    """

    def __init__(
        self,
        env: Any,
        actor: Any,
        multi_observer: MultiObserver,
        actor_optimizer: torch.optim.Optimizer,
        observer_optimizer: torch.optim.Optimizer,
        following_optimizer: torch.optim.Optimizer,
        optimal_actions: Sequence[int] | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.env = env
        self.actor = actor
        self.multi_observer = multi_observer
        self.actor_optimizer = actor_optimizer
        self.observer_optimizer = observer_optimizer
        self.following_optimizer = following_optimizer
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
    ) -> MultiObserverTrainingTrace:
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")
        if eval_interval <= 0:
            raise ValueError("eval_interval must be positive")

        if seed is not None:
            set_global_seed(seed)

        observation, reset_info = self.env.reset(seed=seed)
        state_index = self._state_index_from_info(reset_info)
        trace = MultiObserverTrainingTrace(
            metadata={
                "seed": seed,
                "reset_info": reset_info,
                "num_steps": num_steps,
                "eval_interval": eval_interval,
                "num_observers": self.multi_observer.num_observers,
            }
        )

        for step_index in range(num_steps):
            full_state = self._to_tensor(self.env.get_full_state())
            actor_observation = self._to_tensor(observation)
            observer_index, message, selection_log_prob = (
                self.multi_observer.sample_observer_and_message(full_state)
            )
            actor_input = build_actor_message_input(actor_observation, message)
            action_result = self.actor.act(actor_input, sample=True)
            action = self._action_to_int(action_result.action)

            step_result = self.env.step(action)
            reward = self.multi_observer.report_reward(step_result.reward)
            combined_log_prob = action_result.log_prob + selection_log_prob
            loss = reinforce_loss(reward, combined_log_prob)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite training loss at step {step_index}: "
                    f"reward={reward}, "
                    f"log_pi={float(action_result.log_prob.detach().cpu().item())}, "
                    f"log_rho={float(selection_log_prob.detach().cpu().item())}"
                )

            self.actor_optimizer.zero_grad()
            self.observer_optimizer.zero_grad()
            self.following_optimizer.zero_grad()
            loss.backward()
            trace.actor_grad_norms.append(self._grad_norm(self.actor.parameters()))
            trace.observer_grad_norms.append(
                [
                    self._grad_norm(observer.parameters())
                    for observer in self.multi_observer.observers
                ]
            )
            trace.following_grad_norms.append(
                self._grad_norm(self.multi_observer.following_parameters())
            )
            self.actor_optimizer.step()
            self.observer_optimizer.step()
            self.following_optimizer.step()

            trace.actions.append(action)
            trace.state_indices.append(state_index)
            trace.selected_observer_indices.append(observer_index)
            trace.selection_log_probs.append(
                float(selection_log_prob.detach().cpu().item())
            )
            trace.rewards.append(reward)
            trace.losses.append(float(loss.detach().cpu().item()))
            trace.message_norms.append(float(message.detach().norm().cpu().item()))
            trace.following_weights.append(
                self.multi_observer.following_rates.current_weights().cpu().tolist()
            )

            current_step = step_index + 1
            if current_step % eval_interval == 0 or current_step == num_steps:
                expected_reward, mean_optimal_prob = self.evaluate_policy()
                trace.evaluation_steps.append(current_step)
                trace.expected_rewards.append(expected_reward)
                if mean_optimal_prob is not None:
                    trace.mean_optimal_action_probs.append(mean_optimal_prob)

            if step_result.done:
                observation, reset_info = self.env.reset(seed=None)
                state_index = self._state_index_from_info(reset_info)
            else:
                observation = step_result.observation
                state_index = self._state_index_from_info(step_result.info)

        return trace

    def evaluate_policy(self) -> tuple[float, float | None]:
        """Return expected reward and mean optimal-action probability.

        The expected reward marginalises over both observer selection and
        action sampling:

        `E[r | s] = sum_k rho_k * sum_a pi_theta(a | o, m_k) * r(s, a)`.
        """
        expected_reward = 0.0
        optimal_prob_sum = 0.0

        with torch.no_grad():
            weights = self.multi_observer.following_rates.current_weights()
            for state_index in range(self.env.num_states):
                full_state = self._to_tensor(self.env.full_state_for_index(state_index))
                observation = self._to_tensor(
                    self.env.actor_observation_for_index(state_index)
                )
                marginal_probs = self._marginal_action_probs(
                    full_state=full_state,
                    observation=observation,
                    weights=weights,
                )
                rewards = torch.tensor(
                    self.env.reward_table[state_index],
                    dtype=marginal_probs.dtype,
                    device=marginal_probs.device,
                )
                state_probability = self.env.state_probability(state_index)
                expected_reward += state_probability * float(
                    (marginal_probs * rewards).sum().item()
                )

                if self.optimal_actions is not None:
                    optimal_action = self.optimal_actions[state_index]
                    optimal_prob_sum += float(marginal_probs[..., optimal_action].item())

        mean_optimal_prob = None
        if self.optimal_actions is not None:
            mean_optimal_prob = optimal_prob_sum / len(self.optimal_actions)
        return expected_reward, mean_optimal_prob

    def _marginal_action_probs(
        self,
        full_state: torch.Tensor,
        observation: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Return `sum_k rho_k * pi_theta(. | o, m_k)` for the given state."""
        marginal: torch.Tensor | None = None
        for observer_index, observer in enumerate(self.multi_observer.observers):
            message = observer.message(full_state)
            actor_input = build_actor_message_input(observation, message)
            probs = self.actor.policy.action_distribution(actor_input).probs
            weighted = weights[observer_index] * probs
            marginal = weighted if marginal is None else marginal + weighted
        assert marginal is not None
        return marginal

    def _to_tensor(self, value: Any) -> torch.Tensor:
        return torch.as_tensor(value, dtype=torch.float32, device=self.device)

    @staticmethod
    def _action_to_int(action: torch.Tensor) -> int:
        return int(action.detach().cpu().reshape(-1)[0].item())

    @staticmethod
    def _state_index_from_info(info: dict[str, Any]) -> int:
        state_index = info.get("state_index")
        if state_index is None:
            raise KeyError("multi-observer training requires info['state_index']")
        return int(state_index)

    @staticmethod
    def _grad_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
        total_square_norm = 0.0
        for parameter in parameters:
            if parameter.grad is not None:
                total_square_norm += float(parameter.grad.detach().pow(2).sum().item())
        return total_square_norm**0.5
