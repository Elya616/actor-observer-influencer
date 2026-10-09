"""Training loop for Case 3 partial-observability experiments."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from actor_observer_rl.training.updates import reinforce_loss
from actor_observer_rl.utils.seeding import set_global_seed


@dataclass
class PartialObservableTrainingTrace:
    """Metrics recorded from joint Actor/Observer policy-gradient training."""

    rewards: list[float] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    state_indices: list[int] = field(default_factory=list)
    current_state_optimal_action_probs: list[float] = field(default_factory=list)
    expected_rewards: list[float] = field(default_factory=list)
    mean_optimal_action_probs: list[float] = field(default_factory=list)
    evaluation_steps: list[int] = field(default_factory=list)
    actor_grad_norms: list[float] = field(default_factory=list)
    observer_grad_norms: list[float] = field(default_factory=list)
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


def build_actor_message_input(
    observation: torch.Tensor,
    message: torch.Tensor,
) -> torch.Tensor:
    """Concatenate Actor observation `o_t` and Observer message `m_t`."""
    observation = observation.float()
    message = message.float()
    if observation.ndim == 0:
        observation = observation.unsqueeze(0)
    if message.ndim == 0:
        message = message.unsqueeze(0)
    if observation.ndim != message.ndim:
        raise ValueError("observation and message must have matching rank")
    return torch.cat((observation, message), dim=-1)


class PartialObservableTrainer:
    """Joint trainer for `pi_theta(a | o_t, f_phi(s_t))`.

    Each step keeps the chain rule from the PDF intact:

    `s_t -> f_phi(s_t) -> m_t -> pi_theta(a_t | o_t, m_t) -> log_prob`.

    The Observer message is never detached before `loss.backward()`, so the
    same reward-weighted log-probability loss updates both `theta` and `phi`.
    """

    def __init__(
        self,
        env: Any,
        actor: Any,
        observer: Any,
        actor_optimizer: torch.optim.Optimizer,
        observer_optimizer: torch.optim.Optimizer,
        optimal_actions: Sequence[int] | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        if getattr(observer, "message_module", None) is None:
            raise ValueError("PartialObservableTrainer requires observer.message_module")
        self.env = env
        self.actor = actor
        self.observer = observer
        self.actor_optimizer = actor_optimizer
        self.observer_optimizer = observer_optimizer
        self.optimal_actions = (
            None if optimal_actions is None else tuple(int(action) for action in optimal_actions)
        )
        self.device = torch.device(device)

        if self.optimal_actions is not None and len(self.optimal_actions) != env.num_states:
            raise ValueError("optimal_actions length must match env.num_states")

    def run(
        self,
        num_steps: int,
        seed: int | None = None,
        eval_interval: int = 1,
    ) -> PartialObservableTrainingTrace:
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")
        if eval_interval <= 0:
            raise ValueError("eval_interval must be positive")

        if seed is not None:
            set_global_seed(seed)

        observation, reset_info = self.env.reset(seed=seed)
        state_index = self._state_index_from_info(reset_info)
        trace = PartialObservableTrainingTrace(
            metadata={
                "seed": seed,
                "reset_info": reset_info,
                "num_steps": num_steps,
                "eval_interval": eval_interval,
            }
        )

        for step_index in range(num_steps):
            full_state = self._to_tensor(self.env.get_full_state())
            actor_observation = self._to_tensor(observation)
            message = self.observer.message(full_state)
            actor_input = build_actor_message_input(actor_observation, message)
            action_result = self.actor.act(actor_input, sample=True)
            action = self._action_to_int(action_result.action)

            step_result = self.env.step(action)
            reward = self.observer.report_reward(step_result.reward)
            loss = reinforce_loss(reward, action_result.log_prob)

            self.actor_optimizer.zero_grad()
            self.observer_optimizer.zero_grad()
            loss.backward()
            trace.actor_grad_norms.append(self._grad_norm(self.actor.parameters()))
            trace.observer_grad_norms.append(self._grad_norm(self.observer.parameters()))
            self.actor_optimizer.step()
            self.observer_optimizer.step()

            trace.actions.append(action)
            trace.state_indices.append(state_index)
            trace.rewards.append(reward)
            trace.losses.append(float(loss.detach().cpu().item()))
            trace.message_norms.append(float(message.detach().norm().cpu().item()))

            optimal_prob = self._optimal_action_probability(state_index)
            if optimal_prob is not None:
                trace.current_state_optimal_action_probs.append(optimal_prob)

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
        """Return expected reward and mean optimal-action probability."""
        expected_reward = 0.0
        optimal_prob_sum = 0.0

        with torch.no_grad():
            for state_index in range(self.env.num_states):
                full_state = self._to_tensor(self.env.full_state_for_index(state_index))
                observation = self._to_tensor(
                    self.env.actor_observation_for_index(state_index)
                )
                message = self.observer.message(full_state)
                actor_input = build_actor_message_input(observation, message)
                probs = self.actor.policy.action_distribution(actor_input).probs
                rewards = torch.tensor(
                    self.env.reward_table[state_index],
                    dtype=probs.dtype,
                    device=probs.device,
                )
                state_probability = self.env.state_probability(state_index)
                expected_reward += state_probability * float((probs * rewards).sum().item())

                if self.optimal_actions is not None:
                    optimal_action = self.optimal_actions[state_index]
                    optimal_prob_sum += float(probs[..., optimal_action].item())

        mean_optimal_prob = None
        if self.optimal_actions is not None:
            mean_optimal_prob = optimal_prob_sum / len(self.optimal_actions)
        return expected_reward, mean_optimal_prob

    def _optimal_action_probability(self, state_index: int) -> float | None:
        if self.optimal_actions is None:
            return None
        optimal_action = self.optimal_actions[state_index]
        with torch.no_grad():
            full_state = self._to_tensor(self.env.full_state_for_index(state_index))
            observation = self._to_tensor(self.env.actor_observation_for_index(state_index))
            message = self.observer.message(full_state)
            actor_input = build_actor_message_input(observation, message)
            probs = self.actor.policy.action_distribution(actor_input).probs
            return float(probs[..., optimal_action].item())

    def _to_tensor(self, value: Any) -> torch.Tensor:
        return torch.as_tensor(value, dtype=torch.float32, device=self.device)

    @staticmethod
    def _action_to_int(action: torch.Tensor) -> int:
        return int(action.detach().cpu().reshape(-1)[0].item())

    @staticmethod
    def _state_index_from_info(info: dict[str, Any]) -> int:
        state_index = info.get("state_index")
        if state_index is None:
            raise KeyError("partial-observable training requires info['state_index']")
        return int(state_index)

    @staticmethod
    def _grad_norm(parameters: Any) -> float:
        total_square_norm = 0.0
        for parameter in parameters:
            if parameter.grad is not None:
                total_square_norm += float(parameter.grad.detach().pow(2).sum().item())
        return total_square_norm**0.5
