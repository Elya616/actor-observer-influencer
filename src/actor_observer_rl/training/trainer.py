"""Trainer skeleton for coordinating Actor-Observer experiments."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from actor_observer_rl.agents.observer import Observer
from actor_observer_rl.training.updates import reinforce_loss
from actor_observer_rl.utils.seeding import set_global_seed


@dataclass
class TrainingTrace:
    """Minimal metric container for future benchmark/report generation."""

    rewards: list[float] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    state_indices: list[int] = field(default_factory=list)
    optimal_action_probs: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def final_reward(self) -> float | None:
        return self.rewards[-1] if self.rewards else None

    @property
    def final_optimal_action_prob(self) -> float | None:
        return self.optimal_action_probs[-1] if self.optimal_action_probs else None


class Trainer:
    """Explicit trainer for Cases 1-3.

    The implementation should keep the update visible:
    `loss = -reward * log_prob(action)`.
    """

    def __init__(
        self,
        env: Any,
        actor: Any,
        actor_optimizer: torch.optim.Optimizer | None = None,
        observer: Observer | None = None,
        observer_optimizer: torch.optim.Optimizer | None = None,
        optimal_action: int | None = None,
        optimal_actions: Sequence[int] | None = None,
        reward_baseline: str | float | None = None,
        baseline_momentum: float = 0.01,
        device: str | torch.device = "cpu",
    ) -> None:
        self.env = env
        self.actor = actor
        self.actor_optimizer = actor_optimizer or getattr(actor, "optimizer", None)
        self.observer = observer if observer is not None else Observer()
        self.observer_optimizer = observer_optimizer
        self.optimal_action = optimal_action
        self.optimal_actions = (
            None if optimal_actions is None else tuple(int(action) for action in optimal_actions)
        )
        if isinstance(reward_baseline, str) and reward_baseline != "running_mean":
            raise ValueError("reward_baseline string must be 'running_mean'")
        if not 0.0 < baseline_momentum <= 1.0:
            raise ValueError("baseline_momentum must be in (0, 1]")
        self.reward_baseline = reward_baseline
        self.baseline_momentum = float(baseline_momentum)
        self._running_baseline: float | None = None
        self.device = torch.device(device)

    def run(self, num_steps: int, seed: int | None = None) -> TrainingTrace:
        """Run the explicit REINFORCE loop for bandit-like cases.

        For Cases 1-2, each iteration samples
        `a_t ~ pi_theta(. | actor_observation)`, routes the environment reward
        through `Observer.report_reward`, and applies the PDF update through
        `loss = -reported_reward * log pi_theta(a_t | actor_observation)`.
        """
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")
        if self.actor_optimizer is None:
            raise RuntimeError("Trainer requires an actor optimizer")
        if self.observer_optimizer is not None:
            raise NotImplementedError("observer parameter training is reserved for Case 3")
        if self.observer.message_module is not None:
            raise NotImplementedError("observer message learning is reserved for Case 3")

        if seed is not None:
            set_global_seed(seed)

        observation, reset_info = self.env.reset(seed=seed)
        state_index = self._state_index_from_info(reset_info)
        trace = TrainingTrace(
            metadata={
                "seed": seed,
                "reset_info": reset_info,
                "num_steps": num_steps,
            }
        )

        for _ in range(num_steps):
            actor_input = self._to_tensor(observation)
            action_result = self.actor.act(actor_input, sample=True)
            action = self._action_to_int(action_result.action)

            step_result = self.env.step(action)
            reward = self.observer.report_reward(step_result.reward)
            advantage = self._advantage(reward)
            loss = reinforce_loss(advantage, action_result.log_prob)

            self.actor_optimizer.zero_grad()
            loss.backward()
            self.actor_optimizer.step()

            trace.actions.append(action)
            if state_index is not None:
                trace.state_indices.append(state_index)
            trace.rewards.append(reward)
            trace.losses.append(float(loss.detach().cpu().item()))

            optimal_prob = self._optimal_action_probability(observation, state_index)
            if optimal_prob is not None:
                trace.optimal_action_probs.append(optimal_prob)

            if step_result.done:
                observation, reset_info = self.env.reset(seed=None)
                state_index = self._state_index_from_info(reset_info)
            else:
                observation = step_result.observation
                state_index = self._state_index_from_info(step_result.info)

        return trace

    def _advantage(self, reward: float) -> float:
        """Subtract the optional reward baseline.

        A baseline does not bias the policy-gradient direction but greatly
        reduces variance. It also prevents the rich-get-richer collapse that
        baseline-free REINFORCE suffers when every reward is positive, which is
        the regime of the structured grid task. When ``reward_baseline`` is
        ``None`` (the default), this returns the reward unchanged so the original
        ``loss = -reward * log_prob`` update is recovered exactly.
        """

        if self.reward_baseline is None:
            return reward
        if isinstance(self.reward_baseline, (int, float)):
            return reward - float(self.reward_baseline)
        if self._running_baseline is None:
            self._running_baseline = reward
        else:
            self._running_baseline += self.baseline_momentum * (
                reward - self._running_baseline
            )
        return reward - self._running_baseline

    def _to_tensor(self, value: Any) -> torch.Tensor:
        tensor = torch.as_tensor(value, dtype=torch.float32, device=self.device)
        return tensor

    @staticmethod
    def _action_to_int(action: torch.Tensor) -> int:
        return int(action.detach().cpu().reshape(-1)[0].item())

    def _optimal_action_probability(
        self,
        observation: Any,
        state_index: int | None = None,
    ) -> float | None:
        optimal_action = self._optimal_action_for_state(state_index)
        if optimal_action is None:
            return None
        with torch.no_grad():
            actor_input = self._to_tensor(observation)
            distribution = self.actor.policy.action_distribution(actor_input)
            probs = distribution.probs
            optimal_probs = probs[..., optimal_action]
            return float(optimal_probs.mean().detach().cpu().item())

    def _optimal_action_for_state(self, state_index: int | None) -> int | None:
        if self.optimal_actions is not None:
            if state_index is None:
                return None
            if state_index < 0 or state_index >= len(self.optimal_actions):
                raise IndexError(
                    f"state_index {state_index} is outside optimal_actions range"
                )
            return self.optimal_actions[state_index]
        return self.optimal_action

    @staticmethod
    def _state_index_from_info(info: dict[str, Any]) -> int | None:
        state_index = info.get("state_index")
        return None if state_index is None else int(state_index)
