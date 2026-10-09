"""Budgeted Observer for paper Section 4.

In Section 4 of the paper, a single Observer attends to ``N`` Actors. The
Observer holds a continuous following-rate vector ``mu in R^N`` constrained
to the capped simplex ``{mu >= 0, sum(mu) <= B}``. When Actor ``j`` acts,
the Observer reports its observed reward back to that Actor with
probability ``theta(mu_j) = 1 - exp(-c * mu_j)``. The Observer maintains a
running estimate ``U_hat_j`` of each Actor's average reward and updates
``mu`` by projected gradient ascent on

    max_mu sum_j theta(mu_j) U_hat_j
    s.t.   mu >= 0, sum_j mu_j <= B.

These dynamics are the imperfect-reward analogue of the soft-selection
``MultiObserver`` used in Section 5; the two are kept separate so the
paper's "budgeted attention" and "softmax selection" mechanisms remain
visibly distinct in the codebase.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import torch


def project_capped_simplex(values: torch.Tensor, budget: float) -> torch.Tensor:
    """Project ``values`` onto ``{mu >= 0, sum(mu) <= budget}``.

    This implements the standard Euclidean projection used by the paper's
    update (12). When the non-negative clipping of ``values`` already
    satisfies the budget, that clipping is the projection. Otherwise the
    sum-constraint is active and we find the unique ``tau >= 0`` such that
    ``sum_j max(values_j - tau, 0) = budget`` via the Duchi et al. (2008)
    sorting algorithm.
    """

    if budget <= 0.0:
        raise ValueError("budget must be positive")
    if values.dim() != 1:
        raise ValueError("project_capped_simplex expects a 1-D tensor")

    nonneg = torch.clamp(values, min=0.0)
    if float(nonneg.sum().item()) <= budget:
        return nonneg

    sorted_values, _ = torch.sort(values, descending=True)
    cumulative = torch.cumsum(sorted_values, dim=0)
    indices = torch.arange(1, values.numel() + 1, dtype=values.dtype, device=values.device)
    candidate_offsets = (cumulative - budget) / indices
    feasible = sorted_values - candidate_offsets > 0
    if not bool(feasible.any().item()):
        return torch.zeros_like(values)

    k_star = int(feasible.sum().item())
    tau = (cumulative[k_star - 1] - budget) / float(k_star)
    return torch.clamp(values - tau, min=0.0)


class BudgetedObserver:
    """Observer with budgeted following rates over ``N`` Actors.

    The Observer holds:

    - ``mu`` of shape ``[num_actors]`` on the capped simplex with budget ``B``;
    - ``utilities`` of shape ``[num_actors]`` holding the running estimates
      ``U_hat_j`` of each Actor's average reward.

    The reporting probability for Actor ``j`` is
    ``theta(mu_j) = 1 - exp(-c * mu_j)`` and the derivative used by update
    (11) is ``theta'(mu_j) = c * exp(-c * mu_j)``.
    """

    def __init__(
        self,
        num_actors: int,
        budget: float,
        utility_learning_rate: float,
        following_learning_rate: float,
        c: float = 1.0,
        init_mu: Sequence[float] | None = None,
        init_utilities: Sequence[float] | None = None,
        seed: int | None = None,
    ) -> None:
        if num_actors < 2:
            raise ValueError("BudgetedObserver requires at least two actors")
        if budget <= 0.0:
            raise ValueError("budget must be positive")
        if utility_learning_rate <= 0.0:
            raise ValueError("utility_learning_rate must be positive")
        if following_learning_rate <= 0.0:
            raise ValueError("following_learning_rate must be positive")
        if c <= 0.0:
            raise ValueError("c must be positive")

        self.num_actors = int(num_actors)
        self.budget = float(budget)
        self.utility_learning_rate = float(utility_learning_rate)
        self.following_learning_rate = float(following_learning_rate)
        self.c = float(c)
        self._generator = torch.Generator()
        if seed is not None:
            self._generator.manual_seed(int(seed))

        if init_mu is None:
            mu = torch.full((num_actors,), self.budget / num_actors, dtype=torch.float32)
        else:
            mu = torch.as_tensor(list(init_mu), dtype=torch.float32)
            if mu.shape != torch.Size([num_actors]):
                raise ValueError("init_mu length must match num_actors")
            mu = project_capped_simplex(mu, self.budget)
        self.mu = mu

        if init_utilities is None:
            self.utilities = torch.zeros(num_actors, dtype=torch.float32)
        else:
            initial_u = torch.as_tensor(list(init_utilities), dtype=torch.float32)
            if initial_u.shape != torch.Size([num_actors]):
                raise ValueError("init_utilities length must match num_actors")
            self.utilities = initial_u.clone()

    def report_probabilities(self) -> torch.Tensor:
        """Return ``theta(mu) = 1 - exp(-c * mu)`` for every actor."""

        return 1.0 - torch.exp(-self.c * self.mu)

    def report_probability(self, actor_index: int) -> float:
        return float(self.report_probabilities()[actor_index].item())

    def report_derivative(self) -> torch.Tensor:
        """Return ``theta'(mu) = c * exp(-c * mu)`` for every actor."""

        return self.c * torch.exp(-self.c * self.mu)

    def should_report(self, actor_index: int) -> bool:
        """Sample a Bernoulli with success probability ``theta(mu_{actor_index})``."""

        probability = self.report_probability(actor_index)
        draw = torch.rand((), generator=self._generator).item()
        return bool(draw < probability)

    def report_reward(self, reward: float | torch.Tensor) -> float:
        """Identity reward channel, matching the base ``Observer`` contract."""

        if isinstance(reward, torch.Tensor):
            return float(reward.detach().cpu().item())
        return float(reward)

    def update_utility(self, actor_index: int, reward: float) -> None:
        """Apply update (9) for the currently active actor only."""

        if actor_index < 0 or actor_index >= self.num_actors:
            raise IndexError("actor_index outside valid range")
        if not math.isfinite(reward):
            raise ValueError("reward must be finite")
        current = float(self.utilities[actor_index].item())
        self.utilities[actor_index] = current + self.utility_learning_rate * (
            float(reward) - current
        )

    def update_following_rates(self) -> None:
        """Apply updates (11) and (12) using the current utility estimates."""

        gradient = self.report_derivative() * self.utilities
        unconstrained = self.mu + self.following_learning_rate * gradient
        self.mu = project_capped_simplex(unconstrained, self.budget)

    def current_following_rates(self) -> list[float]:
        return [float(value.item()) for value in self.mu]

    def current_utilities(self) -> list[float]:
        return [float(value.item()) for value in self.utilities]

    def parameters(self) -> Iterable[torch.nn.Parameter]:
        """Return an empty iterator; mu and utilities are not autograd params."""

        return iter(())
