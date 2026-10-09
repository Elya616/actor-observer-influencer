"""Tests for the budgeted observer used in paper Section 4."""

from __future__ import annotations

import math

import pytest
import torch

from actor_observer_rl.agents.budgeted_observer import (
    BudgetedObserver,
    project_capped_simplex,
)


def _is_feasible(mu: torch.Tensor, budget: float, tolerance: float = 1e-5) -> bool:
    return bool(
        (mu >= -tolerance).all().item()
        and float(mu.sum().item()) <= budget + tolerance
    )


def test_project_returns_input_when_already_feasible() -> None:
    values = torch.tensor([0.5, 0.3, 0.2])
    projected = project_capped_simplex(values, budget=2.0)
    assert torch.allclose(projected, values)


def test_project_clips_negative_entries_when_sum_under_budget() -> None:
    values = torch.tensor([0.4, -0.5, 0.1])
    projected = project_capped_simplex(values, budget=2.0)
    assert torch.allclose(projected, torch.tensor([0.4, 0.0, 0.1]))


def test_project_redistributes_when_sum_exceeds_budget() -> None:
    values = torch.tensor([1.0, 1.0, 1.0])
    projected = project_capped_simplex(values, budget=2.0)
    assert _is_feasible(projected, budget=2.0)
    assert math.isclose(float(projected.sum().item()), 2.0, rel_tol=1e-5, abs_tol=1e-5)
    assert torch.allclose(projected, torch.full((3,), 2.0 / 3.0))


def test_project_zeroes_dominated_entries() -> None:
    values = torch.tensor([3.0, -1.0, -1.0])
    projected = project_capped_simplex(values, budget=2.0)
    assert torch.allclose(projected, torch.tensor([2.0, 0.0, 0.0]))


def test_project_returns_zero_when_all_negative_with_active_budget() -> None:
    values = torch.tensor([-1.0, -2.0])
    projected = project_capped_simplex(values, budget=1.0)
    assert torch.allclose(projected, torch.zeros(2))


def test_observer_initialization_distributes_budget_evenly_by_default() -> None:
    observer = BudgetedObserver(
        num_actors=4,
        budget=2.0,
        utility_learning_rate=0.1,
        following_learning_rate=0.01,
    )
    assert observer.mu.shape == torch.Size([4])
    assert torch.allclose(observer.mu, torch.full((4,), 0.5))
    assert _is_feasible(observer.mu, budget=2.0)


def test_observer_init_mu_is_projected_to_feasible() -> None:
    observer = BudgetedObserver(
        num_actors=3,
        budget=1.0,
        utility_learning_rate=0.1,
        following_learning_rate=0.01,
        init_mu=[5.0, -2.0, 1.0],
    )
    assert _is_feasible(observer.mu, budget=1.0)


def test_report_probabilities_match_formula() -> None:
    observer = BudgetedObserver(
        num_actors=2,
        budget=2.0,
        utility_learning_rate=0.1,
        following_learning_rate=0.01,
        c=0.5,
        init_mu=[1.0, 0.0],
    )
    probs = observer.report_probabilities()
    assert math.isclose(float(probs[0].item()), 1.0 - math.exp(-0.5), abs_tol=1e-6)
    assert math.isclose(float(probs[1].item()), 0.0, abs_tol=1e-6)


def test_update_utility_only_touches_active_actor() -> None:
    observer = BudgetedObserver(
        num_actors=3,
        budget=2.0,
        utility_learning_rate=0.5,
        following_learning_rate=0.01,
    )
    observer.update_utility(actor_index=1, reward=2.0)
    assert math.isclose(float(observer.utilities[0].item()), 0.0, abs_tol=1e-7)
    assert math.isclose(float(observer.utilities[1].item()), 1.0, abs_tol=1e-7)
    assert math.isclose(float(observer.utilities[2].item()), 0.0, abs_tol=1e-7)


def test_following_rate_update_moves_mass_to_high_utility_actor() -> None:
    observer = BudgetedObserver(
        num_actors=3,
        budget=2.0,
        utility_learning_rate=0.5,
        following_learning_rate=0.5,
        c=1.0,
        init_mu=[0.5, 0.5, 0.5],
        init_utilities=[0.0, 1.0, 0.0],
    )
    for _ in range(50):
        observer.update_following_rates()
    mu = observer.current_following_rates()
    assert _is_feasible(observer.mu, budget=2.0)
    assert mu[1] > mu[0]
    assert mu[1] > mu[2]


def test_should_report_probability_matches_empirical_frequency() -> None:
    observer = BudgetedObserver(
        num_actors=2,
        budget=2.0,
        utility_learning_rate=0.1,
        following_learning_rate=0.01,
        c=1.0,
        init_mu=[1.0, 0.0],
        seed=42,
    )
    trials = 5000
    successes_zero = sum(int(observer.should_report(0)) for _ in range(trials))
    successes_one = sum(int(observer.should_report(1)) for _ in range(trials))
    expected_zero = 1.0 - math.exp(-1.0)
    expected_one = 0.0
    assert math.isclose(successes_zero / trials, expected_zero, abs_tol=0.03)
    assert math.isclose(successes_one / trials, expected_one, abs_tol=0.01)


def test_constructor_rejects_invalid_arguments() -> None:
    with pytest.raises(ValueError):
        BudgetedObserver(
            num_actors=1,
            budget=1.0,
            utility_learning_rate=0.1,
            following_learning_rate=0.01,
        )
    with pytest.raises(ValueError):
        BudgetedObserver(
            num_actors=2,
            budget=0.0,
            utility_learning_rate=0.1,
            following_learning_rate=0.01,
        )
    with pytest.raises(ValueError):
        BudgetedObserver(
            num_actors=2,
            budget=1.0,
            utility_learning_rate=0.1,
            following_learning_rate=0.01,
            c=-1.0,
        )
