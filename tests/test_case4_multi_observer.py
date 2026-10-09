"""Unit tests for the multi-observer selection used in Case 4."""

from __future__ import annotations

import pytest
import torch

from actor_observer_rl.agents import (
    FollowingRates,
    MultiObserver,
    Observer,
    ObserverMessageModule,
)


def _identity_message_module(num_states: int) -> ObserverMessageModule:
    module = ObserverMessageModule(input_dim=num_states, message_dim=num_states)
    with torch.no_grad():
        module.linear.weight.copy_(torch.eye(num_states))
        module.linear.bias.zero_()
    return module


def _constant_message_module(num_states: int, constant: float) -> ObserverMessageModule:
    module = ObserverMessageModule(input_dim=num_states, message_dim=num_states)
    with torch.no_grad():
        module.linear.weight.zero_()
        module.linear.bias.fill_(constant)
    return module


def test_following_rates_weights_sum_to_one() -> None:
    rates = FollowingRates(num_observers=3)

    weights = rates()

    assert weights.shape == torch.Size([3])
    assert torch.allclose(weights.sum(), torch.tensor(1.0))


def test_following_rates_rejects_invalid_num_observers() -> None:
    with pytest.raises(ValueError, match="num_observers"):
        FollowingRates(num_observers=1)


def test_following_rates_init_logits_length_must_match() -> None:
    with pytest.raises(ValueError, match="init_logits length"):
        FollowingRates(num_observers=2, init_logits=[1.0, 2.0, 3.0])


def test_following_rates_current_weights_match_forward() -> None:
    rates = FollowingRates(num_observers=3, init_logits=[1.0, 2.0, 0.5])

    expected = torch.softmax(torch.tensor([1.0, 2.0, 0.5]), dim=-1)

    assert torch.allclose(rates(), expected)
    assert torch.allclose(rates.current_weights(), expected)
    assert not rates.current_weights().requires_grad


def test_following_rates_distribution_matches_logits() -> None:
    rates = FollowingRates(num_observers=3, init_logits=[1.0, 2.0, 0.5])

    distribution = rates.distribution()

    assert torch.allclose(distribution.probs, rates())
    log_prob = distribution.log_prob(torch.tensor(1))
    assert log_prob.requires_grad


def test_multi_observer_sample_returns_selected_observer_message() -> None:
    observers = [
        Observer(message_module=_constant_message_module(num_states=2, constant=1.0)),
        Observer(message_module=_constant_message_module(num_states=2, constant=-1.0)),
    ]
    rates = FollowingRates(num_observers=2, init_logits=[100.0, -100.0])
    multi_observer = MultiObserver(observers=observers, following_rates=rates)

    full_state = torch.tensor([1.0, 0.0])

    torch.manual_seed(0)
    index, message, log_prob = multi_observer.sample_observer_and_message(full_state)

    assert index == 0
    assert torch.allclose(message, torch.tensor([1.0, 1.0]))
    expected_log_prob = torch.log_softmax(rates.logits.detach(), dim=-1)[0]
    assert torch.allclose(log_prob, expected_log_prob)


def test_multi_observer_sample_keeps_gradient_path_only_for_selected_observer() -> None:
    observers = [
        Observer(message_module=_identity_message_module(num_states=2)),
        Observer(message_module=_identity_message_module(num_states=2)),
    ]
    rates = FollowingRates(num_observers=2, init_logits=[5.0, -5.0])
    multi_observer = MultiObserver(observers=observers, following_rates=rates)

    full_state = torch.tensor([0.5, 0.5])

    torch.manual_seed(0)
    index, message, log_prob = multi_observer.sample_observer_and_message(full_state)

    target = message.sum() + log_prob
    target.backward()

    selected_weight_grad = observers[index].message_module.linear.weight.grad
    other_index = 1 - index
    other_weight_grad = observers[other_index].message_module.linear.weight.grad
    following_grad = rates.logits.grad

    assert selected_weight_grad is not None
    assert torch.linalg.vector_norm(selected_weight_grad) > 0.0
    assert other_weight_grad is None
    assert following_grad is not None
    assert torch.linalg.vector_norm(following_grad) > 0.0


def test_multi_observer_rejects_message_module_count_mismatch() -> None:
    observers = [
        Observer(message_module=_identity_message_module(num_states=2)),
        Observer(message_module=_identity_message_module(num_states=2)),
    ]
    rates = FollowingRates(num_observers=3)

    with pytest.raises(ValueError, match="num_observers"):
        MultiObserver(observers=observers, following_rates=rates)


def test_multi_observer_rejects_missing_message_module() -> None:
    observers = [
        Observer(message_module=_identity_message_module(num_states=2)),
        Observer(),
    ]
    rates = FollowingRates(num_observers=2)

    with pytest.raises(ValueError, match="message_module"):
        MultiObserver(observers=observers, following_rates=rates)


def test_multi_observer_report_reward_uses_first_observer() -> None:
    seen: list[float] = []

    class RecordingObserver(Observer):
        def report_reward(self, reward):
            forwarded = super().report_reward(reward)
            seen.append(forwarded)
            return 2.0 * forwarded

    observers = [
        RecordingObserver(message_module=_identity_message_module(num_states=2)),
        Observer(message_module=_identity_message_module(num_states=2)),
    ]
    rates = FollowingRates(num_observers=2)
    multi_observer = MultiObserver(observers=observers, following_rates=rates)

    forwarded = multi_observer.report_reward(0.5)

    assert seen == [0.5]
    assert forwarded == 1.0


def test_multi_observer_parameter_helpers_split_observers_and_following() -> None:
    observers = [
        Observer(message_module=_identity_message_module(num_states=2)),
        Observer(message_module=_identity_message_module(num_states=2)),
    ]
    rates = FollowingRates(num_observers=2)
    multi_observer = MultiObserver(observers=observers, following_rates=rates)

    observer_param_ids = {id(p) for p in multi_observer.observer_parameters()}
    following_param_ids = {id(p) for p in multi_observer.following_parameters()}
    all_param_ids = {id(p) for p in multi_observer.parameters()}

    assert observer_param_ids.isdisjoint(following_param_ids)
    assert all_param_ids == observer_param_ids | following_param_ids
    assert id(rates.logits) in following_param_ids
