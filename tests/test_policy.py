"""Tests for policy probabilities, shapes, and finite logits."""

from __future__ import annotations

import torch

from actor_observer_rl.policies.softmax_policy import SoftmaxPolicy


def test_learnable_sum_decoder_preserves_distance_growing_message_grad() -> None:
    from actor_observer_rl.policies import LearnableSumDecoderPolicy, RBFPolicy

    policy = LearnableSumDecoderPolicy(num_actions=16, init_sharpness=4.0)
    message = torch.tensor([0.0], requires_grad=True)
    log_prob = policy.action_distribution(message).log_prob(torch.tensor(15))
    (grad,) = torch.autograd.grad(log_prob, message)
    assert float(grad.abs()) > 10.0

    rbf = RBFPolicy(16, 16, sigma=0.5)
    message_rbf = torch.tensor([0.0], requires_grad=True)
    log_prob_rbf = rbf.action_distribution(message_rbf).log_prob(torch.tensor(15))
    (grad_rbf,) = torch.autograd.grad(log_prob_rbf, message_rbf)
    assert float(grad.abs()) > 50.0 * float(grad_rbf.abs())


def test_learnable_sum_decoder_parameters_are_trainable() -> None:
    from actor_observer_rl.policies import LearnableSumDecoderPolicy

    policy = LearnableSumDecoderPolicy(num_actions=8)
    before = [p.detach().clone() for p in policy.parameters()]
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
    message = torch.tensor([3.0])
    loss = -policy.action_distribution(message).log_prob(torch.tensor(3))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    assert any(
        not torch.allclose(old, new.detach())
        for old, new in zip(before, policy.parameters(), strict=True)
    )


def test_softmax_policy_probs_sum_to_one_for_single_input() -> None:
    policy = SoftmaxPolicy(input_dim=1, num_actions=3)
    observation = torch.tensor([1.0])

    distribution = policy.action_distribution(observation)

    assert distribution.probs.shape == (3,)
    assert torch.isclose(distribution.probs.sum(), torch.tensor(1.0))
    assert torch.isfinite(distribution.logits).all()


def test_softmax_policy_probs_have_expected_batch_shape() -> None:
    policy = SoftmaxPolicy(input_dim=2, num_actions=4)
    observations = torch.eye(2)

    distribution = policy.action_distribution(observations)

    assert distribution.probs.shape == (2, 4)
    assert torch.allclose(distribution.probs.sum(dim=-1), torch.ones(2))


def test_greedy_action_uses_highest_probability_action() -> None:
    policy = SoftmaxPolicy(input_dim=1, num_actions=3)
    with torch.no_grad():
        policy.linear.weight.zero_()
        policy.linear.bias.copy_(torch.tensor([0.0, 2.0, -1.0]))

    result = policy.act(torch.tensor([1.0]), sample=False)

    assert result.action.item() == 1
    assert result.log_prob.shape == torch.Size([])
    assert torch.isclose(result.probs.sum(), torch.tensor(1.0))
