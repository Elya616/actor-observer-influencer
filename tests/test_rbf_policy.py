import torch

from actor_observer_rl.policies import RBFPolicy


def test_rbf_features_peak_near_closest_center() -> None:
    policy = RBFPolicy(num_actions=8, sigma=0.5)

    features = policy.features(torch.tensor([4.72]))

    assert features.shape == torch.Size([8])
    assert int(features.argmax().item()) == 5
    assert features[5] > features[4] > features[3]
    assert features[5] > features[6] > features[7]


def test_rbf_policy_returns_action_distribution() -> None:
    policy = RBFPolicy(num_actions=5, sigma=0.5)

    distribution = policy.action_distribution(torch.tensor([2.0]))

    assert distribution.probs.shape == torch.Size([5])
    assert torch.isclose(distribution.probs.sum(), torch.tensor(1.0))


def test_rbf_features_are_differentiable_in_message() -> None:
    policy = RBFPolicy(num_actions=8, sigma=0.5)
    message = torch.tensor([4.72], requires_grad=True)

    # Use a center away from the exact maximum so the derivative is clearly
    # nonzero. This is the path a learned compression function phi needs.
    feature = policy.features(message)[5]
    feature.backward()

    assert message.grad is not None
    assert torch.isfinite(message.grad).all()
    assert abs(float(message.grad.item())) > 1e-6
