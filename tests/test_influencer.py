"""Small, known-answer checks for the direct and relay paths."""

import pytest
import torch

from actor_observer_rl.agents import Influencer
from actor_observer_rl.envs import FeaturePartialObservableStateEnv
from actor_observer_rl.training.combined_group import CombinedGroupTrainer
from actor_observer_rl.training.informational_state import (
    ActorCompressor, MaskedLinearActorCompressor, SelectionLogits,
    fixed_sum_observer_policy,
)


def build(influencer=True, source=2, actor=0, report=True, learn=True):
    env = FeaturePartialObservableStateEnv(
        state_features=[[1.0]], reward_table=[[1.0, 1.0]],
        state_distribution=[1.0], observation_dim=1,
    )
    actors = []
    for weight in [0.3, -0.3]:
        module = MaskedLinearActorCompressor(1)
        with torch.no_grad():
            module.linear.weight.fill_(weight)
            module.linear.bias.zero_()
        actors.append(ActorCompressor(module))
    relay = Influencer(2) if influencer else None
    if relay is not None:
        with torch.no_grad():
            relay.logits[actor] = 20.0
    logits = SelectionLogits(3 if influencer else 2)
    with torch.no_grad():
        logits.logits[source] = 80.0
    return CombinedGroupTrainer(
        env, actors, [fixed_sum_observer_policy(2, sharpness=1)], [logits],
        influencer=relay,
        influencer_optimizer=torch.optim.SGD(relay.parameters(), lr=0.1)
        if relay is not None and learn else None,
        compression_optimizer=torch.optim.SGD(
            [p for a in actors for p in a.parameters()], lr=0.1
        ) if learn else None,
        force_report=report, c=0.0 if not report else 1.0,
        learn_following=True, debug=True, optimal_actions=[0],
    )


def snapshot(module):
    return [p.detach().clone() for p in module.parameters()]


def unchanged(module, before):
    return all(torch.equal(p, old) for p, old in zip(module.parameters(), before))


def test_relay_preserves_message_and_gradient():
    relay = Influencer(1)
    message = torch.tensor([2.0], requires_grad=True)
    assert relay.relay(message) is message
    relay.relay(message).sum().backward()
    assert message.grad.item() == 1.0
    assert relay.select_actor()[0] == 0


@pytest.mark.parametrize("actor", [0, 1])
def test_selected_actor_updates_unselected_actor_does_not(actor):
    trainer = build(actor=actor)
    before = [snapshot(a) for a in trainer.actor_compressors]
    trace = trainer.run(1, seed=3, eval_interval=1)
    row = trace.debug_rows[0]
    assert row["source"] == 2 and row["actor"] == actor
    assert row["via_influencer"] and row["reported"]
    assert row["message"] == pytest.approx([0.3 if actor == 0 else -0.3])
    assert not unchanged(trainer.actor_compressors[actor], before[actor])
    assert unchanged(trainer.actor_compressors[1 - actor], before[1 - actor])
    assert trainer.mu.shape == (1, 3)


def test_no_report_prevents_compression_update():
    trainer = build(report=False)
    before = [snapshot(a) for a in trainer.actor_compressors]
    trace = trainer.run(3, seed=2, eval_interval=3)
    assert all(not row["reported"] for row in trace.debug_rows)
    assert all(unchanged(a, b) for a, b in zip(trainer.actor_compressors, before))


def test_direct_path_does_not_update_influencer():
    trainer = build(source=0)
    before = snapshot(trainer.influencer)
    trace = trainer.run(3, seed=2, eval_interval=3)
    assert all(not row["via_influencer"] for row in trace.debug_rows)
    assert unchanged(trainer.influencer, before)


def test_influencer_selection_learns_from_reward():
    trainer = build()
    with torch.no_grad():
        trainer.influencer.logits.zero_()
    trace = trainer.run(1, seed=4, eval_interval=1)
    chosen = trace.debug_rows[0]["actor"]
    assert trainer.influencer.probabilities()[chosen] > 0.5


def test_exact_evaluation_matches_enumerated_paths():
    trainer = build(learn=False)
    with torch.no_grad():
        trainer.selection_logits_by_observer[0].logits.zero_()
        trainer.influencer.logits.copy_(torch.tensor([0.0, 1.0]))
    state = trainer.env.full_state_for_index(0)
    probs = [trainer.observer_policies[0].action_distribution(a.message(state)).probs[0]
             for a in trainer.actor_compressors]
    relay = trainer.influencer.probabilities()
    expected = (probs[0] + probs[1] + relay[0]*probs[0] + relay[1]*probs[1]) / 3
    reward, correct = trainer.evaluate()
    assert reward == pytest.approx(1.0)
    assert correct == pytest.approx(float(expected.detach()))


def test_disabled_influencer_matches_original_seeded_trace():
    first = build(influencer=False, source=0)
    second = build(influencer=False, source=0)
    second.debug = False
    a = first.run(5, seed=2, eval_interval=1)
    b = second.run(5, seed=2, eval_interval=1)
    assert a.expected_rewards == b.expected_rewards
    assert a.mean_correct_action_probs == b.mean_correct_action_probs
    assert a.following_rates_by_observer == b.following_rates_by_observer
    assert not b.debug_rows
    assert first.mu.shape == (1, 2)


def test_debug_rejects_invalid_budget():
    trainer = build(learn=False)
    trainer.learn_following = False
    trainer.mu.fill_(10.0)
    with pytest.raises(AssertionError, match="budget exceeded"):
        trainer.run(1, seed=2)


def test_invalid_selection_size_is_rejected():
    trainer = build()
    with pytest.raises(ValueError, match="selection size"):
        CombinedGroupTrainer(trainer.env, trainer.actor_compressors,
                             trainer.observer_policies, [SelectionLogits(2)],
                             influencer=Influencer(2))
