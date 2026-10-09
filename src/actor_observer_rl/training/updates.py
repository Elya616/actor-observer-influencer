"""Policy-gradient update helpers."""

from __future__ import annotations

from typing import Any

import torch


def reinforce_loss(reward: float | torch.Tensor, log_prob: torch.Tensor) -> torch.Tensor:
    """Return negative reward-weighted log probability for gradient ascent."""
    reward_tensor = torch.as_tensor(reward, dtype=log_prob.dtype, device=log_prob.device)
    reward_tensor = reward_tensor.detach()
    if reward_tensor.shape not in (torch.Size([]), log_prob.shape):
        raise ValueError(
            f"reward shape {tuple(reward_tensor.shape)} is incompatible with "
            f"log_prob shape {tuple(log_prob.shape)}"
        )
    return -(reward_tensor * log_prob).mean()


def linear_anneal(
    start: float,
    end: float | None,
    step: int,
    total_steps: int,
) -> float:
    """Linearly interpolate from ``start`` to ``end`` over ``total_steps``.

    ``end`` of ``None`` keeps the value constant at ``start`` (no annealing).
    ``step`` is 0-indexed; the final step reaches ``end`` exactly. Matches the
    schedule used by the scalar policy probe.
    """

    if end is None:
        return start
    if total_steps <= 1:
        return end
    frac = step / (total_steps - 1)
    return start + (end - start) * frac


def mixed_action_sample(
    actor: Any,
    actor_input: torch.Tensor,
    exploration_probability: float,
    log_prob_source: str = "mixed",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample an action, optionally mixing a uniform floor into the policy.

    Implements the uniform-exploration recipe Professor Marbach / Amanda asked
    for and that fixed the occasional stuck Actor at large ``N``/``K``:
    ``pi_mix = (1 - eps) * pi_theta + eps / num_actions``. The action is sampled
    from ``pi_mix`` and, when ``log_prob_source == "mixed"`` (the default), the
    REINFORCE update uses ``log pi_mix`` -- the uniform-sampling weight trick
    that keeps the estimator's expectation correct while widening coverage.

    ``actor`` is anything exposing ``act(actor_input, sample) -> ActionResult``
    (an ``Actor`` or a bare policy). Sampling goes through ``actor.act`` so
    Actor-level input transforms are preserved -- e.g. a blind Actor that zeros
    its observation before acting. Calling ``actor.policy.act`` directly would
    bypass that override and silently let the Actor "see" the state.

    When ``exploration_probability <= 0`` this is exactly
    ``actor.act(actor_input, sample=True)``, so the default (eps=0) path is
    byte-for-byte identical to the original REINFORCE loop.

    Returns ``(action, log_prob, probs)`` where ``probs`` is the distribution
    the action was actually sampled from (mixed when exploring).
    """

    if log_prob_source not in {"mixed", "learned"}:
        raise ValueError("log_prob_source must be 'mixed' or 'learned'")

    if exploration_probability <= 0.0:
        result = actor.act(actor_input, sample=True)
        return result.action, result.log_prob, result.probs

    # sample=False consumes no RNG; we only need pi_theta's probabilities here
    # (respecting any Actor input transform) and then draw from the mixture.
    probs = actor.act(actor_input, sample=False).probs
    num_actions = probs.shape[-1]
    mixed = (1.0 - exploration_probability) * probs + exploration_probability / num_actions
    action = torch.distributions.Categorical(probs=mixed).sample()
    if log_prob_source == "mixed":
        log_prob = torch.log(mixed[action].clamp_min(1e-12))
    else:
        log_prob = torch.log(probs[action].clamp_min(1e-12))
    return action, log_prob, mixed
