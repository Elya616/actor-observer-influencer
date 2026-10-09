"""Scalar-state policy-learning probe (single Actor, perfect state + reward).

Professor Marbach asked us to switch the imperfect-reward / perfect-state input
from a one-hot vector to the raw scalar state ID, and to isolate whether the
*policy* is what fails to scale. This script does exactly that in the simplest
possible setting:

* single Actor, single Observer;
* perfect state (the Actor sees the exact state), perfect reward (every step is
  reported -- the ``force_report`` limit, so following rates do not gate the
  update);
* the only thing learned is the policy theta.

It sweeps the number of states K and compares three representations/policies:

* ``scalar + linear``  -- the scalar ID into a single linear layer (the case we
  suspect cannot scale: one affine function of a scalar must produce K distinct
  argmax actions);
* ``scalar + hidden``  -- the scalar ID into a one-hidden-layer policy (the
  "slightly bigger network" fix discussed in the meeting);
* ``one_hot + linear`` -- the previous representation, as a reference ceiling.

For each configuration we report, averaged over seeds, the fraction of states
whose greedy action is optimal (``acc``) and the mean probability mass on the
optimal action (``p_opt``). A configuration "works" when it maps every state to
its correct action (``acc = 1.0``).
"""

from __future__ import annotations

import argparse
from statistics import fmean

import torch

from actor_observer_rl.agents.actor import ActionResult
from actor_observer_rl.envs import IIDStateBanditEnv
from actor_observer_rl.policies import (
    EmbeddingPolicy,
    NeuralPolicy,
    RBFPolicy,
    SoftmaxPolicy,
)
from actor_observer_rl.training.updates import reinforce_loss
from actor_observer_rl.utils.seeding import set_global_seed


class DeepScalarPolicy(torch.nn.Module):
    """Deeper MLP on the scalar state ID: does adding hidden layers help?

    ``num_hidden_layers`` stacked ``Linear -> ReLU`` blocks before the linear
    action head. This tests whether more depth (rather than a smarter feature
    map like the RBF) is enough to fit the scalar->action mapping as K grows.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_actions: int,
        num_hidden_layers: int,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_actions = num_actions
        layers: list[torch.nn.Module] = []
        in_dim = input_dim
        for _ in range(num_hidden_layers):
            layers.append(torch.nn.Linear(in_dim, hidden_dim))
            layers.append(torch.nn.ReLU())
            in_dim = hidden_dim
        layers.append(torch.nn.Linear(in_dim, num_actions))
        self.network = torch.nn.Sequential(*layers)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.network(input_tensor.float())

    def action_distribution(
        self, input_tensor: torch.Tensor
    ) -> torch.distributions.Categorical:
        logits = self(input_tensor)
        if not torch.isfinite(logits).all():
            raise FloatingPointError("policy logits contain non-finite values")
        return torch.distributions.Categorical(logits=logits)

    def act(self, input_tensor: torch.Tensor, sample: bool = True) -> ActionResult:
        distribution = self.action_distribution(input_tensor)
        action = distribution.sample() if sample else distribution.probs.argmax(dim=-1)
        log_prob = distribution.log_prob(action)
        return ActionResult(action=action, log_prob=log_prob, probs=distribution.probs)


def diagonal_reward_table(
    num_states: int, correct_reward: float = 1.0, wrong_reward: float = -0.1
) -> list[list[float]]:
    """State ``k`` rewards action ``k``; every other action is penalized."""
    return [
        [correct_reward if a == s else wrong_reward for a in range(num_states)]
        for s in range(num_states)
    ]


def build_policy(
    kind: str,
    input_dim: int,
    num_actions: int,
    hidden_dim: int,
    num_states: int,
    embedding_dim: int = 16,
):
    if kind == "linear":
        return SoftmaxPolicy(input_dim=input_dim, num_actions=num_actions)
    if kind == "hidden":
        return NeuralPolicy(
            input_dim=input_dim, hidden_dim=hidden_dim, num_actions=num_actions
        )
    if kind == "rbf":
        return RBFPolicy(num_states=num_states, num_actions=num_actions)
    if kind == "embed":
        # Embedding alone + linear head (Peter / Gemini suggestion).
        return EmbeddingPolicy(
            num_states=num_states,
            num_actions=num_actions,
            embedding_dim=embedding_dim,
            num_hidden_layers=0,
        )
    if kind == "embed_mlp1":
        return EmbeddingPolicy(
            num_states=num_states,
            num_actions=num_actions,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_hidden_layers=1,
        )
    if kind.startswith("mlp"):
        num_hidden_layers = int(kind[3:])
        return DeepScalarPolicy(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_actions=num_actions,
            num_hidden_layers=num_hidden_layers,
        )
    raise ValueError(f"unknown policy kind: {kind}")


def _transform(
    observation: torch.Tensor,
    *,
    observation_mode: str,
    normalize: bool,
    num_states: int,
) -> torch.Tensor:
    """Optionally rescale the scalar state ID to ``[0, 1]``.

    A raw scalar ID in ``0..K-1`` is a large, unnormalized input; with ReLU +
    Adam this is hard to fit as ``K`` grows. Dividing by ``K-1`` puts every
    state on a comparable scale, which is the standard fix for feeding raw
    indices into a small network.
    """
    if normalize and observation_mode == "scalar_id" and num_states > 1:
        return observation.float() / float(num_states - 1)
    return observation.float()


def train_once(
    *,
    num_states: int,
    observation_mode: str,
    policy_kind: str,
    hidden_dim: int,
    num_steps: int,
    lr: float,
    seed: int,
    normalize: bool,
    explore_eps: float = 0.0,
    explore_eps_end: float | None = None,
    lr_end: float | None = None,
    use_baseline: bool = False,
    baseline_momentum: float = 0.05,
    embedding_dim: int = 16,
    finetune_steps: int = 0,
    finetune_lr: float | None = None,
    optimizer_name: str = "adam",
) -> tuple[float, float]:
    """Train a single Actor with perfect state + reward; return (acc, p_opt).

    ``explore_eps`` mixes a uniform action distribution into sampling:
    ``π_mix = (1-ε) π_θ + ε / K``. This is the uniform-sampling weight trick:
    every action keeps a nonzero sample probability, which matters when K is
    large and some actions would otherwise never be tried. If ``explore_eps_end``
    is set, ε anneals linearly from ``explore_eps`` down to that value. If
    ``lr_end`` is set, the learning rate anneals linearly as well.

    ``finetune_steps`` runs an extra phase with ε=0 (pure policy sampling) so the
    distribution can sharpen after exploration has covered the action space.

    At large K, ``optimizer_name="sgd"`` (lr≈0.05–0.1) outperforms Adam on
    EmbeddingPolicy: dense Adam momentum drifts unused embedding rows.
    """
    set_global_seed(seed)
    env = IIDStateBanditEnv(
        reward_table=diagonal_reward_table(num_states),
        state_distribution=[1.0 / num_states] * num_states,
        observation_mode=observation_mode,
    )
    input_dim = env.observation_dim
    policy = build_policy(
        policy_kind,
        input_dim,
        num_states,
        hidden_dim,
        num_states,
        embedding_dim=embedding_dim,
    )
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(policy.parameters(), lr=lr)
    else:
        raise ValueError(f"unknown optimizer: {optimizer_name}")
    eps_end = explore_eps if explore_eps_end is None else explore_eps_end
    final_lr = lr if lr_end is None else lr_end
    # Per-state baselines: global mean is useless when reward depends on (s,a).
    baseline_by_state: list[float | None] = [None] * num_states
    total_steps = num_steps + max(0, finetune_steps)

    for step in range(total_steps):
        in_finetune = step >= num_steps
        if in_finetune:
            eps = 0.0
            current_lr = lr if finetune_lr is None else finetune_lr
        else:
            frac = step / max(1, num_steps - 1)
            eps = explore_eps + (eps_end - explore_eps) * frac
            current_lr = lr + (final_lr - lr) * frac
        for group in optimizer.param_groups:
            group["lr"] = current_lr

        observation, info = env.reset()
        state_index = int(info["state_index"])
        actor_input = _transform(
            observation,
            observation_mode=observation_mode,
            normalize=normalize,
            num_states=num_states,
        )
        dist = policy.action_distribution(actor_input)
        if eps > 0.0:
            # Mixture: π_mix = (1-ε)π_θ + ε/K. Sample and update with log π_mix.
            mix_probs = (1.0 - eps) * dist.probs + eps / float(num_states)
            mix = torch.distributions.Categorical(probs=mix_probs)
            action_tensor = mix.sample()
            log_prob = mix.log_prob(action_tensor)
            action = int(action_tensor.detach().cpu().item())
        else:
            action_result = policy.act(actor_input, sample=True)
            action = int(action_result.action.detach().cpu().item())
            log_prob = action_result.log_prob

        reward = env.get_reward(state_index, action)
        # Per-state running-mean baseline cuts REINFORCE variance at large K.
        if use_baseline:
            prev = baseline_by_state[state_index]
            baseline = (
                reward
                if prev is None
                else (1.0 - baseline_momentum) * prev + baseline_momentum * reward
            )
            baseline_by_state[state_index] = baseline
            advantage = reward - baseline
        else:
            advantage = reward
        loss = reinforce_loss(advantage, log_prob)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return evaluate(env, policy, num_states, observation_mode, normalize)


@torch.no_grad()
def evaluate(
    env: IIDStateBanditEnv,
    policy,
    num_states: int,
    observation_mode: str,
    normalize: bool,
) -> tuple[float, float]:
    correct = 0
    prob_opt = 0.0
    for state_index in range(num_states):
        observation = env.observation_for_index(state_index)
        actor_input = _transform(
            observation,
            observation_mode=observation_mode,
            normalize=normalize,
            num_states=num_states,
        )
        probs = policy.action_distribution(actor_input).probs
        greedy = int(probs.argmax().item())
        if greedy == state_index:  # diagonal reward -> optimal action == state id
            correct += 1
        prob_opt += float(probs[state_index].item())
    return correct / num_states, prob_opt / num_states


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--states", type=int, nargs="+", default=[3, 5, 8, 16],
        help="State counts K to sweep (actions == states, diagonal reward).",
    )
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument(
        "--explore-eps",
        type=float,
        default=0.0,
        help="Uniform action mixture weight ε: π_mix=(1-ε)π + ε/K.",
    )
    parser.add_argument(
        "--explore-eps-end",
        type=float,
        default=None,
        help="If set, anneal ε linearly from --explore-eps down to this.",
    )
    parser.add_argument(
        "--lr-end",
        type=float,
        default=None,
        help="If set, anneal Adam lr linearly from --lr down to this.",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Use a running-mean reward baseline to cut REINFORCE variance.",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=16,
        help="Embedding width for EmbeddingPolicy.",
    )
    parser.add_argument(
        "--finetune-steps",
        type=int,
        default=0,
        help="Extra steps with ε=0 after the explore phase (sharpen π).",
    )
    parser.add_argument(
        "--finetune-lr",
        type=float,
        default=None,
        help="Optimizer lr during finetune (default: keep --lr).",
    )
    parser.add_argument(
        "--optimizer",
        choices=("adam", "sgd"),
        default="adam",
        help="Policy optimizer. Prefer sgd for EmbeddingPolicy at large K.",
    )
    parser.add_argument(
        "--only", nargs="+", default=None,
        help="Only run configs whose label contains one of these substrings.",
    )
    args = parser.parse_args()

    # Peter asked (after Gemini): try (1) MLP with 1/2 hidden layers on the raw
    # scalar, and (2) an Embedding layer. one_hot is the ceiling reference.
    configs = [
        ("scalar + mlp1 (norm)", "scalar_id", "mlp1", True),
        ("scalar + mlp2 (norm)", "scalar_id", "mlp2", True),
        ("scalar + embed", "scalar_id", "embed", False),
        ("scalar + embed+mlp1", "scalar_id", "embed_mlp1", False),
        ("scalar + rbf", "scalar_id", "rbf", False),
        ("one_hot + linear", "one_hot", "linear", False),
    ]
    if args.only:
        configs = [
            c
            for c in configs
            if any(
                (k.startswith("=") and c[0] == k[1:]) or (not k.startswith("=") and k in c[0])
                for k in args.only
            )
        ]

    print(
        f"steps={args.steps}+ft{args.finetune_steps} opt={args.optimizer} "
        f"lr={args.lr} lr_end={args.lr_end} "
        f"ft_lr={args.finetune_lr} hidden_dim={args.hidden_dim} seeds={args.seeds} "
        f"eps={args.explore_eps}->{args.explore_eps_end} baseline={args.baseline}"
    )
    print(f"{'config':<24} {'K':>4} {'acc':>7} {'p_opt':>7}")
    for label, mode, kind, normalize in configs:
        for num_states in args.states:
            accs, popts = [], []
            for seed in range(args.seeds):
                acc, p_opt = train_once(
                    num_states=num_states,
                    observation_mode=mode,
                    policy_kind=kind,
                    hidden_dim=args.hidden_dim,
                    num_steps=args.steps,
                    lr=args.lr,
                    seed=seed,
                    normalize=normalize,
                    explore_eps=args.explore_eps,
                    explore_eps_end=args.explore_eps_end,
                    lr_end=args.lr_end,
                    use_baseline=args.baseline,
                    embedding_dim=args.embedding_dim,
                    finetune_steps=args.finetune_steps,
                    finetune_lr=args.finetune_lr,
                    optimizer_name=args.optimizer,
                )
                accs.append(acc)
                popts.append(p_opt)
            print(
                f"{label:<24} {num_states:>4} {fmean(accs):>7.3f} {fmean(popts):>7.3f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
