"""Minimal example: three Actors, three Observers, and one shared Influencer."""

import argparse
from statistics import fmean, pstdev

import torch

from actor_observer_rl.agents import Influencer
from actor_observer_rl.envs import FeaturePartialObservableStateEnv
from actor_observer_rl.training.combined_group import CombinedGroupTrainer
from actor_observer_rl.training.informational_state import (
    SelectionLogits, build_scalar_sum_actor_compressors,
    cumulative_binary_state_features, diagonal_reward_table,
    fixed_sum_observer_policy,
)
from actor_observer_rl.utils.seeding import set_global_seed


def run_experiment(steps=500, seed=0, influencer_lr=0.01, debug=False):
    # Rebuild and reseed each run: only the Influencer learning rate changes.
    set_global_seed(seed)
    env = FeaturePartialObservableStateEnv(
        state_features=cumulative_binary_state_features(3),
        reward_table=diagonal_reward_table(3),
        state_distribution=[1/3] * 3, observation_dim=1,
    )
    actors = build_scalar_sum_actor_compressors(3, learnable=True)
    influencer = Influencer(len(actors))
    selection = [SelectionLogits(len(actors) + 1) for _ in range(3)]
    trainer = CombinedGroupTrainer(
        env, actors, [fixed_sum_observer_policy(3) for _ in range(3)], selection,
        influencer=influencer,
        influencer_optimizer=torch.optim.SGD(influencer.parameters(), lr=influencer_lr),
        compression_optimizer=torch.optim.SGD(
            [p for a in actors for p in a.parameters() if p.requires_grad], lr=0.01),
        selection_optimizer=torch.optim.SGD(
            [p for s in selection for p in s.parameters()], lr=0.01),
        learn_following=True, debug=debug,
    )
    trace = trainer.run(steps, seed=seed, eval_interval=max(1, steps // 5))
    trace.metadata["influencer_learning_rate"] = influencer_lr
    return trace


def compare_learning_rates(steps=500, seed=0, debug=False):
    results = {}
    for lr in (0.001, 0.01, 0.05):
        traces = [run_experiment(steps, seed+i, lr, debug) for i in range(3)]
        results[lr] = traces
        print(f"\nInfluencer learning rate={lr}; seeds={[seed+i for i in range(3)]}")
        for index, step in enumerate(traces[0].evaluation_steps):
            rewards = [t.expected_rewards[index] for t in traces]
            actor_probs = [fmean(t.influencer_actor_probabilities[index][j] for t in traces)
                           for j in range(3)]
            relay_probs = [fmean(t.selection_probabilities_by_observer[index][j][-1]
                                for t in traces) for j in range(3)]
            print(f"step={step}: reward mean={fmean(rewards):.6f}, "
                  f"std={pstdev(rewards):.6f}; "
                  f"Influencer P(A,B,C)={[round(p,4) for p in actor_probs]}; "
                  f"Observer P(Influencer)={[round(p,4) for p in relay_probs]}")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--influencer-lr", type=float, default=0.01)
    parser.add_argument("--compare-lr", action="store_true")
    args = parser.parse_args()
    if args.compare_lr:
        compare_learning_rates(args.steps, args.seed, args.debug)
        return
    trace = run_experiment(args.steps, args.seed, args.influencer_lr, args.debug)
    print("Expected reward:", trace.final_expected_reward)
    print("Influencer Actor probabilities:", trace.influencer_actor_probabilities[-1])
    print("Observer source probabilities:", trace.selection_probabilities_by_observer[-1])
    for row in trace.debug_rows[:10]:
        print(row)


if __name__ == "__main__":
    main()
