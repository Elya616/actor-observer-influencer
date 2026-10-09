"""Run Case 2 benchmark sweeps across state distributions, LRs, and seeds."""

from __future__ import annotations

import argparse
from pathlib import Path

from actor_observer_rl.evaluation import (
    Case2BenchmarkConfig,
    default_case2_scenarios,
    default_case2_state_distributions,
    run_case2_benchmark,
    summarize_case2_results,
    write_case2_per_state_csv,
    write_case2_results_csv,
    write_case2_summary_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/case2_benchmark_runs.csv"),
        help="Path for per-run benchmark CSV output.",
    )
    parser.add_argument("--num-steps", type=int, default=2000)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--reward-window", type=int, default=100)
    parser.add_argument(
        "--learning-rates",
        type=float,
        nargs="+",
        default=[0.001, 0.01, 0.05, 0.1],
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = parser.parse_args()

    config = Case2BenchmarkConfig(
        scenarios=default_case2_scenarios(),
        state_distributions=default_case2_state_distributions(),
        learning_rates=args.learning_rates,
        seeds=args.seeds,
        num_steps=args.num_steps,
        convergence_threshold=args.threshold,
        reward_window=args.reward_window,
    )
    run_results, per_state_results = run_case2_benchmark(config)
    summary_results = summarize_case2_results(run_results)

    output_stem = args.output.stem.removesuffix("_runs")
    summary_path = args.output.with_name(f"{output_stem}_summary.csv")
    per_state_path = args.output.with_name(f"{output_stem}_per_state.csv")
    write_case2_results_csv(run_results, args.output)
    write_case2_per_state_csv(per_state_results, per_state_path)
    write_case2_summary_csv(summary_results, summary_path)

    print(f"wrote_run_results={args.output}")
    print(f"wrote_per_state_results={per_state_path}")
    print(f"wrote_summary_results={summary_path}")
    _print_summary_table(summary_results)


def _print_summary_table(summary_results) -> None:
    header = (
        "scenario",
        "distribution",
        "lr",
        "success",
        "weighted_prob",
        "min_prob",
        "expected_reward",
        "mean_conv_step",
    )
    print(" | ".join(header))
    print(" | ".join("-" * len(column) for column in header))
    for result in summary_results:
        mean_step = (
            "NA"
            if result.mean_overall_convergence_step is None
            else f"{result.mean_overall_convergence_step:.1f}"
        )
        print(
            " | ".join(
                [
                    result.scenario,
                    result.distribution,
                    f"{result.learning_rate:g}",
                    f"{result.success_rate:.2f}",
                    f"{result.mean_weighted_final_optimal_action_prob:.3f}",
                    f"{result.mean_min_final_optimal_action_prob:.3f}",
                    f"{result.mean_expected_reward_policy:.3f}",
                    mean_step,
                ]
            )
        )


if __name__ == "__main__":
    main()
