"""Run Case 1 benchmark sweeps across scenarios, learning rates, and seeds."""

from __future__ import annotations

import argparse
from pathlib import Path

from actor_observer_rl.evaluation import (
    Case1BenchmarkConfig,
    default_case1_scenarios,
    run_case1_benchmark,
    summarize_case1_results,
    write_case1_results_csv,
    write_case1_summary_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/case1_benchmark_runs.csv"),
        help="Path for per-run benchmark CSV output.",
    )
    parser.add_argument("--num-steps", type=int, default=1000)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--reward-window", type=int, default=100)
    parser.add_argument(
        "--learning-rates",
        type=float,
        nargs="+",
        default=[0.001, 0.01, 0.05, 0.1, 0.2],
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = parser.parse_args()

    config = Case1BenchmarkConfig(
        scenarios=default_case1_scenarios(),
        learning_rates=args.learning_rates,
        seeds=args.seeds,
        num_steps=args.num_steps,
        convergence_threshold=args.threshold,
        reward_window=args.reward_window,
    )
    run_results = run_case1_benchmark(config)
    summary_results = summarize_case1_results(run_results)

    summary_path = args.output.with_name(args.output.stem.replace("_runs", "") + "_summary.csv")
    write_case1_results_csv(run_results, args.output)
    write_case1_summary_csv(summary_results, summary_path)

    print(f"wrote_run_results={args.output}")
    print(f"wrote_summary_results={summary_path}")
    _print_summary_table(summary_results)


def _print_summary_table(summary_results) -> None:
    header = (
        "scenario",
        "lr",
        "success",
        "mean_prob",
        "mean_reward",
        "mean_conv_step",
    )
    print(" | ".join(header))
    print(" | ".join("-" * len(column) for column in header))
    for result in summary_results:
        mean_step = (
            "NA"
            if result.mean_convergence_step is None
            else f"{result.mean_convergence_step:.1f}"
        )
        print(
            " | ".join(
                [
                    result.scenario,
                    f"{result.learning_rate:g}",
                    f"{result.success_rate:.2f}",
                    f"{result.mean_final_optimal_action_prob:.3f}",
                    f"{result.mean_average_reward_last_window:.3f}",
                    mean_step,
                ]
            )
        )


if __name__ == "__main__":
    main()
