"""Metric dataclasses for learning curves and benchmark summaries."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkSummary:
    """Report-ready aggregate for repeated-seed benchmark runs."""

    mean_final_reward: float
    std_final_reward: float
    mean_convergence_step: float | None
    num_runs: int
