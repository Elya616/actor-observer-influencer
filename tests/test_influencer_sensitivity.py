import runpy
from pathlib import Path

import pytest


def test_comparison_changes_only_learning_rate_and_records_history(capsys):
    experiment = runpy.run_path(str(Path(__file__).parents[1] /
                                   "experiments/run_influencer.py"))
    results = experiment["compare_learning_rates"](steps=3, seed=7, debug=True)
    assert list(results) == [0.001, 0.01, 0.05]
    for lr, traces in results.items():
        assert [t.metadata["seed"] for t in traces] == [7, 8, 9]
        for trace in traces:
            assert trace.metadata["influencer_learning_rate"] == lr
            assert trace.metadata["num_actor_compressors"] == 3
            assert trace.metadata["num_observers"] == 3
            assert len(trace.debug_rows) == 9
            assert len(trace.influencer_actor_probabilities) == len(trace.evaluation_steps)
            assert trace.influencer_actor_probabilities[0] == pytest.approx([1/3]*3)
    for i in range(3):
        baseline = results[0.001][i]
        for lr in (0.01, 0.05):
            other = results[lr][i]
            assert baseline.expected_rewards[0] == other.expected_rewards[0]
            assert baseline.selection_probabilities_by_observer[0] == other.selection_probabilities_by_observer[0]
    assert "reward mean=" in capsys.readouterr().out
