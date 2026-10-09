"""Systematic stress test for the imperfect-reward side (Sections 4 and 6).

This is the "one rung at a time, check the learning rate and the trajectory at
each rung" procedure Professor Marbach asked for, made systematic. Starting from
the simplest 1-Actor / 1-Observer setting, it scales a single axis gradually and,
at every rung, does three things:

* sweeps the Actor learning rate and keeps the best one (rather than reusing a
  rate tuned on a smaller rung -- the mistake the professor warned about);
* records the greedy-accuracy *trajectory* of that best rate, so we can see
  whether it settled or is still moving / oscillating; and
* judges the rung ``pass`` / ``watch`` / ``fail`` from the settled accuracy and
  how much it still wobbles at the end.

Two axes are covered here, both with one Actor and one Observer so the reward
mechanism is isolated from crowd effects:

* ``states``  -- fix the action count, grow the number of grid states;
* ``actions`` -- fix the state count, grow the number of actions (regions).

The rule is only to trust a rung once the previous one passed, so the ladders
are deliberately gradual. Per-rung trajectories and a summary figure go to
``docs/figures``; a summary CSV goes to ``outputs``.
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".mplcache").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))

from actor_observer_rl.agents import BudgetedObserver
from actor_observer_rl.envs import NoisyActorBanditEnv
from actor_observer_rl.training import MultiActorTrainer, make_fixed_softmax_actor
from actor_observer_rl.utils.seeding import set_global_seed
from run_scaling_actors import concentration as actors_concentration
from run_scaling_actors import top_k_precision
from run_scaling_states import train_one_grid

PASS_THRESHOLD = 0.80
WATCH_THRESHOLD = 0.60
WOBBLE_LIMIT = 0.05


def _save_figure(fig: plt.Figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png", ".svg"):
        fig.savefig(output_base.with_suffix(suffix), bbox_inches="tight")
    plt.close(fig)


def late_wobble(curve: list[float], window: int = 5) -> float:
    tail = curve[-window:] if len(curve) >= window else curve
    return statistics.pstdev(tail) if len(tail) > 1 else 0.0


def verdict(accuracy: float, wobble: float, ceiling: float) -> str:
    """Rate a rung against the best achievable (its representative ceiling).

    We compare to the representative-state accuracy: on this grid the region
    boundaries are genuinely ambiguous, so the honest target is "did the
    in-region cells settle", not a flat 1.0.
    """

    target = min(PASS_THRESHOLD, ceiling - 0.05)
    if accuracy >= target and wobble <= WOBBLE_LIMIT:
        return "pass"
    if accuracy >= WATCH_THRESHOLD:
        return "watch"
    return "fail"


def best_lr_at_rung(
    *,
    grid_size: int,
    num_actions: int,
    hidden_dim: int,
    learning_rates: list[float],
    num_steps: int,
    eval_interval: int,
    seeds: list[int],
) -> dict[str, object]:
    """Sweep the learning rate at one rung and return the best-rate summary."""

    best: dict[str, object] | None = None
    for lr in learning_rates:
        runs = [
            train_one_grid(
                grid_size=grid_size,
                num_actions=num_actions,
                hidden_dim=hidden_dim,
                learning_rate=lr,
                num_steps=num_steps,
                eval_interval=eval_interval,
                seed=seed,
            )
            for seed in seeds
        ]
        num_points = min(len(r["accuracy_curve"]) for r in runs)
        mean_curve = [
            sum(r["accuracy_curve"][k] for r in runs) / len(runs)
            for k in range(num_points)
        ]
        final_acc = sum(r["final_accuracy"] for r in runs) / len(runs)
        rep_mean = sum(r["representative_mean"] for r in runs) / len(runs)
        wobble = sum(late_wobble(r["accuracy_curve"]) for r in runs) / len(runs)
        summary = {
            "learning_rate": lr,
            "steps_axis": runs[0]["steps_axis"][:num_points],
            "accuracy_curve": mean_curve,
            "final_accuracy": final_acc,
            "representative_mean": rep_mean,
            "late_wobble": wobble,
        }
        if best is None or final_acc > best["final_accuracy"]:
            best = summary
    assert best is not None
    return best


def run_axis(
    *,
    axis: str,
    rungs: list[int],
    fixed: int,
    hidden_dim: int,
    learning_rates: list[float],
    num_steps: int,
    eval_interval: int,
    seeds: list[int],
) -> list[dict[str, object]]:
    """Walk one axis rung by rung, tuning the learning rate at each rung."""

    results: list[dict[str, object]] = []
    for rung in rungs:
        if axis == "states":
            grid_size, num_actions = rung, fixed
            size_label = f"{rung * rung} states"
        else:
            grid_size, num_actions = fixed, rung
            size_label = f"{rung} actions"

        best = best_lr_at_rung(
            grid_size=grid_size,
            num_actions=num_actions,
            hidden_dim=hidden_dim,
            learning_rates=learning_rates,
            num_steps=num_steps,
            eval_interval=eval_interval,
            seeds=seeds,
        )
        rung_verdict = verdict(
            float(best["final_accuracy"]),
            float(best["late_wobble"]),
            float(best["representative_mean"]),
        )
        results.append(
            {
                "axis": axis,
                "rung": rung,
                "grid_size": grid_size,
                "num_actions": num_actions,
                "num_states": grid_size * grid_size,
                "size_label": size_label,
                "best_lr": best["learning_rate"],
                "final_accuracy": best["final_accuracy"],
                "representative_mean": best["representative_mean"],
                "late_wobble": best["late_wobble"],
                "verdict": rung_verdict,
                "steps_axis": best["steps_axis"],
                "accuracy_curve": best["accuracy_curve"],
            }
        )
    return results


def train_actors_crowd(
    *,
    num_actors: int,
    num_good: int,
    following_lr: float,
    utility_lr: float,
    budget: float,
    c: float,
    good_quality: float,
    bad_quality: float,
    noise_std: float,
    num_steps: int,
    snapshot_interval: int,
    seed: int,
) -> dict[str, object]:
    """Section 4.1 crowd: fixed-policy Actors, Observer learns whom to follow.

    Returns the budget-concentration and following-rate (mu) trajectories so we
    can watch the Observer move its budget onto the good Actors over time.
    """

    set_global_seed(seed)
    qualities = [good_quality] * num_good + [bad_quality] * (num_actors - num_good)
    env = NoisyActorBanditEnv(qualities=qualities, noise_std=noise_std)
    actors = [
        make_fixed_softmax_actor(
            num_states=1, num_actions=num_actors, target_action=j, sharpness=8.0
        )
        for j in range(num_actors)
    ]
    observer = BudgetedObserver(
        num_actors=num_actors,
        budget=budget,
        utility_learning_rate=utility_lr,
        following_learning_rate=following_lr,
        c=c,
        seed=seed,
    )
    trainer = MultiActorTrainer(
        env=env,
        actors=actors,
        observer=observer,
        actor_selection_probs=[1.0 / num_actors] * num_actors,
        actor_optimizers=None,
        snapshot_interval=snapshot_interval,
    )
    trace = trainer.run(num_steps=num_steps, seed=seed)

    steps_axis = [
        min((i + 1) * snapshot_interval, num_steps)
        for i in range(len(trace.following_rates))
    ]
    num_bad = num_actors - num_good
    concentration_curve = [
        actors_concentration(mu, num_good, budget) for mu in trace.following_rates
    ]
    mu_good_curve = [sum(mu[:num_good]) / num_good for mu in trace.following_rates]
    mu_bad_curve = [
        (sum(mu[num_good:]) / num_bad if num_bad else 0.0)
        for mu in trace.following_rates
    ]
    topk_curve = [top_k_precision(mu, num_good) for mu in trace.following_rates]

    final_mu = trace.final_following_rates or [0.0] * num_actors
    return {
        "num_actors": num_actors,
        "steps_axis": steps_axis,
        "concentration_curve": concentration_curve,
        "mu_good_curve": mu_good_curve,
        "mu_bad_curve": mu_bad_curve,
        "topk_curve": topk_curve,
        "final_concentration": actors_concentration(final_mu, num_good, budget),
        "final_top_k_precision": top_k_precision(final_mu, num_good),
    }


def actors_verdict(top_k: float, concentration: float) -> str:
    """Judge an actors rung on *identification* (top-k precision).

    Budget concentration necessarily dilutes as the crowd grows (a fixed budget
    spread over more Actors), which is expected and not a failure. What must
    hold is that the Observer still ranks the good Actors at the top.
    """

    if top_k >= 0.99:
        return "pass"
    if top_k >= 0.75:
        return "watch"
    return "fail"


def run_actors_axis(
    *,
    rungs: list[int],
    num_good: int,
    following_lrs: list[float],
    utility_lr: float,
    budget: float,
    c: float,
    good_quality: float,
    bad_quality: float,
    noise_std: float,
    num_steps: int,
    snapshot_interval: int,
    seeds: list[int],
) -> list[dict[str, object]]:
    """Walk the actors ladder rung by rung, tuning the following rate per rung."""

    results: list[dict[str, object]] = []
    for num_actors in rungs:
        best: dict[str, object] | None = None
        for following_lr in following_lrs:
            runs = [
                train_actors_crowd(
                    num_actors=num_actors,
                    num_good=num_good,
                    following_lr=following_lr,
                    utility_lr=utility_lr,
                    budget=budget,
                    c=c,
                    good_quality=good_quality,
                    bad_quality=bad_quality,
                    noise_std=noise_std,
                    num_steps=num_steps,
                    snapshot_interval=snapshot_interval,
                    seed=seed,
                )
                for seed in seeds
            ]
            n = len(runs)
            num_points = min(len(r["concentration_curve"]) for r in runs)

            def curve(key: str) -> list[float]:
                return [sum(r[key][k] for r in runs) / n for k in range(num_points)]

            top_k = sum(r["final_top_k_precision"] for r in runs) / n
            summary = {
                "num_actors": num_actors,
                "following_lr": following_lr,
                "steps_axis": runs[0]["steps_axis"][:num_points],
                "concentration_curve": curve("concentration_curve"),
                "mu_good_curve": curve("mu_good_curve"),
                "mu_bad_curve": curve("mu_bad_curve"),
                "final_top_k_precision": top_k,
                "final_concentration": sum(r["final_concentration"] for r in runs) / n,
            }
            if best is None or top_k > best["final_top_k_precision"]:
                best = summary
        assert best is not None
        best["verdict"] = actors_verdict(
            float(best["final_top_k_precision"]),
            float(best["final_concentration"]),
        )
        best["size_label"] = f"{num_actors} actors"
        results.append(best)
    return results


def plot_following_rate_trajectory(
    result: dict[str, object], output_base: Path
) -> None:
    """Plot mean following rate mu on good vs bad Actors over training.

    This is the trajectory the professor asks to see: whether the Observer's
    budget actually moves onto the good Actors and stays there.
    """

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.plot(
        result["steps_axis"],
        result["mu_good_curve"],
        linewidth=2.2,
        label="mean mu on good actors",
    )
    ax.plot(
        result["steps_axis"],
        result["mu_bad_curve"],
        linewidth=1.8,
        linestyle="--",
        label="mean mu on bad actors",
    )
    ax.set_title(
        f"Following-rate trajectory ({result['num_actors']} actors, "
        f"lr={result['following_lr']:g})"
    )
    ax.set_xlabel("training step")
    ax.set_ylabel("mean following rate mu")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    _save_figure(fig, output_base)


def plot_actors_topk(results: list[dict[str, object]], output_base: Path) -> None:
    """Top-k precision and budget concentration vs crowd size (actors axis)."""

    sizes = [r["num_actors"] for r in results]
    top_k = [r["final_top_k_precision"] for r in results]
    conc = [r["final_concentration"] for r in results]

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.plot(sizes, top_k, marker="o", linewidth=2.0, label="top-k precision")
    ax.plot(
        sizes,
        conc,
        marker="s",
        linewidth=1.8,
        linestyle="--",
        label="budget concentration",
    )
    ax.set_title("Stress test: actors axis (1 observer, fixed-policy actors)")
    ax.set_xlabel("number of actors")
    ax.set_ylabel("final value")
    ax.set_xscale("log", base=2)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    _save_figure(fig, output_base)


def plot_trajectories(results: list[dict[str, object]], axis: str, output_base: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for result in results:
        ax.plot(
            result["steps_axis"],
            result["accuracy_curve"],
            linewidth=1.9,
            label=f"{result['size_label']} (lr={result['best_lr']:g}, {result['verdict']})",
        )
    axis_name = "states" if axis == "states" else "actions"
    ax.set_title(f"Stress test: greedy-accuracy trajectory per rung (scaling {axis_name})")
    ax.set_xlabel("training step")
    ax.set_ylabel("greedy accuracy")
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    _save_figure(fig, output_base)


def write_csv(results: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "axis",
        "num_states",
        "num_actions",
        "best_lr",
        "final_accuracy",
        "representative_mean",
        "late_wobble",
        "verdict",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow({field: result[field] for field in fields})


def print_axis(axis: str, results: list[dict[str, object]]) -> None:
    print(f"\n=== axis: {axis} ===")
    print(
        f"{'states':>7} {'actions':>8} {'best_lr':>8} {'acc':>7} "
        f"{'rep':>6} {'wobble':>7} {'verdict':>8}"
    )
    for r in results:
        print(
            f"{r['num_states']:>7} {r['num_actions']:>8} {r['best_lr']:>8g} "
            f"{r['final_accuracy']:>7.3f} {r['representative_mean']:>6.3f} "
            f"{r['late_wobble']:>7.4f} {r['verdict']:>8}"
        )


def write_actors_csv(results: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "num_actors",
        "following_lr",
        "final_top_k_precision",
        "final_concentration",
        "verdict",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow({field: result[field] for field in fields})


def print_actors_axis(results: list[dict[str, object]]) -> None:
    print("\n=== axis: actors (1 observer, fixed-policy actors) ===")
    print(
        f"{'actors':>7} {'foll_lr':>8} {'top_k':>7} {'conc':>7} {'verdict':>8}"
    )
    for r in results:
        print(
            f"{r['num_actors']:>7} {r['following_lr']:>8g} "
            f"{r['final_top_k_precision']:>7.3f} {r['final_concentration']:>7.3f} "
            f"{r['verdict']:>8}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure-dir", type=Path, default=Path("docs/figures"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/stress_test"),
    )
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument(
        "--state-rungs",
        type=int,
        nargs="+",
        default=[2, 3, 4, 6, 8, 11, 16],
        help="Grid sides for the state axis (states = side^2), fixed 4 actions.",
    )
    parser.add_argument(
        "--action-rungs",
        type=int,
        nargs="+",
        default=[2, 3, 4, 6, 8],
        help="Action counts for the action axis, fixed 64 states (8x8).",
    )
    parser.add_argument("--state-axis-actions", type=int, default=4)
    parser.add_argument("--action-axis-grid", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument(
        "--learning-rates",
        type=float,
        nargs="+",
        default=[0.003, 0.01, 0.03],
    )
    parser.add_argument("--num-steps", type=int, default=40000)
    parser.add_argument("--eval-interval", type=int, default=4000)
    parser.add_argument(
        "--actor-rungs",
        type=int,
        nargs="+",
        default=[4, 8, 16, 32, 64, 128],
        help="Crowd sizes for the actors axis (1 Observer, fixed-policy Actors).",
    )
    parser.add_argument("--actor-num-good", type=int, default=2)
    parser.add_argument(
        "--following-lrs",
        type=float,
        nargs="+",
        default=[0.02, 0.05, 0.1],
    )
    parser.add_argument("--actor-utility-lr", type=float, default=0.05)
    parser.add_argument("--actor-budget", type=float, default=2.0)
    parser.add_argument("--actor-c", type=float, default=1.0)
    parser.add_argument("--actor-good-quality", type=float, default=1.0)
    parser.add_argument("--actor-bad-quality", type=float, default=0.0)
    parser.add_argument("--actor-noise-std", type=float, default=1.5)
    parser.add_argument("--actor-num-steps", type=int, default=12000)
    parser.add_argument("--actor-snapshot-interval", type=int, default=1000)
    args = parser.parse_args()

    seeds = list(range(args.seeds))

    state_results = run_axis(
        axis="states",
        rungs=args.state_rungs,
        fixed=args.state_axis_actions,
        hidden_dim=args.hidden_dim,
        learning_rates=args.learning_rates,
        num_steps=args.num_steps,
        eval_interval=args.eval_interval,
        seeds=seeds,
    )
    action_results = run_axis(
        axis="actions",
        rungs=args.action_rungs,
        fixed=args.action_axis_grid,
        hidden_dim=args.hidden_dim,
        learning_rates=args.learning_rates,
        num_steps=args.num_steps,
        eval_interval=args.eval_interval,
        seeds=seeds,
    )

    actor_results = run_actors_axis(
        rungs=args.actor_rungs,
        num_good=args.actor_num_good,
        following_lrs=args.following_lrs,
        utility_lr=args.actor_utility_lr,
        budget=args.actor_budget,
        c=args.actor_c,
        good_quality=args.actor_good_quality,
        bad_quality=args.actor_bad_quality,
        noise_std=args.actor_noise_std,
        num_steps=args.actor_num_steps,
        snapshot_interval=args.actor_snapshot_interval,
        seeds=seeds,
    )

    plot_trajectories(state_results, "states", args.figure_dir / "stress_test_states")
    plot_trajectories(action_results, "actions", args.figure_dir / "stress_test_actions")
    plot_actors_topk(actor_results, args.figure_dir / "stress_test_actors")
    # Representative following-rate trajectory: use the largest crowd rung.
    plot_following_rate_trajectory(
        actor_results[-1], args.figure_dir / "stress_test_actors_following_rate"
    )
    write_csv(
        state_results + action_results,
        args.output_dir / "stress_test_summary.csv",
    )
    write_actors_csv(actor_results, args.output_dir / "stress_test_actors_summary.csv")

    print(f"seeds={args.seeds}, num_steps={args.num_steps}, lr sweep={args.learning_rates}")
    print_axis("states (1 actor, 1 observer, fixed actions)", state_results)
    print_axis("actions (1 actor, 1 observer, fixed states)", action_results)
    print_actors_axis(actor_results)


if __name__ == "__main__":
    main()
