# Paper Section 7 Stabilization Checks

This note documents the follow-up checks for the group imperfect-state
experiments. Section 7 extends the Section 5 setup from one Actor to several
Actors. Each Actor has its own Observer-selection probabilities, but the
Observers and their message functions are shared.

## Failure Mode

The main failure mode is **partial-Observer collapse**. It can happen when one
Actor selects the partial Observer early, receives acceptable reward, and then
reinforces that selection. After that, the full Observer is selected less often
by that Actor, so it receives fewer compression updates. The full Observer is
still the best information source in principle, but it may not become
empirically better quickly enough for every Actor.

This is why Section 7 needs per-Actor diagnostics. Mean reward can hide one
Actor following the wrong Observer.

## Metrics

- `P_j(full)`: probability that Actor `j` selects the full-state Observer.
- `min P(full)`: minimum final `P_j(full)` over all Actor/seed pairs.
- `collapsed Actor/seed pair`: counted when final `P(full) < 0.5` or the
  partial Observer has higher final probability than the full Observer.
- `expected reward`: marginal expected reward under current Actor policies and
  selection probabilities.
- `R2(m, x1+x2)`: fit between the full Observer's message and the known target
  sum.

## Results

The five-seed selection-rate sweep shows that Section 7.2 is stable at
`alpha_w=0.005`, but Section 7.3 is not fully solved:

| Stage | Variant | Final expected | Min `P(full)` | Collapsed pairs |
| --- | --- | ---: | ---: | ---: |
| 7.2 | `alpha_w=0.005` | 0.953 | 0.907 | 0 / 15 |
| 7.2 | warm-start compression | 0.958 | 0.932 | 0 / 15 |
| 7.3 | `alpha_w=0.005` | 0.888 | 0.417 | 1 / 15 |
| 7.3 | per-Actor advantage | 0.912 | 0.366 | 1 / 15 |
| 7.3 | entropy `beta=0.01` | 0.886 | 0.421 | 1 / 15 |
| 7.3 | horizon 100k, `alpha_w=0.001` | 0.890 | 0.372 | 2 / 9 |
| 7.3 | warm-start compression | 0.796 | 0.322 | 5 / 15 |

Interpretation:

- Slower selection updates are not enough. Very small `alpha_w` leaves the
  Actor-specific selection distributions undertrained at 20000 steps.
- Advantage improves average reward in 7.3, but does not remove collapse.
- The tested entropy values do not materially change the outcome.
- Longer training improves reward, but still leaves collapse in the checked
  seeds.
- Warm-start compression helps 7.2 but does not transfer cleanly to the fully
  joint 7.3 setting.

## Collapse-Mechanism Analysis

The follow-up collapse-analysis run logs three internal quantities:

- `selection_probability`: the learned probability `lambda[j, i]` that Actor
  `j` selects Observer `i`.
- `selection_exposure`: the empirical fraction of Actor `j`'s active steps on
  which it selected Observer `i`.
- `observer_gradient_norm_share`: the share of Actor `j`'s cumulative
  compression-gradient signal that went to Observer `i`.

The effective compression-update exposure from Actor `j` to Observer `i` is
proportional to

```text
p_j * lambda[j, i]
```

because Actor `j` must first be active and then select Observer `i`. This is
the core reason that Section 7 is harder than Section 5. In Section 5 there is
only one selector, so the full Observer usually receives enough exposure. In
Section 7 each Actor maintains its own selector, and one Actor can starve the
full Observer even when other Actors learn correctly.

Five-seed collapse-analysis summary:

| Experiment | Final expected | Min `P(full)` | Min full exposure | Collapsed pairs |
| --- | ---: | ---: | ---: | ---: |
| 7.2 fixed Actor decoder | 0.953 | 0.907 | 0.647 | 0 / 15 |
| 7.3 raw reward | 0.888 | 0.417 | 0.401 | 1 / 15 |
| 7.3 per-Actor advantage | 0.912 | 0.366 | 0.377 | 1 / 15 |

The failed raw-reward 7.3 pair is seed 2, Actor 0. That Actor ends with
`P(full)=0.417`, full-Observer exposure `0.401`, and an almost zero final
quality gap between the full and partial Observers. The full Observer is best
on average, but for this Actor it does not become clearly better before the
selection distribution has already moved toward the partial Observer.

## Code Paths

- Trainer: `src/actor_observer_rl/training/group_imperfect_state.py`
- Stabilization runner: `experiments/run_paper_7_stabilization.py`
- Collapse-analysis runner: `experiments/run_paper_7_collapse_analysis.py`
- Tests: `tests/test_paper7_group_imperfect_state.py`
- Report figures:
  - `docs/figures/paper7_stabilization_selection_rate.pdf`
  - `docs/figures/paper7_stabilization_per_actor.pdf`
  - `docs/figures/paper7_collapse_matrices.pdf`
  - `docs/figures/paper7_collapse_failed_timeline.pdf`
- Output summaries:
  - `outputs/paper_7_stabilization_alpha_w5/stabilization_summary.csv`
  - `outputs/paper_7_stabilization_advantage5/stabilization_summary.csv`
  - `outputs/paper_7_stabilization_horizon3/stabilization_summary.csv`
  - `outputs/paper_7_stabilization_entropy_warm5/stabilization_summary.csv`
  - `outputs/paper_7_collapse_analysis/collapse_summary.csv`
