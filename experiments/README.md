# Experiments

Thin runners. They load a config or CLI flags, call a trainer, and write
metrics / figures to `outputs/` (gitignored) and often `docs/figures/`.

**Start here** (under a minute each):

```bash
uv run python experiments/run_case1.py
uv run python experiments/run_case2.py
uv run python experiments/run_case3.py
uv run python experiments/run_case4.py
```

YAML defaults live in [`../configs/`](../configs/). Most paper-section
scripts accept `--help` and override those defaults.

Paper overview and setup: root [`README.md`](../README.md).

---

## Foundation (Cases 1–4)

| Script | What it does |
|---|---|
| `run_case1.py` | 1 static state; learn `π_θ` only |
| `run_case2.py` | i.i.d. states; learn state-dependent `π_θ` |
| `run_case3.py` | Actor sees zeros; Observer message `φ` |
| `run_case4.py` | Multi-observer discrete selection |
| `run_case3_experiments.py` | Case 3 sweep + learning curves |
| `run_case4_experiments.py` | Case 4 sweep + learning curves |
| `run_foundation_experiments.py` | Cases 1–2 verification sweep |
| `run_report_refinement_experiments.py` | Mechanism plots for the technical report (action probs, sum compression, fixed-Observer selection) |
| `benchmark_case1.py` | Case 1 LR / seed benchmark |
| `benchmark_case2.py` | Case 2 LR / seed benchmark |

Configs: `configs/case1_single_state.yaml` … `case4_multi_observer.yaml`.

---

## Paper §4 — imperfect reward, one Observer, many Actors

| Script | What it does |
|---|---|
| `run_paper_4_1.py` | Fixed Actor policies; learn budgeted `μ` |
| `run_paper_4_2.py` | Joint: learn `μ` and Actor policies `θ` |
| `run_paper_4_figures.py` | Controlled `μ` / `Û` figures (known-answer toys) |

Configs: `configs/paper_4_1_multi_actor.yaml`, `paper_4_2_multi_actor.yaml`.

---

## Paper §5 — imperfect state, one Actor, many Observers

| Script | What it does |
|---|---|
| `run_paper_5_1.py` | Fixed messages + decoder; learn selection `w` only |
| `run_paper_5_1_advantage_ablation.py` | Raw-reward vs advantage on `w` |
| `run_paper_5_2.py` | Learn selection `w` and compression `φ` |
| `run_paper_5_3.py` | Joint `w` + `φ` + `θ` |
| `run_updated_section_5_1_scalar_sum_scaling.py` | §5.1 scalar-sum `φ` scaling |
| `run_updated_section_5_2_scalar_sum_selection_scaling.py` | §5.2 `φ+w` scaling |
| `run_updated_section_5_3_scalar_sum_group_scaling.py` | §5.3 many-to-many |
| `run_joint_phi_theta_1x1_diag.py` | 1×1 `policy_only` / `compression_only` / `joint` |
| `run_fused_vs_cut_backprop.py` | Fused NN vs cut-module REINFORCE control |

Configs: `configs/paper_5_1_observer_selection.yaml`,
`paper_5_2_selection_compression.yaml`,
`paper_5_3_joint_imperfect_state.yaml`.

---

## Paper §6 — group imperfect reward

| Script | What it does |
|---|---|
| `run_paper_6_1.py` | Many Observers, each with private `R_i`; fixed Actors; learn `μ_i` |
| `run_paper_6_2.py` | Joint: learn `μ_i` and Actor policies |
| `run_paper_6_figures.py` | Assortative / joint `μ` figures |

Configs: `configs/paper_6_1_group.yaml`, `paper_6_2_group.yaml`.

---

## Paper §7 — group imperfect state

| Script | What it does |
|---|---|
| `run_paper_7.py` | Per-Actor selection `λ_j`, shared compressors |
| `run_paper_7_stabilization.py` | §7.2 / §7.3 rate / advantage / entropy / horizon checks |
| `run_paper_7_collapse_analysis.py` | Partial-Observer collapse diagnostics |

Configs: `configs/paper_7_1_group_observer_selection.yaml`,
`paper_7_2_group_selection_compression.yaml`,
`paper_7_3_group_joint_imperfect_state.yaml`.

---

## Combined / §8-style ladder

| Script | What it does |
|---|---|
| `run_combine_ladder.py` | Small fused loop: freeze/learn `φ`, `w`, `θ`, `μ` one knob at a time |

---

## Imperfect-reward scaling (grid + scalar)

| Script | What it does |
|---|---|
| `run_scaling_states.py` | Phase 1: grow grid states, 1 Actor / 1 Observer |
| `run_scaling_actors.py` | Phase 2: grow Actors, tiny 1-state task |
| `run_scaling_states_actors.py` | Phase 3 / crowd scaling |
| `run_previous_cases_scaling.py` | Pre-§8 controlled scaling of already-tested mechanisms |
| `run_structured_scaling_check.py` | Grid vs unstructured best-action sanity |
| `run_scalar_policy_probe.py` | Large-K scalar ID + `EmbeddingPolicy` (contrast, not headline) |
| `run_scalar_multi_scale.py` | Scale Actor/Observer count at fixed K (scalar + embed) |
| `run_stress_test.py` | One-rung-at-a-time LR + trajectory sweep (§4 / §6) |
| `run_learning_rate_check.py` | Trajectory vs LR (too high oscillates, too low stuck) |
| `run_reward_budget_sanity.py` | Raise budget `B` → report rate → perfect-reward limit |
| `run_actor_selection_check.py` | Residual gap from samples-per-Actor, not the reward gate |

---

## Outputs

| Location | What |
|---|---|
| `outputs/` | CSVs, JSON, per-run logs (gitignored) |
| `docs/figures/` | SVG / PDF / PNG curves (SVG usually committed; PDF/PNG gitignored) |
| `docs/updated_section_5_*.md` | Some §5 scalers also write a result note |

Set `MPLCONFIGDIR=/tmp/mplconfig` if matplotlib complains about a read-only
config directory.
