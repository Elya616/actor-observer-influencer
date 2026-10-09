# Actor–Observer RL

Snapshot of prior University of Toronto ROP work on **Actor–Observer
learning dynamics**. Use it as a starting point for a new project; it is
not a project you are expected to continue.

The paper:

> Izadimand and Marbach, *Observer-Actor Systems: A Sociologically Inspired
> Framework for Emergent Coordination in Multi-Agent Systems* (22 May 2026).
> PDF: [`docs/materials/izadimand_marbach_observer_actor_systems_2026-05-22.pdf`](docs/materials/izadimand_marbach_observer_actor_systems_2026-05-22.pdf).

---

## 1. What this is about

```text
environment produces state s
  → an Actor (or several) chooses action a ~ π_θ
  → reward / social feedback is produced
  → Observers may (a) report reward, and/or (b) compress state into a message m
  → selected parameters update by REINFORCE / projected gradient
```

The cases isolate **one learnable mechanism at a time**, use a toy setup
whose correct answer is known, and plot that internal quantity — not only
reward.

Two tracks; do not merge their “following” variables:

| Track | What is imperfect | What is learned |
|---|---|---|
| Imperfect reward / perfect state | reward / social feedback | following rates `μ`, utilities `Û`, optionally policies `θ` |
| Imperfect state / perfect reward | state information | compressors `φ`, selection logits `w`, optionally `θ` |

Foundation Cases 1–4 sit under both tracks and are the right place to start.

---

## 2. Where to read

- Peter’s paper: [`docs/materials/izadimand_marbach_observer_actor_systems_2026-05-22.pdf`](docs/materials/izadimand_marbach_observer_actor_systems_2026-05-22.pdf)
- Technical report: [`docs/materials/actor_observer_cases1_5_refined_report.pdf`](docs/materials/actor_observer_cases1_5_refined_report.pdf)
- Imperfect-reward scaling: [`docs/materials/scaling_imperfect_reward_report.pdf`](docs/materials/scaling_imperfect_reward_report.pdf)
- Section notes: [`docs/paper_section_4_report.md`](docs/paper_section_4_report.md),
  [`5`](docs/paper_section_5_report.md),
  [`6`](docs/paper_section_6_report.md),
  [`7`](docs/paper_section_7_stabilization.md)
- Math → code: [`docs/math_to_code_mapping.md`](docs/math_to_code_mapping.md)
- Runners: [`experiments/README.md`](experiments/README.md)
- Docs index: [`docs/README.md`](docs/README.md)

---

## 3. Setup

Requires **Python ≥ 3.10**. Prefer `uv`:

```bash
git clone https://github.com/xubo16/actor-observer-rl-handover.git
cd actor-observer-rl-handover
uv sync --extra dev
uv run --extra dev pytest
```

Or a local venv:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Dependencies (see `pyproject.toml`): `torch`, `numpy`, `matplotlib`, `pyyaml`;
dev extras `pytest`, `pandas`.

Package code is under `src/actor_observer_rl/`. Experiment scripts under
`experiments/` write CSVs / figures into `outputs/` (gitignored) and often
`docs/figures/`. `outputs/` is not in a fresh clone; re-run a script to
regenerate numbers.

If matplotlib complains about a read-only config directory:

```bash
export MPLCONFIGDIR=/tmp/mplconfig
```

---

## 4. Quick start

```bash
uv run python experiments/run_case1.py   # 1 state, learn π_θ only
uv run python experiments/run_case2.py   # i.i.d. states, learn π_θ
uv run python experiments/run_case3.py   # Actor sees zeros; needs Observer message
uv run python experiments/run_case4.py   # multi-observer selection
```

Longer diagnostic sweeps (write figures):

```bash
uv run python experiments/run_foundation_experiments.py
uv run python experiments/run_case4_experiments.py
uv run python experiments/run_report_refinement_experiments.py
```

Then:

```bash
uv run --extra dev pytest
```

---

## 5. How the cases are built

- Custom bandit-style environments (`reset` / `step`), not Gymnasium.
- Policy gradient is descent on `loss = −reward * log π(a | ·)`.
- Messages stay in the autograd graph so `φ` receives gradients.
- Discrete Observer selection uses a second REINFORCE term `−r log λ_k`.
- Budgeted following rates use projected gradient onto `{μ ≥ 0, ∑ μ ≤ B}`.
- On the reward track, utility steps `α_U` are faster than `α_μ` / `α_θ`.

Useful policy heads when reading the later scripts:

- `FixedSumDecoderPolicy` — paper decoder `logit_a ∝ −β(m − a)²`
- `LearnableSumDecoderPolicy` — same shape, learnable `(scale, shift, β)`
- `SoftmaxPolicy` + `init_as_sum_decoder(...)` — joint `θ` warm start
- `EmbeddingPolicy` — large-K scalar IDs on the imperfect-reward track

Paper sections in this repo:

```text
§3   dyadic foundation                         Cases 1–3
§4   imperfect reward, 1 Observer, many Actors μ, Û, then θ
§5   imperfect state, 1 Actor, many Observers  w, then φ, then θ
§6   group imperfect reward                    many μ_i, private R_i
§7   group imperfect state                     per-Actor λ_j, shared φ
```

| Paper | Scripts |
|---|---|
| Cases 1–4 | `experiments/run_case1.py` … `run_case4.py` |
| §4 | `run_paper_4_1.py`, `run_paper_4_2.py`, `run_paper_4_figures.py` |
| §5 | `run_paper_5_1.py`, `run_paper_5_2.py`, `run_paper_5_3.py` |
| §6 | `run_paper_6_1.py`, `run_paper_6_2.py`, `run_paper_6_figures.py` |
| §7 | `run_paper_7.py`, `run_paper_7_stabilization.py` |

Most scripts accept `--help`. Later scaling runners are listed in
[`experiments/README.md`](experiments/README.md).

---

## 6. Repository layout

```text
src/actor_observer_rl/
  envs/          # single-state, i.i.d., partial-obs, grid, noisy-actor
  agents/        # Actor, Observer, MultiObserver, BudgetedObserver
  policies/      # Softmax, RBF, Embedding, fixed / learnable sum decoder
  training/      # REINFORCE loops: foundation, §4–7
  evaluation/    # metrics, plotting, benchmarks
  utils/         # seed, log, YAML config
experiments/     # runners — see experiments/README.md
tests/           # pytest suite
configs/         # YAML for paper-section runners
docs/            # see docs/README.md
  materials/     # paper + compiled report PDFs
  figures/
outputs/         # local run artifacts (not cloned)
```

| File | Role |
|---|---|
| `training/trainer.py` | Cases 1–2 |
| `training/partial_observable.py` | Case 3 |
| `training/multi_observer.py` | Case 4 |
| `training/staged_multi_observer.py` | paper §5.1–5.3 staged |
| `training/informational_state.py` | scalar-sum §5 scaling |
| `training/multi_actor.py` | paper §4 |
| `training/group_multi_actor.py` | paper §6 |
| `training/group_imperfect_state.py` | paper §7 |
