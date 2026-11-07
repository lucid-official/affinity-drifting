# Changelog

Code, config, directory-convention and rule changes on branch `affinity-drifting` (CLAUDE.md §8.5).
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Experiment results live in `EXPERIMENTS.md`, not here.

## [Unreleased]

### Added

- `affdrift` package (Python 3.14, `lucid-dl==3.16.4`), installed editable by `scripts/setup_env.sh`.
- `core/drift.py`: `DriftLoss`, the Lucid release drifting loss split into overridable steps and bit-identical
  to `lucid.models.generative.drifting._model._drift_loss` (loss and gradient); `RawDriftLoss` (release field
  without per-τ normalisation, the L0 baseline, D4); `AffinityDrift(DriftLoss)` overriding `distances`, `scale`
  and `goal` for AffD's caption-weighted Gaussian mean-shift field. Single positive and K1–K5 are options of it.
- `core/kernels.py`, `core/bandwidth.py`: k_x (Gaussian / Laplace / none), caption weights with exact h → 0 and
  h → ∞ limits, K2 neighbour shuffling, N_eff bisection for h_p.
- `core/identities.py`: Lemma 1 autograd reference, Prop E closed form.
- `data/particles.py` (L0 single bank, Gaussian pair, particle integrator with `lucid.compile` and eager
  fallback), `data/toy2d.py` and `models/generator.py` (L1, not run yet), `data/shapes_unique.py` (TODO).
- `config.py`: B0 ⊕ Level ⊕ common ⊕ arm resolution; Δ check against B0 and against the reference arm;
  mandatory baseline arm `drifting` (D11).
- `tracking/`: run directories, JSONL + wandb (offline by default) logging, `EXPERIMENTS.md` ledger.
- `eval/`: metrics, paired bootstrap CIs, pre-registered judges, figure catalog with the plan's reading criteria,
  blind figures with keys, observation sheets that survive regeneration.
- `levels/level0.py`: runners `lemma1` (E0.1) and `particles` (E0.2–E0.6).
- `scripts/`: `run.py`, `sweep.py`, `aggregate.py`, `make_figures.py`, `make_probe.py`, `setup_env.sh`.
- `configs/base_B0.yaml`, `probe/probe_L0.json`, `probe/probe_L1.json`.
- `CLAUDE.md` §7 (code rules) and §8 (records, logging and figure rules), rule §2.11 (baseline arm).
- `docs/deviations.md` D1–D12.

- `data/particles.py`: `gaussian_closure_field` for the E0.5b diagnostic (D13); B0 key `level0.density`
  (`kde` default, so no pre-registered experiment changes).
- `tracking/ledger.py`: file lock around `EXPERIMENTS.md` writes so concurrent sweeps cannot drop entries.

- `tracking/logger.py`: wandb's own usage telemetry and error reporting are disabled outside `online` mode.

- Level 1: `levels/level1.py` (runners `train`, `lemma0`; every arm of plan §2.3 as an option of the same field,
  baseline `drifting`), `data/toy2d.Bank`, standard metrics (precision/recall, density/coverage, MMD), marginal and
  data–data W2, σ-smoothed NW TV for E1.5, `composite` judge. B0 `levels.L1` (model, train, D15 σ rule).
- `tests/test_lucid_crosscheck.py`: Lucid field, Adam, EMA, spectral norm, quantisation and generator gradients
  checked against independent NumPy/float64 references before any L1 run.

### Changed

- Figures moved out of `experiments/`: every figure now lives under the root `figures/<Level>/<ID>/` (run-level under
  `runs/<arm>/seed<k>/`). `Experiment.figure_dir` / `run_figure_dir` are the only source of these paths; run
  directories no longer get a `figs/` folder. Existing L0 figures were moved, not regenerated (user request,
  2026-11-08).

- `CLAUDE.md` §1: directory layout follows the experiment hierarchy (`experiments/<Level>/<ID>/runs/<arm>/seed<k>/`)
  instead of `runs/`, `results/<ID>.md`, `configs/exp/<ID>.yaml` (D12).
- `docs/plan.md`: the two display equations (§1 Lemma 0, §4 Prop E) moved from ```` ```latex ```` code fences to `$$`
  blocks so Markdown previews render them. Formatting only: the equation text is byte-identical.
- `docs/plan.md`: inline math wrapped in `$...$` (KaTeX) — Greek letters, subscripted symbols, norms, sets and
  inequalities; IDs (E0.1, K1, C2, …), metric names and `↔` comparisons left as text. Meaning unchanged; `1e-6` is now
  written $10^{-6}$. Pre-registration files keep their verbatim copies of the old text.
- `CLAUDE.md` §6: repository `affinity-drifting`, branch `affinity-drifting`, Lucid on Metal (D1, D12).
