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

- `core/drift.MultiScaleAffinityDrift(AffinityDrift)`: sum of per-scale normalised AffD fields (MMD-GAN style
  bandwidth mixture; the release does the same per temperature); L1 `x_kernel.sigma_rule: multiscale` (kNN width →
  median heuristic, 3 geometric scales). Not used by any registered experiment (candidate D20, awaiting approval).
- L1 runs also save F1.3's ε-slice outputs.

- L1 `Positives.neighbourhood`: the caption-weighted field is computed over the 1,024 nearest conditions when every
  dropped weight is < 1e-12 × the row maximum (zero in float32); otherwise the whole bank. Speed only; tested equal
  to the full computation.

- `eval.metrics.evaluate_conditions`: per-condition L1 metrics split over `eval.workers` = 5 processes
  (`ProcessPoolExecutor`); evaluation was single-threaded and left the GPU idle for ~1 min per run. Tested equal to
  the serial computation.

- `core/drift.DriftNorm` and B0 `normalization.drift_target` (`term` default = behaviour of every run so far;
  `total` = normalise the whole field per batch and per kernel scale, as the release does per temperature).

### Changed

- B0 `levels.L1`: `normalization.drift_target: total` (D24) and `train.lr: null` until Gate G4c. L1 runs made under
  per-term normalisation moved to `experiments/L1/<ID>/runs_superseded/D24/` (kept, not deleted).

- B0 `levels.L1.train.lr` = 1e-3, chosen by G4b's pre-registered rule; G4b itself failed narrowly and the user decided
  to proceed (D23). Affects every L1 experiment.

- B0 `levels.L1`: σ rule `multiscale` (D20), `gen.M` 64 and `train.batch_conditions` 8 (D22). Affects every L1
  experiment; E1.x pre-registrations get an addendum before any run. G4 is re-registered as G4b with full-batch
  unconditional repulsion (D21).

- L1 training materialises loss, parameters, EMA copy and Adam state every `train.eval_every` = 10 steps. Evaluating
  only the loss let the update graphs grow until Metal's resource limit (first G4 run failed at ~18k steps); every
  step works but costs 11.3 ms/step against 3.7 ms/step at 10. CLAUDE.md §7.4 updated.

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
