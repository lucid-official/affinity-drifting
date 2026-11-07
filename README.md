# Affinity Drifting (AffD) — experiments

Bottom-up, single-factor attribution experiments for Affinity Drifting: drifting whose attraction is a
caption-affinity mean-shift over the positives of similar conditions, so that one positive per condition still
teaches a conditional *distribution*. Reference: Deng et al., *Generative Modeling via Drifting*, arXiv 2602.04770.

| | |
| --- | --- |
| Plan (source of truth) | [docs/plan.md](docs/plan.md) |
| Rules | [CLAUDE.md](CLAUDE.md) |
| Lab notebook | [EXPERIMENTS.md](EXPERIMENTS.md) |
| Deviations from the plan | [docs/deviations.md](docs/deviations.md) |
| Code changes | [CHANGELOG.md](CHANGELOG.md) |
| Levels | [L0](experiments/L0/README.md) |

## Setup

```bash
scripts/setup_env.sh            # .venv with lucid-dl==3.16.4, no bytecode caches
.venv/bin/python -m pytest
```

Apple Silicon (Metal) via [Lucid](https://github.com/lucid-official/lucid). wandb logs offline into `wandb/`
(`AFFD_WANDB=offline|online|disabled`); nothing is uploaded unless you run `wandb sync`.

## Workflow for one experiment

```bash
# 1. write experiments/<Level>/<ID>/{experiment.yaml,prereg.md}, commit them ("<ID>: pre-register")
.venv/bin/python scripts/sweep.py E0.3 --smoke     # seed 0, capped steps, runs_smoke/
.venv/bin/python scripts/sweep.py E0.3             # every arm × seed (resumable)
.venv/bin/python scripts/aggregate.py E0.3         # pre-registered verdict → aggregate.json, report.md
.venv/bin/python scripts/make_figures.py E0.3      # figures/L0/E0.3/<FIG>/ with blind twins and reading sheets
```

`sweep.py` refuses to start if the pre-registration is uncommitted or modified, if an arm changes more than its
experiment declares (Δ check), or if the original-Drifting baseline arm `drifting` is missing.

## Layout

```
affdrift/            library: core (kernels, bandwidth, drift), data, models, eval, tracking, levels
configs/base_B0.yaml B0 and its per-Level sections
experiments/<Level>/<ID>/
    experiment.yaml  prereg.md  sigma.json  aggregate.json  report.md
    runs/<arm>/seed<k>/   config.yaml config_hash.txt delta.json env.json status.json log.jsonl metrics.json
figures/<Level>/<ID>/      every figure, and only here
    <FIG>/                <FIG>.png <FIG>_blind.png <FIG>_key.json <FIG>.md
    runs/<arm>/seed<k>/<FIG>/
probe/  scripts/  tests/  docs/
```
