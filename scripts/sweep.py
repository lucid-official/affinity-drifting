"""Run every arm × seed of an experiment (resumes: complete runs are skipped)."""

from _cli import experiment, parser

from affdrift.pipeline import sweep

if __name__ == "__main__":
    p = parser(__doc__)
    p.add_argument("--arms", nargs="*")
    p.add_argument("--seeds", nargs="*", type=int)
    p.add_argument("--smoke", action="store_true", help="seed 0 only, capped steps, runs_smoke/")
    p.add_argument("--force", action="store_true")
    p.add_argument("--no-wandb", action="store_true")
    args = p.parse_args()
    outcomes = sweep(experiment(args), arms=args.arms, seeds=args.seeds, smoke=args.smoke,
                     force=args.force, use_wandb=not args.no_wandb)
    raise SystemExit(1 if any(o.state.startswith("failed") for o in outcomes) else 0)
