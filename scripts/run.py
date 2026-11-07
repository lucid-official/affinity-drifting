"""Run one (experiment, arm, seed).  Refuses on a Δ violation or a missing pre-registration."""

from _cli import experiment, parser

from affdrift.pipeline import assert_preregistered, run_one

if __name__ == "__main__":
    p = parser(__doc__)
    p.add_argument("--arm", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--smoke", action="store_true", help="capped steps, written to runs_smoke/")
    p.add_argument("--force", action="store_true", help="redo a complete run")
    p.add_argument("--no-wandb", action="store_true")
    args = p.parse_args()
    exp = experiment(args)
    assert_preregistered(exp)
    o = run_one(exp, args.arm, args.seed, smoke=args.smoke, force=args.force, use_wandb=not args.no_wandb)
    print(o.name, o.state, f"{o.seconds:.1f}s")
