"""Judge an experiment with its pre-registered rule and write aggregate.json + report.md."""

from _cli import experiment, parser

from affdrift.pipeline import aggregate

if __name__ == "__main__":
    p = parser(__doc__)
    p.add_argument("--allow-partial", action="store_true")
    args = p.parse_args()
    j = aggregate(experiment(args), allow_partial=args.allow_partial)
    print(f"{args.exp}: {j.verdict}")
    for c in j.checks:
        print(f"  {c.verdict:13s} {c.name}: {c.detail}")
