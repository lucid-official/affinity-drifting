"""Draw an experiment's figure set (figures: in experiment.yaml) from its complete runs."""

from _cli import experiment, parser

from affdrift.pipeline import make_figures

if __name__ == "__main__":
    p = parser(__doc__)
    p.add_argument("--only", nargs="*")
    args = p.parse_args()
    for path in make_figures(experiment(args), only=args.only):
        print(path)
