"""Shared argument parsing for the scripts."""

import argparse

from affdrift.config import Experiment


def parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("exp", help="experiment ID, e.g. E0.3")
    return p


def experiment(args: argparse.Namespace) -> Experiment:
    return Experiment.load(args.exp)
