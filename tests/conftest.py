"""Shared fixtures.  Refuses to run if bytecode caching is on (CLAUDE.md §7)."""

import sys

import numpy as np
import pytest

from affdrift.runtime import default_device

if not sys.dont_write_bytecode:
    raise RuntimeError("bytecode caching is on: run scripts/setup_env.sh (no __pycache__ allowed)")


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(1234)


@pytest.fixture(scope="session")
def device() -> str:
    return default_device()
