"""Type aliases shared across the package."""

from collections.abc import Mapping
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

type FloatArray = NDArray[np.float64]
type IntArray = NDArray[np.int64]
type Device = Literal["metal", "cpu"]
type Config = Mapping[str, Any]
type Metrics = dict[str, Any]
