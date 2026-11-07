r"""Level 1 generator :math:`f(\epsilon, g(c))` with the two capacity knobs of plan §5.1.

* ``delta`` — :math:`c` is snapped to the centre of its grid cell of size
  :math:`\delta` before :math:`g`, so conditions in one cell are
  indistinguishable to the generator (E1.4).
* ``lipschitz`` — :math:`g = L \cdot g_\mathrm{sn}` with every linear layer
  of :math:`g_\mathrm{sn}` spectrally normalised and 1-Lipschitz
  activations, so :math:`g` is at most :math:`L`-Lipschitz in :math:`c`
  (E1.4b).
"""

from dataclasses import dataclass
from typing import override

import lucid
import lucid.nn as nn
from lucid import Tensor


@dataclass(frozen=True, slots=True, kw_only=True)
class GeneratorSpec:
    eps_dim: int = 2
    cond_dim: int = 2
    out_dim: int = 2
    hidden: int = 128
    depth: int = 3
    cond_hidden: int = 64
    delta: float = 0.0
    lipschitz: float | None = None


def quantize(c: Tensor, delta: float) -> Tensor:
    """Centre of the δ-cell containing ``c``; identity for ``delta = 0``."""
    if delta == 0:
        return c
    return (lucid.floor(c / delta) + 0.5) * delta


class ConditionPath(nn.Module):
    def __init__(self, spec: GeneratorSpec) -> None:
        super().__init__()
        layers = [nn.Linear(spec.cond_dim, spec.cond_hidden), nn.Linear(spec.cond_hidden, spec.cond_hidden)]
        if spec.lipschitz is not None:
            layers = [nn.utils.spectral_norm(layer) for layer in layers]
        self.l1, self.l2 = layers
        self.act = nn.ReLU()
        self.spec = spec

    @override
    def forward(self, c: Tensor) -> Tensor:
        h = self.l2(self.act(self.l1(quantize(c, self.spec.delta))))
        return h if self.spec.lipschitz is None else h * self.spec.lipschitz


class MLPGenerator(nn.Module):
    def __init__(self, spec: GeneratorSpec) -> None:
        super().__init__()
        self.spec = spec
        self.cond = ConditionPath(spec)
        dims = [spec.eps_dim + spec.cond_hidden, *[spec.hidden] * spec.depth]
        self.body = nn.Sequential(*[m for a, b in zip(dims, dims[1:]) for m in (nn.Linear(a, b), nn.SiLU())])
        self.head = nn.Linear(spec.hidden, spec.out_dim)

    @override
    def forward(self, eps: Tensor, c: Tensor) -> Tensor:
        """``eps`` ``(..., eps_dim)`` and ``c`` ``(..., cond_dim)`` broadcast together."""
        g = self.cond(c)
        shape = tuple(max(a, b) for a, b in zip(eps.shape[:-1], g.shape[:-1], strict=True))
        h = lucid.cat([eps.expand(*shape, eps.shape[-1]), g.expand(*shape, g.shape[-1])], dim=-1)
        return self.head(self.body(h))
