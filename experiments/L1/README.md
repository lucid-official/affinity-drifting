# Level 1 — low-dimensional neural toy

Plan §5. Conditions c ∈ [0, 1]², one positive per condition, true p(x|c) a 3-mode GMM; an MLP generator whose
c-resolution is set by δ (quantisation) or L (Lipschitz). Decides the paper's spine C1–C3.

| ID | Claim | Status |
| --- | --- | --- |
| [G4](G4/) / [G4b](G4b/) / [G4c](G4c/) | Gate G4 (lr, unconditional fit) | FAIL / FAIL / **PASS** (lr 1e-3) |
| [E1.1](E1.1/) | C0 (= G2) | **PASS** |
| [E1.2](E1.2/) | C1 (+ Gate G3) | **PASS** |
| [E1.3](E1.3/) | C2 | **PASS** |
| [E1.4](E1.4/) | C3 (quantisation δ) | **FAIL** |
| [E1.4b](E1.4b/) | C3 (Lipschitz L of g) | **INCONCLUSIVE** (knob does not bind) |
| [E1.4c](E1.4c/) | C3 re-test, revised prediction (Fourier scale s; D25) | pre-registered 2026-11-09 |
| [E1.5](E1.5/) | C4 | **FAIL** |
| [E1.6](E1.6/) | h selection | **PASS** |
| E1.7–E1.9 | C5, C7, ψ quality | not yet pre-registered |

Level-wide settings: B0 `levels.L1` (Adam lr 1e-3 from G4c, EMA 0.999, 20k steps, 8 conditions × 64 samples (D22),
three Gaussian scales (D20), whole-field drift normalisation (D24)). Level summary: [summary.md](summary.md).
