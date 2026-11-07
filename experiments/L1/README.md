# Level 1 — low-dimensional neural toy

Plan §5. Conditions c ∈ [0, 1]², one positive per condition, true p(x|c) a 3-mode GMM; an MLP generator whose
c-resolution is set by δ (quantisation) or L (Lipschitz). Decides the paper's spine C1–C3.

| ID | Claim | Status |
| --- | --- | --- |
| [G4](G4/) | Gate G4 (lr, unconditional fit) | **FAIL** (2026-11-08) — see [diagnosis](G4/diagnosis.md) |
| [E1.1](E1.1/) | C0 (= G2) | pre-registered, not run |
| [E1.2](E1.2/) | C1 (+ Gate G3) | pre-registered, not run |
| [E1.3](E1.3/) | C2 | pre-registered, not run |
| [E1.4](E1.4/) / [E1.4b](E1.4b/) | C3 | pre-registered, not run |
| [E1.5](E1.5/) | C4 | pre-registered, not run |
| [E1.6](E1.6/) | h selection | pre-registered, not run |
| E1.7–E1.9 | C5, C7, ψ quality | not yet pre-registered |

Level-wide settings: B0 `levels.L1` (Adam, EMA 0.999, 20k steps, 64 conditions × 8 samples, σ by D15), D15–D19.
L1 is stopped at G4 until the user decides D20–D22.
