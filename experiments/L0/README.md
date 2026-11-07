# Level 0 — non-parametric particle simulation

Plan §4. No network: each condition owns M particles moved by x ← x + ηV(x|c), i.e. a generator of unlimited
capacity. L0 checks the field's fixed points before any training (Lemma 1, Prop A, C, E) and becomes the
"no capacity limit" reference for later Levels.

| ID | Claim | Question | Runner |
| --- | --- | --- | --- |
| [E0.1](E0.1/) | Lemma 1 | V = σ²∇log(p̂/q̂) for a Gaussian kernel | `lemma1` |
| [E0.2](E0.2/) | Prop A | symmetric h_q = h_p memorises | `particles` |
| [E0.3](E0.3/) | Prop C | h_q = 0 gives the NW mixture w̃ | `particles` |
| [E0.4](E0.4/) | Prop A–C | h_q interpolates monotonically between the two | `particles` |
| [E0.5](E0.5/) | Prop E | counterfactual negatives tilt to a closed-form Gaussian | `particles` |
| [E0.6](E0.6/) | 단서 1 | joint vs separate softmax: same fixed point | `particles` |

Every experiment also runs the original Drifting (Lucid release) as arm `drifting` (D11).

Level-wide settings (B0 `levels.L0`, docs/deviations.md): raw field, no drift normalisation (D4); σ = 0.5 ×
median pairwise distance of the positive bank, frozen per experiment (D5); self-included repulsion (D6);
G2 logged, not enforced (D7). Gates: G2 is the E0.1 identity plus the release-equality test; G0/G1 do not apply
(φ = identity, ψ = oracle); G3/G4 start at L1.

Go/no-go after L0 (plan §9): "L0 고정점이 예측과 일치하나?" — decided by the user from [summary.md](summary.md).
