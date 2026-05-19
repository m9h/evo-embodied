# Math Primitives Soft-Body Robotics Surfaces for jaxctrl

A research note on which mathematical primitives a soft-body / voxel-evolution layer in `evo-embodied` would need, framed by the criterion that *new applications are valuable only when they surface math that benefits every other application of the library*.

The target library is [`jaxctrl`](https://github.com/m9h/jaxctrl) — a JAX-native differentiable control-theory stack with roots in network control theory from neuroscience. The existing four layers (L0 system ID → L1 LTI control → L2 tensor control → L3 hypergraph control) are domain-agnostic by design. Adding soft-body-specific code paths is the wrong move; identifying primitives that soft-body, GRNs, neural networks, and active-inference generative models all need but the library doesn't yet have is the right one.

The ranking below is the result of two passes: an initial robotics-pulled ranking, then a reranking against the gaps in an active-inference / free-energy library (`alf`) — which sharpened priorities #1 and #2 by adding a second, independent project as evidence of domain-generality.

## The four-regime framing

`evo-embodied`'s planned task surface spans four genuinely different dynamics regimes, and each one stresses different mathematics:

| Regime | Hybridity | Dimensionality | Math demand |
|---|---|---|---|
| Walking on rigid ground | High (impacts, contact mode switching) | Low-moderate | Saltation matrices + hybrid Floquet |
| Gravity-only / aerial phase | None (smooth) | Low-moderate | Standard Floquet |
| Swimming / fluid coupling | None (smooth, continuous reaction) | High (voxels + fluid) | Randomized / streaming DMD |
| Granular media | Partial (persistent + breaking contacts) | High | **Both** saltation **and** randomized DMD |

The breadth of this surface is the strongest argument for the principle: any primitive that lands in jaxctrl as a consequence of this work gets exercised across genuinely different conditions, not just relabeled for legged-robot papers.

## A note on the two soft-body cultures

Before the ranking, one structural observation that affects how heavily to weight saltation/hybrid Floquet: the soft-body community has split into two cultures with incompatible attitudes toward contact.

**Hybrid / rigorous culture** (legged robotics, Hybrid Zero Dynamics, biomechanics): treats contact as a real switching event. Uses saltation matrices, monodromy matrices, Filippov dynamics. Mathematically clean, integration-step-respecting.

**Smooth / gradient culture** (DiffTaichi, SoftZoo, DiffPD, [RoboDiff](https://robodiff.github.io/) — Matthews, Spielberg, Rus, Kriegman, Bongard, *PNAS* 2023): smooths contact with penalty forces or viscoelastic relaxation specifically so the whole simulation stays differentiable. Used by everyone doing gradient-based morphology design, including the direct xenobot lineage. RoboDiff is the differentiable successor to voxcraft from the same authors, and it explicitly avoids hybrid contact in favour of smooth penalties + "void interpolation" for differentiable morphology.

This is not a math gap — it's a *culture* gap. evo-embodied's task surface spans both: walking on rigid ground belongs in the hybrid culture, swimming and MPM-based locomotion belong in the smooth culture. The math primitives below are correctly chosen for the hybrid side; the smooth side mostly needs a JAX-native MPM (which is what JAX-MPM is becoming) and is *not* solved by saltation.

**Implication for ranking:** the saltation/hybrid-Floquet primitives are essential for *one half* of the soft-body application surface, not all of it. Randomized DMD and the Riccati-ODE propagator are essential for *both halves* plus ALF — which is why they rerank higher under the multi-project criterion.

## Six candidate primitives, ranked

### #1 — Saltation matrix as a JAX primitive (HIGHEST priority)

The standard tool for linearizing through a hybrid jump (Kong, Payne, Zhu, Johnson, *T-RO* 2024 — [arXiv:2306.06862](https://arxiv.org/abs/2306.06862)). Every legged-robotics paper re-implements it. **No JAX-native implementation exists.**

Signature: `saltation(f_pre, f_post, guard, x_event, t_event) -> S_matrix`. Must be `vmap`-able and differentiable.

Generality beyond robotics (already cited in the saltation literature):
- Integrate-and-fire neural networks (the reset is a hybrid jump)
- Transcriptional bursting in GRNs (discrete state jumps in mRNA count)
- Power circuits, switched control systems
- Computational neuroscience (spike events)

### #2 — Hybrid Floquet monodromy (HIGH priority)

`monodromy_hybrid(flow_segments, saltations) -> Phi_T`. Combines smooth Floquet between events with saltation across them. Eigenvalues = hybrid Floquet multipliers, which is the stability test for limit-cycle behaviour with discrete jumps.

Generality: same as saltation, *plus* all the smooth-limit-cycle cases (GRN repressilator, circadian oscillators, brain rhythms). One primitive serves both audiences — robotics gets the hybrid case, biology/neuro gets the smooth case as the special-case-with-zero-saltations.

### #3 — Randomized / streaming DMD in JAX (MEDIUM priority, low risk)

Math is settled (Erichson-Mathelin-Kutz 2017 for randomized; Hemati et al. for streaming). GPU CUDA implementations exist. The Python ecosystem has [PyDMD (1.2k★, last push Dec 2025)](https://github.com/PyDMD/PyDMD) and [PyKoopman (428★, last push Jan 2026)](https://github.com/dynamicslab/pykoopman), both healthy, both scipy-based. **Neither is JAX-native.** jaxctrl's `KoopmanEstimator` is exact DMD only — caps at a few hundred dimensions, breaks at voxel/neural-population scale.

Generality: every domain in jaxctrl's README scales out — single-cell GRN data, neural population recordings, voxel sims, CFD. This is pure porting work, not new math, but it unblocks every high-dimensional use case the library already targets.

### #4 — Switched / mode-conditioned Koopman estimator (MEDIUM priority)

`SwitchedKoopman(mode_indicator)` fits per-mode EDMD operators and exposes a single switched system to L1. Older and well-understood mathematically.

Generality: legged robots (per-gait-phase Koopman) *and* GRNs with regime changes (drug applied/withdrawn) *and* neural networks with state-dependent gating.

### #5 — Global Koopman lifting absorbing contact (SPECULATIVE)

The Nov 2025 paper ([arXiv:2511.06515](https://arxiv.org/abs/2511.06515)) claims a single global Koopman operator can subsume contact mode switching under a viscoelastic contact assumption. No code yet. The underlying question — *when can a single global Koopman absorb apparent non-smoothness?* — is genuinely deep and domain-general, but the affirmative answer is currently robot-specific. Worth following; not worth implementing speculatively.

### #6 — Time-varying Riccati propagator (HIGH priority, surfaced by ALF cross-reference)

jaxctrl L1 currently has `solve_continuous_are` for *steady-state* Riccati. The missing piece is `riccati_ode_propagate(A_fn, B_fn, Q_fn, R_fn, P0, ts) -> P_traj` — the Riccati equation as a time-varying ODE, integrated forward in time.

This is the same primitive that:
- Propagates state covariance through a saltation event (covariance propagation in #1)
- Implements continuous Kalman-Bucy filtering for LTV systems (ALF's continuous-state-space gap)
- Solves finite-horizon LQR for non-LTI plants (every limit-cycle controller in any domain)
- Is the algebraic core of DEM under Gaussian noise

Diffrax handles the ODE; the math is the standard Riccati ODE with autodiff rules through the Lyapunov solver chain. Smaller in scope than the saltation work; arguably higher in dual-use density.

## Cross-project overlap with `alf` (active inference / free-energy)

The `alf` project has its own list of missing maths. Cross-checking the two lists is the cleanest test of the user's "new math that all domains benefit" criterion: any primitive that lands in jaxctrl as a consequence of this work should also fill an `alf` gap, or fail the test.

| `alf` gap | Overlap with the primitives above | Notes |
|---|---|---|
| Continuous-state-space inference (DEM, Kalman-Bucy, SDE generatives) | **Strong** — partially filled by existing L1 (`solve_continuous_are` IS the steady-state Kalman-Bucy gain solver). Closed by **#6** (Riccati-ODE propagator). | One implementation, two consumers. |
| Multi-scale / RG / coarse-graining operator | **Strongest** — slow Koopman modes ARE the coarse-grained variables; randomized DMD on multi-scale systems extracts the slow manifold directly. Closed by **#3**. | Mori-Zwanzig formalism is mathematically dual to Koopman analysis. This is the single cleanest overlap. |
| Factor graphs / message passing (RxInfer) | **Structural** — L3 hypergraph primitives (`adjacency_tensor`, `laplacian_tensor`, `tensor_kalman_rank`) are algebra over graph topology and could be the substrate underneath a sum-product / EP engine. Not closed by any primitive here; latent overlap. | Worth flagging as a future direction. |
| Bayesian Model Reduction / BMC | None — Bayesian info theory, no dynamics primitive shared. | Out of scope. |
| Object-centric / slot-based (AXIOM) | None — representation-learning question, orthogonal. | Out of scope. |
| Amortized inference / normalising flows / diffusion | None — generative-modeling stack. Diffusion models are SDEs, but the score-matching training stack doesn't share primitives with the linear/multilinear control stack. | Out of scope. |
| Saltation / hybrid Floquet (#1, #2) | None *today* — `alf` is discrete-POMDP-based, no continuous hybrid system to linearize. | **Future overlap if `alf` extends to continuous hybrid generative models** (movement onset, decision triggers, sensory transients are all hybrid events). |

## Reranking under the multi-project criterion

The original ranking was robotics-pulled. The `alf` cross-reference reweights:

| Primitive | evo-embodied | `alf` | Combined priority |
|---|---|---|---|
| **#3 Randomized/streaming DMD** | Medium (scales L0 for voxel sims) | **High** (coarse-graining + scale-out) | **#1 by domain-generality** |
| **#6 Time-varying Riccati propagator** | Medium (covariance through impacts, LTV LQR) | **High** (continuous Kalman-Bucy / DEM) | **#2 by domain-generality** |
| **#1 Saltation matrix** | High (hybrid contact) | None today | #3 — robotics-pulled but defensibly general |
| **#2 Hybrid Floquet monodromy** | High (gait stability) | None today | #4 — companion to saltation |
| #4 Switched Koopman | Medium | Low | Hold |
| #5 Global Koopman absorbing contact | Speculative | Speculative | Hold |

## Recommendation

**Tier 1 (build now):**
- **#3 randomized/streaming DMD** — pure porting work of settled math (Erichson-Mathelin-Kutz 2017, Hemati streaming), no JAX-native version exists, unblocks every high-DOF use case across robotics, neural populations, single-cell trajectories, and coarse-graining for `alf`.
- **#6 time-varying Riccati propagator** — small in scope, large in dual-use density, builds on existing Diffrax integration in L1.

**Tier 2 (build next):**
- **#1 saltation matrix + #2 hybrid Floquet together** — inseparable mathematically, both universally missing, justify themselves on the hybrid-locomotion side alone and have clean future extensions to spiking neural nets and transcriptional bursting.

**Hold:** #4 switched Koopman (too narrow), #5 global-Koopman-absorbing-contact (too speculative).

## Key references

- [Saltation Matrices: The Essential Tool for Linearizing Hybrid Dynamical Systems — Kong et al., *T-RO* 2024](https://arxiv.org/abs/2306.06862)
- [Robust Bipedal Locomotion: Leveraging Saltation Matrices for Gait Optimization](https://arxiv.org/pdf/2209.10452)
- [Koopman Global Linearization of Contact Dynamics — Nov 2025](https://arxiv.org/abs/2511.06515)
- [Koopman Operators in Robot Learning (survey)](https://arxiv.org/abs/2408.04200)
- [Modeling Quadruped Leg Dynamics on Deformable Terrains via Koopman](https://www.sciencedirect.com/science/article/pii/S2405896322028622)
- [Koopman Operators for Modeling and Control of Soft Robotics](https://arxiv.org/abs/2301.09708)
- [MPC of Fluid-Structure Interaction via Koopman ROM — *J. Fluid Mech.*](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/abs/model-predictive-control-of-fluidstructure-interaction-via-koopmanbased-reducedorder-model/2829DE4F0B9D0D6539DD1C2461B6112B)
- [Randomized DMD — *SIAM J. Appl. Dyn. Syst.*](https://epubs.siam.org/doi/10.1137/18M1215013)
- [Beneath Our Feet: Strategies for Locomotion in Granular Media — *Annu. Rev. Fluid Mech.*](https://crablab.gatech.edu/pages/publications/pdf/annurev-fluid-010313-141324.pdf)
- [PyDMD](https://github.com/PyDMD/PyDMD) · [PyKoopman](https://github.com/dynamicslab/pykoopman)
- [Efficient automatic design of robots — Matthews, Spielberg, Rus, Kriegman, Bongard, *PNAS* 2023 (RoboDiff)](https://www.pnas.org/doi/abs/10.1073/pnas.2305180120) · [project page](https://robodiff.github.io/) · [code](https://github.com/robodiff/robodiff)
- [DiffPD: Differentiable Projective Dynamics — Du, Wu, Spielberg, Matusik, Rus, Matusik, *ACM TOG* 2021](https://dl.acm.org/doi/abs/10.1145/3490168)
