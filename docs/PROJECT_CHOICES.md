# Project mission, platform choices and integration priorities

Reviewed 2026-11-07 against MuJoCo 4.0 and the upstream sources linked below.
The project combines evolutionary robotics teaching with a research path toward
body–controller co-design, behavioral diversity, development and lifetime
plasticity. The central research question is whether those mechanisms improve
adaptation under matched evaluation and compute budgets.

The course and virtual-rat pathway remain part of that mission. Competitive
performance, paper reproduction and biological fidelity require the evidence in
[VALIDATION_PLAN.md](VALIDATION_PLAN.md); they are not established by installing
more libraries. [VALIDATION_STATUS.md](VALIDATION_STATUS.md) records what ran.

## Physics and execution choices

| Layer | Choice | Reason and boundary |
|---|---|---|
| Native reference | MuJoCo 4.0 | Model inspection, individual rollouts and CPU reference traces; exact XML and solver settings are part of the task |
| Current population backend | MJX-JAX 4.0 with JAX 0.12 | Batched fixed-topology models, JIT and supported gradients; existing long-horizon/contact discrepancies remain open |
| Future NVIDIA comparison | MJX-Warp / MuJoCo Warp | Test contact-heavy throughput and broader feature support on NVIDIA hardware; a separate backend experiment, not an automatic replacement |
| Apple GPU | Isolated JAX-MPS profile | Experimental; its package compatibility and float32 checks do not establish CPU/Metal research parity |
| Soft-body reference | Upstream EvoGym in a separate environment | Preserve its task definitions, material model and historical reproduction version |
| Differentiable soft-material research | Evaluate SoftZoo/RoboDiff separately | Their physics and objectives differ from EvoGym; match their native protocols before comparing methods |

`mujoco-mjx==4.0.0` identifies the physics package. `jax==0.12.0.1` identifies the
numerical framework. MJX exposes both `impl="jax"` and `impl="warp"`; a JAX-facing
API does not imply a JAX physics implementation. Our validation probes select
JAX explicitly. MJX-JAX supports autodiff with feature/contact limitations;
MJX-Warp currently does not support physics autodiff. Evolution and standard PPO
do not require differentiating through the simulator.
[MJX implementation and feature matrix](https://mujoco.readthedocs.io/en/stable/mjx.html).

Body topology changes may require recompilation because structural model fields
are static to JAX. Start continuous morphology searches with fixed topology;
for structural evolution, group compatible shapes and charge compilation,
padding and invalid-design costs. The common runner must record implementation,
device, precision, model hash and evaluation semantics for each result.

## What MuJoCo 4 changes for this project

- **Studio and Filament:** modular visual tools, hot reload and browser viewing
  can support inspecting evolved bodies and failed contacts. The viewer is an
  inspection tool, not a substitute for numerical replay evidence.
- **Deformables:** rods, shells, solids and persistent plastic deformation expand
  native-engine research options. MJX-JAX does not support flex deformables, so
  these features cannot simply be enabled in the current JAX evaluator.
- **Speculative contacts:** 4.0 adds anti-tunneling support for convex objects on
  CPU. MJX/Warp support was still forthcoming at review time; do not attribute
  this capability to our batched backend.
- **Model authoring and calibration:** `MjSpec`, model composition and
  `mujoco.sysid` are candidates for grammar-based morphology construction and
  fitting Bittle dynamics to recorded trajectories. Neither adapter is implemented
  here yet. Several capabilities matured during 3.x before the 4.0 milestone.

Sources: [4.0 announcement](https://github.com/google-deepmind/mujoco/discussions/3773),
[release changes](https://github.com/google-deepmind/mujoco/releases/tag/4.0.0),
[model editing](https://mujoco.readthedocs.io/en/stable/programming/modeledit.html).

The split `mujoco-core`/`mujoco-studio` distribution separates minimal simulation
bindings from visualization. We retain the ordinary `mujoco` installation for
the teaching environment; a headless worker profile needs its own resolution
and checks. [Python packaging](https://mujoco.readthedocs.io/en/latest/python.html#installation).

## Projects selected for the mission

| Project | Role | Integration status and next evidence |
|---|---|---|
| [evosax](https://github.com/RobertTLange/evosax) | Weight-search baselines: CMA-ES/OpenES versus hill climbing | Installed optional `strategies` extra; ask/evaluate/tell smoke passes. Next: matched B1 robot evaluator |
| [QDax](https://github.com/adaptive-intelligent-robotics/QDax) | Diverse behavior repertoires with MAP-Elites | Released 0.5.0 in optional `qd` extra; runnable planar-arm/random-search comparison and archive validation. Dynamic robot adapter remains planned |
| [TensorNEAT](https://github.com/EMI-Group/tensorneat) | Evolving network structure | Source reviewed; next: isolated dependency lock and B2/B9 adapter with matched node/parameter caps |
| [Revolve2](https://github.com/ci-group/revolve2) | Modular robot body/brain co-design, including physical robots | Source reviewed; next: isolated worker, one fixed body/controller fixture, then matched co-design protocol |
| [EvoRL](https://github.com/EMI-Group/evorl) | Evolution plus within-evaluation RL and population-based training | Source reviewed; next: independently locked workflow and shared B1 task; charge learning updates and environment transitions |
| [EvoGym](https://github.com/EvolutionGym/evogym) | Soft voxel-body/control benchmark | Adapter planned; separate historical paper and modern-reference environments |
| [RoboMorph](https://github.com/kevinxqiu/robomorph) | LLM-guided design grammar and learned controllers | Example 07 is a local adaptation. Reproduce the upstream grammar, inner learner and terrain protocol before comparison |
| `devograph` / `hgx` / `bl1` | Developmental and plastic controller hypotheses | Proposed local integrations; fixed/growth-disabled/plasticity-disabled controls required |
| `jaxctrl` | Identification, feedback and controllability-aware design | Proposed B7 integration; identify dynamics from state and input histories and validate feedback in the nonlinear simulator |

The last two rows refer to the user's repositories linked in
[RESEARCH_COMPARISONS.md](RESEARCH_COMPARISONS.md#existing-m9h-components).
Source snapshots there identify reviewed code, not completed installations.

## Runnable quality-diversity baseline

```bash
uv run --locked --extra qd python examples/12_quality_diversity.py \
  --seed 0 --seed 1 --seed 2 --output /tmp/qd-arm.json
```

The example uses QDax's analytic planar arm. Genotypes are normalized joint
angles, descriptors are normalized endpoint x/y, and fitness is the negative
standard deviation of the angles. MAP-Elites and random search share the initial
population and archive grid. Defaults charge **192 candidate evaluations per
method per seed** (32 initial + 5 × 32 new); elite replay and analytic fixtures
are reported separately. QD score uses a declared 0.5 offset and coverage is a
percentage of all grid cells, including unreachable ones.

The JSON saves configuration, versions, source/lock hashes, per-generation
metrics and occupied elites. Checks cover independent kinematic fixtures,
reevaluated elite fitness/descriptors, finite occupied cells, elitism and seeded
repeatability in the stack probe. Random search winning one of these tiny runs
is a valid observation. This integration demo does not establish MAP-Elites
superiority or dynamic robot validity.

The release is deliberately pinned to [QDax 0.5.0](https://github.com/adaptive-intelligent-robotics/QDax/releases/tag/v0.5.0).
The inspected development branch reports 0.5.1 and different dependencies; it is
not the package tested here. Advanced QDax actor-critic, Jumanji and probability
backends are not validated merely because their dependencies install.

## Reference runtimes and boundaries

**Playground:** `demos` retains the 0.2 JAX/MJX reference. Current upstream uses
PyTorch, `reaf-sim` and MuJoCo Warp, with the original JAX suite under `rss2025`.
Treat current Playground as a separately locked control/manipulation and
sim-to-real comparison. PyTorch is a legitimate comparison runtime; it is not a
core dependency of the JAX teaching environment.
[Upstream README](https://github.com/google-deepmind/mujoco_playground).

**Brax:** upstream maintains `brax/training`; the legacy environments/physics
pipelines are no longer the recommended foundation for new work. Our pinned
Ant/Humanoid demos remain recognizable reference tasks and explicitly use MJX.
New JAX task adapters should target the preserved Playground suite or direct
MJX, with Brax training used where appropriate. The PPO smoke check constructs
a policy and runs inference; it does not run a learning campaign.
[Maintenance status](https://github.com/google/brax).

**EvoJAX:** upstream is archived. Keep it as a historical neuroevolution reference;
use evosax/QDax for new search adapters and evaluate EvoRL in its own workflow.
[Archived repository](https://github.com/google/evojax).

**Priority order:** close the B0 reset/evaluator failures; use evosax/QDax through
a shared fixed-body evaluator; establish Revolve2/EvoGym co-design references;
then test development, plasticity, controllability and LLM proposals against
those controls. The new integrations extend that sequence rather than bypassing
its correctness requirements.
