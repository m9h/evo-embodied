# Research demonstrations and comparison projects

Reviewed 2026-11-06 against evo-embodied commit
`fde9fc82d41b4aff541df86d023a80d1e43805e7`. This is a source-backed comparison
catalog, not a list of reproduced results. Execution protocols, budgets, and
decision rules are in [VALIDATION_PLAN.md](VALIDATION_PLAN.md); observed local
results are in [VALIDATION_STATUS.md](VALIDATION_STATUS.md).

## What each comparison can establish

| Source | Demonstration to test | Reference implementation | Role in evo-embodied |
|---|---|---|---|
| Dvoretskii et al., [Braitenberg Vehicles as Developmental Neurosimulation, v3](https://arxiv.org/abs/2003.07689v3), pp. 6-13, Fig. 3 | Body proportions, sensor-effector spacing, and 5/10-node connectomes affect movement-score diversity | [BraGenBrain / Stefan](https://github.com/Orthogonal-Research-Lab/GSoC-Braitenberg-Vehicles/tree/11188e8ab8909b6aecb9f4c7ecd653e309b2a448/Stefan) | B2: implement the original planar dynamics first; a MuJoCo vehicle is a separate embodiment extension |
| Same paper, pp. 13-17, Figs. 4-6 | Odor-taste associations and preference/avoidance through Li-Hopfield dynamics and generalized Hebbian learning | [MultiLearn / Ziyi](https://github.com/Orthogonal-Research-Lab/GSoC-Braitenberg-Vehicles/tree/11188e8ab8909b6aecb9f4c7ecd653e309b2a448/Ziyi) | B3: distinguish within-lifetime learning from evolution; test forgetting as well as acquisition |
| Same paper, pp. 17-23, Figs. 7-9 and Table 2 | Fixed-wiring vehicles produce vortices, crossings, and jamming-like density patterns around common stimuli | [BV collectives](https://github.com/ankiitgupta7/Simulations-of-Braitenberg-Vehicles/tree/3ee2a8b5aaec3cd34d538613b337e93f8f92075d) | B4: a low-cost population benchmark with no learned policy or inter-agent communication |
| Bhatia et al., [Evolution Gym, NeurIPS 2021](https://arxiv.org/abs/2201.09863), Figs. 3-4, Appendix D | Voxel morphology search with a controller trained for each body | [Paper release 1.0.0](https://github.com/EvolutionGym/evogym/tree/9a1a5e7b26702184821e6e64587220ead2ab0e21) | B5/B6: fixed-body control first, then co-design on an unchanged upstream task |
| Qiu et al., [RoboMorph](https://arxiv.org/abs/2407.08626), [project and demonstrations](https://robomorph.github.io/) | LLM-generated body designs evaluated through controller training across terrains | [Official code](https://github.com/kevinxqiu/robomorph) | B8: compare proposal mechanisms under one evaluator; existing example 07 is an adaptation, not a reproduction |
| [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground), [technical report](https://playground.mujoco.org/assets/playground_technical_report.pdf) | Original JAX/MJX environments and RL training pipelines; 1.0 uses a different runtime | Local extra pinned to 0.2.0; original upstream suite under [rss2025](https://github.com/google-deepmind/mujoco_playground/tree/rss2025); [upgrade evidence](DEPENDENCIES.md) | B1: preserve the JAX benchmark protocol and evaluate 1.0 separately |

The Braitenberg paper presents three separate implementations. Its collective
agents respond to common stimuli without communicating; adding collision forces,
alignment rules, or MARL changes the model. BraGenBrain's reported size conditions
are configured experiments, not proof that neurons grow during a lifetime.
MultiLearn reports loss of recall after repeated environmental replacement.
Traveling-salesman optimization and cumulative image classification are proposed
applications; they are not benchmark results to claim reproduced.

## External software to compare or reuse

| Project | Use | Fair-comparison boundary |
|---|---|---|
| [evosax](https://github.com/RobertTLange/evosax/tree/e493325ec25e59350cce17f546f2e994971c958d) | CMA-ES/OpenES and GA alternatives to the current parallel hill climber | Same weights/encoding, initial-state distribution, action/observation contract, and charged evaluations; a categorical voxel decoder must be specified and audited |
| [TensorNEAT](https://github.com/EMI-Group/tensorneat/tree/ee204900e8e55ffe0ed756651ea871c70230dd34), [NEAT-Python](https://github.com/CodeReclaimers/neat-python) | Topology-evolution baselines for B2 and B9 | Separate weight-only from topology-changing search; account for parameter/node limits and total evaluations |
| [EvoGym modern snapshot](https://github.com/EvolutionGym/evogym/tree/f63c649c688adda1accad5f86731aad96606388a), [examples](https://github.com/EvolutionGym/evogym/blob/f63c649c688adda1accad5f86731aad96606388a/examples/README.md) | Maintained task interface; GA, BO, CPPN-NEAT plus PPO baselines | Upstream explicitly directs exact paper reproduction to release 1.0.0; modern runs get a separate results table |
| [PyDMD / DMDc](https://pydmd.github.io/PyDMD/dmdc.html), [source snapshot](https://github.com/PyDMD/PyDMD/tree/024464ff667e4b0226f9ecbe933bc67f3fe21e81) | Independent controlled-system identification reference for B7 | Fit from state **and input** histories; autonomous exact DMD does not identify an actuator matrix B |
| [Brian2](https://github.com/brian-team/brian2) | Independent neural-dynamics/plasticity implementation when developing B3 extensions | STDP spiking networks and the paper's rate/oscillator model are different mechanisms |
| [Mesa](https://github.com/mesa/mesa) | An independently implemented CPU collective baseline | Match equations, integration order, timestep, and stimuli before measuring performance |
| [JaxMARL](https://github.com/bold-lab-ai/JaxMARL) | Multi-agent learning comparisons after B4 | Learned cooperation/communication is an extension; it cannot validate the paper's fixed-wiring collective model |
| [SoftZoo](https://github.com/zswang666/softzoo), [RoboDiff](https://github.com/robodiff/robodiff) | Later soft-material/differentiable design comparisons | Different constitutive/contact models; use their native tasks and reference configurations, not cross-engine raw-return rankings |
| [OpenWorm](https://github.com/openworm/OpenWorm) | Later biological connectome-to-body comparison | Different scale, mechanics, and biological evidence; not an initial locomotion leaderboard baseline |

Snapshots above identify inspected source, not validated dependency locks. Every
actual run must record its resolved commit, submodules, patches, package versions,
license, and environment lock. Do not upgrade a reference midway through a study.
Retain attribution when adapting code; review dependency/license compatibility
before distributing a combined implementation (TensorNEAT is GPL-3.0).

## Existing m9h components

| Repository | Candidate integration | What must be supplied or checked |
|---|---|---|
| [evo-embodied](https://github.com/m9h/evo-embodied) | Physics, feedforward controllers, hill climbing, checkpoint/video examples | Shared evaluator, reliable resets, provenance, independent seeds, held-out evaluation |
| [devograph](https://github.com/m9h/devograph), [NDP source](https://github.com/m9h/devograph/blob/main/devograph/_ndp.py), [hgx](https://github.com/m9h/hgx) | B9: local developmental rules, growth/division, dynamic connectivity | Sensor/motor interface, node cap, growth schedule, and frozen/growth-disabled ablations |
| [bl1](https://github.com/m9h/bl1) | B3/B9 extension: plastic spiking controller | Stimulus encoder, motor decoder, plasticity-off controls, real biological versus simulation claims kept distinct |
| [jaxctrl](https://github.com/m9h/jaxctrl) | B7: controllability and LQR on identified dynamics | Controlled identification or an explicit B; state/input normalization, stabilizability checks, constrained nonlinear evaluation |
| [RatInABox fork](https://github.com/m9h/RatInABox) | 2D navigation and spatial representations | Vehicle kinematics, odor/taste fields, and matching integration semantics; optional environment adapter |
| [alf](https://github.com/m9h/alf), [neuro-nav fork](https://github.com/m9h/neuro-nav) | Active-inference and RL navigation comparators | Equivalent observations/actions and an explicit information-access audit |
| [SustainHub-jaxmarl](https://github.com/m9h/SustainHub-jaxmarl) | Patterns for seed batching and population analysis | Reuse experiment infrastructure, not its unrelated sustainability reward |
| [neurosim](https://github.com/m9h/neurosim) | Presentation/overview of the stack | No executable benchmark runner was found in the inspected tree |

## Integration constraints that affect the plan

- **Separate environments:** evo-embodied requires Python >=3.12. The inspected
  EvoGym 2.0.0 [metadata](https://github.com/EvolutionGym/evogym/blob/f63c649c688adda1accad5f86731aad96606388a/pyproject.toml)
  requires Python >=3.7,<3.11 and NumPy <2. Use a pinned Python 3.10
  worker/container for modern EvoGym and a separately locked legacy worker for
  release 1.0.0. Exchange body arrays, action/observation contracts and result
  records through a file or subprocess interface; do not add incompatible core
  dependencies to this project.
- **Differentiability:** EvoGym's mass-spring/penalty-contact model does not by
  itself supply a JAX-autodiff simulator. For B7, run feedback in upstream EvoGym
  through an adapter. Differentiable SoftZoo experiments are a separate track.
- **MJX backend:** the current code targets MJX-JAX. [MJX documentation](https://mujoco.readthedocs.io/en/stable/mjx.html)
  distinguishes this from MJX-Warp, which does not support autodiff. Record the
  implementation and precision; benchmark hardware/backend combinations separately.
- **RoboMorph scope:** example 07 asks an LLM for MJCF and evolves controller
  weights. The official project uses a structured design grammar and RL-trained
  controllers. Match those choices before using the word reproduction; otherwise
  label the result a controlled adaptation and record API/model costs.

## Questions this platform could answer

1. Does developmental growth or lifetime plasticity improve adaptation beyond
   fixed and genetically optimized controllers at the same compute budget?
2. Does selecting bodies for controllability improve perturbed task performance,
   beyond simply selecting them for training return?
3. Can JAX batching reduce time-to-solution without changing reference behavior,
   evaluation budgets, or reported uncertainty?

Each is a falsifiable extension. None follows merely from producing plausible
motion or placing the component libraries in one repository.
