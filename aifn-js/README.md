# aifn-js: core and applications

The AI Field Notes library in two workspace packages. The decision and its reasoning are in
`docs/aifn-architecture.md` ("Decided: core and applications").

| Package        | Folder          | Imported as                                                       | Holds                                                              |
| -------------- | --------------- | ----------------------------------------------------------------- | ------------------------------------------------------------------ |
| `aifn`         | `core/`         | `aifn/<family>/<module>` (`aifn` for foundation's common surface) | Tensor-native numerics, generic engines and every protocol         |
| `aifn-applied` | `applications/` | `aifn-applied/<area>/…`                                           | Named models, problems, environments, datasets and worked examples |

Both packages are trees (`.scratch/aifn/module-tree.md`): core has 12 families in tiers, each holding modules in local
tiers and optional shared files at the family root; applications have 15 areas holding groups and modules, where a
group's root files are its shared layer. `modules.json` is the one source for the tree: families, modules, local
tiers, shared files, areas and their DAG (`dependsOn`), and the taxonomy topics each serves. The layer lint
(`node scripts/aifn-layers.ts`, in `make lint` and `make test`), the tables below and the lab's sidebar read it.

## What core is

A module or export belongs in core when it passes all of C1–C5, or C6 alone:

- **C1** Its interface names no model, problem or dataset.
- **C2** At least two areas (or core modules) use it.
- **C3** It is testable against a reference (NumPy, SciPy, scikit-learn, a textbook value) or a law.
- **C4** It is Tensor-native: inputs are `number | Tensor` or a declared protocol object; outputs are Tensors, numbers
  or protocol objects.
- **C5** It does not change when a note or a figure does.
- **C6** Other core code depends on it. Foundations such as `zeros`, the `*Like` input types and `Value` are core
  because the rest of core is built on them, not because they are capabilities.

Core is the numerical library: primitives and their derivatives; linear algebra, special functions, randomness and
statistics; the solvers (optimisation, roots, quadrature, ODE and SDE integration, mathematical programming, optimal
transport, interpolation); the generic inference engines (message passing, EP, MCMC, VI, Kalman-type filters); signal
and system operations; general learning parts (kernels, standard losses and metrics, neural-network layers, pipelines
and validation); and every protocol the rest implements. Its module contract is `core/README.md`.

## What an application is

An application is a named model, problem, environment, dataset or worked example built from core: k-means, a GMM
fitted by EM, the dishonest casino, a multi-armed bandit, the heat equation. Applications live in 15 areas named after
the site's subjects, not its folders; a note moving category changes nothing here. Didactic code that is itself the
lesson lives here too, written to be read, and notes show it rather than copy it. Details: `applications/README.md`.

## Import rules

| From               | May import                                                                                                                                                             | May not                                                                                            |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| core file          | modules of lower family tiers; modules of its own family in lower local tiers (D5); its ancestors' shared files by relative path; a family index only for shared names | `aifn-applied`; the package root; its own family's index; anything outside aifn; the DOM           |
| core test          | core                                                                                                                                                                   | `aifn-applied`                                                                                     |
| application file   | any core node; its ancestors' shared files by relative path; areas below it in the DAG                                                                                 | siblings or other nodes of its own area (siblings share through the parent); anything outside aifn |
| lab, site, widgets | core and applications, by public path                                                                                                                                  | defining numerics; relative paths into aifn                                                        |

- No relative import leaves its module except upward to an ancestor's shared file; a parent never imports a child
  (except an `index.ts` re-exporting it).
- Tests of a core combinator with an application (a pipeline of `standardScaler` and a model) are application tests.

## Promotion and demotion

- **Promote** an application export to core when C1–C5 hold: its interface names no model, a second area (or a core
  module) needs it, a reference fixture exists, and its surface is Tensor-native.
- **Demote** a core export to an application when it names a problem, has consumers in one area only, or exists for
  one figure.
- **How.** Move the file (`node scripts/aifn-moves.ts` rewrites importers from `moves.json`), update `modules.json`,
  move or add the fixture, and rewrite importers in the same change.

## Layers (generated)

Families import only strictly lower tiers; modules of a family import only lower local tiers of it.

<!-- aifn-layers:start -->

<!-- Generated from aifn-js/modules.json by `node scripts/aifn-layers.ts --write`; do not edit. -->

| Tier | Family      | Modules (local tiers, low to high; * gap)                                                              | Shared             |
| ---- | ----------- | ------------------------------------------------------------------------------------------------------ | ------------------ |
| 0    | foundation  | contracts, errors · registry · tensor · pytree, fourier · convolution, autodiff, random · space, trace |                    |
| 1    | numerics    | special · linalg · polynomial, quadrature, roots, geometry · interpolate                               |                    |
| 2    | graph       | traversal, shortest-paths, spanning-trees, structures, matrices · flows, structured, propagation       | graph, tree, heap  |
| 3    | probability | stats, bijectors, samplers · distributions · likelihoods, information                                  |                    |
| 3    | optim       | line-search · first-order, second-order, proximal, derivative-free, programming · minimize             | options, schedules |
| 3    | systems     | (one module)                                                                                           |                    |
| 4    | inference   | model · exact, message-passing, expectation-propagation, variational, stochastic, filtering · engines  |                    |
| 4    | dynamics    | ode, sde · fields, control                                                                             |                    |
| 4    | signal      | windows · filters, spectral, time-frequency, wavelets, statistical, multirate* · decompositions        | signal             |
| 4    | transport   | (one module)                                                                                           |                    |
| 5    | learning    | estimators, kernels · losses, metrics, compose, validate                                               |                    |
| 6    | nn          | functional, init · layers · training                                                                   |                    |

<!-- aifn-layers:end -->

## Areas (generated)

An area imports core freely and the areas it depends on (transitively).

<!-- aifn-areas:start -->

<!-- Generated from aifn-js/modules.json by `node scripts/aifn-layers.ts --write`; do not edit. -->

| Area         | Nodes (group/{children} [shared]; * gap)                                                                                                                                                                                                     | Depends on       | Serves topics                                                   |
| ------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------- | --------------------------------------------------------------- |
| learning     | generalised/{glm, gam, ordinal*} [irls, residuals, backfitting, smoothing], linear, generative-classifiers, kernel-methods, gaussian-processes, trees-and-ensembles/{bagging, boosting} [tree], neighbours, reductions, preprocessing [util] | unsupervised     | supervised-learning, learning-foundations                       |
| unsupervised | clustering, embedding/{linear, manifold, neighbour} [neighbourhoods, centring, util]                                                                                                                                                         |                  | unsupervised-learning, anomaly-detection                        |
| inference    | sequence-models, topic-models, rating-models, lattice-models, mixture-models, conjugate-models, classifier-models                                                                                                                            |                  | probabilistic-inference, probability, statistics                |
| timeseries   | (one module)                                                                                                                                                                                                                                 |                  | time-series                                                     |
| signals      | audio                                                                                                                                                                                                                                        |                  | signal-processing                                               |
| vision       | filters                                                                                                                                                                                                                                      |                  | computer-vision                                                 |
| dynamics     | maps, pde, nonlinear, control                                                                                                                                                                                                                |                  | maths/differential-equations, control-theory                    |
| decisions    | bandits, reinforcement-learning/{planning, learning} [mdp]                                                                                                                                                                                   |                  | online-experimentation, reinforcement-learning, decision-making |
| generative   | diffusion                                                                                                                                                                                                                                    | neural           | generative-models                                               |
| neural       | architectures, ordinal*                                                                                                                                                                                                                      |                  | neural-networks, transformers, sequence-models                  |
| retrieval    | losses                                                                                                                                                                                                                                       |                  | recommendation-and-retrieval, losses                            |
| evaluation   | text, detection, quality, generative, fairness, beyond-accuracy                                                                                                                                                                              | algorithms       | metrics, trustworthy-machine-learning                           |
| information  | channels, coding, projection                                                                                                                                                                                                                 |                  | probability/information-theory                                  |
| algorithms   | dynamic-programming                                                                                                                                                                                                                          |                  | maths/optimisation, natural-language-processing                 |
| data         | synthetic, real, objectives, targets, environments, signals [truth, sizes, types, rows]                                                                                                                                                      | every other area | (support)                                                       |

<!-- aifn-areas:end -->

## To the lab in phase 1

Presentation helpers still in core, to move to the lab (the paper's "aifn-ui"):

- `fields`: grid sampling for drawing (direction and slope fields, nullclines, level sets).
- `geometry`: `lttb` and `minMaxDecimate` (decimation for charts).
