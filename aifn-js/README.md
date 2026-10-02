# aifn-js: core and applications

The AI Field Notes library in two workspace packages. The decision and its reasoning are in
`docs/aifn-architecture.md` ("Decided: core and applications").

| Package        | Folder          | Imported as                                                       | Holds                                                              |
| -------------- | --------------- | ----------------------------------------------------------------- | ------------------------------------------------------------------ |
| `aifn`         | `core/`         | `aifn/<family>/<module>` (`aifn` for foundation's common surface) | Tensor-native numerics, generic engines and every protocol         |
| `aifn-applied` | `applications/` | `aifn-applied/<area>/…`                                           | Named models, problems, environments, datasets and worked examples |

Both packages are trees (`.scratch/aifn/module-tree.md`): core has 14 families in tiers, each holding modules in local
tiers and optional shared files at the family root; applications have 17 areas holding groups and modules, where a
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
fitted by EM, the dishonest casino, a multi-armed bandit, the heat equation. Applications live in 17 areas named after
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

## Registries and the catalog

Every named variant in aifn is a registry entry: `define(info, value)` (`aifn/foundation/registry`) attaches a frozen
`info` to the value itself and returns it, so `adam.info`, `Normal.info` and `metricRegistry.auroc.info` are the same
kind of object. `definer(kind, module)` makes a `define` for one module, with `stability` defaulting to
`experimental`. Each module builds its registry statically from its own namespaces with `entries(kind, …)` (no global
`register()`), usually in a `registry.ts` beside the code, and exports the table from its index.

| Kind                 | Table (core)                                                    | Kind-specific info                                                                                         |
| -------------------- | --------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `distribution`       | `distributionRegistry`                                          | `params: Space` (argument order), `support`, `discrete`, `eventRank`, `expFamily`                          |
| `kl-rule`            | `klRegistry` (keyed `p\|q`; `kl` dispatches on it)              | `p`, `q` family keys                                                                                       |
| `bijector`           | `bijectorRegistry`                                              | `domain`, `codomain`, `params`, `factory`                                                                  |
| `link`, `likelihood` | `linkRegistry`, `likelihoodRegistry`                            | mean space; support, canonical link, links, dispersion                                                     |
| `kernel`             | `kernelRegistry`                                                | `hyper: Space`, `stationary`                                                                               |
| `window`             | `windowRegistry` (functions of the length)                      | `params`, `mainLobeWidth` (bins), `sideLobeDb`                                                             |
| `wavelet`            | `waveletRegistry`                                               | `family`, `vanishingMoments`, `taps`, `continuous`                                                         |
| `filter-design`      | `filterDesignRegistry`                                          | `family: 'iir' \| 'fir'`, `bands`, `honours`, `params`                                                     |
| `algorithm`          | `<module>Algorithms` in each module (`firstOrderAlgorithms`, …) | `problem`, `state` roles (`iterate`, `objective`, `grad`, `stepSize`, `flags`), `random`                   |
| `function`           | `<module>Functions` in each module (`statsFunctions`, …)        | `role` (transform, estimator, test, construction, property, fit, simulation, solver, inference), `returns` |
| `metric`, `loss`     | `metricRegistry`, `lossRegistry`                                | inputs, direction, range, capability; family, paired metric                                                |
| `primitive`          | the primitive table (`registry.list()` in `foundation/tensor`)  | arity, rule sources                                                                                        |

Applications register models, datasets, modifiers, environments, objectives and log-densities the same way (table in
`applications/README.md`). Every entry carries `key` (its export or lookup name), `kind`, `module`, `name`,
`stability`, and where they apply `notes`, `glossary`, `cite` (keys of `content/references.yaml`), `summary` and
`random`. An entry's address is `<module>/<key>` (`optim/first-order/adam`); applications may prefix `applied/`.

**Generated tests.** The core tests load the registries as the catalog does (`test/registries.ts`) and test every
entry without a hand-kept list: every algorithm passes the trace protocol on a case keyed by its address and has the
fields its state roles name (`test/foundation/trace/algorithms.test.ts`, which fails for an algorithm without a case);
every distribution family matches its info, its cdf (Kolmogorov–Smirnov) and its moments
(`test/probability/distributions/families.test.ts`); windows, wavelets, kernels, bijectors, links, likelihood families,
filter designs and KL rules match their declared metadata (`test/foundation/registry/conformance.test.ts`).

**The catalog.** `make catalog` (`scripts/aifn-catalog.ts`) loads every module of both packages, collects each entry
once and writes `generated/catalog.json`: `{ counts, entries: { <kind>: [{ address, package, key, kind, module, name,
stability, …info }] } }`, with `Space`s inlined and sorted by address. `make check` runs `make catalog-check`, which
fails when the committed catalog is stale, an entry's module does not exist, two entries share an address, a `notes`
slug is not a note in `content/notes`, a `glossary` key is not in `content/glossary.yaml` (or names a note the entry
does not list), or a `cite` key is not in `content/references.yaml`. It also reports fixture coverage: stable entries
without a reference case in the fixtures (report only for now). Today the catalog holds 1,899 entries of 21 kinds
(`counts` at its top), 187 of them stable.

**How a note links to aifn.** The link is declared once, on the aifn side: list the note's slug in the entry's
`notes` (the first is the defining note), and the catalog check verifies it. Notes never list their implementations
in frontmatter; the site computes each note's "In aifn" backlinks from the catalog's reverse index of `notes` (site
phase). A glossary entry stays independent: an aifn entry may name its `glossary` key, and the glossary never names
aifn keys.

## Running

| Command                                                         | What it does                                                                                                                                |
| --------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `make test`                                                     | the layer lint, then vitest for core and applications, then the Python core tests                                                           |
| `npx vitest run --config aifn-js/<pkg>/vitest.config.ts <path>` | one package, folder or file (`<pkg>` is `core` or `applications`; paths are relative to the package, e.g. `test/numerics/linalg`)           |
| `make bench`                                                    | core micro-benchmarks (`core/bench/core.bench.ts`): primitive dispatch, kernels, gradients, linear algebra, draws, algorithms on primitives |
| `make fixtures`                                                 | regenerate golden values from Python for both packages (`FIXTURES="numerics/linalg …"` for some)                                            |
| `make catalog`, `make catalog-check`                            | rebuild `generated/catalog.json`; check that it is fresh and that its links exist                                                           |
| `make aifn-layers`, `make aifn-names`                           | the layer lint and the name-collision lint (both in `make lint`); `node scripts/aifn-layers.ts --write` regenerates the tables below        |
| `make lab`, `make lab-check`, `make lab-shots`                  | the lab's dev server, its server-side render check, and screenshots into `.scratch/lab-shots/` (`ARGS="--only <family>/<module>"`)          |

`make check` runs the lints, the tests and `make catalog-check`; `make bench`, `make lab-check` and `make lab-shots` run
on demand.

## Layers (generated)

Families import only strictly lower tiers; modules of a family import only lower local tiers of it.

<!-- aifn-layers:start -->

<!-- Generated from aifn-js/modules.json by `node scripts/aifn-layers.ts --write`; do not edit. -->

| Tier | Family      | Modules (local tiers, low to high; * gap)                                                                                      | Shared             |
| ---- | ----------- | ------------------------------------------------------------------------------------------------------------------------------ | ------------------ |
| 0    | foundation  | contracts, errors · registry · tensor · pytree, fourier · convolution, autodiff, random · space, trace                         |                    |
| 1    | numerics    | special · linalg · polynomial, quadrature, roots, implicit, geometry · interpolate                                             |                    |
| 2    | graph       | traversal, shortest-paths, spanning-trees, structures, matrices · flows, structured, propagation                               | graph, tree, heap  |
| 3    | probability | stats, bijectors, samplers · distributions · likelihoods, information, tests                                                   |                    |
| 3    | optim       | line-search · first-order, second-order, proximal, derivative-free, programming · minimize                                     | options, schedules |
| 3    | systems     | (one module)                                                                                                                   |                    |
| 4    | inference   | model · exact, message-passing, expectation-propagation, variational, stochastic, filtering · engines                          |                    |
| 4    | dynamics    | ode, sde · fields, control                                                                                                     |                    |
| 4    | signal      | windows · filters, spectral, time-frequency, wavelets, statistical, cepstrum · multirate, decompositions                       | signal             |
| 4    | transport   | (one module)                                                                                                                   |                    |
| 4    | text        | normalise, tokenise, stem, hyphenation · vocabulary · subword, features, cooccurrence · pipeline, representations · statistics | aligned            |
| 5    | learning    | estimators, kernels, calibration · losses, metrics, compose, validate                                                          |                    |
| 6    | nn          | functional, init, decoding · layers · attention, training, experts · sequence                                                  |                    |
| 7    | interpreter | (one module)                                                                                                                   |                    |

<!-- aifn-layers:end -->

## Areas (generated)

An area imports core freely and the areas it depends on (transitively).

<!-- aifn-areas:start -->

<!-- Generated from aifn-js/modules.json by `node scripts/aifn-layers.ts --write`; do not edit. -->

| Area         | Nodes (group/{children} [shared]; * gap)                                                                                                                                                                                                                                                             | Depends on                                                                                                                                         | Serves topics                                                   |
| ------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| learning     | generalised/{glm, gam, ordinal} [irls, residuals, backfitting, smoothing, registry], linear, generative-classifiers, kernel-methods, gaussian-processes, trees-and-ensembles/{bagging, boosting} [tree, registry], neighbours, reductions, preprocessing, mixture-of-experts, mixture-density [util] | unsupervised                                                                                                                                       | supervised-learning, learning-foundations                       |
| unsupervised | clustering, embedding/{linear, manifold, neighbour} [neighbourhoods, centring, util]                                                                                                                                                                                                                 |                                                                                                                                                    | unsupervised-learning, anomaly-detection                        |
| inference    | sequence-models, topic-models, rating-models, lattice-models, mixture-models, conjugate-models, classifier-models                                                                                                                                                                                    |                                                                                                                                                    | probabilistic-inference, probability, statistics                |
| timeseries   | (one module)                                                                                                                                                                                                                                                                                         |                                                                                                                                                    | time-series                                                     |
| signals      | audio                                                                                                                                                                                                                                                                                                |                                                                                                                                                    | signal-processing                                               |
| text         | corpora, tokenisers, hyphenation                                                                                                                                                                                                                                                                     | inference                                                                                                                                          | natural-language-processing                                     |
| vision       | filters                                                                                                                                                                                                                                                                                              |                                                                                                                                                    | computer-vision                                                 |
| dynamics     | maps, pde, nonlinear, control                                                                                                                                                                                                                                                                        |                                                                                                                                                    | maths/differential-equations, control-theory                    |
| gym          | environments/{control} [bandits, gridworlds, registry], agents/{control} [random, bandits, tabular, planning, dqn, registry] [rollout, mdp, train, registry]                                                                                                                                         |                                                                                                                                                    | online-experimentation, reinforcement-learning, decision-making |
| generative   | diffusion, gan, energy [densities]                                                                                                                                                                                                                                                                   | neural                                                                                                                                             | generative-models                                               |
| neural       | language-models, contrastive, grokking, ode, ode-mixtures, full-batch                                                                                                                                                                                                                                |                                                                                                                                                    | neural-networks, transformers, sequence-models                  |
| retrieval    | losses                                                                                                                                                                                                                                                                                               |                                                                                                                                                    | recommendation-and-retrieval, losses                            |
| evaluation   | text, detection, quality, generative, fairness, beyond-accuracy                                                                                                                                                                                                                                      | algorithms                                                                                                                                         | metrics, trustworthy-machine-learning                           |
| information  | channels, coding, projection                                                                                                                                                                                                                                                                         |                                                                                                                                                    | probability/information-theory                                  |
| algorithms   | dynamic-programming                                                                                                                                                                                                                                                                                  |                                                                                                                                                    | maths/optimisation, natural-language-processing                 |
| data         | synthetic, real/{fonts, hyphenation} [embedded, real], objectives, targets, signals [truth, sizes, types, rows, define, recipe]                                                                                                                                                                      | learning, unsupervised, inference, timeseries, signals, vision, dynamics, generative, neural, retrieval, evaluation, information, algorithms, text | (support)                                                       |
| interpreter  | (one module)                                                                                                                                                                                                                                                                                         | learning                                                                                                                                           | (support)                                                       |

<!-- aifn-areas:end -->

## Presentation code lives in the lab

Presentation helpers that the design assigned to "aifn-ui" moved to the lab with the module tree
(`aifn-lab/src/viz/drawing`): `fields.ts` samples direction and slope fields, contours, level sets and nullclines for
drawing (on core's `aifn/dynamics/fields` grids), and `decimate.ts` holds `lttb` and `minMaxDecimate`.
