# aifn architecture (target, draft 1; decisions adopted; phases 0–5 built)

The target design for the whole system: the numerical core, the shared objects built on it, the registries that list
them, the view layer, the Python side and the packages. It supersedes the ad-hoc conventions that grew during the fast
build. Each operation, object type and list is defined once; everything else refers to it.

Detail and evidence live in three working documents, cited by letter:

- **K**: `.scratch/aifn/design-core.md` (tensor, autodiff, primitives, randomness, traces, the DSP spine).
- **S**: `.scratch/aifn/design-system.md` (components, registries, views, Python, packages, DSP coverage).
- **C**: `.scratch/aifn/consolidation.md` (naming, collisions, capability conformance, layer order); **R**:
  `.scratch/aifn/refinement.md` (known gaps).

Decisions that need the owner are collected in §10, each with a recommendation. What is built and what is still
planned is in "Status (2026-10-01)" below and in a status line at the head of each section.

> **Decided (2026-09-30): core and applications.** aifn-js holds two workspace packages:
> `aifn-js/core` (package `aifn`, imported as `aifn/<family>/<module>`) and `aifn-js/methods` (package `aifn-methods`,
> imported as `aifn-methods/<area>`). Details and the per-module classification: `.scratch/aifn/core-vs-apps.md`.
>
> - **Core** holds what passes all of: C1 its interface names no model, problem or dataset; C2 at least two areas (or
>   core modules) use it; C3 it is testable against a reference or a law; C4 it is Tensor-native; C5 it does not change
>   when a note or figure does. **Or** C6: other core code depends on it (foundations such as `zeros`, the `*Like`
>   input types and `Value` are core because the rest of core is built on them, not because they are capabilities).
> - **Contracts.** `aifn/foundation/contracts` (tier 0, types only) is where the shapes and signatures are defined once: `Value`,
>   `TensorLike`/`VectorLike`/`MatrixLike` and the other input aliases, the protocol interfaces (Algorithm, Trace,
>   Status, Distribution, LogDensity, Bijector, Objective, Kernel, Model and capabilities, Dataset and Recipe types,
>   Metric/Loss info, Graph/Tree, Signal, Spectrum, LtiSystem, Decomposition, Space, registry Info), and the function
>   signature types of each family (e.g. `Sampler`, `Kernel`, `MetricFn`). Mathematical values are typed `Scalar` (an alias of
>   `number`), so signatures read in mathematical terms: `erf(x: Scalar | Tensor)`, `Normal(mean: Scalar | Tensor, …)`.
>   Integer metadata keeps its own names (`Size`, `Axis`, `Index`, `Shape`), so a scalar value and a count are never
>   confused in a signature. Implementations are checked against them
>   (`satisfies`); no module defines its own alias. The generated API reports record every concrete export.
> - **Applications** are named models, problems, environments, datasets and worked examples, in 17 areas (learning,
>   unsupervised, inference, timeseries, signals, text, vision, dynamics, gym, generative, neural, retrieval,
>   evaluation, information, algorithms, data, interpreter). All datasets, generators, modifiers, test objectives and
>   environments are in `data`. Applications import core and lower areas; core never imports applications; didactic
>   code that is itself a lesson lives here, written to be read, not in notes. An application is promoted to core when
>   C1–C5 hold.
> - Presentation code that the paper assigned to "aifn-ui" (grid sampling for drawing, LTTB decimation) goes to the lab.
>
> **Decided (2026-09-30): break freely.** Until the site consumes aifn, refactors go straight to the target shape:
> importers are rewritten in the same pass, with no alias barrels, shims, `Legacy…` types or deprecation aliases.
> The deprecation policy in §8 applies only from site migration on.
>
> **Decided (2026-09-30): graphs as one core structure, intentional first.** One base `Graph` (and `Tree`) carries
> typed nodes and edges. On it, **structured graphs** describe intentional models: node roles (observed, latent,
> factor, deterministic, parameter), **groups** (plates, with sizes and nesting) and **templates** (chains, lattices,
> trees, repeated slices), expandable by `unroll` and queryable by `shape` so algorithms take fast paths
> (forward–backward on a chain, exact BP on a tree). The pgm model language, LDA's plate diagram, a linear-chain CRF
> and the lab's diagrams all use this one structure. Data graphs (k-NN, ε-ball, random) and graph matrices
> (adjacency, Laplacian) sit on the same base, with a differentiable propagation primitive. Details:
> `.scratch/aifn/refinement.md` ("Graph structures in core").
>
> **Decided (2026-09-30): nested structure.** `foundation/` (contracts, tensor with its primitives, autodiff, random,
> trace, registry, space, errors, pytrees) is its own tier: all of core may import it, and it imports nothing else in
> core. Families form the second level (e.g. `inference/{exact,message-passing,stochastic,variational,…}`,
> `nn/{functional,layers,…}` after torch), with sub-modules only where they earn it. Import paths follow the tree
> (`aifn/inference/variational`), each family has an index for its common surface, `modules.json` records the tree,
> and the lint checks order at both levels. Applications nest the same way inside their areas. The full tree is
> `.scratch/aifn/module-tree.md`; its decisions D1–D15 were adopted as recommended (2026-09-30).
>
> **Decided (2026-09-30): named chain models are applications; core keeps the chain engines.** By C1, a named model
> is an application. `aifn/inference/exact` holds only generic chain engines: `forwardBackward`, `viterbi`, their
> `…Steps` and `sampleHiddenPath` (forward filtering, backward sampling) on a `ChainPotentials` (node potentials
> N × K, transition K × K), `chainForwardBackward`/`chainViterbi` on log-potentials, and `factorChain`/`chainSumProduct`
> on chain-shaped factor graphs. The HMM (`Hmm`, `hmm`, `hmmChain` building its potentials from observations,
> `hmmModel`, the casino, sampling) and the linear-chain CRF live in `aifn-methods/inference/sequence-models`, and so
> do future named chain models (factorial HMM, MEMM).

## Status (2026-10-01)

Phases 0–5 are built; phase 7 (coverage) is mostly done; phases 6 (languages, `mlc` → `aifn`) and 8 (the site) wait
for the owner. Open items are in R (`.scratch/aifn/refinement.md`, "Open" sections at the top).

| Section                | Built                                                                                                                                                                                                | Planned                                                                                            |
| ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| §2 Shape of the system | aifn-js as `core/` (`aifn`, 12 families, 124 modules) and `applications/` (`aifn-methods`, 15 areas); aifn-lab                                                                                       | aifn-py; the site consuming aifn                                                                   |
| §3 Numerical core      | brand, dtypes incl. `complex128`, promotion, errors; 134 primitives; three interpreters; custom rules; implicit differentiation; linalg rules; plain-data randomness; Algorithm/Trace; the DSP spine | a `Kernels` seam object for WASM                                                                   |
| §4 Shared objects      | every row, typed once in `aifn/foundation/contracts`                                                                                                                                                 | `toWire`/`fromWire` converters (phase 6)                                                           |
| §5 Registries, catalog | 19 kinds, `generated/catalog.json` (`make catalog`, `make catalog-check`)                                                                                                                            | the site's "In aifn" backlinks (phase 8); fixture coverage as a gate                               |
| §6 View layer          | `Plot`, axis model, layers; figure state, probes, `useComputed` (inline and worker), `Equation`; view registry and `Show`; every lab page reviewed; `XYChart`, `Heatmap`, `Subplots` retired         | DSP views (spectrogram, scalogram, pole–zero, responses) and audio                                 |
| §7 Python              | golden fixtures generated in Python (`aifn-js/*/test/fixtures`, `make fixtures`)                                                                                                                     | wire shapes, `aifn.contracts`, fixtures keyed by catalog entry, `python/mlc` → `aifn-py` (phase 6) |
| §8 Packages and checks | layer lint, name-collision lint, catalog freshness and links, registry conformance, generated primitive tests, algorithm protocol tests, fixture-coverage report, `lab-check`, `lab-shots`           | API reports per module, docs coverage, deprecation lint, stability promotion                       |

Module names and paths that changed since this document was drafted:

- Implicit differentiation (`implicitFixedPoint`, `implicitRoot`, `atConvergence`) is `aifn/numerics/implicit`, on
  `aifn/numerics/linalg`'s `solve`; it is not in `foundation/autodiff`. `unrolled` stays in `aifn/foundation/trace`.
- Scalar minimisation (`minimizeScalar`: Brent and golden section) is in `aifn/numerics/roots`.
- `LinearOperator` (a matrix or a function v ↦ Av) is in `aifn/numerics/linalg`, beside the matrix-free Lanczos
  eigensolver `eigsh`.
- Multirate processing is the module `aifn/signal/multirate` (`resamplePoly`, `decimateSignal`, `polyphase`,
  `dftFilterBank`, and `upfirdn` from `aifn/foundation/convolution`).
- Expectation propagation over model-language programs (Gaussian factors and interval factors, as in TrueSkill) is
  `aifn/inference/expectation-propagation/model.ts` (`compileGaussianModel`, `modelExpectationPropagation`); `infer`
  picks it when a model compiles.
- Ordinal models: the likelihoods (cumulative, continuation-ratio, adjacent-category) are
  `aifn/probability/likelihoods` (`ordinal.ts`) and the metrics (ordinal MAE, ranked probability score, C-index) are
  `aifn/learning/metrics` (`ordinal.ts`); the models are applications, in `aifn-methods/learning/generalised/ordinal`
  (thresholds, ordinal GLMs, deep heads) and `aifn-methods/learning/gaussian-processes/ordinal.ts`.
- KL rules are a static table, `klRegistry` in `aifn/probability/distributions` (keyed `p|q`, read by `kl`); the
  former `registerKl` is gone.
- Polynomials are `aifn/numerics/polynomial` (`polynomialRoots` on the one `hqr`, in linalg's `eig`).
- Presentation helpers (`lttb`, `minMaxDecimate`, field drawing) moved to the lab (`aifn-lab/src/viz/drawing`).

## 1. Goals and non-goals

- **Stable and durable.** Public shapes are few, typed and versioned by an API report. Renames go through deprecation.
- **Defined once.** One implementation per operation, one type per concept, one list per enumeration. Duplicates are
  errors caught by checks, not by review.
- **Readable.** A learner can read any algorithm and step through it. Eager evaluation, plain data, one file per idea.
- **Inspectable.** Every iterative computation is a trace; every object can be shown by a registered view.
- **Not** a general-purpose framework: no jit, lazy graphs, fusion, GPU tensors, sparse tensors or exotic dtypes (K §13).
  Speed is sized to figures (a drag stays under 8 ms); heavier work is offloaded whole, never per operation.

## 2. The shape of the system

```
                  ┌────────────────────────── site (notes, MDX, search) ─────────────────────────┐
                  │                                                                              │
aifn-lab: the UI home (design, controls, Plot + layers, views, figures, diagram, audio, specimens) ◀─┘ (later)
                                   │
                                   ▼
aifn-js (package aifn): numerics and models, no React, no DOM ◀── generated types ── aifn-py (package aifn)
    tier 0  foundation (contracts, errors, registry, tensor, pytree, fourier, convolution,           wire shapes,
            autodiff, random, space, trace)                                                          fixtures,
    tier 1  numerics   2  graph   3  probability, optim, systems                                        figure data,
    tier 4  inference, dynamics, signal, transport   5  learning   6  nn                                examples
aifn-methods: 17 areas over core (learning, …, data)
```

**Status:** built, except aifn-py (phase 6) and the site's use of aifn (phase 8). The tier list in the diagram above
follows the module tree (`.scratch/aifn/module-tree.md` §4.1, applied
2026-09-30, and moved in phase 1 when `graph` came to use `numerics/linalg`: families `foundation` 0; `numerics` 1;
`graph` 2; `probability`, `optim`, `systems` 3; `inference`, `dynamics`, `signal`, `transport` 4; `learning` 5; `nn` 6). The layer order lives in one file,
`aifn-js/modules.json`; a lint enforces it, and the README tables and the lab sidebar are generated from it (S §5.2).

## 3. The numerical core (K)

**Status:** built (phases 0–3), except the `Kernels` seam object; the kernels are `foundation/tensor/kernels.ts`.

**Tensor.** Keep the strided, plain-object, immutable `Tensor`. Add a brand (no duck typing), `bool` and interleaved
`complex128` dtypes, one promotion table, and one error hierarchy (`ShapeError`, `DTypeError`,
`NotDifferentiableError`, `NumericalError`). Kernels gain row-broadcast and streaming-reduction fast paths. A
`Kernels` object is the seam where WASM could replace the CPU loops later; Workers and WebGPU take whole computations,
not operations (K §3).

**Primitives, one registry.** About 60 primitives were planned, 30 of them non-elementwise (134 are registered today), each declared once with `impl`, a
`shape` rule, `vjp`, `jvp`, `transpose` (linear ones) and a `batch` rule, plus docs metadata (formula, note slug).
Registering an id twice throws, so "defined once" is checked at import. Everything else is a composition. `special`'s
functions register as elementwise primitives with **one** derivative each; today's duplicate scalar derivatives go
(K §3.4, §5).

**Autodiff as interpreters.** JAX's structure at teaching scale, with PyTorch's eagerness: reverse (tape), forward
(dual) and batch interpreters nesting by level. This gives true forward mode, `vmap`, `jacobian` over pytrees, and
honest types. Add `customVjp`/`customJvp`, `checkpoint`, implicit differentiation (`implicitFixedPoint`, `implicitRoot`, in
`aifn/numerics/implicit` because it needs linalg's `solve`), unrolled (`aifn/foundation/trace`) and at-convergence
differentiation of Algorithms, the ODE adjoint (`odeAdjoint`, `aifn/dynamics/ode`), and linear-algebra rules (cholesky, lu, eigh, svd, qr, expm). One
pytree module in tensor. The public API keeps its names (K §4).

**Randomness as plain data.** Philox keys stay. `Stream` becomes `{ key, position }` with no hidden state (the
Box–Muller spare goes), samplers are batched over `randomBits`, and distributions gain `rsample` where a pathwise draw
exists. `vmap` and `replicate` give the same draws (K §6).

**Algorithm and Trace.** Factory form only. `step(state, ctx)` receives its stream from the runner, so states are
plain data and never hold a stream. States carry a `Status` (`t`, `converged`, `diverged`, `stalled`, `terminated`)
that the runner reads. Trace columns are growable buffers with O(1) snapshots; `keep` and `timing` options bound
memory. An algorithm registry with serialisable refs lets a Worker rebuild and run any algorithm (K §7).

**The signal-processing spine.** Complex numbers are a dtype, differentiated as pairs of reals. `fft`, `ifft`,
`rfft`, `irfft` are primitives in tensor with transpose rules, and the DFT matrix is their definition (law-tested
against them). One `conv`/`convTranspose`/`pad` family replaces four convolutions (dsp 1-D, dsp image, nn, stats), with
FFT and overlap-add as methods, and multirate as strided convolution. `linearFilter` (IIR) is a differentiable
primitive. Polynomials are `aifn/numerics/polynomial`, with one `hqr` (in linalg's `eig`) (K §8).

**Robustness.** IEEE semantics in elementwise ops; flags on factorisations; one jitter ladder in `cholesky`;
`{ atol, rtol }` tolerances; pairwise summation; differentiating a flagged result throws rather than returning a wrong
gradient (K §9).

## 4. The shared objects (S §2)

Every value that crosses a module, a view or a language is one of these. Each has one home, one interface, a `kind`
brand where it can be shown, and a registry where it has named variants.

**Status:** built. Every type is defined once in `aifn/foundation/contracts`; the Home column gives the module that
implements it.

| Object                                                                                                             | Home                                                                    | Replaces                                                                                                |
| ------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| `Space` (serialisable parameter schema: real, int, choice, bool, nested, variants; conditions as data)             | `aifn/foundation/space`                                                 | five parameter descriptions: lab controls, search spaces, recipes, hyperparameters, distribution ranges |
| `Distribution`, `LogDensity`, `Bijector`                                                                           | `aifn/probability/{distributions,bijectors}`                            | two distribution protocols; mcmc's `Target`; compose's target maps                                      |
| `Objective` (value built from primitives, optional domain and known minimisers)                                    | `aifn/optim`; test surfaces in `aifn-methods/data/objectives`           | three objective shapes; test surfaces become registered objectives                                      |
| `Model` + capabilities (`decide`, `scores`, `predictive`, `expect`, `transform`, `sample`), declared per estimator | `aifn/learning/estimators` (`defineModel`)                              | uneven conformance (C §3); separate `probabilities` methods                                             |
| `Dataset`, `Recipe`, truth as a `Model`                                                                            | contracts (types); `aifn-methods/data` (generators, recipe interpreter) | two dataset types; hand-kept recipe lists; truth that re-implements densities                           |
| `Metric`, `Loss` (functions with metadata; one capability vocabulary)                                              | `aifn/learning/{metrics,losses}`                                        | two metric types and two `defineMetric`s                                                                |
| `Kernel`, `Graph`, `Tree`, `Curve`, confusion results                                                              | `aifn/learning/kernels`, `aifn/graph`, `aifn/learning/metrics`          | hand-kept wire mirrors                                                                                  |
| `Signal`, `Spectrum`, `TimeFrequency`, `Filter`, `FilterBank`                                                      | `aifn/signal` (shared `signal.ts`) and its modules                      | `Tensor \| ArrayLike` signals with no sample rate                                                       |
| `LtiSystem` (tf, zpk, ss, sos; continuous or discrete; conversions, responses)                                     | **`aifn/systems`**, below signal and control                            | two transfer-function types and two frequency-response shapes                                           |
| `Decomposition` (components that sum back to the signal)                                                           | contracts; `aifn/signal/decompositions`, wavelets, STL, GAM             | EMD results, wavelet decompositions, STL, GAM partial effects                                           |
| `Algorithm`, `Trace`                                                                                               | `aifn/foundation/trace`                                                 | two algorithm forms (C N1)                                                                              |

"Target" disappears as a name (it meant three things).

## 5. Registries and the catalog (S §3)

**Status:** built (phase 4). Core registers primitives, distributions, KL rules (`klRegistry`), bijectors, links,
likelihoods, kernels, windows, wavelets, filter designs, metrics, losses and algorithms (`<module>Algorithms`);
applications register models, datasets, modifiers, environments, objectives and log-densities. Views are registered in
the lab, not in the catalog. `catalog.json` is read by its own check today; the site's backlinks and aifn-py's fixture
keys are planned. Details: `aifn-js/README.md`, "Registries and the catalog".

One pattern, `define(info, value)`, static per module, for primitives, metrics, losses, distribution families,
bijectors, kernels, windows, wavelets, filter designs, datasets, objectives, models, algorithms and views. `info`
carries `key`, `kind`, `stability`, `params: Space`, `notes` (site slugs), references, and kind-specific fields
(declared capabilities, state roles, `random: true`). A build step collects the registries into `catalog.json`, which
the site's content check (note links exist), the lab (reference pages, pickers), aifn-py (fixture coverage) and the
docs read. Note links are declared on the aifn side; notes get computed backlinks ("In aifn"). The glossary stays
independent.

## 6. The view layer (S §4, lab DESIGN.md)

**Status:** built (phase 5), with the page review of every lab page done and `XYChart`, `Heatmap`, `Subplots` and the
frame-owning view wrappers retired for `Plot` (5d). Planned: the DSP views and audio playback below.

Three levels, each with one job:

- **Layers** draw inside a `Plot` that owns its axes (fit, hold, fixed or support-bound ranges; equal aspect; sharing;
  zoom; a compact toolbar): curve, points, bars, area, signed area, histogram, density, rug, support band, vectors,
  raster, contours, handle, probe, annotation. `EChart.tsx` stays the only file that touches ECharts.
- **Views** show one kind of object as a panel, never a frame. They are registered by `kind`, so `Show(object)` picks
  the right one. About 50 views over 22 kinds cover every lab page. The 18 current views that draw their own `Figure`
  are split.
- **Figures** are the frame: title, purpose, state (`Space` + presentation), the `Player`, the equation band, grouped
  readouts, caption, anchors and URL state. `Plots`/`Subplots` and `Dashboard` lay out their panels. The `useComputed`
  scheduler keeps drags immediate (DESIGN.md §8a).

DSP adds views for waveform, spectrum (magnitude and phase, dB), spectrogram and scalogram, pole–zero, frequency,
impulse and step response, filter banks, and audio playback.

## 7. Python and TypeScript (S §5)

**Status:** planned (phase 6, waiting on the owner). Today the golden fixtures are generated by Python scripts inside
aifn-js (`aifn-js/core/test/fixtures/generate.py`, `make fixtures`), and `make catalog-check` reports stable entries
without a fixture.

aifn-py complements aifn-js; it does not mirror it. Each concept has one home language:

| Concept                            | Home                                          | Carried by                                                                                    |
| ---------------------------------- | --------------------------------------------- | --------------------------------------------------------------------------------------------- |
| Wire shapes crossing languages     | pydantic (`aifn.contracts`)                   | JSON Schema → generated `aifn/wire` types, with typed `toWire`/`fromWire` converters          |
| Implementations and their metadata | TypeScript registries                         | `catalog.json`                                                                                |
| Reference values for tests         | aifn-py fixtures, keyed by catalog entry      | `aifn-js/test/fixtures/*.json`; `aifn fixtures --check` reports stable entries without a case |
| Palette, module order              | `design/palette.json`, `aifn-js/modules.json` | read directly                                                                                 |

`python/mlc` becomes `aifn-py` (package `aifn`) as its own step (plan §2).

## 8. Packages and checks (S §6)

**Status:** the package boundaries hold and most checks run in `make check` (layers, names, catalog, conformance,
primitive and protocol tests); `lab-check` and `lab-shots` run on demand. Planned: API reports, docs coverage, the
deprecation lint and promotion to `stable` (164 of 680 entries are stable).

| Package                                                                 | May import                                                                   |
| ----------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| aifn-js                                                                 | lower tiers of itself only; no DOM                                           |
| aifn-lab (holds all UI: design, controls, viz, views, figures, diagram) | aifn, React, ECharts, Base UI, KaTeX, lucide, the palette                    |
| site, note widgets                                                      | aifn; how the site reuses the lab's UI is decided at site migration, not now |

Checks: layer lint; package boundaries; name collisions; catalog freshness and links; registry conformance (declared
capabilities, state roles, kinds); generated primitive tests (values, broadcasting, dtypes, vjp and jvp against finite
differences and each other, batch rules, input immutability); protocol tests for every registered algorithm; fixture
coverage; API reports per module; docs coverage; the page standard and `lab-shots`.

**Stability tiers.** Exports and entries are `stable`, `experimental` or `deprecated`. A thing is stable when it has
reference tests, conforms, is documented and cited. Protocol types become stable first.

## 9. Roadmap

One sequence merging K §11, S §8 and C §13. Phases overlap where their files do not. What each landed:
`docs/aifn-plan.md` §0.

| Phase                           | Work                                                                                                                                                                                                                                                                                                                                                                                                                                               | Size | Status                                |
| ------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---- | ------------------------------------- |
| **0 Safety net and layout**     | aifn-js split into `core/` and `applications/` workspaces with the 12 whole application modules moved; `aifn/contracts`; error hierarchy, tensor brand, numerics constants; `definePrimitive` registry scaffold with today's `define*` as wrappers; generated primitive test suite and benchmarks against today's rules; `aifn/registry` (`define`, `Info`, kinds); `modules.json` + layer lint (WP0 has the lint); DOM lib out of aifn's tsconfig | M    | done                                  |
| **1 One of each, one protocol** | the §4 merges (Dataset, Metric, distribution protocol, LogDensity, Objective, capability vocabulary, kind brands); Algorithm protocol in its final form in one pass: factory only, `step(state, ctx)`, `Status`, plain-data streams, no Box–Muller spare, batched samplers; naming and collisions (C §1, §10); capability conformance (C §3)                                                                                                       | L    | done                                  |
| **2 Core internals**            | special with one derivative per function; structural primitives collapsed; kernel fast paths; interpreters (reverse, forward, batch), `vmap`, pytrees once; custom rules, checkpoint, implicit differentiation; linalg derivatives, single-factor cholesky, lu factor/solve                                                                                                                                                                        | L    | done                                  |
| **3 Signal spine**              | complex128; FFT primitives and the DFT matrix; the conv family; `linearFilter`; polynomials; `aifn/systems`; `Signal`, `Spectrum`, `TimeFrequency`, `Decomposition`; dsp and control moved onto them                                                                                                                                                                                                                                               | L    | done                                  |
| **4 Catalog and Space**         | `aifn/space` under the lab's param builders; registries for every kind; `catalog.json` and its checks                                                                                                                                                                                                                                                                                                                                              | L    | done                                  |
| **5 Views v2**                  | Plot, axis model and layers; figure state, probes, scheduler, `Equation`; view registry and `Show`; frame-owning views split; the page review of every lab page against DESIGN.md §2                                                                                                                                                                                                                                                               | L    | done                                  |
| **6 Languages**                 | wire shapes and converters; fixtures moved into aifn-py and keyed by the catalog; `mlc` → `aifn`                                                                                                                                                                                                                                                                                                                                                   | M    | waiting on owner                      |
| **7 Coverage and hardening**    | DSP gaps by chapter (multirate, adaptive, time–frequency, audio, spectral, wavelets; S §9); the rest of R; reference fixtures to promote modules to stable                                                                                                                                                                                                                                                                                         | L    | mostly done; open list in R           |
| **8 The site**                  | later: decide how the site reuses the lab's UI, then migrate figures by domain                                                                                                                                                                                                                                                                                                                                                                     | L    | waiting on owner                      |
| **Throughout**                  | API reports, deprecation lint, generated docs                                                                                                                                                                                                                                                                                                                                                                                                      | S    | checks run; API reports, docs not yet |

Phase 1 replaces consolidation WP1–WP5 (their renames and conformance work, plus the Algorithm changes done once
rather than twice). Phases 2 and 3 are internal to the core and can run beside 4 and 5.

## 10. Decisions for the owner

All sixteen adopted as recommended (2026-09-30). The core/applications split (above) is decided separately.

| #   | Decision                                                                                                                           | Recommendation                                                                                                          |
| --- | ---------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| 1   | Autodiff as three interpreters (reverse, forward, batch) over one primitive registry, every general primitive with vjp **and** jvp | **Adopt.** About 1,500 internal lines change; the public API stays.                                                     |
| 2   | Randomness without hidden state: no Box–Muller spare (normal draws change once more), streams passed to `step`                     | **Yes, both, in phase 1**, together with the factory-only Algorithm change, so every stochastic algorithm changes once. |
| 3   | Complex numbers as an interleaved dtype, differentiated as real pairs; FFT and convolution as core primitives                      | **Yes.** It is the spine of the signal-processing wing.                                                                 |
| 4   | Static registries per module plus a generated catalog                                                                              | **Yes.** No import-order side effects; one catalog for every consumer.                                                  |
| 5   | A `kind` brand on every displayable object                                                                                         | **Yes.** Views and wire decoding dispatch exactly.                                                                      |
| 6   | `Space` in aifn (data), presentation in the lab                                                                                    | **Yes.** Search, recipes, families, kernels and figure state share one schema; URL state becomes generic.               |
| 7   | Note links declared on aifn entries, backlinks computed                                                                            | **Yes.** The code knows what it implements.                                                                             |
| 8   | One LTI system in `aifn/systems` below dsp and control                                                                             | **Yes** (named `systems`, not `lti`: modules spell ideas out).                                                          |
| 9   | One `Decomposition` shape for EMD, wavelets, STL and partial effects                                                               | **Yes.**                                                                                                                |
| 10  | Pydantic stays the source of cross-language shapes                                                                                 | **Yes**, with typed converters in aifn-js.                                                                              |
| 11  | A separate aifn-ui package                                                                                                         | **No.** The owner's focus is the lab; all UI lives in aifn-lab. The site question waits for migration.                  |
| 12  | One shadcn install shared with the site                                                                                            | **Not now.** No UI packaging work; revisit at site migration.                                                           |
| 13  | Register all 165 algorithms                                                                                                        | **Yes.** Workers and the generic trace view need every one.                                                             |
| 14  | Frequency axes: Hz with `fs` for sampled data, rad/s for continuous systems, always tagged                                         | **Yes.**                                                                                                                |
| 15  | Audio playback in the lab (user-gesture start, peak normalisation reported, A/B comparison)                                        | **Yes.**                                                                                                                |
| 16  | Execution order: phase 0, then 1, then 2–5 overlapping                                                                             | **Yes.** Phase 1 is the last big breaking change; after it, changes are additive or internal.                           |
