# aifn architecture (target, draft 1; decisions adopted)

The target design for the whole system: the numerical core, the shared objects built on it, the registries that list
them, the view layer, the Python side and the packages. It supersedes the ad-hoc conventions that grew during the fast
build. Each operation, object type and list is defined once; everything else refers to it.

Detail and evidence live in three working documents, cited by letter:

- **K**: `.scratch/aifn/design-core.md` (tensor, autodiff, primitives, randomness, traces, the DSP spine).
- **S**: `.scratch/aifn/design-system.md` (components, registries, views, Python, packages, DSP coverage).
- **C**: `.scratch/aifn/consolidation.md` (naming, collisions, capability conformance, layer order); **R**:
  `.scratch/aifn/refinement.md` (known gaps).

Decisions that need the owner are collected in §10, each with a recommendation.

> **Pending: core against applications.** The owner wants aifn-js split into a clean core of general capabilities and
> an application layer of compositions (e.g. a GMM fitted by EM), the latter loosely following the field notes. The
> options are being analysed in `.scratch/aifn/core-vs-apps.md`; the tiers, registries and packages below will be
> revised once that is decided.

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
    tier 0  tensor (+ primitives, pytrees, fourier, conv), space, registry, errors                    wire shapes,
    tier 1  special, linalg (+ polynomials)                                                            fixtures,
    tier 2  random, kernels                                                                            figure data,
    tier 3  autodiff transforms (interpreters), trace, data (Dataset, Decomposition types)             examples
    tier 4+ distributions, optim, solve, quadrature, graph, systems, stats, …  (full order: C §11, updated here)
```

The layer order lives in one file, `aifn-js/modules.json`; a lint enforces it, and the README tables and the lab
sidebar are generated from it (S §5.2).

## 3. The numerical core (K)

**Tensor.** Keep the strided, plain-object, immutable `Tensor`. Add a brand (no duck typing), `bool` and interleaved
`complex128` dtypes, one promotion table, and one error hierarchy (`ShapeError`, `DTypeError`,
`NotDifferentiableError`, `NumericalError`). Kernels gain row-broadcast and streaming-reduction fast paths. A
`Kernels` object is the seam where WASM could replace the CPU loops later; Workers and WebGPU take whole computations,
not operations (K §3).

**Primitives, one registry.** About 60 primitives, 30 of them non-elementwise, each declared once with `impl`, a
`shape` rule, `vjp`, `jvp`, `transpose` (linear ones) and a `batch` rule, plus docs metadata (formula, note slug).
Registering an id twice throws, so "defined once" is checked at import. Everything else is a composition. `special`'s
functions register as elementwise primitives with **one** derivative each; today's duplicate scalar derivatives go
(K §3.4, §5).

**Autodiff as interpreters.** JAX's structure at teaching scale, with PyTorch's eagerness: reverse (tape), forward
(dual) and batch interpreters nesting by level. This gives true forward mode, `vmap`, `jacobian` over pytrees, and
honest types. Add `customVjp`/`customJvp`, `checkpoint`, implicit differentiation (`fixedPoint`, `rootOf`), unrolled
and at-convergence differentiation of Algorithms, and linear-algebra rules (cholesky, lu, eigh, svd, qr, expm). One
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
primitive. Polynomials live in linalg with one `hqr` (K §8).

**Robustness.** IEEE semantics in elementwise ops; flags on factorisations; one jitter ladder in `cholesky`;
`{ atol, rtol }` tolerances; pairwise summation; differentiating a flagged result throws rather than returning a wrong
gradient (K §9).

## 4. The shared objects (S §2)

Every value that crosses a module, a view or a language is one of these. Each has one home, one interface, a `kind`
brand where it can be shown, and a registry where it has named variants.

| Object | Home | Replaces |
| --- | --- | --- |
| `Space` (serialisable parameter schema: real, int, choice, bool, nested, variants; conditions as data) | `aifn/space`, tier 0 | five parameter descriptions: lab controls, search spaces, recipes, hyperparameters, distribution ranges |
| `Distribution`, `LogDensity`, `Bijector` | distributions | two distribution protocols; mcmc's `Target`; compose's target maps |
| `Objective` (value built from primitives, optional domain and known minimisers) | optim | three objective shapes; test surfaces become registered objectives |
| `Model` + capabilities (`decide`, `scores`, `predictive`, `expect`, `transform`, `sample`), declared per estimator | estimators | uneven conformance (C §3); separate `probabilities` methods |
| `Dataset`, `Recipe`, truth as a `Model` | `aifn/data` (types), datasets (generators) | two dataset types; hand-kept recipe lists; truth that re-implements densities |
| `Metric`, `Loss` (functions with metadata; one capability vocabulary) | registry + metrics, losses | two metric types and two `defineMetric`s |
| `Kernel`, `Graph`, `Tree`, `Curve`, confusion results | kernels, graph, metrics | hand-kept wire mirrors |
| `Signal`, `Spectrum`, `TimeFrequency`, `Filter`, `FilterBank` | dsp | `Tensor \| ArrayLike` signals with no sample rate |
| `LtiSystem` (tf, zpk, ss, sos; continuous or discrete; conversions, responses) | **`aifn/systems`**, below dsp and control | two transfer-function types and two frequency-response shapes |
| `Decomposition` (components that sum back to the signal) | `aifn/data` | EMD results, wavelet decompositions, STL, GAM partial effects |
| `Algorithm`, `Trace` | trace | two algorithm forms (C N1) |

"Target" disappears as a name (it meant three things).

## 5. Registries and the catalog (S §3)

One pattern, `define(info, value)`, static per module, for primitives, metrics, losses, distribution families,
bijectors, kernels, windows, wavelets, filter designs, datasets, objectives, models, algorithms and views. `info`
carries `key`, `kind`, `stability`, `params: Space`, `notes` (site slugs), references, and kind-specific fields
(declared capabilities, state roles, `random: true`). A build step collects the registries into `catalog.json`, which
the site's content check (note links exist), the lab (reference pages, pickers), aifn-py (fixture coverage) and the
docs read. Note links are declared on the aifn side; notes get computed backlinks ("In aifn"). The glossary stays
independent.

## 6. The view layer (S §4, lab DESIGN.md)

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

aifn-py complements aifn-js; it does not mirror it. Each concept has one home language:

| Concept | Home | Carried by |
| --- | --- | --- |
| Wire shapes crossing languages | pydantic (`aifn.contracts`) | JSON Schema → generated `aifn/wire` types, with typed `toWire`/`fromWire` converters |
| Implementations and their metadata | TypeScript registries | `catalog.json` |
| Reference values for tests | aifn-py fixtures, keyed by catalog entry | `aifn-js/test/fixtures/*.json`; `aifn fixtures --check` reports stable entries without a case |
| Palette, module order | `design/palette.json`, `aifn-js/modules.json` | read directly |

`python/mlc` becomes `aifn-py` (package `aifn`) as its own step (plan §2).

## 8. Packages and checks (S §6)

| Package | May import |
| --- | --- |
| aifn-js | lower tiers of itself only; no DOM |
| aifn-lab (holds all UI: design, controls, viz, views, figures, diagram) | aifn, React, ECharts, Base UI, KaTeX, lucide, the palette |
| site, note widgets | aifn; how the site reuses the lab's UI is decided at site migration, not now |

Checks: layer lint; package boundaries; name collisions; catalog freshness and links; registry conformance (declared
capabilities, state roles, kinds); generated primitive tests (values, broadcasting, dtypes, vjp and jvp against finite
differences and each other, batch rules, input immutability); protocol tests for every registered algorithm; fixture
coverage; API reports per module; docs coverage; the page standard and `lab-shots`.

**Stability tiers.** Exports and entries are `stable`, `experimental` or `deprecated`. A thing is stable when it has
reference tests, conforms, is documented and cited. Protocol types become stable first.

## 9. Roadmap

One sequence merging K §11, S §8 and C §13. Phases overlap where their files do not.

| Phase | Work | Size |
| --- | --- | --- |
| **0 Safety net** | error hierarchy, tensor brand, numerics constants; `definePrimitive` registry scaffold with today's `define*` as wrappers; generated primitive test suite and benchmarks against today's rules; `aifn/registry` (`define`, `Info`, kinds); `modules.json` + layer lint (WP0 has the lint); DOM lib out of aifn's tsconfig | M |
| **1 One of each, one protocol** | the §4 merges (Dataset, Metric, distribution protocol, LogDensity, Objective, capability vocabulary, kind brands); Algorithm protocol in its final form in one pass: factory only, `step(state, ctx)`, `Status`, plain-data streams, no Box–Muller spare, batched samplers; naming and collisions (C §1, §10); capability conformance (C §3) | L |
| **2 Core internals** | special with one derivative per function; structural primitives collapsed; kernel fast paths; interpreters (reverse, forward, batch), `vmap`, pytrees once; custom rules, checkpoint, implicit differentiation; linalg derivatives, single-factor cholesky, lu factor/solve | L |
| **3 Signal spine** | complex128; FFT primitives and the DFT matrix; the conv family; `linearFilter`; polynomials; `aifn/systems`; `Signal`, `Spectrum`, `TimeFrequency`, `Decomposition`; dsp and control moved onto them | L |
| **4 Catalog and Space** | `aifn/space` under the lab's param builders; registries for every kind; `catalog.json` and its checks | L |
| **5 Views v2** | Plot, axis model and layers; figure state, probes, scheduler, `Equation`; view registry and `Show`; frame-owning views split; the page review of every lab page against DESIGN.md §2 | L |
| **6 Languages** | wire shapes and converters; fixtures moved into aifn-py and keyed by the catalog; `mlc` → `aifn` | M |
| **7 Coverage and hardening** | DSP gaps by chapter (multirate, adaptive, time–frequency, audio, spectral, wavelets; S §9); the rest of R; reference fixtures to promote modules to stable | L |
| **8 The site** | later: decide how the site reuses the lab's UI, then migrate figures by domain | L |
| **Throughout** | API reports, deprecation lint, generated docs | S |

Phase 1 replaces consolidation WP1–WP5 (their renames and conformance work, plus the Algorithm changes done once
rather than twice). Phases 2 and 3 are internal to the core and can run beside 4 and 5.

## 10. Decisions for the owner

All sixteen adopted as recommended (2026-09-30). The core/applications split (above) is decided separately.

| # | Decision | Recommendation |
| --- | --- | --- |
| 1 | Autodiff as three interpreters (reverse, forward, batch) over one primitive registry, every general primitive with vjp **and** jvp | **Adopt.** About 1,500 internal lines change; the public API stays. |
| 2 | Randomness without hidden state: no Box–Muller spare (normal draws change once more), streams passed to `step` | **Yes, both, in phase 1**, together with the factory-only Algorithm change, so every stochastic algorithm changes once. |
| 3 | Complex numbers as an interleaved dtype, differentiated as real pairs; FFT and convolution as core primitives | **Yes.** It is the spine of the signal-processing wing. |
| 4 | Static registries per module plus a generated catalog | **Yes.** No import-order side effects; one catalog for every consumer. |
| 5 | A `kind` brand on every displayable object | **Yes.** Views and wire decoding dispatch exactly. |
| 6 | `Space` in aifn (data), presentation in the lab | **Yes.** Search, recipes, families, kernels and figure state share one schema; URL state becomes generic. |
| 7 | Note links declared on aifn entries, backlinks computed | **Yes.** The code knows what it implements. |
| 8 | One LTI system in `aifn/systems` below dsp and control | **Yes** (named `systems`, not `lti`: modules spell ideas out). |
| 9 | One `Decomposition` shape for EMD, wavelets, STL and partial effects | **Yes.** |
| 10 | Pydantic stays the source of cross-language shapes | **Yes**, with typed converters in aifn-js. |
| 11 | A separate aifn-ui package | **No.** The owner's focus is the lab; all UI lives in aifn-lab. The site question waits for migration. |
| 12 | One shadcn install shared with the site | **Not now.** No UI packaging work; revisit at site migration. |
| 13 | Register all 165 algorithms | **Yes.** Workers and the generic trace view need every one. |
| 14 | Frequency axes: Hz with `fs` for sampled data, rad/s for continuous systems, always tagged | **Yes.** |
| 15 | Audio playback in the lab (user-gesture start, peak normalisation reported, A/B comparison) | **Yes.** |
| 16 | Execution order: phase 0, then 1, then 2–5 overlapping | **Yes.** Phase 1 is the last big breaking change; after it, changes are additive or internal. |
