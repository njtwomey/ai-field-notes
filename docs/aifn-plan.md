# aifn: a shared library for the field notes

Status: **plan for review**. Nothing here is built yet. Points that need the owner's decision are marked **[decide]**.
Evidence for every claim about today's code is in the stage-1 survey (`.scratch/aifn/survey.md`).

## 1. Why

The site's value is in its interactive models: about 900 TypeScript files under `content/notes`, 25,500 lines of
algorithm code in 69 `_shared` folders, and roughly 1,150 lines of central code in `site/src/lib`. The code is local
to notes and heavily duplicated: `mean` is written 73 times, Cholesky 11 times with four different diagonal floors,
Gaussian elimination 11 times, a symmetric eigensolver 7 times, a gamma sampler 6 times and logistic regression fitting
5 times. The copies disagree on conventions and carry bugs:

- a Cholesky that returns NaN on a singular matrix;
- five solvers with no zero-pivot guard;
- two ROC modules with swapped argument order;
- a normal CDF with no tail accuracy below about 1e-7;
- an O(n²) effective sample size that reruns on every slider tick;
- about 40 ad-hoc seed formulas, some of which can collide.

Nothing in TypeScript is tested.

aifn collects this into one tested library. It is organised by what each part does, it exposes every algorithm's
internals, and every iterative algorithm can be stepped, scrubbed, replicated, profiled and resumed. Packaged libraries
(TF.js, ml.js) hide the internals a teaching figure needs to show, which is why we build our own.

## 2. Names and packages

| Today                                                             | After                                                                                  | Import                                             |
| ----------------------------------------------------------------- | -------------------------------------------------------------------------------------- | -------------------------------------------------- |
| `python/`, package `mlc`                                          | `aifn-py/`, package `aifn` (same internal layout: examples, figures, build, contracts) | `from aifn.contracts import …`; `uv run aifn runs` |
| `site/src/lib/{math,distributions,dsp}` and the `_shared` folders | `aifn-js/`, an npm workspace package named `aifn`                                      | `import { cholesky } from 'aifn/linalg'`           |

- `aifn-js` is TypeScript source, consumed directly by Vite and `tsc -b` (no build step). It has one subpath export per
  module, so figures import only what they use.
- aifn-py and aifn-js are independent. They share names where that is natural, but not code or compatibility.
- The Python rename is a separate, mechanical step. It rebuilds every committed run and figure asset, because the
  asset hashes include source paths.

## 3. Organisation: by what things do

There is no "core" module. Modules are grouped into families for documentation only; the import paths are flat
(`aifn/linalg`, `aifn/pgm`). Dependencies point one way (listed families first), and a lint rule forbids cycles.

**Numerics**

- `special`: erf, erfc, log erfc, Φ, log Φ, Φ⁻¹, logΓ, digamma, trigamma, incomplete beta and gamma, Student t, and
  the stable forms: softplus, sigmoid, logit, logsumexp, softmax.
- `tensor`: the n-dimensional array type everything numeric builds on: shape, strides and dtype (`float64`,
  `float32`, `int32`) over typed-array storage; views, slicing, reshaping and transposition without copies;
  broadcasting elementwise operations; reductions along axes; a small `einsum`; and conversion to and from plain arrays
  at the chart boundary. `Vector` and `Matrix` are rank-1 and rank-2 tensors with their own fast paths.
- `linalg`: products; Cholesky with jitter reporting; triangular solves; LU; QR; inverse; log-determinant; symmetric
  eigendecomposition; SVD; Kronecker product; fast paths for 2×2 and 3×3. All on `tensor`.
- `autodiff`: a reverse-mode tape over scalars and tensors; forward-mode duals; Hessian-vector products; a
  gradient check. Graphs are inspectable (nodes, values, adjoints).
- `optim`: gradient descent, momentum, Nesterov, AdaGrad, RMSProp, Adam, damped Newton, Armijo and Wolfe line
  searches, L-BFGS, Nelder–Mead, box QP, proximal gradient and ISTA/FISTA. All are traceable.
- `programming`: classical exact solvers for small problems, correct rather than state of the art: linear programming
  (simplex with the tableau exposed, and an interior-point method), quadratic programming (active set, and interior
  point for convex QPs), integer and mixed-integer programming (branch and bound, with the search tree exposed, and
  Gomory cutting planes), dynamic programming (knapsack, shortest paths, sequence alignment), the assignment problem
  (Hungarian algorithm), and KKT and duality reporting (multipliers, slacks, duality gap). Each is traceable, so a
  figure can walk through simplex pivots or a branch-and-bound tree.
- `solve`: nonlinear equations and roots: bisection, Brent, secant, Newton and damped Newton for systems,
  fixed-point iteration with convergence diagnostics, continuation. All are traceable.
- `quadrature`: trapezoid, Simpson, Gauss–Legendre and Gauss–Hermite, adaptive quadrature, Monte Carlo integration.

**Probability**

- `random`: keyed streams (§6); samplers (uniform, normal, gamma, beta, Dirichlet, categorical and alias, Poisson,
  binomial, multinomial, permutation); `replicate`.
- `distributions`: distribution objects, univariate and multivariate, batched (§5.2).
- `stats`: descriptive statistics, ranks, histograms with an explicit edge policy, KDE, bootstrap, permutation tests,
  classical tests.
- `info`: information theory: entropy, joint and conditional entropy, cross-entropy, KL and Jensen–Shannon divergence
  and the f-divergence family, mutual information (discrete, Gaussian, and estimators from samples such as
  Kraskov–Stögbauer–Grassberger), differential entropy, channel capacity by Blahut–Arimoto, rate–distortion, and
  coding (Huffman and Shannon–Fano codes, Kraft's inequality, arithmetic coding, Hamming distance and error-correcting
  codes with their bounds).
- `pgm`: model descriptions, plates, factor graphs, inference engines, and translation to plate diagrams (§7).
- `ep`: expectation propagation and assumed density filtering as a first-class module, usable with or without `pgm`:
  Gaussian and exponential-family message algebra (natural parameters, multiply, divide, damping), cavity and tilted
  distributions, moment matching for probit, step and truncated-Gaussian factors (win and draw), power EP, and
  convergence diagnostics. It underlies TrueSkill, the Bayes point machine, GP classification and the clutter problem.
- `mcmc`: Metropolis–Hastings, Gibbs, HMC with leapfrog, unadjusted Langevin and MALA, SGLD, particle filters;
  diagnostics (ACF via FFT, IACT, ESS, split R̂, importance ESS).
- `vi`: coordinate-ascent VI, mean field, black-box VI with score and reparameterisation estimators, the ELBO.
- `gp`: kernels (lengthscale convention), Gram matrices, posteriors, sampling, log marginal likelihood and its gradient.

**Dynamics** (deterministic and stochastic evolution, kept apart from probability)

- `trace`: the trace protocol, runners and recorders (§4). Used by every family, not only dynamics.
- `ode`: initial-value solvers as steppable algorithms: explicit Euler, Heun, midpoint, RK4, adaptive Runge–Kutta
  (Dormand–Prince RK45 with error control and step-size history), implicit Euler, trapezoid and BDF for stiff systems,
  symplectic integrators (symplectic Euler, leapfrog and velocity Verlet) with energy tracking, event detection, and
  the matrix exponential for linear systems. Every solver reports its steps, step sizes, error estimates and function
  evaluations, so figures can show stability regions, stiffness and energy drift.
- `fields`: vector and scalar fields: evaluation on grids, direction fields, divergence, curl and Jacobians (via
  `autodiff`), gradient and Hamiltonian fields, flow maps and streamlines, fixed points and their linearisation
  (eigenvalues, stability classification, stable and unstable manifolds), nullclines, limit cycles and Poincaré
  sections, Lyapunov functions, and transport of densities along a flow.
- `pde`: method-of-lines finite differences for the heat, transport, wave and Fokker–Planck equations, with stability
  (CFL) reporting.
- `maps`: discrete-time dynamical systems: iteration of maps, cobweb diagrams, orbits, bifurcation diagrams,
  Lyapunov exponents.
- `control`: linear state-space systems, controllability and observability, pole placement, LQR, stability margins,
  and discretisation. The Kalman filter stays in `timeseries`.
- `sde`: Euler–Maruyama, Milstein, particle clouds, Fokker–Planck solvers.
- `diffusion`: noise schedules, forward noising, small denoisers and score models, DDPM, DDIM and probability-flow
  sampling (§8.2).
- `timeseries`: ARMA simulation and fitting, ACF and PACF, Levinson and Burg, Kalman filter and RTS smoother, EM,
  exponential smoothing, GARCH.

**Learning**

- `estimators`: the fit and predict capability mixins (§5).
- `preprocess`: transforms with `fit` and `transform` (and `inverse` where it exists): standard, min–max and robust
  scaling, one-hot and target encoding, imputation, polynomial and spline features, random Fourier features,
  whitening, and Box–Cox and Yeo–Johnson power transforms.
- `compose`: pipelines, column-wise transforms and target transforms (§5.4).
- `validate`: splitters, cross-validation, nested cross-validation and hyperparameter search (§5.5).
- `glm`: links and families, IRLS, logistic and multinomial regression, Poisson, gamma and negative binomial.
- `smooth`: splines and interpolation: polynomial and piecewise interpolation, cubic and natural splines, PCHIP and
  Akima, B-spline bases, P-splines and their difference penalties, smoothing splines, thin-plate splines, and tensor-
  product and cyclic bases.
- `gam`: generalised additive models on `smooth` and `glm`: penalised IRLS, backfitting, smoothing-parameter selection
  by GCV and REML, effective degrees of freedom, posterior bands and draws, `by` terms, shape constraints (monotone and
  convex), expectile GAMs, and explainable boosting machines.
- `classify`: SVM (SMO and active set), trees, ensembles, k-NN, multiclass reductions and output codes.
- `cluster`: k-means (steppable, k-means++), GMM by EM, agglomerative clustering, DBSCAN.
- `embed`: PCA, MDS, Isomap, spectral embeddings, t-SNE, UMAP.
- `nn`: layers built on `autodiff` (§8.1).
- `losses`: classification, regression, ranking, sampled softmax and logQ, contrastive.
- `metrics`: one implementation for every metric the site has a note on (§5.3): classification, probabilistic and
  calibration, regression and forecasting, ranking and retrieval, clustering, detection and segmentation, text, image
  and audio quality, generative models, and fairness.

**Signals and decisions**

- `dsp`: FFT, windows, spectral estimation, filters, wavelets, EMD, audio features, image operators.
- `bandits`, `rl`: environments and policies.
- `ot`: cost matrices, Hungarian algorithm, log-domain Sinkhorn, 1-D Wasserstein.

**Support**

- `datasets`: seeded generators and small embedded sets (§9).
- `geometry`: covariance and precision ellipses, contours, grids, decimation for drawing.

## 4. Traces (`trace`)

"Trace" is the word used throughout: every iterative algorithm, whether a solver, a sampler, a training loop, a message
schedule or a simplex, produces a trace of typed steps. An algorithm is a pure description:

```ts
interface Algorithm<Opts, State> {
  init(opts: Opts, stream?: Stream): State // the state carries its internals, fully typed
  step(state: State): State // pure: returns the next state
  done?(state: State): boolean
}
```

For example, a k-means state is `{ centroids, labels, inertia, step }`, an EP state is
`{ sites, cavities, tilted, marginals, sweep }`, and an SMO state is `{ alpha, workingPair, dualObjective, activeSet }`.

Running an algorithm produces a **trace**: the list of typed steps, packaged with timing and metadata.

```ts
interface Trace<State> {
  steps: State[] // decimated by `every`; step 0 is the initial state
  index: number[] // the step number of each kept step
  series: Record<string, Tensor> // recorded quantities, stacked over kept steps
  timing: { stepMs: Float64Array; totalMs: number; perSecond: number }
  meta: { algorithm: string; opts: unknown; seed?: string; stopped: 'done' | 'limit' | 'diverged' }
}
```

- The runners are `run(alg, opts, n)` (final state only), `trace(alg, opts, n, { every, record })`,
  `seek(alg, opts, i)` (restores from checkpoints every k steps), `extend(trace, m)` (resumes, so raising a steps
  slider only computes the new steps) and `live(alg, opts)` (a generator for play loops).
- **Recorders** turn states into the series a figure plots, so learning is traced as fully as dynamics. A training
  loop's trace can record the weights, the loss, gradient norms, learning rate and any `metrics` on training and
  held-out data at every kept step:

  ```ts
  const run = trace(train(model, data, adam({ lr: 0.05 })), opts, 500, {
    every: 5,
    record: { weights: (s) => s.params, loss: (s) => s.loss, auroc: (s) => metrics.auroc(heldOut, s.model) },
  })
  run.series.loss // Float64Array, one value per kept step
  run.series.weights // tensor stacked over steps, e.g. for a weight-trajectory plot
  ```

  `run.series` holds each recorded quantity stacked over steps, aligned with `run.index` and the timing, ready for a
  chart of weights, loss and metrics over time.

- Timing is recorded per step (and, optionally, per named phase within a step via `profile(name, fn)`), so a figure or
  a test can show where the time goes.
- States are immutable and serialisable (plain objects and typed arrays). They can be cloned, checkpointed, sent to a
  Web Worker and displayed in a readout.
- On the site side: `useTrace`, `useStepper` and `usePlayLoop` hooks, a time-sliced runner, and a worker runner, all
  on the same protocol. `StepControls` and the scrubbing sliders read a trace directly.

## 5. Estimators and output modes

### 5.1 Capability mixins

Following scikit-learn's mixins, but typed. A fitted model declares what it can produce, and figures can only ask for
capabilities it has.

```ts
interface Fitted<X> {
  forward(x: X): Head // partial forward pass: logits, latent means, scores
}
interface Decides<X, Y> {
  decide(x: X): Y
} // argmax, cluster label, action
interface Predicts<X, D extends Distribution> {
  predictive(x: X): D
} // a distribution object (§5.2)
interface Expects<X> {
  expect(x: X, f?: (y: number) => number): number | Float64Array
} // E[f(y) | x]
interface Scores<X> {
  score(x: X): Float64Array
} // per-class or per-item scores
interface Transforms<X, Z> {
  transform(x: X): Z
} // embeddings, projections
interface Samples<X, Y> {
  sample(stream: Stream, x: X, n?: number): Y
}
```

Capabilities are composed by mixins, not written again for each model:

- `withDecision(model, rule)`: builds `decide` from `predictive` or `score`. The rule is `'argmax'`, `'mode'`, a
  threshold, or a cost matrix (Bayes decision under costs).
- `withExpectation(model)`: builds `expect` from `predictive`. For ordinal regression, `predictive` is an ordered
  categorical and `expect` gives E[y].
- `withSampling(model)`: builds `sample` from `predictive`.
- `readout(model, complete)`: the custom case. It takes the model's partial forward (`forward(x) → Head`) and a
  callback that completes it, and returns a new fitted model whose capabilities follow from the callback's return type.

Type guards (`hasPredictive(m)` and the like) let generic figure code branch safely. Clustering (`decide` = label,
`predictive` = responsibilities), classifiers, regressors, density models and ordinal models all use the same
vocabulary.

### 5.2 Distribution objects

Predictions return real distribution objects, shaped to the data:

- A GP posterior at M test points is `MvNormal(mean: Vector(M), cov: Matrix(M×M))`.
- A logistic model over N inputs is a batch of N `Bernoulli`s, and a multinomial model a batch of N `Categorical`s.
- A mixture model's density is a `Mixture` of components.

Every distribution provides `logProb`, `prob`, `cdf` and `logcdf` (where defined), `quantile` (univariate),
`sample(stream, n)`, `mean`, `variance` or `covariance`, `entropy`, `support`, and its `batchShape` and `eventShape`.
Exponential-family members also expose natural parameters and sufficient statistics, which `pgm` uses for conjugate
updates and message passing. Parameterisation is fixed: Gaussians by mean and standard deviation (covariance for the
multivariate case), like scipy's `scale`, with natural parameters available.

### 5.3 Metrics

The site has about 80 metric notes in 16 categories (classification, classification curves, probabilistic, regression,
forecasting, ranking, clustering, correlation and agreement, distances and similarities, detection and segmentation,
text, image quality, audio, generative models, fairness, and metric foundations). `metrics` gives each of them one
implementation, with the conventions the notes use:

- Every metric is a function of predictions and targets, with a declared input type (labels, scores, probabilities,
  distributions, rankings, sets, sequences or images) and a declared direction (higher or lower is better).
- Metrics compose with estimators: `evaluate(model, data, [auroc, logLoss, ece])` asks each metric for the capability
  it needs (`decide`, `score` or `predictive`), so a type error catches, for example, a calibration metric applied to a
  model that only decides.
- Averaging for multiclass and multilabel problems (micro, macro, weighted) and confidence intervals (bootstrap,
  analytic where known) are shared utilities, not rewritten per metric.
- Curves (ROC, PR, calibration and reliability, cost and threshold curves) return typed objects that charts draw
  directly.
- Metrics can be recorded in traces (§4), so a figure can plot a held-out metric against training steps.
- Each metric is tested against a Python reference (scikit-learn, scipy or the metric's reference implementation)
  through the fixtures of §13.

### 5.4 Pipelines and target transforms

- `pipeline(standardScaler(), logistic({ l2: 0.1 }))` chains transforms and a final estimator. Fitting fits each step
  on the output of the previous one; the pipeline has exactly the capabilities of its final step (`decide`,
  `predictive`, `expect` and so on), and the types enforce it.
- `columns({ age: standardScaler(), city: oneHot() })` applies transforms to named columns.
- `transformTarget(regressor, log())` fits on a transformed target and maps predictions back. A point prediction is
  inverted directly; a predictive distribution is pushed forward through the inverse (a Gaussian on log y becomes a
  log-normal on y), so the pipeline still returns a proper distribution. This covers targets on small or skewed
  scales.
- Every fitted pipeline exposes each step's fitted state (the scaler's means and scales, the model's weights), so a
  figure can show what each stage learned.

### 5.5 Validation

- Splitters are pure functions of a dataset and a stream: k-fold, stratified k-fold, grouped k-fold, leave-one-out,
  repeated k-fold, shuffle-split, and time-series rolling-origin and expanding-window splits.
- `crossValidate(pipeline, data, splitter, metrics)` returns a typed result: the fold assignment (a matrix a figure can
  draw), and for each fold the fitted pipeline, its predictions, its metrics and its training trace.
- `nested(outer, inner, search)` gives nested cross-validation: the inner loop selects hyperparameters, the outer loop
  estimates performance, and both levels' results are kept so a figure can show the optimism of the unnested
  estimate.
- `gridSearch` and `randomSearch` over a typed parameter space, with the full results table kept.
- Most figures fit directly on the whole dataset, because the site teaches ideas; these wrappers exist so that the
  same model object can also be validated properly when a note is about validation.

## 6. Randomness

- Streams are counter-based and keyed: `stream(seed)`, `s.child('chain', k)`, `s.child('env', run)`. Children never
  collide and never depend on how many values the parent drew. This replaces the ~40 seed formulas.
- `replicate(n, stream, fn)` runs replicate k on `stream.child(k)` and caches results by key. Raising a count reuses
  the replicates already computed (prefix reuse), and comparing methods on the same child streams gives common random
  numbers.
- Samplers take the stream as their first argument. The normal sampler uses both Box–Muller outputs, with no tail clamp.
- **[decide]** The generator switch changes every random figure's exact draws once (the distributions stay the same).
  Recommended: switch.

## 7. Probabilistic graphical models, set up properly (`pgm`)

### 7.1 A model description language

Models are described once, in a small typed builder, with variables, factors, observations and plates:

```ts
const lda = model('Latent Dirichlet allocation', (m) => {
  const K = m.size('K'),
    V = m.size('V')
  const topics = m.plate('topics', K)
  const docs = m.plate('documents', 'D')
  const words = docs.plate('words', 'N_d')

  const alpha = m.constant('α'),
    beta = m.constant('β')
  const phi = topics.variable('φ', Dirichlet(beta, V)) // one word distribution per topic
  const theta = docs.variable('θ', Dirichlet(alpha, K)) // one topic mixture per document
  const z = words.variable('z', Categorical(theta)) // a topic per word
  words.observed('w', Categorical(phi.at(z))) // the word itself
})
```

- The description is data. `toFactorGraph(model)` expands it into variables and factors.
- `toPlateDiagram(model)` produces a `DiagramSpec` for the site's `Diagram` component: variables as circles (observed
  ones shaded), plates as labelled groups, and edges from parents. It uses the existing `variable`, `factor` and
  `link` components, and the plate layout can be adjusted by hand.
- `toFactorDiagram(model)` draws the factor graph. It can highlight a Markov blanket, which generalises the Gibbs
  figure.
- The language is deliberately small: discrete variables, Gaussians, and the conjugate exponential families (Beta,
  Dirichlet, Gamma, Wishart); deterministic links (index, linear, probit and logistic); plates and nested plates. It
  does not need to be fully generic.

### 7.2 Inference on a description

Inference engines take a model plus data and follow the step protocol, so every engine can be stepped and every
message and marginal inspected.

| Engine                                     | Applies to                                                                              | Built                                                                         |
| ------------------------------------------ | --------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| Enumeration and variable elimination       | small discrete models                                                                   | by construction                                                               |
| Sum-product and max-product                | trees, chains (HMM and linear-chain CRF as special cases)                               | by construction                                                               |
| Loopy belief propagation                   | discrete graphs with loops (Ising models, grids)                                        | by construction, with schedules and damping                                   |
| Expectation propagation                    | Gaussian and probit or step factors (TrueSkill, Bayes point machine, GP classification) | by construction, from Gaussian message algebra and truncated-Gaussian moments |
| Variational message passing and mean field | conjugate exponential families                                                          | by construction, from natural parameters                                      |
| Gibbs sampling                             | discrete or conjugate full conditionals                                                 | by construction, from the Markov blanket                                      |
| Metropolis–Hastings, HMC, Langevin         | continuous latent variables                                                             | from `logDensity`, with gradients by `autodiff`                               |
| Custom routines                            | e.g. collapsed Gibbs for LDA, forward-filtering backward-sampling for HMMs              | registered against a model shape                                              |

- `infer(model, data, 'gibbs' | 'vmp' | 'ep' | …)` picks an engine. A registered custom routine (such as LDA's
  collapsed Gibbs) takes precedence when it matches the model.
- `schedule(engine)` exposes the message order. This is what MBML-style step-through figures need.
- The existing implementations (HMM, CRF, FHMM, EP, the skill and crowd models, loopy Ising) become test cases:
  each must be reproduced from its description.

## 8. Neural networks and diffusion

### 8.1 `nn`

- Small tensors: rank ≤ 4, `Float64Array` storage, CPU by default.
- Layers on `autodiff`: dense, 1-D and 2-D convolution (with stride, padding and dilation), pooling, batch and layer
  normalisation, RMS normalisation, dropout (keyed stream), embeddings, attention (masked, multi-head), recurrent
  cells (RNN, GRU, LSTM), and residual blocks.
- Every layer exposes its activations and gradients, for figures such as receptive fields, attention maps and gradient
  flow.
- Training loops are steppable algorithms: an optimiser from `optim` plus a loss from `losses`, producing traces with
  per-step timing.

### 8.2 `diffusion`

- Noise schedules (linear, cosine, VP and VE SDEs).
- The forward process in closed form, and training of small denoisers on 1-D and 2-D data (an MLP noise or score
  predictor).
- Samplers as steppable algorithms: DDPM ancestral sampling, DDIM, and probability-flow ODE via `ode`, plus reverse
  SDEs via `sde`.
- Enough to show the ideas on toy data in the browser, not to train image models. Image-scale results stay precomputed
  in aifn-py.

## 9. Datasets

- One `datasets` module of seeded generators with fixed conventions. Each returns `{ x, y?, meta }` plus a description
  used in captions:
  - 2-D and 3-D sets: blobs, moons, rings, spirals, XOR, Swiss roll;
  - the occasionally dishonest casino;
  - Zipf catalogues, ratings matrices and click logs;
  - synthetic time series;
  - test images and optimisation test surfaces.
- Small real datasets are embedded as JSON modules (e.g. Iris). Large or precomputed ones stay in aifn-py and are loaded
  with `useFigure`.
- Every generator takes a stream, so figures, tests and replicates agree.

## 10. Numbers and conventions

- **[decide]** Typed arrays inside aifn (`Float64Array` vectors; `Matrix { rows, cols, data }` row-major) and plain
  arrays only at the chart boundary (charts accept `ArrayLike<number>`). Recommended over plain `number[][]` for speed,
  workers and a future GPU path.
- Conventions fixed once:
  - Gaussians by standard deviation; kernels by lengthscale ℓ.
  - Eigenvalues in descending order.
  - Histograms return `{ edges, counts, density }` with a stated edge policy.
  - Ranks average ties; nDCG gain is explicit (default 2^rel − 1).
  - The stream comes first in every sampler.
  - Divergence and singularity are reported, never silently clipped or floored.

## 11. Performance, workers and the GPU

- Budget: a handle drag gets one frame (about 8 ms). Anything slower runs incrementally (`extend`, `replicate`), is
  time-sliced, or runs in a Web Worker.
- The FFT-based ACF and ESS, prefix reuse of chains and draws, and checkpointed `seek` remove today's worst hot spots.
- An `accel` backend interface covers elementwise operations, matmul, convolution and particle updates. It runs on CPU
  typed arrays by default, with WebGL2 or WebGPU kernels later for particle clouds, lattice Gibbs, grid evaluation and
  convolution. This is an implementation detail behind the same APIs; nothing in v1 depends on it.

## 12. The lab

A standalone app for developing and exploring aifn, separate from the site for now. `make lab` starts it.

- **Where.** `aifn-lab/`, its own small Vite app and workspace package. It depends on `aifn` (aifn-js) and may import
  the site's chart primitives (`XYChart`, `Heatmap`, `Diagram`, the controls) read-only, but nothing in the site depends
  on it.
- **Purpose.** Seeing what aifn produces, not composing pipelines: search and browse modules, run their specimens,
  and inspect every output.
- **Discovery.** An index of modules and specimens with a search box (by module, algorithm, object type or tag). Each
  specimen page shows its inputs, its outputs and the source of the call that produced them.
- **Specimens.** Each module ships small declarative specimens (inputs, a run, what to show). They drive the lab pages
  and double as smoke tests.
- **Views: the generic layer between aifn and the interface.** A registry maps each aifn object type to a view, so any
  output can be shown without a bespoke widget:

  | aifn object                    | View                                                                                                                                          |
  | ------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------- |
  | `Trace`                        | scrub or play steps; plot any recorded series (weights, loss, metrics); inspect the current step's typed state; timing per step and per phase |
  | `Tensor`, `Matrix`             | heatmap, table, or line plot by rank, with shape and dtype                                                                                    |
  | `Distribution` (and batches)   | density or mass, CDF, samples against the density, moments                                                                                    |
  | fitted estimators              | decision regions, predictive bands, calibration, per-step training traces                                                                     |
  | `CrossValidation` results      | fold-assignment matrix, per-fold metrics, nested-level comparison                                                                             |
  | PGM descriptions and inference | plate diagram, factor graph, messages and marginals per step                                                                                  |
  | `programming` results          | simplex tableaux, branch-and-bound trees, KKT multipliers                                                                                     |

- **Quality-of-life work** happens here first: better trace scrubbing, linked brushing between a trace's series and its
  state, consistent readouts, timing overlays and export of any view's data. The views and improvements are developed
  in the lab against real aifn objects, then moved into the site as part of the migration (stage 4), where the note
  figures adopt them.
- **Later.** A builder that lets readers assemble pipelines and models in the browser. The serialisable descriptions
  (§5.4, §7.1) keep this possible; it is out of scope for v1.

## 13. Testing

- vitest in `aifn-js/test`, one file per module, run by `make check`.
- Python generates the test cases. An aifn-py command (`uv run aifn fixtures`) writes JSON fixtures from
  numpy, scipy, scikit-learn, statsmodels-style computations and torch, covering most modules: tensor operations and
  linear algebra; special functions and distributions; information-theoretic quantities (`scipy.stats.entropy` and
  friends); optimisation and exact solvers (`scipy.optimize` for LP, QP via KKT systems, and MILP); ODE solutions
  (`scipy.integrate`); every metric (scikit-learn and reference implementations); smoothing and GAM fits against
  penalised-regression references; EP moments against numerical quadrature; HMM posteriors against brute-force
  enumeration; GP posteriors; Kalman/RTS; Sinkhorn; k-means and EM from fixed inits; and autodiff gradients against
  torch. Where no library reference exists, the Python side computes one directly (e.g. by enumeration or quadrature).
- Stochastic code gets statistical tests (moments, KS tests against scipy CDFs) and frozen snapshots.
- Protocol tests check that:
  - the same seed gives the same trace;
  - `seek(i)` equals `run(i)`;
  - `extend` equals a longer trace;
  - replicate prefixes are reused;
  - each PGM engine reproduces the existing hand-written implementations.

## 14. Stages

1. **Survey.** Done (`.scratch/aifn/survey.md`).
2. **Design.** This document, iterated with the owner until approved.
3. **Build.** The whole of aifn-js with tests, before any figure changes, with the lab (§12) growing alongside it so
   each module can be seen working as it lands. The lab's generic views are built in this stage too. Order: `tensor`, `random`, `special`,
   `linalg`, `stats`, `distributions`, `trace`, `autodiff`, `optim`, `programming`, `solve`, `quadrature`, `info`,
   `estimators`, `metrics`, `preprocess`, `compose`, `validate`, then the domains (`ep`, `pgm`, `mcmc`, `vi`, `gp`, `glm`, `smooth`, `gam`, `classify`,
   `cluster`, `embed`, `timeseries`, `dsp`, `ode`, `fields`, `pde`, `maps`, `control`, `sde`, `diffusion`, `nn`,
   `losses`, `ot`, `bandits`, `rl`), then `datasets` and `geometry`. `make check` runs the tests.
4. **Migrate.** The lab's views and quality-of-life improvements move into the site first; then figures migrate,
   one by one, by domain. A snapshot harness compares each figure's computed series before and
   after, within a tolerance. Known fixes (Cholesky floors, ESS, normal tails, the RNG switch) are expected
   differences and are reviewed visually. Didactic code that is itself the lesson may stay in its note, but it uses
   aifn primitives. The Python rename (`mlc` → `aifn`, `python/` → `aifn-py/`) is its own step within stage 4.

## 15. Decisions for the owner

1. Switch the random generator to keyed, counter-based streams (a one-off change in every random figure's exact draws).
2. Typed arrays inside aifn and plain arrays at the chart boundary.
3. Migration with numeric tolerance and visual review, rather than bit-identical output.
4. The scope of the PGM language and its first engines (§7), and which existing models must be reproduced first.
5. How far `nn` and `diffusion` go in v1 (§8), and how far `programming` goes (e.g. whether MILP by branch and bound
   on small problems is enough).
6. Tensor scope: ranks up to 4 and float64 by default, or a general n-dimensional type from the start.
