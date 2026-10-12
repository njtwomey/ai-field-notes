/**
 * SplineVM: a prototype of sparse Bayesian learning (the RVM's automatic relevance determination) on spline knots.
 * Local numerics while the design is explored; it moves into the engine once it settles.
 *
 * Each input dimension is rescaled to u_d = (x_d − lo_d)/span_d on [0, 1]. The latent function is a degree-p spline in
 * truncated-power form,
 *
 *   f(u) = Σ_e c_e u^e + Σ_j w_j B_j(u),   B_j(u) = Π_{d ∈ D_j} (u_d − t_jd)₊^p,
 *
 * where the polynomial part runs over the exponent vectors e with every e_d ≤ p, and a term j is a knot on one axis
 * (D_j = {d}: w_j is the jump in the p-th derivative along u_d at t_jd) or the product of knots on two axes (a corner
 * at (t_j1, t_j2), as in MARS). Regression observes y = f + ε, ε ~ N(0, 1/β); classification observes y ∈ {0, 1}
 * with P(y = 1) = σ(f), the logistic sigmoid, and replaces the posterior by its Laplace approximation at the mode
 * (Newton's method, as in Tipping's 2001 RVM classifier). The polynomial part has a fixed prior; each term weight
 * has its own precision, w_j ~ N(0, 1/α_j). Maximising the evidence does three things at once:
 *
 * - selection: α_j → ∞ switches term j off. Each α_j has a closed-form best value given the others (Tipping and Faul,
 *   2003): with its sparsity and quality factors s_j, q_j, α_j = s_j²/(q_j² − s_j) when q_j² > s_j and ∞ (the term
 *   is removed) otherwise. An iteration updates every α_j at once when that raises the objective, and otherwise makes
 *   the single update with the largest gain;
 * - weights: the posterior mean μ and covariance Σ of (c, w);
 * - locations: gradient ascent on the knot locations with α (and β) held fixed, using ∂L/∂Φ = e μᵀ − W Φ Σ, where
 *   e_n = ∂ ln p(y_n | f_n)/∂f_n at the mode and W = diag(w_n) its negative second derivative (β r_n and β for
 *   regression; y_n − σ_n and σ_n(1 − σ_n) for classification, where W is held fixed, so the classification gradient
 *   is approximate; a step is kept only when the objective rises).
 *
 * The objective is the log evidence minus λ per term: a prior p(K) ∝ e^(−λK) on the number of terms. λ = 0 is the
 * plain evidence, whose Occam factor alone sets the sparsity; λ > 0 asks each term to earn λ nats to stay
 * (λ = ½ ln N is BIC's price per parameter).
 */

export type Likelihood = 'gaussian' | 'bernoulli'

/** A basis term: the product over `dims` of (u_d − at_k)₊^p. */
export type Term = { dims: readonly number[]; at: number[] }

export type SplineVmOptions = {
  /** Candidate knots per axis, evenly spaced inside (0, 1) at the start. */
  candidates: number
  /** Two or more inputs: also start with a corner term at every pair of candidate knots on two axes. */
  interactions?: boolean
  /** Move the knots by the objective's gradient (false: selection among the fixed candidates only). */
  moveKnots: boolean
  /** The cost λ of a term, in nats (default 0: the plain evidence). */
  penalty?: number
  maxIter?: number
  /** Stop when the last `window` iterations together raise the objective by less than this (default 1e-2 nats). */
  tolerance?: number
  window?: number
}

/** One iteration of the fit. Locations are in u units; `ids` keep each term's identity across iterations. */
export type SplineVmState = {
  iter: number
  ids: number[]
  terms: Term[]
  alpha: number[]
  /** The posterior mean of each term's weight w_j. */
  weights: number[]
  /** The noise precision (regression; 1 and unused for classification). */
  beta: number
  logEvidence: number
  /** The log evidence minus λ per term: what the fit maximises. */
  objective: number
  /** Terms removed at this iteration (at their last location). */
  pruned: Term[]
  /** The largest knot move at this iteration (u). */
  moved: number
  converged: boolean
}

export type SplineVmProblem = {
  /** Inputs on [0, 1]^D, row-major N × D. */
  u: Float64Array
  n: number
  dims: number
  y: Float64Array
  lo: number[]
  span: number[]
  degree: number
  likelihood: Likelihood
  /** The polynomial part's exponent vectors. */
  poly: number[][]
}

const TINY = 1e-12
/** The polynomial part's prior precision: vague for regression, weakly informative for classification so that a
 * separable sample keeps a finite mode. */
const polyPrecision = (likelihood: Likelihood) => (likelihood === 'gaussian' ? 1e-6 : 1e-2)

/** `x`: one number per point (1-D) or one row per point. */
export function splineVmProblem(
  x: ArrayLike<number> | readonly ArrayLike<number>[],
  y: ArrayLike<number>,
  degree: number,
  likelihood: Likelihood = 'gaussian',
): SplineVmProblem {
  const rows: ArrayLike<number>[] =
    typeof x[0] === 'number'
      ? Array.from(x as ArrayLike<number>, (v) => [v])
      : Array.from(x as readonly ArrayLike<number>[])
  const n = rows.length
  const dims = rows[0]?.length ?? 1
  const lo = Array.from({ length: dims }, (_, d) => Math.min(...rows.map((r) => r[d])))
  const span = Array.from({ length: dims }, (_, d) => Math.max(...rows.map((r) => r[d])) - lo[d] || 1)
  const u = new Float64Array(n * dims)
  rows.forEach((r, i) => {
    for (let d = 0; d < dims; d++) u[i * dims + d] = (r[d] - lo[d]) / span[d]
  })
  // Exponent vectors with every entry ≤ p.
  let poly: number[][] = [[]]
  for (let d = 0; d < dims; d++) poly = poly.flatMap((e) => Array.from({ length: degree + 1 }, (_, k) => [...e, k]))
  return { u, n, dims, y: Float64Array.from(y), lo, span, degree, likelihood, poly }
}

/** Rescale points (one number or one row each) to the problem's u, row-major. */
export function toUnit(pr: SplineVmProblem, x: ArrayLike<number> | readonly ArrayLike<number>[]): Float64Array {
  const rows = typeof x[0] === 'number' ? Array.from(x as ArrayLike<number>, (v) => [v]) : (x as ArrayLike<number>[])
  const u = new Float64Array(rows.length * pr.dims)
  rows.forEach((r, i) => {
    for (let d = 0; d < pr.dims; d++) u[i * pr.dims + d] = (r[d] - pr.lo[d]) / pr.span[d]
  })
  return u
}

/** The design matrix [polynomial part, B_1, …, B_T] at the rows of `u` (row-major, rows × (P + T)). */
export function design(pr: SplineVmProblem, u: Float64Array, terms: readonly Term[]): Float64Array {
  const D = pr.dims
  const p = pr.degree
  const rows = u.length / D
  const P = pr.poly.length
  const m = P + terms.length
  const phi = new Float64Array(rows * m)
  for (let i = 0; i < rows; i++) {
    const row = i * m
    for (let k = 0; k < P; k++) {
      let v = 1
      for (let d = 0; d < D; d++) v *= u[i * D + d] ** pr.poly[k][d]
      phi[row + k] = v
    }
    for (let j = 0; j < terms.length; j++) {
      const { dims, at } = terms[j]
      let v = 1
      for (let k = 0; k < dims.length && v !== 0; k++) {
        const g = u[i * D + dims[k]] - at[k]
        v = g > 0 ? v * g ** p : 0
      }
      phi[row + P + j] = v
    }
  }
  return phi
}

export type Posterior = {
  mu: Float64Array
  sigma: Float64Array
  logEvidence: number
  phi: Float64Array
  /** ∂ ln p(y_n | f_n)/∂f_n at the mode, and its negative second derivative (per row). */
  e: Float64Array
  w: Float64Array
  /** ‖y − Φμ‖² (regression). */
  rss: number
}

/** H = A + ΦᵀWΦ for row weights `w` and prior precisions `prec`. */
function precisionMatrix(phi: Float64Array, n: number, m: number, w: ArrayLike<number>, prec: (k: number) => number) {
  const h = new Float64Array(m * m)
  for (let i = 0; i < n; i++) {
    const row = i * m
    const wi = w[i]
    for (let r = 0; r < m; r++) {
      const v = phi[row + r]
      if (v === 0) continue
      for (let c = r; c < m; c++) h[r * m + c] += wi * v * phi[row + c]
    }
  }
  for (let r = 0; r < m; r++) {
    h[r * m + r] += prec(r)
    for (let c = 0; c < r; c++) h[r * m + c] = h[c * m + r]
  }
  return h
}

const logSigmoid = (f: number) => (f >= 0 ? -Math.log1p(Math.exp(-f)) : f - Math.log1p(Math.exp(f)))
const sigmoid = (f: number) => (f >= 0 ? 1 / (1 + Math.exp(-f)) : Math.exp(f) / (1 + Math.exp(f)))

/**
 * The weight posterior (exact for regression, Laplace for classification) and the log evidence for terms, their
 * precisions α and the noise precision β. `warm` starts Newton's method (classification) from a mode.
 */
export function posterior(
  pr: SplineVmProblem,
  terms: readonly Term[],
  alpha: readonly number[],
  beta: number,
  warm?: ArrayLike<number>,
): Posterior {
  const { y, n } = pr
  const P = pr.poly.length
  const m = P + terms.length
  const phi = design(pr, pr.u, terms)
  const a0 = polyPrecision(pr.likelihood)
  const prec = (k: number) => (k < P ? a0 : alpha[k - P])
  let sumLogA = 0
  for (let k = 0; k < m; k++) sumLogA += Math.log(prec(k))

  if (pr.likelihood === 'gaussian') {
    const w = new Float64Array(n).fill(beta)
    const h = precisionMatrix(phi, n, m, w, prec)
    const rhs = new Float64Array(m)
    for (let i = 0; i < n; i++) for (let r = 0; r < m; r++) rhs[r] += beta * phi[i * m + r] * y[i]
    const l = cholesky(h, m)
    const mu = cholSolve(l, m, rhs)
    const sigma = cholInverse(l, m)
    const e = new Float64Array(n)
    let rss = 0
    for (let i = 0; i < n; i++) {
      let f = 0
      for (let k = 0; k < m; k++) f += phi[i * m + k] * mu[k]
      rss += (y[i] - f) ** 2
      e[i] = beta * (y[i] - f)
    }
    let muAmu = 0
    for (let k = 0; k < m; k++) muAmu += prec(k) * mu[k] ** 2
    const logEvidence =
      0.5 * (n * Math.log(beta) + sumLogA - logDet(l, m) - beta * rss - muAmu - n * Math.log(2 * Math.PI))
    return { mu, sigma, logEvidence, phi, e, w, rss }
  }

  // Bernoulli: Newton's method for the mode of ln p(y | w) − ½ wᵀAw, with step halving.
  let mu = warm && warm.length === m ? Float64Array.from(warm) : new Float64Array(m)
  const objective = (wt: Float64Array) => {
    let ll = 0
    for (let i = 0; i < n; i++) {
      let f = 0
      for (let k = 0; k < m; k++) f += phi[i * m + k] * wt[k]
      ll += y[i] > 0.5 ? logSigmoid(f) : logSigmoid(-f)
    }
    let pen = 0
    for (let k = 0; k < m; k++) pen += prec(k) * wt[k] ** 2
    return ll - 0.5 * pen
  }
  let obj = objective(mu)
  let l: Float64Array = new Float64Array(0)
  const e = new Float64Array(n)
  const w = new Float64Array(n)
  const curvature = () => {
    for (let i = 0; i < n; i++) {
      let f = 0
      for (let k = 0; k < m; k++) f += phi[i * m + k] * mu[k]
      const s = sigmoid(f)
      e[i] = y[i] - s
      w[i] = Math.max(s * (1 - s), 1e-10)
    }
    l = cholesky(precisionMatrix(phi, n, m, w, prec), m)
  }
  for (let it = 0; it < 50; it++) {
    curvature()
    const g = new Float64Array(m)
    for (let i = 0; i < n; i++) for (let k = 0; k < m; k++) g[k] += phi[i * m + k] * e[i]
    for (let k = 0; k < m; k++) g[k] -= prec(k) * mu[k]
    const delta = cholSolve(l, m, g)
    let size = 0
    for (const d of delta) size = Math.max(size, Math.abs(d))
    if (size < 1e-6) break
    let step = 1
    let accepted = false
    for (let tries = 0; tries < 20; tries++, step /= 2) {
      const trial = mu.map((v, k) => v + step * delta[k])
      const o = objective(trial)
      if (o >= obj) {
        mu = trial
        obj = o
        accepted = true
        break
      }
    }
    if (!accepted) break
  }
  curvature()
  const sigma = cholInverse(l, m)
  // Laplace: ln p(y | α) ≈ ln p(y | μ) − ½ μᵀAμ + ½ ln|A| − ½ ln|H|.
  const logEvidence = obj + 0.5 * sumLogA - 0.5 * logDet(l, m)
  return { mu, sigma, logEvidence, phi, e, w, rss: NaN }
}

/** ℓ(α) = ½[ln α − ln(α + s) + q²/(α + s)], the part of the evidence that depends on one term's α (0 at ∞). */
const ell = (a: number, s: number, q: number) =>
  a === Infinity ? 0 : 0.5 * (Math.log(a) - Math.log(a + s) + (q * q) / (a + s))

/**
 * Each term's best α given the others under the penalised objective, and the gain of moving to it. Every term is in
 * the model, so its marginal posterior w_j ~ N(q_j/(α_j + s_j), 1/(α_j + s_j)) gives s_j = 1/Σ_jj − α_j and
 * q_j = μ_j/Σ_jj: stable for the nearly collinear cubic columns, where S_j = βφᵀφ − (βΦᵀφ)ᵀΣ(βΦᵀφ) cancels
 * catastrophically. A term stays only when its best ℓ exceeds the penalty λ. For classification these are the
 * factors of the Laplace approximation's Gaussian.
 */
export function bestAlphas(
  pr: SplineVmProblem,
  alpha: readonly number[],
  post: Pick<Posterior, 'mu' | 'sigma'>,
  penalty = 0,
) {
  const P = pr.poly.length
  const m = P + alpha.length
  return alpha.map((a, j) => {
    const c = P + j
    const sjj = post.sigma[c * m + c]
    const s = Math.max(1 / sjj - a, TINY)
    const q = post.mu[c] / sjj
    const theta = q * q - s
    const star = theta > 0 ? (s * s) / theta : Infinity
    const next = star < Infinity && ell(star, s, q) > penalty ? star : Infinity
    const value = (b: number) => (b === Infinity ? 0 : ell(b, s, q) - penalty)
    return { next, gain: value(next) - value(a) }
  })
}

/** ∂L/∂(knot locations), one array per term (exact for regression; W held fixed for classification). */
export function knotGradient(pr: SplineVmProblem, terms: readonly Term[], post: Posterior): number[][] {
  const { u, n, dims: D, degree: p } = pr
  const P = pr.poly.length
  const m = P + terms.length
  const { mu, sigma, phi, e, w } = post
  const g = terms.map((t) => t.at.map(() => 0))
  const gaps = new Float64Array(2)
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < terms.length; j++) {
      const { dims, at } = terms[j]
      let inside = true
      for (let k = 0; k < dims.length; k++) {
        gaps[k] = u[i * D + dims[k]] - at[k]
        if (gaps[k] <= 0) inside = false
      }
      if (!inside) continue
      const col = P + j
      let ps = 0
      for (let k = 0; k < m; k++) ps += phi[i * m + k] * sigma[k * m + col]
      const dLdPhi = e[i] * mu[col] - w[i] * ps
      for (let k = 0; k < dims.length; k++) {
        // ∂/∂at_k Π_l gap_l^p = −p gap_k^(p−1) Π_{l≠k} gap_l^p
        let d = -p * gaps[k] ** (p - 1)
        for (let l = 0; l < dims.length; l++) if (l !== k) d *= gaps[l] ** p
        g[j][k] += dLdPhi * d
      }
    }
  }
  return g
}

/** The starting terms: evenly spaced knots on every axis, and (with `interactions`) a corner at every pair. */
function candidateTerms(pr: SplineVmProblem, opts: SplineVmOptions): Term[] {
  const k0 = opts.candidates
  const knots = Array.from({ length: k0 }, (_, j) => (j + 1) / (k0 + 1))
  const terms: Term[] = []
  for (let d = 0; d < pr.dims; d++) for (const t of knots) terms.push({ dims: [d], at: [t] })
  if (opts.interactions && pr.dims >= 2)
    for (let d1 = 0; d1 < pr.dims; d1++)
      for (let d2 = d1 + 1; d2 < pr.dims; d2++)
        for (const t1 of knots) for (const t2 of knots) terms.push({ dims: [d1, d2], at: [t1, t2] })
  return terms
}

/** The whole fit, one state per iteration: update α, re-estimate β (regression), then one knot step. */
export function splineVmFit(pr: SplineVmProblem, opts: SplineVmOptions): SplineVmState[] {
  const P = pr.poly.length
  const gaussian = pr.likelihood === 'gaussian'
  const lambda = opts.penalty ?? 0
  const maxIter = opts.maxIter ?? 300
  const tolerance = opts.tolerance ?? 1e-2
  const window = opts.window ?? 10
  const n = pr.n
  let ybar = 0
  for (const v of pr.y) ybar += v / n
  let vy = 0
  for (const v of pr.y) vy += (v - ybar) ** 2 / n
  vy = gaussian ? vy || 1 : 1

  let terms = candidateTerms(pr, opts)
  let ids = terms.map((_, j) => j)
  let alpha = terms.map(() => 1 / vy)
  let beta = gaussian ? 10 / vy : 1
  let step = 0.02
  let post = posterior(pr, terms, alpha, beta)
  const objectiveOf = (pst: Posterior, count: number) => pst.logEvidence - lambda * count
  const snapshot = (iter: number, pruned: Term[], moved: number, converged: boolean): SplineVmState => ({
    iter,
    ids,
    terms: terms.map((t) => ({ dims: t.dims, at: [...t.at] })),
    alpha,
    weights: Array.from(post.mu.slice(P)),
    beta,
    logEvidence: post.logEvidence,
    objective: objectiveOf(post, terms.length),
    pruned,
    moved,
    converged,
  })
  const states: SplineVmState[] = [snapshot(0, [], 0, false)]
  /** The current mode restricted to the kept terms, to warm-start Newton's method. */
  const warmFor = (keep: boolean[]) => [...post.mu.slice(0, P), ...post.mu.slice(P).filter((_, j) => keep[j])]
  const keepOnly = (keep: boolean[]) => {
    terms = terms.filter((_, j) => keep[j])
    ids = ids.filter((_, j) => keep[j])
    alpha = alpha.filter((_, j) => keep[j])
  }

  for (let iter = 1; iter <= maxIter; iter++) {
    const before = objectiveOf(post, terms.length)
    let pruned: Term[] = []

    // 1 · Selection and weights: every α_j at once if the objective rises, else the single best update.
    if (terms.length > 0) {
      const best = bestAlphas(pr, alpha, post, lambda)
      const all = best.map((b) => b.next)
      const keepAll = all.map((a) => a < Infinity)
      const kept = terms.filter((_, j) => keepAll[j])
      const trial = posterior(
        pr,
        kept,
        all.filter((_, j) => keepAll[j]),
        beta,
        warmFor(keepAll),
      )
      if (objectiveOf(trial, kept.length) > objectiveOf(post, terms.length)) {
        pruned = terms.filter((_, j) => !keepAll[j])
        alpha = all
        keepOnly(keepAll)
        post = trial
      } else {
        const j = best.reduce((b, v, i) => (v.gain > best[b].gain ? i : b), 0)
        if (best[j].gain > 0) {
          const keep = terms.map((_, i) => i !== j || best[j].next < Infinity)
          const warm = warmFor(keep)
          if (best[j].next === Infinity) {
            pruned = [terms[j]]
            keepOnly(keep)
          } else alpha = alpha.map((a, i) => (i === j ? best[j].next : a))
          // Exact for regression; under the Laplace approximation the single update may not raise the objective.
          post = posterior(pr, terms, alpha, beta, warm)
        }
      }
    }

    // 2 · Noise (regression): β ← (N − Σγ)/‖r‖², kept when it raises the evidence.
    if (gaussian) {
      const m = P + terms.length
      const a0 = polyPrecision(pr.likelihood)
      let gammaSum = 0
      for (let k = 0; k < m; k++) gammaSum += 1 - (k < P ? a0 : alpha[k - P]) * post.sigma[k * m + k]
      const nb = Math.max(n - gammaSum, 1) / Math.max(post.rss, TINY * vy)
      const tp = posterior(pr, terms, alpha, nb)
      if (tp.logEvidence > post.logEvidence) {
        beta = nb
        post = tp
      }
    }

    // 3 · Locations: one step along the gradient (largest coordinate move `step`), with backtracking.
    let moved = 0
    if (opts.moveKnots && terms.length > 0) {
      const g = knotGradient(pr, terms, post)
      let gmax = 0
      for (const gj of g) for (const v of gj) gmax = Math.max(gmax, Math.abs(v))
      if (gmax > 0) {
        for (let tries = 0; tries < 6; tries++) {
          const trial = terms.map((t, j) => ({
            dims: t.dims,
            at: t.at.map((a, k) => Math.min(1 - 1e-4, Math.max(1e-4, a + (step * g[j][k]) / gmax))),
          }))
          const tp = posterior(pr, trial, alpha, beta, post.mu)
          if (tp.logEvidence > post.logEvidence) {
            for (let j = 0; j < terms.length; j++)
              for (let k = 0; k < terms[j].at.length; k++)
                moved = Math.max(moved, Math.abs(trial[j].at[k] - terms[j].at[k]))
            terms = trial
            post = tp
            step = Math.min(step * 1.3, 0.05)
            break
          }
          step *= 0.5
        }
        step = Math.max(step, 1e-4)
      }
    }

    // Converged when an iteration gains almost nothing, or the last `window` together gain under `tolerance`.
    const now = objectiveOf(post, terms.length)
    const back = states.length >= window ? states[states.length - window].objective : -Infinity
    const converged = Math.abs(now - before) < 1e-7 || now - back < tolerance
    states.push(snapshot(iter, pruned, moved, converged))
    if (converged) break
  }
  return states
}

/**
 * The latent posterior f at points x (one number or one row each) for a state of the fit: mean and variance (plus
 * the noise 1/β with `noise`, for regression). For classification, `probability` is the moderated
 * P(y = 1) ≈ σ(μ/√(1 + πv/8)) (MacKay, 1992).
 */
export function predict(
  pr: SplineVmProblem,
  s: SplineVmState,
  x: ArrayLike<number> | readonly ArrayLike<number>[],
  noise = true,
) {
  const post = posterior(pr, s.terms, s.alpha, s.beta)
  const u = toUnit(pr, x)
  const rows = u.length / pr.dims
  const m = pr.poly.length + s.terms.length
  const phi = design(pr, u, s.terms)
  const mean = new Float64Array(rows)
  const variance = new Float64Array(rows)
  const probability = new Float64Array(rows)
  const gaussian = pr.likelihood === 'gaussian'
  for (let i = 0; i < rows; i++) {
    let f = 0
    let v = 0
    for (let r = 0; r < m; r++) {
      const a = phi[i * m + r]
      f += a * post.mu[r]
      if (a === 0) continue
      for (let c = 0; c < m; c++) v += a * post.sigma[r * m + c] * phi[i * m + c]
    }
    mean[i] = f
    variance[i] = v + (noise && gaussian ? 1 / s.beta : 0)
    probability[i] = sigmoid(f / Math.sqrt(1 + (Math.PI * v) / 8))
  }
  return { mean, variance, probability }
}

// ── Small dense Cholesky helpers (row-major m × m) ──

function logDet(l: Float64Array, m: number) {
  let s = 0
  for (let k = 0; k < m; k++) s += 2 * Math.log(l[k * m + k])
  return s
}

/**
 * Cholesky factor L (A = L Lᵀ). The truncated-power basis is badly conditioned (columns for knots near the right end
 * are tiny), so A is scaled to unit diagonal first, S = D⁻¹AD⁻¹, factored without jitter, and L = D L_S is returned.
 * An earlier version floored each pivot at 10⁻¹⁰ of its diagonal, which on this basis distorted the posterior.
 */
export function cholesky(a: Float64Array, m: number): Float64Array {
  const d = Array.from({ length: m }, (_, k) => Math.sqrt(Math.max(a[k * m + k], 1e-300)))
  const l = new Float64Array(m * m)
  for (let j = 0; j < m; j++) {
    let piv = 1
    for (let k = 0; k < j; k++) piv -= l[j * m + k] ** 2
    const ljj = Math.sqrt(Math.max(piv, 1e-300))
    l[j * m + j] = ljj
    for (let i = j + 1; i < m; i++) {
      let v = a[i * m + j] / (d[i] * d[j])
      for (let k = 0; k < j; k++) v -= l[i * m + k] * l[j * m + k]
      l[i * m + j] = v / ljj
    }
  }
  for (let i = 0; i < m; i++) for (let j = 0; j <= i; j++) l[i * m + j] *= d[i]
  return l
}

export function cholSolve(l: Float64Array, m: number, b: ArrayLike<number>): Float64Array {
  const z = new Float64Array(m)
  for (let i = 0; i < m; i++) {
    let s = b[i]
    for (let k = 0; k < i; k++) s -= l[i * m + k] * z[k]
    z[i] = s / l[i * m + i]
  }
  const x = new Float64Array(m)
  for (let i = m - 1; i >= 0; i--) {
    let s = z[i]
    for (let k = i + 1; k < m; k++) s -= l[k * m + i] * x[k]
    x[i] = s / l[i * m + i]
  }
  return x
}

function cholInverse(l: Float64Array, m: number): Float64Array {
  const inv = new Float64Array(m * m)
  const e = new Float64Array(m)
  for (let c = 0; c < m; c++) {
    e.fill(0)
    e[c] = 1
    const col = cholSolve(l, m, e)
    for (let r = 0; r < m; r++) inv[r * m + c] = col[r]
  }
  return inv
}
