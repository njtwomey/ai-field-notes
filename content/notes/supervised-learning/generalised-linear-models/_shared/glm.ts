/**
 * Generalised linear models with one feature, fitted by iteratively reweighted least squares (IRLS), for the figures
 * in this category. Two coefficients only, so each weighted least-squares step is a 2 × 2 solve.
 */

export type FamilyName = 'gaussian' | 'poisson' | 'binomial' | 'gamma'
export type LinkName = 'identity' | 'log' | 'logit' | 'cloglog' | 'inverse' | 'sqrt'

export type Link = {
  /** η = g(μ). */
  g: (mu: number) => number
  /** μ = g⁻¹(η). */
  inv: (eta: number) => number
  /** dμ/dη at η. */
  dmu: (eta: number) => number
}

export const LINKS: Record<LinkName, Link> = {
  identity: { g: (m) => m, inv: (e) => e, dmu: () => 1 },
  log: { g: Math.log, inv: Math.exp, dmu: Math.exp },
  logit: {
    g: (m) => Math.log(m / (1 - m)),
    inv: (e) => 1 / (1 + Math.exp(-e)),
    dmu: (e) => {
      const p = 1 / (1 + Math.exp(-e))
      return p * (1 - p)
    },
  },
  cloglog: {
    g: (m) => Math.log(-Math.log(1 - m)),
    inv: (e) => 1 - Math.exp(-Math.exp(e)),
    dmu: (e) => Math.exp(e - Math.exp(e)),
  },
  inverse: { g: (m) => 1 / m, inv: (e) => 1 / e, dmu: (e) => -1 / (e * e) },
  sqrt: { g: Math.sqrt, inv: (e) => e * e, dmu: (e) => 2 * e },
}

export type Family = {
  label: string
  /** Variance function V(μ): var(y) = φ V(μ). */
  variance: (mu: number) => number
  /** Unit deviance d(y, μ); the deviance is its sum. */
  unitDeviance: (y: number, mu: number) => number
  /** Whether μ lies in the family's mean space. */
  valid: (mu: number) => boolean
  canonical: LinkName
  links: readonly LinkName[]
}

const xlogy = (y: number, r: number) => (y === 0 ? 0 : y * Math.log(r))

export const FAMILIES: Record<FamilyName, Family> = {
  gaussian: {
    label: 'Gaussian',
    variance: () => 1,
    unitDeviance: (y, mu) => (y - mu) ** 2,
    valid: () => true,
    canonical: 'identity',
    links: ['identity', 'log'],
  },
  poisson: {
    label: 'Poisson',
    variance: (mu) => mu,
    unitDeviance: (y, mu) => 2 * (xlogy(y, y / mu) - (y - mu)),
    valid: (mu) => mu > 0,
    canonical: 'log',
    links: ['log', 'identity', 'sqrt'],
  },
  binomial: {
    label: 'Bernoulli',
    variance: (mu) => mu * (1 - mu),
    unitDeviance: (y, mu) => 2 * (xlogy(y, y / mu) + xlogy(1 - y, (1 - y) / (1 - mu))),
    valid: (mu) => mu > 0 && mu < 1,
    canonical: 'logit',
    links: ['logit', 'cloglog'],
  },
  gamma: {
    label: 'gamma',
    variance: (mu) => mu * mu,
    unitDeviance: (y, mu) => 2 * (-Math.log(y / mu) + (y - mu) / mu),
    valid: (mu) => mu > 0,
    canonical: 'inverse',
    links: ['inverse', 'log', 'identity'],
  },
}

export type Iterate = { beta: [number, number]; deviance: number }

export type IrlsResult = { iterates: Iterate[]; converged: boolean }

/** Deviance of fitted means for responses y. */
export function deviance(family: Family, y: number[], mu: number[]): number {
  return y.reduce((s, yi, i) => s + family.unitDeviance(yi, mu[i]), 0)
}

/**
 * Fit η = β₀ + β₁x by IRLS, starting from the constant fit μ = ȳ, which is valid for every family and link. A
 * step that leaves the mean space (non-canonical links can propose negative Poisson means) or raises the deviance is
 * halved towards the previous iterate. Returns every iterate, starting with the constant fit μ = ȳ.
 */
export function irls(x: number[], y: number[], family: Family, link: Link, maxIter = 25, tol = 1e-8): IrlsResult {
  const n = x.length
  const ybar = y.reduce((a, b) => a + b, 0) / n
  let eta = y.map(() => link.g(ybar))
  // Iterate 0 is the constant fit μ = ȳ: always in the mean space, and the fallback for step halving.
  let prev: [number, number] = [link.g(ybar), 0]
  let dev = deviance(
    family,
    y,
    y.map(() => ybar),
  )
  const iterates: Iterate[] = [{ beta: prev, deviance: dev }]
  for (let it = 0; it < maxIter; it++) {
    // Working response z and weights w from the current linear predictor.
    let s0 = 0,
      s1 = 0,
      s11 = 0,
      t0 = 0,
      t1 = 0
    for (let i = 0; i < n; i++) {
      const mu = link.inv(eta[i])
      const d = link.dmu(eta[i])
      const z = eta[i] + (y[i] - mu) / d
      const w = (d * d) / family.variance(mu)
      s0 += w
      s1 += w * x[i]
      s11 += w * x[i] * x[i]
      t0 += w * z
      t1 += w * z * x[i]
    }
    const det = s0 * s11 - s1 * s1
    let beta: [number, number] = [(s11 * t0 - s1 * t1) / det, (s0 * t1 - s1 * t0) / det]
    // Step halving towards the previous iterate until every mean is valid and the deviance does not rise.
    let mu = x.map((xi) => link.inv(beta[0] + beta[1] * xi))
    for (let h = 0; h < 30; h++) {
      const ok = mu.every(family.valid) && deviance(family, y, mu) <= dev * (1 + 1e-10)
      if (ok) break
      beta = [(beta[0] + prev[0]) / 2, (beta[1] + prev[1]) / 2]
      mu = x.map((xi) => link.inv(beta[0] + beta[1] * xi))
    }
    if (!mu.every(family.valid)) return { iterates, converged: false }
    const newDev = deviance(family, y, mu)
    iterates.push({ beta, deviance: newDev })
    eta = x.map((xi) => beta[0] + beta[1] * xi)
    if (Math.abs(newDev - dev) < tol * (Math.abs(newDev) + 0.1)) return { iterates, converged: true }
    dev = newDev
    prev = beta
  }
  return { iterates, converged: false }
}

/** Seeded data from each family's true model on x ∈ [−2, 2]. `uniform` and `normal` come from rng(seed). */
export function simulate(
  family: FamilyName,
  n: number,
  r: { uniform: () => number; normal: () => number },
): { x: number[]; y: number[]; truth: (x: number) => number } {
  const truth: Record<FamilyName, (x: number) => number> = {
    gaussian: (x) => 1 + 0.8 * x,
    poisson: (x) => Math.exp(0.5 + 0.6 * x),
    binomial: (x) => 1 / (1 + Math.exp(-(0.3 + 1.5 * x))),
    gamma: (x) => Math.exp(0.8 + 0.5 * x),
  }
  const f = truth[family]
  const x = Array.from({ length: n }, () => -2 + 4 * r.uniform())
  const y = x.map((xi) => {
    const mu = f(xi)
    switch (family) {
      case 'gaussian':
        return mu + 0.7 * r.normal()
      case 'poisson': {
        // Knuth's multiplication method; means stay below about 10.
        const limit = Math.exp(-mu)
        let k = 0
        let p = r.uniform()
        while (p > limit) {
          k++
          p *= r.uniform()
        }
        return k
      }
      case 'binomial':
        return r.uniform() < mu ? 1 : 0
      case 'gamma':
        // Shape 2 (dispersion φ = 1/2): the sum of two exponentials with mean μ/2 each.
        return (-Math.log(Math.max(r.uniform(), 1e-12)) - Math.log(Math.max(r.uniform(), 1e-12))) * (mu / 2)
    }
  })
  return { x, y, truth: f }
}
