import { describe, expect, it } from 'vitest'
import {
  blobs,
  checkerboard,
  classCounts,
  decodeRecipe,
  describeRecipe,
  encodeRecipe,
  flippedMask,
  gaussians,
  moons,
  recipe,
  regression1d,
  rotation2d,
  split,
  withCovariateShift,
  withLabelNoise,
  withMissing,
  withNuisanceFeatures,
  withOutliers,
  withPrevalence,
  withTransform,
  xor,
  type ClassificationTruth,
  type Dataset,
  type DatasetRecipe,
} from 'aifn/datasets'
import { stream } from 'aifn/random'
import { toFlat } from 'aifn/tensor'

const counts = (d: Dataset, k = 2) => {
  const c = new Array<number>(k).fill(0)
  toFlat(d.y!).forEach((v) => c[v]++)
  return c
}
const truthOf = (d: Dataset) => d.meta.truth as ClassificationTruth
const rows = (d: Dataset) => {
  const [n, dim] = d.x.shape
  const x = toFlat(d.x)
  return Array.from({ length: n }, (_, i) => x.slice(i * dim, (i + 1) * dim))
}

describe('class sizes', () => {
  it('classCounts uses the largest remainder and sums exactly', () => {
    expect(classCounts(100, [1, 2, 3])).toEqual([17, 33, 50])
    expect(classCounts(10, [1, 1, 1])).toEqual([4, 3, 3])
    expect(classCounts(7, [0.5, 0.5])).toEqual([4, 3])
    expect(classCounts(0, [1, 2])).toEqual([0, 0])
  })

  it('every labelled generator gives exact counts for a prevalence or weights', () => {
    expect(counts(moons(stream(1), { n: 400, prevalence: 0.2 }))).toEqual([320, 80])
    expect(counts(xor(stream(1), { n: 101, prevalence: 0.3 }))).toEqual([71, 30])
    expect(counts(xor(stream(1), { n: 50, prevalence: 0.1, kind: 'gaussian' }))).toEqual([45, 5])
    expect(counts(checkerboard(stream(1), { n: 90, classWeights: [2, 1] }))).toEqual([60, 30])
    expect(counts(blobs(stream(1), { n: 100, classWeights: [1, 2, 3] }), 3)).toEqual([17, 33, 50])
    expect(counts(gaussians(stream(1), { n: [10, 20] }))).toEqual([10, 20])
  })

  it('class-conditional xor and checkerboard keep their labelling rule', () => {
    const d = xor(stream(2), { n: 200, prevalence: 0.25 })
    const x = toFlat(d.x)
    toFlat(d.y!).forEach((l, i) => expect(l).toBe(x[2 * i] * x[2 * i + 1] < 0 ? 1 : 0))
    const c = checkerboard(stream(2), { n: 200, prevalence: 0.7, tiles: 3 })
    const xc = toFlat(c.x)
    toFlat(c.y!).forEach((l, i) => expect(l).toBe((Math.floor(xc[2 * i]) + Math.floor(xc[2 * i + 1])) % 2))
  })

  it('withPrevalence hits the target exactly by subsampling and oversampling', () => {
    const d = moons(stream(3), { n: 400 })
    expect(counts(withPrevalence(stream(4), d, 0.2))).toEqual([200, 50])
    expect(counts(withPrevalence(stream(4), d, 0.2, { method: 'oversample' }))).toEqual([800, 200])
    expect(counts(withPrevalence(stream(4), d, 0.1, { n: 100 }))).toEqual([90, 10])
  })
})

describe('label noise', () => {
  it('symmetric flips happen at the stated rate, and clean labels are kept', () => {
    const d = blobs(stream(5), { n: 20000, centers: 2, separation: 2 })
    const noisy = withLabelNoise(stream(6), d, { rate: 0.1 })
    const f = flippedMask(noisy)
    const rate = f.reduce((a, b) => a + b, 0) / f.length
    expect(Math.abs(rate - 0.1)).toBeLessThan(0.01)
    expect(toFlat(noisy.meta.cleanLabels!)).toEqual(toFlat(d.y!))
  })

  it('class-conditional flips follow the matrix row by row', () => {
    const d = blobs(stream(7), { n: 20000, centers: 2, separation: 2 })
    const noisy = withLabelNoise(stream(8), d, {
      matrix: [
        [0.9, 0.1],
        [0.3, 0.7],
      ],
    })
    const clean = toFlat(d.y!)
    const f = flippedMask(noisy)
    const rate = (j: number) => {
      const idx = clean.map((c, i) => (c === j ? i : -1)).filter((i) => i >= 0)
      return idx.reduce((a, i) => a + f[i], 0) / idx.length
    }
    expect(Math.abs(rate(0) - 0.1)).toBeLessThan(0.01)
    expect(Math.abs(rate(1) - 0.3)).toBeLessThan(0.015)
  })

  it('symmetric noise maps the Bayes error e to ρ + (1 − 2ρ) e', () => {
    const d = gaussians(stream(9), { n: 100, separation: 2 })
    const e = truthOf(d).bayesError
    const noisy = truthOf(withLabelNoise(stream(10), d, { rate: 0.1 }))
    expect(noisy.bayesErrorMethod).toBe('closed form')
    expect(noisy.bayesError).toBeCloseTo(0.1 + 0.8 * e, 12)
    expect(noisy.probability([0.3, 0.1])).toBeCloseTo(0.1 + 0.8 * truthOf(d).probability([0.3, 0.1]), 12)
  })
})

describe('known truth', () => {
  it('the Bayes posterior matches Monte Carlo label frequencies for Gaussian classes', () => {
    const d = gaussians(stream(11), {
      n: 40000,
      prevalence: 0.3,
      means: [
        [0, 0],
        [1.5, 0.5],
      ],
      covariances: [
        [
          [1, 0.3],
          [0.3, 1],
        ],
        [
          [0.5, 0],
          [0, 2],
        ],
      ],
    })
    const t = truthOf(d)
    const y = toFlat(d.y!)
    const bins = Array.from({ length: 10 }, () => ({ p: 0, y: 0, n: 0 }))
    rows(d).forEach((r, i) => {
      const p = t.probability(r)
      const b = bins[Math.min(9, Math.floor(p * 10))]
      b.p += p
      b.y += y[i]
      b.n++
    })
    for (const b of bins.filter((b) => b.n > 500)) expect(Math.abs(b.y / b.n - b.p / b.n)).toBeLessThan(0.03)
  })

  it('the Bayes error of two 1-D Gaussians matches quadrature (closed form and Monte Carlo)', () => {
    const quad = (m: number[], sd: number[], pi: number[]) => {
      const pdf = (x: number, j: number) =>
        Math.exp(-0.5 * ((x - m[j]) / sd[j]) ** 2) / (sd[j] * Math.sqrt(2 * Math.PI))
      let s = 0
      const h = 1e-3
      for (let x = -15; x <= 15; x += h) s += Math.min(pi[0] * pdf(x, 0), pi[1] * pdf(x, 1)) * h
      return s
    }
    const equal = truthOf(
      gaussians(stream(12), { n: 10, prevalence: 0.3, means: [[0], [1.7]], covariances: [[[1]], [[1]]] }),
    )
    expect(equal.bayesErrorMethod).toBe('closed form')
    expect(equal.bayesError).toBeCloseTo(quad([0, 1.7], [1, 1], [0.7, 0.3]), 5)
    const unequal = truthOf(
      gaussians(stream(12), { n: 10, prevalence: 0.4, means: [[0], [1.5]], covariances: [[[1]], [[0.25]]] }),
    )
    expect(unequal.bayesErrorMethod).toBe('monte carlo')
    const q = quad([0, 1.5], [1, 0.5], [0.6, 0.4])
    expect(Math.abs(unequal.bayesError - q)).toBeLessThan(4 * unequal.bayesErrorSe + 1e-3)
  })

  it('a prevalence change gives the posterior under the new priors', () => {
    const d = gaussians(stream(13), { n: 400, separation: 2 })
    const shifted = truthOf(withPrevalence(stream(14), d, 0.2))
    const fresh = truthOf(gaussians(stream(13), { n: 10, separation: 2, prevalence: 0.2 }))
    for (const x of [
      [0, 0],
      [0.8, -1],
      [-1.5, 2],
    ])
      expect(shifted.probability(x)).toBeCloseTo(fresh.probability(x), 10)
    expect(shifted.prevalence[1]).toBeCloseTo(0.2, 12)
    expect(shifted.bayesError).toBeCloseTo(fresh.bayesError, 12)
  })

  it('covariate shift keeps the posterior; transforms and nuisance features carry it along', () => {
    const d = moons(stream(15), { n: 600, noise: 0.2 })
    const t = truthOf(d)
    const x = [0.4, 0.2]
    expect(truthOf(withCovariateShift(stream(16), d)).probability(x)).toBeCloseTo(t.probability(x), 10)
    const r = rotation2d(0.7)
    const moved = truthOf(withTransform(d, r, [1, -2]))
    const x2 = [r[0][0] * x[0] + r[0][1] * x[1] + 1, r[1][0] * x[0] + r[1][1] * x[1] - 2]
    expect(moved.probability(x2)).toBeCloseTo(t.probability(x), 10)
    const wide = withNuisanceFeatures(stream(17), d, { count: 3 })
    expect(wide.x.shape).toEqual([600, 5])
    expect(truthOf(wide).probability([...x, 5, -3, 1])).toBeCloseTo(t.probability(x), 10)
  })

  it('moons have a small Monte Carlo Bayes error that grows with the noise', () => {
    const low = truthOf(moons(stream(18), { n: 10, noise: 0.1 })).bayesError
    const high = truthOf(moons(stream(18), { n: 10, noise: 0.4 })).bayesError
    expect(low).toBeLessThan(0.01)
    expect(high).toBeGreaterThan(0.08)
  })

  it('regression generators carry the mean function and noise sd', () => {
    const r = regression1d(stream(19), { n: 20, fn: 'sine', noise: 0.3 })
    const t = r.meta.truth!
    expect(t.kind).toBe('regression')
    if (t.kind === 'regression') {
      expect(t.mean([1])).toBeCloseTo(Math.sin(1), 14)
      expect(t.noiseSd).toBe(0.3)
      expect(t.bayesRisk).toBeCloseTo(0.09, 14)
    }
  })
})

describe('modifiers', () => {
  it('outliers, missing values and splits are marked and exact', () => {
    const d = moons(stream(20), { n: 200 })
    const o = withOutliers(stream(21), d, { fraction: 0.05 })
    expect(toFlat(o.meta.outliers!).reduce((a, b) => a + b, 0)).toBe(10)
    const m = withMissing(stream(22), d, { rate: 0.2, mechanism: 'mar' })
    const mask = toFlat(m.meta.missing!)
    expect(mask.filter((_, i) => i % 2 === 0).every((v) => v === 0)).toBe(true)
    toFlat(m.x).forEach((v, i) => expect(Number.isNaN(v)).toBe(mask[i] === 1))
    const { train, test } = split(stream(23), withPrevalence(stream(24), d, 0.2), { test: 0.3 })
    expect(counts(test)).toEqual([30, 8])
    expect(counts(train)).toEqual([70, 17])
  })
})

describe('recipes', () => {
  const r: DatasetRecipe = {
    base: 'moons',
    n: 400,
    prevalence: 0.2,
    noise: 0.2,
    labelNoise: 0.05,
    outliers: 0.02,
    nuisance: 2,
    missing: { rate: 0.1, mechanism: 'mnar' },
    seed: 7,
  }

  it('round-trips through JSON and a URL string', () => {
    const back = decodeRecipe(encodeRecipe(r))
    expect(back).toEqual(r)
    const a = recipe(r)
    const b = recipe(back)
    expect(toFlat(a.x)).toEqual(toFlat(b.x))
    expect(toFlat(a.y!)).toEqual(toFlat(b.y!))
    expect(counts(recipe({ ...r, labelNoise: 0 }))).toEqual([320, 80])
    expect(a.meta.recipe!.map((s) => s.op)).toEqual(['moons', 'outliers', 'nuisance', 'labelNoise', 'missing'])
  })

  it('is deterministic in its seed', () => {
    expect(toFlat(recipe(r).x)).toEqual(toFlat(recipe(r).x))
    expect(toFlat(recipe({ ...r, seed: 8 }).x)).not.toEqual(toFlat(recipe(r).x))
    // Changing one step leaves the base draws alone.
    const base = toFlat(recipe({ base: 'moons', n: 100, seed: 1 }).x)
    const noisy = recipe({ base: 'moons', n: 100, seed: 1, labelNoise: 0.2 })
    expect(toFlat(noisy.x)).toEqual(base)
  })

  it('describes itself in one line', () => {
    expect(describeRecipe(r)).toBe(
      '400 points of moons (noise 0.2), 20% in class 1, 2% outliers, 2 nuisance features, 5% labels flipped, 10% missing (MNAR), seed 7',
    )
  })
})
