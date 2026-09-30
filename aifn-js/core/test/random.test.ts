import { describe, expect, it } from 'vitest'
import * as R from 'aifn/random'
import { chiSquareSf } from 'aifn/numerics/special'
import { fromRows, tensor, toFlat, toRows, type Tensor } from 'aifn/foundation/tensor'
import { fixture } from './fixtures'

// All tests use fixed streams, so they are deterministic. Thresholds are set at significance levels near 1e-3 (or
// several standard errors), so a correct sampler passes for almost any seed and the fixed seeds cannot mask a bug.

type Continuous = { name: string; args: Record<string, number>; p: number[]; x: number[]; mean: number; var: number }
type Discrete = { name: string; args: Record<string, number>; k: number[]; pmf: number[]; mean: number; var: number }
const ref = fixture<{ continuous: Continuous[]; discrete: Discrete[] }>('random')

const N = 20_000

function moments(xs: ArrayLike<number>): { mean: number; variance: number } {
  let m = 0
  for (let i = 0; i < xs.length; i++) m += xs[i]
  m /= xs.length
  let v = 0
  for (let i = 0; i < xs.length; i++) v += (xs[i] - m) ** 2
  return { mean: m, variance: v / (xs.length - 1) }
}

function correlation(a: ArrayLike<number>, b: ArrayLike<number>): number {
  const ma = moments(a).mean
  const mb = moments(b).mean
  let sab = 0
  let saa = 0
  let sbb = 0
  for (let i = 0; i < a.length; i++) {
    sab += (a[i] - ma) * (b[i] - mb)
    saa += (a[i] - ma) ** 2
    sbb += (b[i] - mb) ** 2
  }
  return sab / Math.sqrt(saa * sbb)
}

/** Pearson chi-square p-value for observed counts against expected probabilities, pooling cells with expectation < 5. */
function chiSquarePValue(observed: number[], probabilities: number[], n: number): number {
  let stat = 0
  let cells = 0
  let obsPool = 0
  let expPool = 0
  for (let i = 0; i < observed.length; i++) {
    obsPool += observed[i]
    expPool += n * probabilities[i]
    if (expPool >= 5) {
      stat += (obsPool - expPool) ** 2 / expPool
      cells++
      obsPool = 0
      expPool = 0
    }
  }
  if (expPool > 0) {
    stat += (obsPool - expPool) ** 2 / Math.max(expPool, 1e-300)
    cells++
  }
  return chiSquareSf(stat, cells - 1)
}

describe('philox4x32-10', () => {
  it('matches the Random123 known-answer vectors', () => {
    const out = new Uint32Array(4)
    R.philox4x32(0, 0, 0, 0, 0, 0, out)
    expect([...out]).toEqual([0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8])
    const f = 0xffffffff
    R.philox4x32(f, f, f, f, f, f, out)
    expect([...out]).toEqual([0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd])
    R.philox4x32(0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344, 0xa4093822, 0x299f31d0, out)
    expect([...out]).toEqual([0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1])
  })
})

describe('streams', () => {
  it('are deterministic, and their first values are fixed (a change here changes every figure)', () => {
    const a = R.stream(42)
    const b = R.stream(42)
    const xs = Array.from({ length: 10 }, () => a.uint32())
    expect(Array.from({ length: 10 }, () => b.uint32())).toEqual(xs)
    expect(xs.slice(0, 4)).toMatchInlineSnapshot(`
      [
        450079237,
        2369274609,
        3627297871,
        3524454626,
      ]
    `)
    expect(R.stream(7).child('chain', 3).uniform()).toMatchInlineSnapshot(`0.911639436531366`)
  })

  it('have readable, one-to-one keys', () => {
    expect(R.stream(7).child('chain', 3).child('env').key).toBe('7/chain:3/env')
    expect(R.stream('a/b').child('x:y').key).toBe('a%2Fb/x%3Ay')
    expect(R.stream(7).child('a', 'b').key).not.toBe(R.stream(7).child('a:b').key)
  })

  it('treat numbers and their decimal strings as the same name', () => {
    expect(R.stream(7).uint32()).toBe(R.stream('7').uint32())
    expect(R.stream(1).child(3).uint32()).toBe(R.stream(1).child('3').uint32())
  })

  it('give different sequences for different paths, including ones that look alike', () => {
    const first = (s: R.Stream) => [s.uint32(), s.uint32()].join(',')
    const root = R.stream(1)
    const variants = [
      root,
      root.child('a', 'b'),
      root.child('a:b'),
      root.child('a').child('b'),
      root.child('ab'),
      root.child(''),
      root.child(),
      R.stream(2),
      R.stream('1/a'),
    ]
    expect(new Set(variants.map(first)).size).toBe(variants.length)
    // Many siblings: no two share their first 64 bits.
    const seen = new Set<string>()
    for (let k = 0; k < 20_000; k++) seen.add(first(root.child(k)))
    expect(seen.size).toBe(20_000)
  })

  it("child draws do not depend on the parent's or siblings' usage", () => {
    const fresh = R.stream(3).child('chain', 1)
    const expected = Array.from({ length: 50 }, () => fresh.uniform())
    const p = R.stream(3)
    for (let i = 0; i < 1234; i++) p.uint32()
    const sibling = p.child('chain', 0)
    for (let i = 0; i < 99; i++) sibling.uniform()
    const late = p.child('chain', 1)
    expect(Array.from({ length: 50 }, () => late.uniform())).toEqual(expected)
  })

  it('produce uncorrelated sequences across parent, children and siblings', () => {
    const s = R.stream(11)
    const draw = (t: R.Stream) => Float64Array.from({ length: N }, () => t.uniform())
    const parent = draw(R.stream(11))
    const pairs: [Float64Array, Float64Array][] = [
      [parent, draw(s.child(0))],
      [draw(s.child(0)), draw(s.child(1))],
      [draw(s.child('x')), draw(s.child('x', ''))],
      [draw(R.stream(11).child(5)), draw(R.stream(12).child(5))],
    ]
    const bound = 4 / Math.sqrt(N)
    for (const [a, b] of pairs) expect(Math.abs(correlation(a, b))).toBeLessThan(bound)
    // Lag-1 autocorrelation within a stream.
    expect(Math.abs(correlation(parent.subarray(0, N - 1), parent.subarray(1)))).toBeLessThan(bound)
  })

  it('uniform lies in [0, 1) with the right moments, and every bit is fair', () => {
    const s = R.stream('bits')
    const u = Float64Array.from({ length: N }, () => s.uniform())
    expect(Math.min(...u)).toBeGreaterThanOrEqual(0)
    expect(Math.max(...u)).toBeLessThan(1)
    const m = moments(u)
    expect(Math.abs(m.mean - 0.5)).toBeLessThan(4 * Math.sqrt(1 / 12 / N))
    expect(Math.abs(m.variance - 1 / 12)).toBeLessThan(0.003)
    const ones = new Array(32).fill(0)
    for (let i = 0; i < N; i++) {
      const w = s.uint32()
      for (let b = 0; b < 32; b++) ones[b] += (w >>> b) & 1
    }
    for (const c of ones) expect(Math.abs(c / N - 0.5)).toBeLessThan(4 * Math.sqrt(0.25 / N))
  })

  it('int(n) is uniform, including n above 2^32', () => {
    const s = R.stream('int')
    const counts = new Array(7).fill(0)
    for (let i = 0; i < N; i++) counts[s.int(7)]++
    expect(chiSquarePValue(counts, new Array(7).fill(1 / 7), N)).toBeGreaterThan(1e-3)
    const big = 3 * 2 ** 40 + 1
    const xs = Float64Array.from({ length: N }, () => s.int(big))
    expect(xs.every((x) => Number.isInteger(x) && x >= 0 && x < big)).toBe(true)
    expect(Math.abs(moments(xs).mean / big - 0.5)).toBeLessThan(4 * Math.sqrt(1 / 12 / N))
    expect(() => s.int(0)).toThrow(RangeError)
  })
})

/** Draw N values of the continuous sampler named in a fixture case. */
function drawContinuous(c: Continuous, s: R.Stream): Float64Array {
  const a = c.args
  const kind = c.name.split(' ')[0]
  const one: () => number = {
    uniform: () => R.uniform(s, a.a, a.b),
    normal: () => R.normal(s, a.mean, a.sd),
    exponential: () => R.exponential(s, a.rate),
    gamma: () => R.gamma(s, a.shape, a.scale),
    beta: () => R.beta(s, a.a, a.b),
    chiSquare: () => R.chiSquare(s, a.df),
    studentT: () => R.studentT(s, a.df, a.loc, a.scale),
  }[kind]!
  return Float64Array.from({ length: N }, one)
}

describe('continuous samplers against scipy.stats', () => {
  for (const c of ref.continuous) {
    it(`${c.name}: empirical cdf and moments`, () => {
      const xs = drawContinuous(c, R.stream('continuous').child(c.name)).sort()
      // Kolmogorov–Smirnov on scipy's quantile grid: |F_N(x) − F(x)| at 199 points. 1.95/√N is the 0.1% critical value.
      let worst = 0
      let j = 0
      c.x.forEach((x, i) => {
        // Skip quantiles that coincide after rounding (Beta(0.05, 0.08) puts 1.5% of its mass within 1e-16 of 1).
        if (x === c.x[i - 1] || x === c.x[i + 1]) return
        while (j < N && xs[j] <= x) j++
        worst = Math.max(worst, Math.abs(j / N - c.p[i]))
      })
      expect(worst, c.name).toBeLessThan(1.95 / Math.sqrt(N))
      // The mean within 4.5 standard errors where the variance is finite (heavy-tailed t skips this).
      if (Number.isFinite(c.var)) {
        expect(Math.abs(moments(xs).mean - c.mean), c.name).toBeLessThan(4.5 * Math.sqrt(c.var / N))
      }
    })
  }
})

describe('discrete samplers against scipy.stats', () => {
  for (const c of ref.discrete) {
    it(`${c.name}: chi-square goodness of fit and moments`, () => {
      const s = R.stream('discrete').child(c.name)
      const [kind] = c.name.split(' ')
      const counts = new Array(c.k.length).fill(0)
      const xs = new Float64Array(N)
      for (let i = 0; i < N; i++) {
        const x = kind === 'poisson' ? R.poisson(s, c.args.lambda) : R.binomial(s, c.args.n, c.args.p)
        xs[i] = x
        const idx = x - c.k[0]
        expect(idx >= 0 && idx < c.k.length, `${x} in the support range`).toBe(true)
        counts[idx]++
      }
      expect(chiSquarePValue(counts, c.pmf, N), c.name).toBeGreaterThan(1e-3)
      expect(Math.abs(moments(xs).mean - c.mean)).toBeLessThan(4.5 * Math.sqrt(c.var / N))
    })
  }
})

describe('other samplers', () => {
  it('normal uses both Box–Muller outputs and is unclamped', () => {
    const s = R.stream('normal')
    const z = Float64Array.from(toFlat(R.normals(s, 2 * N)))
    const m = moments(z)
    expect(Math.abs(m.mean)).toBeLessThan(4 / Math.sqrt(2 * N))
    expect(Math.abs(m.variance - 1)).toBeLessThan(0.05)
    // The pair (cos, sin) of one draw is uncorrelated.
    const even = z.filter((_, i) => i % 2 === 0)
    const odd = z.filter((_, i) => i % 2 === 1)
    expect(Math.abs(correlation(even, odd))).toBeLessThan(4 / Math.sqrt(N))
    // Two normals per two uniforms: n normals use n uniforms (2n words).
    const t = R.stream('count')
    R.normals(t, 10)
    const u = R.stream('count')
    for (let i = 0; i < 20; i++) u.uint32()
    expect(t.uint32()).toBe(u.uint32())
  })

  it('bernoulli', () => {
    const s = R.stream('bernoulli')
    let k = 0
    for (let i = 0; i < N; i++) k += R.bernoulli(s, 0.3) ? 1 : 0
    expect(Math.abs(k / N - 0.3)).toBeLessThan(4 * Math.sqrt(0.21 / N))
  })

  it('categorical and the alias table match the weights', () => {
    const w = [0.1, 0, 0.5, 0.25, 0.15]
    const p = w.map((x) => x / 1)
    const s = R.stream('categorical')
    const table = R.aliasTable(w.map((x) => 7 * x))
    for (const draw of [() => R.categorical(s, w), () => R.aliasSample(s, table)]) {
      const counts = new Array(w.length).fill(0)
      for (let i = 0; i < N; i++) counts[draw()]++
      expect(counts[1]).toBe(0)
      expect(
        chiSquarePValue(
          counts.filter((_, i) => i !== 1),
          p.filter((_, i) => i !== 1),
          N,
        ),
      ).toBeGreaterThan(1e-3)
    }
    expect(() => R.categorical(s, [0, 0])).toThrow(RangeError)
  })

  it('dirichlet has the right means and variances, and small concentrations give no NaN', () => {
    const alpha = [0.5, 2, 7.5]
    const a0 = 10
    const s = R.stream('dirichlet')
    const draws = Array.from({ length: N }, () => toFlat(R.dirichlet(s, alpha)))
    alpha.forEach((a, k) => {
      const m = moments(draws.map((d) => d[k]))
      const mean = a / a0
      const variance = (mean * (1 - mean)) / (a0 + 1)
      expect(Math.abs(m.mean - mean)).toBeLessThan(4.5 * Math.sqrt(variance / N))
      expect(Math.abs(m.variance / variance - 1)).toBeLessThan(0.08)
    })
    for (const d of draws.slice(0, 100)) expect(d.reduce((x, y) => x + y, 0)).toBeCloseTo(1, 12)
    const tiny = toFlat(R.dirichlet(s, [1e-4, 1e-4, 1e-4]))
    expect(tiny.every(Number.isFinite)).toBe(true)
    expect(tiny.reduce((x, y) => x + y, 0)).toBeCloseTo(1, 12)
    expect(Number.isFinite(R.beta(s, 1e-5, 1e-5))).toBe(true)
  })

  it('multinomial counts sum to n with the right means and covariances', () => {
    const p = [0.2, 0.5, 0.3]
    const n = 30
    const s = R.stream('multinomial')
    const draws = Array.from({ length: N }, () => toFlat(R.multinomial(s, n, p)))
    for (const d of draws) expect(d[0] + d[1] + d[2]).toBe(n)
    const c0 = draws.map((d) => d[0])
    const c1 = draws.map((d) => d[1])
    const m0 = moments(c0)
    expect(Math.abs(m0.mean - n * p[0])).toBeLessThan(4.5 * Math.sqrt((n * p[0] * (1 - p[0])) / N))
    // Corr(c0, c1) = −√(p0 p1 / ((1 − p0)(1 − p1))).
    const rho = -Math.sqrt((p[0] * p[1]) / ((1 - p[0]) * (1 - p[1])))
    expect(Math.abs(correlation(c0, c1) - rho)).toBeLessThan(0.03)
  })

  it('permutation and shuffle are uniform over all 24 orders of 4 items', () => {
    const s = R.stream('perm')
    const counts = new Map<string, number>()
    for (let i = 0; i < N; i++) {
      const key = toFlat(R.permutation(s, 4)).join('')
      counts.set(key, (counts.get(key) ?? 0) + 1)
    }
    expect(counts.size).toBe(24)
    expect(chiSquarePValue([...counts.values()], new Array(24).fill(1 / 24), N)).toBeGreaterThan(1e-3)
    const arr = ['a', 'b', 'c']
    expect(R.shuffle(s, arr)).toBe(arr)
    expect([...arr].sort()).toEqual(['a', 'b', 'c'])
  })

  it('choice: uniform and weighted, with and without replacement', () => {
    const s = R.stream('choice')
    // Uniform without replacement: distinct, uniform marginals.
    const marg = new Array(10).fill(0)
    for (let i = 0; i < N / 10; i++) {
      const c = toFlat(R.choice(s, 10, 4, { replace: false }))
      expect(new Set(c).size).toBe(4)
      for (const k of c) marg[k]++
    }
    expect(chiSquarePValue(marg, new Array(10).fill(0.1), (N / 10) * 4)).toBeGreaterThan(1e-3)
    // Weighted without replacement: ordered pairs (i, j) have probability wᵢ/W · wⱼ/(W − wᵢ).
    const w = [1, 2, 3, 0]
    const pairs = new Map<string, number>()
    for (let i = 0; i < N; i++) {
      const c = toFlat(R.choice(s, 4, 2, { replace: false, weights: w }))
      pairs.set(`${c[0]}${c[1]}`, (pairs.get(`${c[0]}${c[1]}`) ?? 0) + 1)
    }
    const keys: string[] = []
    const probs: number[] = []
    for (let i = 0; i < 3; i++)
      for (let j = 0; j < 3; j++)
        if (i !== j) {
          keys.push(`${i}${j}`)
          probs.push((w[i] / 6) * (w[j] / (6 - w[i])))
        }
    expect(
      chiSquarePValue(
        keys.map((k) => pairs.get(k) ?? 0),
        probs,
        N,
      ),
    ).toBeGreaterThan(1e-3)
    expect([...pairs.keys()].every((k) => keys.includes(k))).toBe(true)
    // Weighted with replacement.
    const counts = new Array(4).fill(0)
    for (const k of toFlat(R.choice(s, 4, N, { weights: w }))) counts[k]++
    expect(counts[3]).toBe(0)
    expect(chiSquarePValue(counts.slice(0, 3), [1 / 6, 2 / 6, 3 / 6], N)).toBeGreaterThan(1e-3)
    expect(() => R.choice(s, 3, 4, { replace: false })).toThrow(RangeError)
  })

  it('multivariateNormal has covariance L Lᵀ', () => {
    const L = fromRows([
      [2, 0],
      [-1.5, 0.5],
    ])
    const s = R.stream('mvn')
    const draws = toRows(R.multivariateNormal(s, [1, -2], { choleskyFactor: L }, { shape: [N] }))
    const x = draws.map((d) => d[0])
    const y = draws.map((d) => d[1])
    expect(Math.abs(moments(x).mean - 1)).toBeLessThan(4.5 * Math.sqrt(4 / N))
    expect(moments(x).variance).toBeCloseTo(4, 0)
    expect(moments(y).variance).toBeCloseTo(2.5, 0)
    // Corr = Σ₀₁/√(Σ₀₀Σ₁₁) = −3/√(4 · 2.5).
    expect(Math.abs(correlation(x, y) + 3 / Math.sqrt(10))).toBeLessThan(0.02)
  })
})

describe('replicate', () => {
  it('runs fn on child k, caches by key and reuses a prefix', () => {
    const s = R.stream('rep')
    let calls = 0
    const fn = (r: R.Stream) => {
      calls++
      return r.uniform()
    }
    const five = R.replicate(5, s, fn)
    expect(calls).toBe(5)
    expect(five).toEqual([0, 1, 2, 3, 4].map((k) => s.child(k).uniform()))
    const eight = R.replicate(8, s, fn)
    expect(calls).toBe(8)
    expect(eight.slice(0, 5)).toEqual(five)
    // Another stream has other keys, so nothing is reused.
    R.replicate(2, R.stream('other'), fn)
    expect(calls).toBe(10)
    // An explicit cache, and no cache.
    const cache = new Map<string, number>()
    R.replicate(3, s, fn, { cache })
    expect(cache.size).toBe(3)
    R.replicate(3, s, fn, { cache: false })
    expect(calls).toBe(16)
  })
})

describe('samplers: shapes, broadcasting and element order', () => {
  const scalarCalls: [string, (s: R.Stream) => number, (s: R.Stream, shape: number[]) => Tensor][] = [
    ['uniform', (s) => R.uniform(s, -1, 3), (s, shape) => R.uniform(s, -1, 3, { shape })],
    ['normal', (s) => R.normal(s, 1, 2), (s, shape) => R.normal(s, 1, 2, { shape })],
    ['exponential', (s) => R.exponential(s, 2), (s, shape) => R.exponential(s, 2, { shape })],
    ['logGammaVariate', (s) => R.logGammaVariate(s, 0.3), (s, shape) => R.logGammaVariate(s, 0.3, { shape })],
    ['gamma', (s) => R.gamma(s, 2.5, 0.5), (s, shape) => R.gamma(s, 2.5, 0.5, { shape })],
    ['beta', (s) => R.beta(s, 0.5, 2), (s, shape) => R.beta(s, 0.5, 2, { shape })],
    ['chiSquare', (s) => R.chiSquare(s, 3), (s, shape) => R.chiSquare(s, 3, { shape })],
    ['studentT', (s) => R.studentT(s, 4, 1, 2), (s, shape) => R.studentT(s, 4, 1, 2, { shape })],
    ['bernoulli', (s) => R.bernoulli(s, 0.3), (s, shape) => R.bernoulli(s, 0.3, { shape })],
    ['poisson', (s) => R.poisson(s, 12), (s, shape) => R.poisson(s, 12, { shape })],
    ['binomial', (s) => R.binomial(s, 80, 0.4), (s, shape) => R.binomial(s, 80, 0.4, { shape })],
  ]
  for (const [name, one, many] of scalarCalls) {
    it(`${name}: numbers give a number; a shape gives the same draws in row-major order`, () => {
      expect(typeof one(R.stream('shape').child(name))).toBe('number')
      const a = R.stream('shape').child(name)
      const want = Array.from({ length: 24 }, () => one(a))
      for (const shape of [[24], [4, 6], [2, 3, 4], [1, 24, 1]]) {
        const t = many(R.stream('shape').child(name), shape)
        expect(t.shape).toEqual(shape)
        expect(t.dtype).toBe('float64')
        expect(toFlat(t)).toEqual(want)
      }
      // Rank 0: a scalar tensor holding the first draw.
      const r0 = many(R.stream('shape').child(name), [])
      expect(r0.shape).toEqual([])
      expect(toFlat(r0)).toEqual([want[0]])
    })
  }

  it('normals(s, n) is a tensor equal to n successive normal draws', () => {
    const a = R.stream('normals')
    const want = Array.from({ length: 7 }, () => R.normal(a))
    const z = R.normals(R.stream('normals'), 7)
    expect(z.shape).toEqual([7])
    expect(toFlat(z)).toEqual(want)
    expect(R.normals(R.stream('normals'), [7, 1]).shape).toEqual([7, 1])
  })

  it('tensor parameters broadcast, element by element, against each other and the shape', () => {
    const mean = tensor([0, 10, 20], [3, 1])
    const sd = tensor([1, 0.5])
    const t = R.normal(R.stream('bc'), mean, sd)
    expect(t.shape).toEqual([3, 2])
    const s = R.stream('bc')
    const want = [0, 10, 20].flatMap((m) => [1, 0.5].map((d) => R.normal(s, m, d)))
    expect(toFlat(t)).toEqual(want)
    // A shape the parameters broadcast to; and one they do not.
    expect(R.normal(R.stream('bc'), mean, sd, { shape: [4, 3, 2] }).shape).toEqual([4, 3, 2])
    expect(() => R.normal(R.stream('bc'), mean, sd, { shape: [3] })).toThrow(RangeError)
    // A number next to a tensor, and an int32 tensor parameter.
    const lambdas = tensor([1, 5, 50], [3], 'int32')
    const counts = R.poisson(R.stream('bc'), lambdas)
    expect(counts.shape).toEqual([3])
    const u = R.stream('bc')
    expect(toFlat(counts)).toEqual([1, 5, 50].map((l) => R.poisson(u, l)))
    expect(R.binomial(R.stream('bc'), 10, tensor([0.1, 0.9])).shape).toEqual([2])
  })

  it('broadcast parameters give the right distribution in every cell', () => {
    const t = toRows(
      R.normal(R.stream('cells'), tensor([-5, 0, 5], [3, 1]), tensor([1, 2, 3], [3, 1]), { shape: [3, N] }),
    )
    t.forEach((row, i) => {
      const m = moments(row)
      expect(Math.abs(m.mean - (5 * i - 5))).toBeLessThan(4.5 * Math.sqrt((i + 1) ** 2 / N))
      expect(Math.abs(m.variance / (i + 1) ** 2 - 1)).toBeLessThan(0.05)
    })
    const g = toRows(R.gamma(R.stream('cells'), tensor([0.5, 4], [2, 1]), 2, { shape: [2, N] }))
    g.forEach((row, i) => {
      const a = [0.5, 4][i]
      expect(Math.abs(moments(row).mean - 2 * a)).toBeLessThan(4.5 * Math.sqrt((4 * a) / N))
    })
  })

  it('categorical and aliasSample: numbers for one draw, int32 tensors for batches and shapes', () => {
    const w = [0.2, 0.5, 0.3]
    expect(typeof R.categorical(R.stream('cat'), w)).toBe('number')
    expect(R.categorical(R.stream('cat'), w)).toBe(R.categorical(R.stream('cat'), tensor(w)))
    const s = R.stream('cat')
    const want = Array.from({ length: 6 }, () => R.categorical(s, w))
    const t = R.categorical(R.stream('cat'), w, { shape: [2, 3] }) as Tensor
    expect(t.dtype).toBe('int32')
    expect(t.shape).toEqual([2, 3])
    expect(toFlat(t)).toEqual(want)
    // A batch of weight vectors: row i always picks index i.
    const batch = fromRows([
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
      [0, 1, 0],
    ])
    expect(toFlat(R.categorical(R.stream('cat'), batch) as Tensor)).toEqual([0, 1, 2, 1])
    expect((R.categorical(R.stream('cat'), batch, { shape: [5, 4] }) as Tensor).shape).toEqual([5, 4])
    expect(() => R.categorical(R.stream('cat'), batch, { shape: [3] })).toThrow(RangeError)
    const table = R.aliasTable(tensor(w))
    const a = R.aliasSample(R.stream('alias'), table, { shape: [3, 2] })
    expect(a.dtype).toBe('int32')
    const u = R.stream('alias')
    expect(toFlat(a)).toEqual(Array.from({ length: 6 }, () => R.aliasSample(u, table)))
  })

  it('index samplers return int32 tensors', () => {
    const p = R.permutation(R.stream('idx'), 5)
    expect(p.dtype).toBe('int32')
    expect([...toFlat(p)].sort()).toEqual([0, 1, 2, 3, 4])
    const c = R.choice(R.stream('idx'), 10, [2, 3], { replace: false, weights: tensor(Array(10).fill(1)) })
    expect(c.dtype).toBe('int32')
    expect(c.shape).toEqual([2, 3])
    expect(new Set(toFlat(c)).size).toBe(6)
  })

  it('dirichlet and multinomial append the event axis and broadcast batches', () => {
    const alpha = fromRows([
      [1, 2, 3],
      [50, 50, 50],
    ])
    const d = R.dirichlet(R.stream('dir'), alpha, { shape: [4, 2] })
    expect(d.shape).toEqual([4, 2, 3])
    for (const row of toRows(R.dirichlet(R.stream('dir'), alpha)))
      expect(row.reduce((a, b) => a + b)).toBeCloseTo(1, 12)
    // Row-major order: the first row of a batch is the draw from the first alpha.
    const first = toFlat(R.dirichlet(R.stream('dir'), [1, 2, 3]))
    expect(toFlat(d).slice(0, 3)).toEqual(first)
    const m = R.multinomial(R.stream('mult'), tensor([5, 50], [2, 1]), [0.2, 0.8], { shape: [2, 3] })
    expect(m.shape).toEqual([2, 3, 2])
    toRows(R.multinomial(R.stream('mult'), tensor([5, 50]), [0.2, 0.8])).forEach((row, i) =>
      expect(row[0] + row[1]).toBe([5, 50][i]),
    )
  })

  it('multivariateNormal takes a covariance or a Cholesky factor, and reports a covariance that does not factor', () => {
    const cov = fromRows([
      [4, -3],
      [-3, 2.5],
    ])
    const L = fromRows([
      [2, 0],
      [-1.5, 0.5],
    ])
    const a = R.multivariateNormal(R.stream('mvn2'), tensor([1, -2]), { covariance: cov }, { shape: [5] })
    const b = R.multivariateNormal(R.stream('mvn2'), [1, -2], { choleskyFactor: L }, { shape: [5] })
    expect(a.shape).toEqual([5, 2])
    toFlat(a).forEach((v, i) => expect(v).toBeCloseTo(toFlat(b)[i], 12))
    expect(R.multivariateNormal(R.stream('mvn2'), [1, -2], { covariance: cov }).shape).toEqual([2])
    // A batch of means broadcast against the shape.
    const means = fromRows([
      [0, 0],
      [100, 100],
    ])
    const batch = R.multivariateNormal(R.stream('mvn2'), means, { choleskyFactor: L }, { shape: [3, 2] })
    expect(batch.shape).toEqual([3, 2, 2])
    // Element [i, 1, :] is drawn around the second mean.
    expect(toFlat(batch)[2]).toBeGreaterThan(50)
    const singular = fromRows([
      [1, 1],
      [1, 1],
    ])
    expect(() => R.multivariateNormal(R.stream('mvn2'), [0, 0], { covariance: singular })).toThrow(/positive definite/)
    expect(R.multivariateNormal(R.stream('mvn2'), [0, 0], { covariance: singular }, { jitter: 'auto' }).shape).toEqual([
      2,
    ])
    expect(() => R.multivariateNormal(R.stream('mvn2'), [0, 0, 0], { choleskyFactor: L })).toThrow(RangeError)
  })
})
