import { describe, expect, it } from 'vitest'
import {
  bool,
  clamp,
  clampReport,
  decode,
  defaults,
  encode,
  encodedSize,
  grid,
  int,
  isDimActive,
  oneOf,
  real,
  sample,
  space,
  subspace,
  variants,
  when,
} from 'aifn/foundation/space'
import { DomainError } from 'aifn/foundation/errors'
import { stream } from 'aifn/foundation/random'

const S = space({
  rate: real(1e-4, 1, { scale: 'log' }),
  layers: int(1, 4),
  act: oneOf(['relu', 'tanh']),
  momentum: bool(),
  beta: real(0, 1, { step: 0.25, when: when('momentum', true) }),
  prior: subspace(space({ scale: real(0, 10, { default: 1 }) })),
  kernel: variants({ rbf: space({ length: real(0.1, 10) }), linear: space({}) }),
})

describe('dimensions and defaults', () => {
  it('defaults: log midpoint, min, first option, false, nested and first variant; inactive dims are absent', () => {
    const d = defaults(S)
    expect(d.rate).toBeCloseTo(1e-2, 12)
    expect(d.layers).toBe(1)
    expect(d.act).toBe('relu')
    expect(d.momentum).toBe(false)
    expect('beta' in d).toBe(false)
    expect(d.prior).toEqual({ scale: 1 })
    expect(d.kernel).toEqual({ case: 'rbf', params: { length: (0.1 + 10) / 2 } })
  })
  it('invalid dimensions are DomainErrors', () => {
    expect(() => real(2, 1)).toThrow(DomainError)
    expect(() => real(0, 1, { scale: 'log' })).toThrow(DomainError)
    expect(() => int(0.5, 2)).toThrow(DomainError)
    expect(() => oneOf([])).toThrow(DomainError)
    expect(() => variants({})).toThrow(DomainError)
  })
  it('conditions are data', () => {
    expect(isDimActive(S.dims.beta, { momentum: true })).toBe(true)
    expect(isDimActive(S.dims.beta, { momentum: false })).toBe(false)
    expect(JSON.parse(JSON.stringify(S))).toEqual(S)
  })
})

describe('clamp', () => {
  it('clips, rounds, snaps, fills defaults and drops unknown or inactive keys (reported)', () => {
    const r = clampReport(S, {
      rate: 5,
      layers: 2.6,
      act: 'gelu',
      momentum: true,
      beta: 0.6,
      extra: 1,
      prior: { scale: -3, junk: 0 },
      kernel: { case: 'linear', params: { length: 2 } },
    })
    expect(r.values).toEqual({
      rate: 1,
      layers: 3,
      act: 'relu',
      momentum: true,
      beta: 0.5,
      prior: { scale: 0 },
      kernel: { case: 'linear', params: {} },
    })
    expect(r.dropped).toEqual(['prior.junk', 'kernel.params.length', 'extra'])
    expect(clamp(S, { momentum: false, beta: 0.3 })).not.toHaveProperty('beta')
  })
})

describe('encoding into the unit cube', () => {
  it('decode ∘ encode is the identity on points of the space, and codes lie in [0, 1]', () => {
    const v = clamp(S, {
      rate: 0.003,
      layers: 3,
      act: 'tanh',
      momentum: true,
      beta: 0.75,
      kernel: { case: 'rbf', params: { length: 2 } },
    })
    const u = encode(S, v)
    expect(u.length).toBe(encodedSize(S))
    for (const x of u) expect(x >= 0 && x <= 1).toBe(true)
    const back = decode(S, u)
    expect(back.layers).toBe(3)
    expect(back.act).toBe('tanh')
    expect(back.beta).toBe(0.75)
    expect(back.rate).toBeCloseTo(0.003, 12)
  })
  it('a log dimension encodes log x linearly', () => {
    const L = space({ x: real(1, 100, { scale: 'log' }) })
    expect(encode(L, { x: 10 })[0]).toBeCloseTo(0.5, 12)
  })
  it('sample draws decoded uniform codes; the same stream gives the same point', () => {
    const a = sample(stream(3), S)
    expect(sample(stream(3), S)).toEqual(a)
    expect(clamp(S, a)).toEqual(a)
    // Integers cover their range uniformly: layers ∈ {1, …, 4} each about a quarter of the time.
    const s = stream(4)
    const counts = [0, 0, 0, 0]
    for (let k = 0; k < 4000; k++) counts[(sample(s, S).layers as number) - 1]++
    for (const c of counts) expect(Math.abs(c - 1000)).toBeLessThan(150)
  })
})

describe('grid', () => {
  it('every combination, conditions respected', () => {
    const G = space({ a: oneOf(['x', 'y']), on: bool(), k: int(1, 3, { when: when('on', true) }) })
    const g = grid(G)
    // on = false: 2 points; on = true: 2 × 3 points.
    expect(g.length).toBe(8)
    expect(g.filter((p) => p.on === false).every((p) => !('k' in p))).toBe(true)
    expect(new Set(g.map((p) => JSON.stringify(p))).size).toBe(8)
    expect(grid(space({ r: real(0, 1) }), { points: 5 }).map((p) => p.r)).toEqual([0, 0.25, 0.5, 0.75, 1])
  })
})
