/**
 * Smoke tests for aifn/info: one check per family of exports. References: scipy.stats.entropy,
 * scipy.spatial.distance.jensenshannon and sklearn.metrics.mutual_info_score (values hard-coded), known capacities
 * (BSC 1 − H₂(p), BEC 1 − ε, the Z channel), R(D) = 1 − H₂(D), and hand-checked codes and bounds.
 */

import { describe, expect, it } from 'vitest'
import * as I from 'aifn/info'
import { normalProjection } from 'aifn/info'
import { Mixture, Normal } from 'aifn/distributions'
import { normals, stream } from 'aifn/random'
import { binaryEntropy } from 'aifn/special'
import { fromRows, tensor, toFlat, unwrap, type Value } from 'aifn/tensor'
import { extend, run, seek, trace } from 'aifn/trace'
import { checkGradient } from './tensor/mock-tape'

const n = (v: Value) => {
  const r = unwrap(v)
  return typeof r === 'number' ? r : toFlat(r)[0]
}
const close = (a: number, b: number, tol = 1e-12) =>
  expect(Math.abs(a - b) / Math.max(1, Math.abs(b))).toBeLessThan(tol)
const P = [0.1, 0.2, 0.3, 0.4]
const Q = [0.25, 0.25, 0.4, 0.1]

describe('info: measures', () => {
  it('against scipy and sklearn', () => {
    close(n(I.entropy(P, { base: 2 })), 1.8464393446710157)
    close(n(I.entropy([1, 2, 3, 4])), n(I.entropy(P)))
    close(n(I.klDivergence(P, Q)), 0.3319553392621645)
    close(n(I.jensenShannonDistance(P, Q, { base: 2 })), 0.317253708730743)
    close(
      n(
        I.mutualInformation([
          [10, 2, 3],
          [1, 8, 4],
        ]),
      ),
      0.22147334222306372,
    )
    expect(n(I.klDivergence([0.5, 0.5], [1, 0]))).toBe(Infinity)
  })

  it('identities between the measures', () => {
    const joint = [
      [10, 2, 3],
      [1, 8, 4],
    ]
    close(n(I.conditionalEntropy(joint)), n(I.jointEntropy(joint)) - n(I.entropy([15, 13])))
    close(n(I.crossEntropy(P, Q)), n(I.entropy(P)) + n(I.klDivergence(P, Q)))
    close(I.fDivergence(P, Q, I.fGenerators.kl), n(I.klDivergence(P, Q)))
    close(I.fDivergence(P, Q, I.fGenerators.reverseKl), n(I.klDivergence(Q, P)))
    close(I.fDivergence(P, Q, I.fGenerators.jensenShannon), n(I.jensenShannonDivergence(P, Q)))
    close(I.fDivergence(P, Q, I.fGenerators.totalVariation), n(I.totalVariation(P, Q)))
    close(Math.sqrt(I.fDivergence(P, Q, I.fGenerators.squaredHellinger)), n(I.hellingerDistance(P, Q)))
    const pmi = toFlat(I.pointwiseMutualInformation(joint) as never)
    close(pmi[0], Math.log(10 / 28 / ((15 / 28) * (11 / 28))))
    close(n(I.differentialEntropy(Normal(0, 2), { base: 2 })), n(Normal(0, 2).entropy()) / Math.LN2)
    close(
      n(
        I.gaussianMutualInformation(
          fromRows([
            [1, 0.6],
            [0.6, 1],
          ]),
          [0],
          [1],
        ),
      ),
      -0.5 * Math.log(1 - 0.36),
    )
  })

  it('are differentiable', () => {
    checkGradient((p, q) => I.klDivergence(p, q), [tensor(P), tensor(Q)], { tol: 1e-6 })
    checkGradient((p) => I.entropy(p), [tensor(P)])
    checkGradient((p, q) => I.jensenShannonDivergence(p, q), [tensor(P), tensor(Q)], { tol: 1e-6 })
  })

  it('KSG estimate is close to the Gaussian closed form', () => {
    const s = stream('ksg')
    const a = toFlat(normals(s.child('a'), 1000))
    const e = toFlat(normals(s.child('e'), 1000))
    const b = a.map((v, i) => 0.6 * v + 0.8 * e[i])
    expect(Math.abs(I.ksgMutualInformation(a, b) - -0.5 * Math.log(1 - 0.36))).toBeLessThan(0.03)
  })
})

describe('info: Blahut–Arimoto', () => {
  it('known capacities', () => {
    close(
      I.channelCapacity(
        [
          [0.9, 0.1],
          [0.1, 0.9],
        ],
        { base: 2 },
      ).capacity,
      1 - binaryEntropy(0.1, 2),
      1e-9,
    )
    close(
      I.channelCapacity(
        [
          [0.7, 0.3, 0],
          [0, 0.3, 0.7],
        ],
        { base: 2 },
      ).capacity,
      0.7,
      1e-9,
    )
    const z = I.channelCapacity(
      [
        [1, 0],
        [0.4, 0.6],
      ],
      { base: 2 },
    )
    expect(z.converged).toBe(true)
    close(z.capacity, Math.log2(1 + 0.6 * 0.4 ** (0.4 / 0.6)), 1e-9)
  })

  it('rate–distortion of a binary source under Hamming distortion', () => {
    for (const beta of [1, 2, 4]) {
      const r = I.rateDistortion(
        [0.5, 0.5],
        [
          [0, 1],
          [1, 0],
        ],
        beta,
        { base: 2 },
      )
      close(r.rate, 1 - binaryEntropy(r.distortion, 2), 1e-9)
    }
    const p = 0.2
    const r = I.rateDistortion(
      [1 - p, p],
      [
        [0, 1],
        [1, 0],
      ],
      3,
      { base: 2 },
    )
    close(r.rate, binaryEntropy(p, 2) - binaryEntropy(r.distortion, 2), 1e-8)
  })

  it('follows the trace protocol', () => {
    const opts = {
      channel: [
        [0.8, 0.2, 0],
        [0.1, 0.6, 0.3],
        [0.1, 0.3, 0.6],
      ],
    }
    const full = trace(I.blahutArimotoCapacity, opts, 30)
    expect(seek(I.blahutArimotoCapacity, opts, 12).input).toEqual(run(I.blahutArimotoCapacity, opts, 12).input)
    const longer = extend(trace(I.blahutArimotoCapacity, opts, 10), I.blahutArimotoCapacity, opts, 20)
    expect(longer.steps[longer.steps.length - 1].lower).toEqual(full.steps[full.steps.length - 1].lower)
  })
})

describe('info: coding', () => {
  it('Huffman, Shannon–Fano, Shannon and Kraft', () => {
    const p = [0.25, 0.25, 0.2, 0.15, 0.15]
    const h = I.huffmanCode(p)
    expect(h.codewords).toEqual(['01', '10', '00', '110', '111'])
    close(h.expectedLength, 2.3)
    expect(h.expectedLength).toBeGreaterThanOrEqual(h.entropy)
    close(I.kraftSum(h.lengths), 1)
    expect(I.shannonFanoCode(p).codewords).toEqual(['00', '01', '10', '110', '111'])
    expect(I.shannonCode(p).codewords).toEqual(['00', '01', '100', '101', '110'])
    expect(I.huffmanCode([1]).codewords).toEqual(['0'])
  })

  it('arithmetic interval, Hamming distances and bounds', () => {
    const a = I.arithmeticInterval([0, 1, 0], [0.8, 0.2])
    close(a.low, 0.64)
    close(a.width, 0.8 * 0.2 * 0.8)
    expect(a.bits).toBe(4)
    expect(I.hammingDistance('10110', '11100')).toBe(2)
    expect(I.hammingWeight([1, 0, 1, 1])).toBe(3)
    expect(I.minimumDistance(['0000000', '1101001', '0101010', '1000011'])).toBe(3)
    // The [7, 4, 3] Hamming code is perfect: it attains the sphere-packing bound.
    expect(I.hammingBound(7, 3)).toBe(16)
    expect(I.singletonBound(7, 3)).toBe(32)
    expect(I.plotkinBound(7, 3)).toBe(16)
    expect(I.plotkinBound(10, 6)).toBe(6)
    expect(I.plotkinBound(20, 3)).toBe(Infinity)
  })
})

describe('info: KL projections onto the normal family', () => {
  it('forward matches moments; reverse seeks the mode nearest the start', () => {
    const p = Mixture([0.5, 0.5], [Normal(-2, 0.5), Normal(2, 0.5)])
    const forward = normalProjection(p, 'forward', { init: { loc: 1.5, scale: 0.5 } })
    expect(forward.converged).toBe(true)
    close(forward.loc, 0, 1e-5)
    close(forward.scale, Math.sqrt(0.25 + 4), 1e-5)
    const right = normalProjection(p, 'reverse', { init: { loc: 1.5, scale: 1 } })
    const left = normalProjection(p, 'reverse', { init: { loc: -1.5, scale: 1 } })
    expect(right.path.length).toBeGreaterThan(2)
    close(right.loc, 2, 1e-3)
    close(right.scale, 0.5, 1e-3)
    close(left.loc, -2, 1e-3)
    // One mode of an equal mixture of well-separated components: KL(q ‖ p) ≈ log 2.
    close(right.objective, Math.LN2, 1e-3)
  })
})
