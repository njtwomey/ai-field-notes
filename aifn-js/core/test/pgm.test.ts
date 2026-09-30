import { describe, expect, it } from 'vitest'
import { fromEdges } from 'aifn/graph'
import { inverse } from 'aifn/numerics/linalg'
import {
  beliefPropagation,
  beliefPropagationSteps,
  decodeBeliefs,
  gaussianBeliefPropagation,
  gaussianBeliefPropagationSteps,
} from 'aifn/inference/message-passing'
import {
  chainForwardBackward,
  enumerate,
  enumerationSteps,
  forwardBackward,
  forwardBackwardSteps,
  sampleHiddenPath,
  variableElimination,
  variableEliminationSteps,
  viterbi,
  viterbiSteps,
  type Hmm,
} from 'aifn/inference/exact'
import {
  crfGradient,
  crfLogLikelihood,
  crfMarginals,
  crfScore,
  crfViterbi,
  dishonestCasino,
  linearChainCrf,
  sampleHmm,
} from 'aifn-applied/inference/sequence-models'
import {
  discreteFactor,
  discreteFactorGraph,
  dist,
  expandModel,
  factorMarginalise,
  factorProduct,
  isTree,
  logJoint,
  markovBlanket,
  model,
  sampleModel,
  toDiscreteFactorGraph,
  toFactorDiagram,
  toFactorGraph,
  toPlateDiagram,
  type DiscreteFactorGraph,
} from 'aifn/inference/model'
import {
  factorGraphGibbsSteps,
  gibbsMarginals,
  gibbsSteps,
  type ModelGibbsState as GibbsState,
} from 'aifn/inference/stochastic'
import { gridEdges, isingModel } from 'aifn-applied/inference/lattice-models'
import { infer } from 'aifn/inference/engines'
import { ldaCollapsedGibbsSteps, ldaModel, type LdaState } from 'aifn-applied/inference/topic-models'
import { stream } from 'aifn/foundation/random'
import { fromData, tensor, toFlat, toRows, type Matrix } from 'aifn/foundation/tensor'
import { extend, run, seek, trace, type Algorithm } from 'aifn/foundation/trace'

const close = (a: ArrayLike<number>, b: ArrayLike<number>, tol: number) => {
  expect(a.length).toBe(b.length)
  for (let i = 0; i < a.length; i++) expect(Math.abs(a[i] - b[i])).toBeLessThanOrEqual(tol)
}

// ── Fixtures ────────────────────────────────────────────────────────────────────────────────────────────────────────

/** The sprinkler network (Russell & Norvig), written in the description language with CPTs indexed by parents. */
const sprinkler = model('Sprinkler', (m) => {
  const cloudy = m.variable('cloudy', dist.Bernoulli(0.5))
  const pS = m.constant('pS', [0.5, 0.1])
  const pR = m.constant('pR', [0.2, 0.8])
  const pW = m.constant('pW', [
    [0.01, 0.9],
    [0.9, 0.99],
  ])
  const s = m.variable('sprinkler', dist.Bernoulli(pS.at(cloudy)))
  const r = m.variable('rain', dist.Bernoulli(pR.at(cloudy)))
  const pw = m.deterministic('pw', 'index', [pW, s, r])
  m.observed('wet', dist.Bernoulli(pw))
})

/** P(rain = 1 | wet = 1) and P(wet = 1) by summing the joint by hand. */
function sprinklerBrute() {
  const pS = [0.5, 0.1]
  const pR = [0.2, 0.8]
  const pW = [
    [0.01, 0.9],
    [0.9, 0.99],
  ]
  const b = (p: number, x: number) => (x ? p : 1 - p)
  let evidence = 0
  let rain = 0
  let cloudy = 0
  for (const c of [0, 1])
    for (const s of [0, 1])
      for (const r of [0, 1]) {
        const p = 0.5 * b(pS[c], s) * b(pR[c], r) * pW[s][r]
        evidence += p
        if (r) rain += p
        if (c) cloudy += p
      }
  return { rain: rain / evidence, cloudy: cloudy / evidence, evidence }
}

/** A tree-shaped factor graph: a chain x0 – x1 – x2 with a branch x1 – x3, unary factors, random positive tables. */
function randomTree(seed: number): DiscreteFactorGraph {
  const s = stream(seed)
  const cards = [2, 3, 2, 2]
  const table = (scope: number[], k: number) =>
    discreteFactor(scope, cards, (a) => 0.2 + s.child(k, ...Array.from(a)).uniform())
  return discreteFactorGraph(cards, [
    table([0], 0),
    table([0, 1], 1),
    table([1, 2], 2),
    table([1, 3], 3),
    table([3], 4),
  ])
}

/** Enumerate every hidden path of an HMM: marginals, pairwise marginals, log p(x) and the best path. */
function hmmBrute(h: Hmm, obs: number[]) {
  const K = h.initial.shape[0]
  const N = obs.length
  const A = toRows(h.transition)
  const B = toRows(h.emission)
  const marg = Array.from({ length: N }, () => new Array<number>(K).fill(0))
  const pair = Array.from({ length: N - 1 }, () => Array.from({ length: K }, () => new Array<number>(K).fill(0)))
  let total = 0
  let best = -Infinity
  let bestPath: number[] = []
  for (let code = 0; code < K ** N; code++) {
    const y = Array.from({ length: N }, (_, n) => Math.floor(code / K ** (N - 1 - n)) % K)
    let p = h.initial.data[y[0]] * B[y[0]][obs[0]]
    for (let n = 1; n < N; n++) p *= A[y[n - 1]][y[n]] * B[y[n]][obs[n]]
    total += p
    y.forEach((k, n) => (marg[n][k] += p))
    for (let n = 0; n + 1 < N; n++) pair[n][y[n]][y[n + 1]] += p
    if (p > best) [best, bestPath] = [p, y]
  }
  return {
    marginals: marg.map((r) => r.map((v) => v / total)),
    pairwise: pair.map((m) => m.map((r) => r.map((v) => v / total))),
    logLikelihood: Math.log(total),
    path: bestPath,
    logBest: Math.log(best),
  }
}

// ── Factor algebra and exact inference ──────────────────────────────────────────────────────────────────────────────

describe('factor algebra', () => {
  it('multiplies and marginalises tables', () => {
    const cards = [2, 3]
    const a = discreteFactor([0], cards, [1, 2])
    const b = discreteFactor([0, 1], cards, [1, 2, 3, 4, 5, 6])
    const ab = factorProduct(a, b, cards)
    expect(ab.scope).toEqual([0, 1])
    expect(toFlat(ab.table)).toEqual([1, 2, 3, 8, 10, 12])
    expect(toFlat(factorMarginalise(ab, [0]).table)).toEqual([9, 12, 15])
    expect(toFlat(factorMarginalise(ab, [1], 'max').table)).toEqual([3, 12])
  })
})

describe('sprinkler: a model described in the language', () => {
  const bindings = { data: { wet: 1 } }
  const brute = sprinklerBrute()
  it('enumeration of its discrete factor graph gives the brute-force posterior and evidence', () => {
    const { graph, keys, logConstant } = toDiscreteFactorGraph(sprinkler, bindings)
    expect(keys).toEqual(['cloudy', 'sprinkler', 'rain'])
    const r = enumerate(graph)
    expect(r.marginals[2].data[1]).toBeCloseTo(brute.rain, 12)
    expect(r.marginals[0].data[1]).toBeCloseTo(brute.cloudy, 12)
    expect(r.logZ + logConstant).toBeCloseTo(Math.log(brute.evidence), 12)
  })
  it('variable elimination agrees with enumeration for every order heuristic', () => {
    const { graph } = toDiscreteFactorGraph(sprinkler, bindings)
    for (const order of ['min-fill', 'min-degree', [0, 1]] as const) {
      const r = variableElimination(graph, [2], { order })
      expect(r.marginal.data[1]).toBeCloseTo(brute.rain, 12)
      expect(r.logZ).toBeCloseTo(Math.log(brute.evidence), 12)
    }
  })
  it('infer picks variable elimination for an all-discrete model', () => {
    expect(infer(sprinkler, bindings).engine).toBe('variable-elimination')
  })
  it('Markov blanket of sprinkler: parents, children and co-parents', () => {
    const mb = markovBlanket(sprinkler, 'sprinkler', bindings)
    expect(mb.parents).toEqual(['cloudy'])
    expect(mb.children).toEqual(['wet'])
    expect(mb.coParents).toEqual(['rain'])
  })
  it('Gibbs on the model matches the exact posterior', () => {
    const s = run(gibbsSteps, { model: sprinkler, bindings }, 4000, { stream: stream(3) })
    expect(s.sweep).toBe(4000)
    // The running estimate from the trace of `rain`.
    const t = trace(gibbsSteps, { model: sprinkler, bindings }, 4000, {
      stream: stream(3),
      record: { rain: (st: GibbsState) => st.values.rain as number },
    })
    const mean =
      toFlat(t.series.rain)
        .slice(1)
        .reduce((a, b) => a + b, 0) / 4000
    expect(Math.abs(mean - brute.rain)).toBeLessThan(0.03)
  })
})

describe('trees: sum-product and max-product are exact', () => {
  const g = randomTree(11)
  const exact = enumerate(g)
  it('recognises a tree', () => expect(isTree(g)).toBe(true))
  it('tree schedule: marginals and Bethe log Z equal enumeration after one sweep', () => {
    const r = beliefPropagation(g)
    expect(r.sweeps).toBe(1)
    r.marginals.forEach((m, v) => close(m.data, exact.marginals[v].data, 1e-12))
    expect(r.logZ).toBeCloseTo(exact.logZ, 10)
  })
  it('flooding converges to the same marginals', () => {
    const r = beliefPropagation(g, { schedule: 'flooding' })
    expect(r.converged).toBe(true)
    r.marginals.forEach((m, v) => close(m.data, exact.marginals[v].data, 1e-9))
  })
  it('max-product decodes the MAP assignment', () => {
    const r = beliefPropagation(g, { mode: 'max' })
    expect(Array.from(decodeBeliefs(r.state).data)).toEqual(Array.from(exact.map.data))
  })
  it('evidence: BP agrees with variable elimination', () => {
    const bp = beliefPropagation(g, { evidence: { 3: 1 } })
    const ve = variableElimination(g, [0], { evidence: { 3: 1 } })
    close(bp.marginals[0].data, ve.marginal.data, 1e-12)
    expect(bp.logZ).toBeCloseTo(ve.logZ, 10)
  })
  it('message granularity exposes one update per step', () => {
    const t = trace(beliefPropagationSteps, { graph: g }, 100)
    expect(t.steps[1].updated).toHaveLength(1)
    expect(t.meta.stopped).toBe('done')
    expect(t.meta.steps).toBe(t.steps[0].schedule.length)
  })
})

describe('loopy BP and Gibbs on a 3 × 3 Ising grid', () => {
  const grid = fromEdges(9, gridEdges(3, 3), { directed: false })
  const g = isingModel(grid, 0.3, 0.1)
  const exact = enumerate(g)
  it('loopy BP converges near the exact marginals for weak coupling', () => {
    const r = beliefPropagation(g)
    expect(isTree(g)).toBe(false)
    expect(r.converged).toBe(true)
    r.marginals.forEach((m, v) => expect(Math.abs(m.data[1] - exact.marginals[v].data[1])).toBeLessThan(0.02))
    expect(Math.abs(r.logZ - exact.logZ)).toBeLessThan(0.05)
  })
  it('damping and a sequential schedule reach the same fixed point', () => {
    const a = beliefPropagation(g)
    const b = beliefPropagation(g, { schedule: 'sequential', damping: 0.5 })
    a.marginals.forEach((m, v) => close(m.data, b.marginals[v].data, 1e-6))
  })
  it('factor-graph Gibbs estimates the marginals', () => {
    const s = run(factorGraphGibbsSteps, { graph: g }, 5000, { stream: stream(1) })
    gibbsMarginals(s).forEach((m, v) => expect(Math.abs(m.data[1] - exact.marginals[v].data[1])).toBeLessThan(0.04))
  })
})

// ── Gaussian BP ─────────────────────────────────────────────────────────────────────────────────────────────────────

describe('Gaussian belief propagation', () => {
  const solve = (J: number[][], h: number[]) => {
    const S = toRows(inverse(tensor(J)))
    return { mean: S.map((r) => r.reduce((s, v, j) => s + v * h[j], 0)), variance: S.map((r, i) => r[i]) }
  }
  it('is exact on a tree', () => {
    const J = [
      [3, -1, 0, 0.5],
      [-1, 4, 1, 0],
      [0, 1, 2, 0],
      [0.5, 0, 0, 2],
    ]
    const h = [1, -2, 0.5, 1]
    const r = gaussianBeliefPropagation(J, h)
    const ex = solve(J, h)
    expect(r.converged).toBe(true)
    close(r.means.data, ex.mean, 1e-9)
    close(r.variances.data, ex.variance, 1e-9)
  })
  it('gets the means right on a loop (diagonally dominant), variances only approximately', () => {
    const J = [
      [4, 1, 0, 1],
      [1, 4, 1, 0],
      [0, 1, 4, 1],
      [1, 0, 1, 4],
    ]
    const h = [1, 2, 3, 4]
    const r = gaussianBeliefPropagation(J, h, { schedule: 'sequential' })
    const ex = solve(J, h)
    close(r.means.data, ex.mean, 1e-8)
    close(r.variances.data, ex.variance, 0.02)
  })
})

// ── Chains ──────────────────────────────────────────────────────────────────────────────────────────────────────────

describe('HMM: the dishonest casino against enumeration', () => {
  const h = dishonestCasino()
  const obs = [5, 5, 0, 5, 2, 5, 5]
  const brute = hmmBrute(h, obs)
  const fb = forwardBackward(h, obs)
  it('forward–backward marginals, pairwise marginals and log-likelihood', () => {
    toRows(fb.marginals).forEach((r, n) => close(r, brute.marginals[n], 1e-12))
    const K = 2
    for (let n = 0; n < obs.length - 1; n++)
      close(Array.from(fb.pairwise.data.slice(n * K * K, (n + 1) * K * K)), brute.pairwise[n].flat(), 1e-12)
    expect(fb.logLikelihood).toBeCloseTo(brute.logLikelihood, 12)
  })
  it('Viterbi finds the best path', () => {
    const v = viterbi(h, obs)
    expect(Array.from(v.path.data)).toEqual(brute.path)
    expect(v.logProbability).toBeCloseTo(brute.logBest, 12)
  })
  it('the same HMM described in the language gives the same posterior', () => {
    const N = obs.length
    const casino = model('Casino', (m) => {
      const A = m.constant('A', toRows(h.transition))
      const B = m.constant('B', toRows(h.emission))
      let prev = m.variable('y0', dist.Categorical([0.5, 0.5]))
      m.observed('x0', dist.Categorical(B.at(prev)))
      for (let n = 1; n < N; n++) {
        const y = m.variable(`y${n}`, dist.Categorical(A.at(prev)))
        m.observed(`x${n}`, dist.Categorical(B.at(y)))
        prev = y
      }
    })
    const data = Object.fromEntries(obs.map((x, n) => [`x${n}`, x]))
    const { graph, logConstant } = toDiscreteFactorGraph(casino, { data })
    const bp = beliefPropagation(graph)
    expect(bp.sweeps).toBe(1)
    bp.marginals.forEach((m, n) => close(m.data, brute.marginals[n], 1e-12))
    expect(bp.logZ + logConstant).toBeCloseTo(brute.logLikelihood, 10)
  })
  it('log-space chain forward–backward agrees with the scaled version', () => {
    const logU = fromData(Float64Array.from(toRows(fb.psi).flatMap((r) => r.map(Math.log))), [obs.length, 2])
    const logA = fromData(Float64Array.from(toFlat(h.transition).map(Math.log)), [2, 2])
    const c = chainForwardBackward(logU, logA)
    close(c.marginals.data, fb.marginals.data, 1e-12)
    expect(c.logZ).toBeCloseTo(fb.logLikelihood, 12)
  })
  it('sampling and FFBS are reproducible and valid paths', () => {
    const a = sampleHmm(stream(4), h, 50)
    const b = sampleHmm(stream(4), h, 50)
    expect(Array.from(a.observations.data)).toEqual(Array.from(b.observations.data))
    const path = sampleHiddenPath(stream(5), h, Array.from(a.observations.data))
    expect(path.shape).toEqual([50])
    expect(Array.from(path.data).every((k) => k === 0 || k === 1)).toBe(true)
  })
  it('stepping reveals the same result', () => {
    const s = run(forwardBackwardSteps, { model: h, observations: obs }, 100)
    close(s.marginals.data, fb.marginals.data, 0)
    const v = run(viterbiSteps, { model: h, observations: obs }, 100)
    expect(Array.from(v.path.data)).toEqual(brute.path)
  })
})

describe('linear-chain CRF', () => {
  const s = stream(21)
  const K = 3
  const F = 2
  const N = 5
  const rnd = (k: string, n: number) => Array.from({ length: n }, (_, i) => s.child(k, i).uniform() * 2 - 1)
  const crf = linearChainCrf(
    [rnd('w0', F), rnd('w1', F), rnd('w2', F)],
    [rnd('t0', K), rnd('t1', K), rnd('t2', K)],
    rnd('s', K),
  )
  const x = Array.from({ length: N }, (_, n) => rnd(`x${n}`, F))
  const y = [0, 2, 1, 1, 0]
  const all = () => {
    const out: number[][] = []
    for (let c = 0; c < K ** N; c++) out.push(Array.from({ length: N }, (_, n) => Math.floor(c / K ** (N - 1 - n)) % K))
    return out
  }
  it('log Z, marginals and Viterbi against enumeration', () => {
    const scores = all().map((lab) => ({ lab, s: crfScore(crf, x, lab) }))
    const logZ = Math.log(scores.reduce((t, { s }) => t + Math.exp(s), 0))
    const m = crfMarginals(crf, x)
    expect(m.logZ).toBeCloseTo(logZ, 12)
    const p0 = scores.filter(({ lab }) => lab[2] === 1).reduce((t, { s }) => t + Math.exp(s - logZ), 0)
    expect(m.marginals.data[2 * K + 1]).toBeCloseTo(p0, 12)
    const best = scores.reduce((a, b) => (b.s > a.s ? b : a))
    expect(Array.from(crfViterbi(crf, x).path.data)).toEqual(best.lab)
  })
  it('the gradient matches finite differences', () => {
    const g = crfGradient(crf, x, y)
    expect(g.logLikelihood).toBeCloseTo(crfLogLikelihood(crf, x, y), 12)
    const h = 1e-6
    const perturb = (block: 'weights' | 'transitions' | 'start', i: number, d: number) => {
      const copy = { weights: crf.weights, transitions: crf.transitions, start: crf.start }
      const data = Float64Array.from(copy[block].data)
      data[i] += d
      return { ...copy, [block]: fromData(data, copy[block].shape) }
    }
    for (const block of ['weights', 'transitions', 'start'] as const)
      for (let i = 0; i < crf[block].data.length; i++) {
        const fd =
          (crfLogLikelihood(perturb(block, i, h), x, y) - crfLogLikelihood(perturb(block, i, -h), x, y)) / (2 * h)
        expect(Math.abs(g[block].data[i] - fd)).toBeLessThan(1e-6)
      }
  })
})

// ── Conjugate Gibbs and LDA ─────────────────────────────────────────────────────────────────────────────────────────

describe('Gibbs from conjugate conditionals', () => {
  it('Beta–Bernoulli: the chain mean approaches the posterior mean', () => {
    const coin = model('coin', (m) => {
      const p = m.variable('p', dist.Beta(2, 2))
      m.plate('flips', 'n').observed('x', dist.Bernoulli(p))
    })
    const x = [1, 1, 1, 0, 1, 1, 0, 1, 1, 1]
    const t = trace(gibbsSteps, { model: coin, bindings: { data: { x } } }, 3000, {
      stream: stream(9),
      record: { p: (s: GibbsState) => s.values.p as number },
    })
    const mean =
      toFlat(t.series.p)
        .slice(1)
        .reduce((a, b) => a + b, 0) / 3000
    expect(Math.abs(mean - (2 + 8) / (4 + 10))).toBeLessThan(0.01)
  })
  it('Normal mean with a mixture indicator: enumerated and conjugate conditionals together', () => {
    const mix = model('two means', (m) => {
      const mu = m.plate('components', 2).variable('μ', dist.Normal(0, 10))
      const points = m.plate('points', 'n')
      const z = points.variable('z', dist.Categorical([0.5, 0.5]))
      points.observed('x', dist.Normal(mu.at(z), 1))
    })
    const x = [-5.1, -4.8, -5.3, 4.9, 5.2, 5.0]
    const s = run(gibbsSteps, { model: mix, bindings: { data: { x } } }, 200, { stream: stream(2) })
    const means = [s.values['μ[0]'] as number, s.values['μ[1]'] as number].sort((a, b) => a - b)
    expect(Math.abs(means[0] + 5)).toBeLessThan(1.5)
    expect(Math.abs(means[1] - 5)).toBeLessThan(1.5)
    expect(s.kinds['μ[0]']).toBe('normal')
    expect(s.kinds['z[0]']).toBe('enumerate')
  })
})

describe('LDA', () => {
  const lda = ldaModel()
  const bindings = {
    sizes: { K: 2, V: 4 },
    constants: { α: 0.5, β: 0.1 },
    data: {
      w: [
        [0, 1, 0, 1, 0],
        [2, 3, 3, 2],
        [0, 0, 1, 1, 1],
        [3, 2, 2, 3, 3],
      ],
    },
  }
  it('the description expands, samples and has the expected Markov blanket', () => {
    const em = expandModel(lda, bindings)
    expect(em.byNode.get('z')!.length).toBe(19)
    const mb = markovBlanket(lda, 'z[0,0]', bindings)
    expect(mb.parents).toEqual(['θ[0]'])
    expect(mb.children).toEqual(['w[0,0]'])
    expect(mb.coParents.sort()).toEqual(['φ[0]', 'φ[1]'])
    const draw = sampleModel(stream(1), lda, bindings)
    expect(Number.isFinite(logJoint(em, draw))).toBe(true)
  })
  it('infer picks the registered collapsed Gibbs engine; the collapsed likelihood improves', () => {
    const inf = infer(lda, bindings)
    expect(inf.engine).toBe('lda-collapsed-gibbs')
    const t = trace(inf.algorithm as Algorithm<unknown, LdaState>, inf.options, 50, {
      stream: stream(8),
      record: { ll: (s) => s.logLikelihood },
    })
    const ll = toFlat(t.series.ll)
    expect(ll[ll.length - 1]).toBeGreaterThan(ll[0])
  })
  it('uncollapsed Gibbs runs from conjugate Dirichlet conditionals', () => {
    const s = run(gibbsSteps, { model: lda, bindings }, 5, { stream: stream(3) })
    expect(s.kinds['φ[0]']).toBe('dirichlet')
    expect(s.kinds['θ[1]']).toBe('dirichlet')
  })
  it('diagrams: plates around their nodes, a factor graph with a highlighted blanket', () => {
    const plate = toPlateDiagram(lda)
    expect(plate.nodes.map((n) => n.id)).toEqual(['α', 'β', 'φ', 'θ', 'z', 'w'])
    expect(plate.nodes.find((n) => n.id === 'w')!.filled).toBe(true)
    expect(plate.groups.find((g) => g.id === 'plate:words')!.around).toEqual(['z', 'w'])
    expect(plate.groups.find((g) => g.id === 'plate:documents')!.around).toEqual(['θ', 'z', 'w'])
    const small = { ...bindings, data: { w: [[0, 1]] } }
    const fg = toFactorDiagram(lda, { bindings: small, highlight: 'z[0,0]' })
    expect(fg.nodes.find((n) => n.id === 'var z[0,0]')!.state).toBe('active')
    expect(fg.nodes.find((n) => n.id === 'var φ[1]')!.state).toBe('done')
    expect(fg.nodes.find((n) => n.id === 'var z[0,1]')!.state).toBe('idle')
    expect(toFactorGraph(lda, small).factors.length).toBe(2 + 1 + 2 + 2)
  })
})

// ── Trace protocol ──────────────────────────────────────────────────────────────────────────────────────────────────

describe('protocol', () => {
  const grid = isingModel(fromEdges(4, gridEdges(2, 2), { directed: false }), 0.4, 0.1)
  const h = dishonestCasino()
  const lda = {
    documents: [
      [0, 1, 0, 1],
      [2, 3, 2],
    ],
    topics: 2,
    vocabulary: 4,
    alpha: 0.5,
    beta: 0.1,
  }
  const J: Matrix = tensor([
    [3, 1, 0],
    [1, 3, 1],
    [0, 1, 3],
  ])
  const cases: [string, Algorithm<unknown, unknown>, unknown, (s: never) => number][] = [
    ['enumeration', enumerationSteps as never, { graph: grid }, (s: { logZ: number }) => s.logZ],
    [
      'variable elimination',
      variableEliminationSteps as never,
      { graph: grid },
      (s: { logScale: number }) => s.logScale,
    ],
    ['belief propagation', beliefPropagationSteps as never, { graph: grid }, (s: { change: number }) => s.change],
    [
      'gaussian bp',
      gaussianBeliefPropagationSteps as never,
      { precision: J, shift: [1, 0, 1] },
      (s: { means: { data: Float64Array } }) => s.means.data[0],
    ],
    [
      'forward-backward',
      forwardBackwardSteps as never,
      { model: h, observations: [5, 0, 5] },
      (s: { logLikelihood: number }) => s.logLikelihood,
    ],
    ['viterbi', viterbiSteps as never, { model: h, observations: [5, 0, 5] }, (s: { position: number }) => s.position],
    [
      'factor-graph gibbs',
      factorGraphGibbsSteps as never,
      { graph: grid },
      (s: { assignment: { data: Int32Array } }) => s.assignment.data[0],
    ],
    [
      'model gibbs',
      gibbsSteps as never,
      { model: sprinkler, bindings: { data: { wet: 1 } } },
      (s: { logJoint: number }) => s.logJoint,
    ],
    ['lda collapsed gibbs', ldaCollapsedGibbsSteps as never, lda, (s: { logLikelihood: number }) => s.logLikelihood],
  ]
  it.each(cases)('%s: same stream → same trace; seek equals run; extend equals a longer trace', (_, alg, opts, rec) => {
    const record = { v: rec as (s: unknown) => number }
    const n = 6
    const a = trace(alg, opts, n, { record, stream: stream(5), stopOnNonFinite: false })
    const b = trace(alg, opts, n, { record, stream: stream(5), stopOnNonFinite: false })
    expect(toFlat(a.series.v)).toEqual(toFlat(b.series.v))
    const k = Math.min(3, a.meta.steps)
    expect(rec(seek(alg, opts, k, { stream: stream(5) }) as never)).toEqual(
      rec(run(alg, opts, k, { stream: stream(5) }) as never),
    )
    const short = trace(alg, opts, 2, { record, stream: stream(5), stopOnNonFinite: false })
    const longer = extend(short, alg, opts, n - 2)
    expect(toFlat(longer.series.v)).toEqual(toFlat(a.series.v))
  })
})
