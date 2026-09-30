import { describe, expect, it } from 'vitest'
import { cellState, tabularMdp } from 'aifn-applied/decisions/reinforcement-learning'
import { cliffWalking, frozenLake, gridworld, maze } from 'aifn-applied/data/environments'
import {
  evaluatePolicy,
  policyEvaluation,
  policyIteration,
  valueIteration,
} from 'aifn-applied/decisions/reinforcement-learning/planning'
import {
  expectedSarsa,
  greedyPath,
  monteCarloControl,
  nStepSarsa,
  qLearning,
  reinforce,
  sarsa,
  tdPrediction,
} from 'aifn-applied/decisions/reinforcement-learning/learning'
import { stream } from 'aifn/foundation/random'
import { toFlat } from 'aifn/foundation/tensor'
import { extend, run, seek, trace } from 'aifn/foundation/trace'

describe('planning', () => {
  it('value iteration solves a hand-worked corridor', () => {
    const corridor = gridworld({ width: 3, height: 1, walls: [], terminals: [{ x: 2, y: 0, value: 1 }], noise: 0 })
    const s = run(valueIteration(corridor), {}, 100)
    expect(s.converged).toBe(true)
    const V = toFlat(s.V)
    expect(V[1]).toBeCloseTo(0.9, 12)
    expect(V[0]).toBeCloseTo(0.81, 12)
    expect(toFlat(s.policy)).toEqual([1, 1, -1])
  })

  it("value iteration reproduces Russell and Norvig's 4 × 3 utilities", () => {
    const g = gridworld({ stepReward: -0.04, gamma: 1, noise: 0.2 })
    const V = toFlat(run(valueIteration(g, { tolerance: 1e-12 }), {}, 2000).V)
    const at = (x: number, y: number) => V[cellState(4, x, y)]
    expect(at(0, 0)).toBeCloseTo(0.705, 3)
    expect(at(0, 2)).toBeCloseTo(0.812, 3)
    expect(at(2, 2)).toBeCloseTo(0.918, 3)
    expect(at(2, 1)).toBeCloseTo(0.66, 3)
    expect(at(3, 0)).toBeCloseTo(0.388, 3)
  })

  it('policy iteration and value iteration agree, and exact evaluation matches iterative', () => {
    for (const mdp of [gridworld({ noise: 0.2 }), frozenLake(), maze(['..G', '.#.', 'S..'], { slip: 0.1 })]) {
      const vi = run(valueIteration(mdp), {}, 5000)
      const pi = run(policyIteration(mdp), {}, 50)
      expect(pi.stable).toBe(true)
      const a = toFlat(vi.V)
      toFlat(pi.V).forEach((v, i) => expect(v).toBeCloseTo(a[i], 6))
      const ev = run(policyEvaluation(mdp, pi.policy), {}, 5000)
      toFlat(ev.V).forEach((v, i) => expect(v).toBeCloseTo(toFlat(evaluatePolicy(mdp, pi.policy))[i], 6))
    }
  })

  it('tabularMdp builds a general MDP', () => {
    // Two states: stay pays 1, leave to a terminal pays 5.
    const m = tabularMdp({
      transitions: [
        [
          [1, 0],
          [0, 1],
        ],
        [
          [0, 1],
          [0, 1],
        ],
      ],
      rewards: [
        [1, 5],
        [0, 0],
      ],
      gamma: 0.5,
      terminal: [1],
    })
    expect(toFlat(run(valueIteration(m), {}, 200).V)[0]).toBeCloseTo(5, 10)
  })

  it('value iteration follows the trace protocol', () => {
    const alg = valueIteration(gridworld())
    expect(toFlat(seek(alg, {}, 5).V)).toEqual(toFlat(run(alg, {}, 5).V))
    const rec = { record: { residual: (s: { residual: number }) => s.residual } }
    const short = trace(alg, {}, 3, rec)
    const long = trace(alg, {}, 8, rec)
    expect(long.meta.stopped).toBe('limit')
    expect(long.index.length).toBe(9)
    expect(toFlat(extend(short, alg, {}, 5).series.residual)).toEqual(toFlat(long.series.residual))
  })
})

describe('learning', () => {
  const cliff = cliffWalking()

  it('Q-learning finds the cliff-edge path; SARSA earns more online by walking further away', () => {
    const rec = { record: { reward: (s: { rewardSum: number }) => s.rewardSum }, stream: stream(1) }
    const q = trace(qLearning(cliff, { alpha: 0.5, epsilon: 0.1 }), {}, 500, rec)
    const s = trace(sarsa(cliff, { alpha: 0.5, epsilon: 0.1 }), {}, 500, rec)
    const qPath = toFlat(greedyPath(cliff, q.steps.at(-1)!.policy))
    const sPath = toFlat(greedyPath(cliff, s.steps.at(-1)!.policy))
    expect(qPath.length).toBe(14)
    expect(qPath.at(-1)).toBe(11)
    expect(sPath.length).toBeGreaterThan(14)
    const tail = (t: typeof q) =>
      toFlat(t.series.reward)
        .slice(-100)
        .reduce((a, b) => a + b, 0) / 100
    expect(tail(s)).toBeGreaterThan(tail(q))
  })

  it('expected SARSA, n-step SARSA and Monte Carlo control learn to reach the goal', () => {
    const m = maze(['...G', '.#..', 'S...'])
    for (const alg of [expectedSarsa(m), nStepSarsa(m, { n: 3 }), monteCarloControl(m, { epsilon: 0.2 })]) {
      const last = run(alg, {}, 300, { stream: stream(2) })
      const path = toFlat(greedyPath(m, last.policy))
      expect(path.at(-1)).toBe(cellState(4, 3, 2))
    }
  })

  it('TD(0) prediction approaches the exact value of a fixed policy', () => {
    const g = gridworld({ noise: 0.2 })
    const policy = run(valueIteration(g), {}, 500).policy
    const exact = toFlat(evaluatePolicy(g, policy))
    const est = toFlat(run(tdPrediction(g, policy, { alpha: 0.02 }), {}, 3000, { stream: stream(3) }).V)
    expect(Math.abs(est[0] - exact[0])).toBeLessThan(0.08)
  })

  it('REINFORCE improves the return on the corridor', () => {
    const corridor = maze(['S....G'], { rewards: { step: -1, goal: 10 } })
    const t = trace(reinforce(corridor, { alpha: 0.05 }), {}, 400, {
      record: { reward: (s) => s.rewardSum },
      stream: stream(4),
    })
    const r = toFlat(t.series.reward)
    const mean = (a: number[]) => a.reduce((x, y) => x + y, 0) / a.length
    expect(mean(r.slice(-50))).toBeGreaterThan(mean(r.slice(1, 51)))
  })

  it('learners follow the trace protocol', () => {
    const alg = qLearning(cliff)
    const opts = { stream: stream(5) }
    expect(toFlat(seek(alg, {}, 7, opts).Q)).toEqual(toFlat(run(alg, {}, 7, opts).Q))
    const rec = { record: { g: (s: { return: number }) => s.return }, stream: stream(5) }
    const short = trace(alg, {}, 4, rec)
    const long = trace(alg, {}, 10, rec)
    expect(long.index.length).toBe(11)
    expect(toFlat(extend(short, alg, {}, 6).series.g)).toEqual(toFlat(long.series.g))
  })
})
