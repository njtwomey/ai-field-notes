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
import { run, trace } from 'aifn/foundation/trace'
import { expectProtocol } from '../../protocol'

describe('planning', () => {
  it('value iteration solves a hand-worked corridor', () => {
    const corridor = gridworld({ width: 3, height: 1, walls: [], terminals: [{ x: 2, y: 0, value: 1 }], noise: 0 })
    const s = run(valueIteration(corridor), undefined, 100)
    expect(s.converged).toBe(true)
    const V = toFlat(s.V)
    expect(V[1]).toBeCloseTo(0.9, 12)
    expect(V[0]).toBeCloseTo(0.81, 12)
    expect(toFlat(s.policy)).toEqual([1, 1, -1])
  })

  it("value iteration reproduces Russell and Norvig's 4 × 3 utilities", () => {
    const g = gridworld({ stepReward: -0.04, gamma: 1, noise: 0.2 })
    const V = toFlat(run(valueIteration(g, { tolerance: 1e-12 }), undefined, 2000).V)
    const at = (x: number, y: number) => V[cellState(4, x, y)]
    expect(at(0, 0)).toBeCloseTo(0.705, 3)
    expect(at(0, 2)).toBeCloseTo(0.812, 3)
    expect(at(2, 2)).toBeCloseTo(0.918, 3)
    expect(at(2, 1)).toBeCloseTo(0.66, 3)
    expect(at(3, 0)).toBeCloseTo(0.388, 3)
  })

  it('policy iteration and value iteration agree, and exact evaluation matches iterative', () => {
    for (const mdp of [gridworld({ noise: 0.2 }), frozenLake(), maze(['..G', '.#.', 'S..'], { slip: 0.1 })]) {
      const vi = run(valueIteration(mdp), undefined, 5000)
      const pi = run(policyIteration(mdp), undefined, 50)
      expect(pi.converged).toBe(true)
      const a = toFlat(vi.V)
      toFlat(pi.V).forEach((v, i) => expect(v).toBeCloseTo(a[i], 6))
      const ev = run(policyEvaluation(mdp, pi.policy), undefined, 5000)
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
    expect(toFlat(run(valueIteration(m), undefined, 200).V)[0]).toBeCloseTo(5, 10)
  })

  it('planners follow the trace protocol', () => {
    const g = gridworld({ noise: 0.2 })
    const record = { residual: (s: { residual: number }) => s.residual }
    expectProtocol(valueIteration(g), undefined, { record })
    expectProtocol(policyEvaluation(g, run(policyIteration(g), undefined, 50).policy), undefined, { record })
    expectProtocol(policyIteration(g), undefined, { n: 4 })
  })
})

describe('learning', () => {
  const cliff = cliffWalking()

  it('Q-learning finds the cliff-edge path; SARSA earns more online by walking further away', () => {
    const rec = { record: { reward: (s: { rewardSum: number }) => s.rewardSum }, stream: stream(1) }
    const q = trace(qLearning(cliff, { learningRate: 0.5, epsilon: 0.1 }), undefined, 500, rec)
    const s = trace(sarsa(cliff, { learningRate: 0.5, epsilon: 0.1 }), undefined, 500, rec)
    const qPath = toFlat(greedyPath(cliff, q.final.policy))
    const sPath = toFlat(greedyPath(cliff, s.final.policy))
    expect(qPath.length).toBe(14)
    expect(qPath.at(-1)).toBe(11)
    // SARSA's greedy path leaves the row next to the cliff for a safer one (states 24 and up are two rows away).
    expect(Math.max(...sPath)).toBeGreaterThanOrEqual(24)
    const tail = (t: typeof q) =>
      toFlat(t.series.reward)
        .slice(-100)
        .reduce((a, b) => a + b, 0) / 100
    expect(tail(s)).toBeGreaterThan(tail(q))
  })

  it('expected SARSA, n-step SARSA and Monte Carlo control learn to reach the goal', () => {
    const m = maze(['...G', '.#..', 'S...'])
    for (const alg of [expectedSarsa(m), nStepSarsa(m, { n: 3 }), monteCarloControl(m, { epsilon: 0.2 })]) {
      const last = run(alg, undefined, 300, { stream: stream(2) })
      const path = toFlat(greedyPath(m, last.policy))
      expect(path.at(-1)).toBe(cellState(4, 3, 2))
    }
  })

  it('TD(0) prediction approaches the exact value of a fixed policy', () => {
    const g = gridworld({ noise: 0.2 })
    const policy = run(valueIteration(g), undefined, 500).policy
    const exact = toFlat(evaluatePolicy(g, policy))
    const est = toFlat(run(tdPrediction(g, policy, { learningRate: 0.02 }), undefined, 3000, { stream: stream(3) }).V)
    expect(Math.abs(est[0] - exact[0])).toBeLessThan(0.08)
  })

  it('REINFORCE improves the return on the corridor', () => {
    const corridor = maze(['S....G'], { rewards: { step: -1, goal: 10 } })
    const t = trace(reinforce(corridor, { learningRate: 0.05 }), undefined, 400, {
      record: { reward: (s) => s.rewardSum },
      stream: stream(4),
    })
    const r = toFlat(t.series.reward)
    const mean = (a: number[]) => a.reduce((x, y) => x + y, 0) / a.length
    expect(mean(r.slice(-50))).toBeGreaterThan(mean(r.slice(1, 51)))
  })

  it('learners follow the trace protocol', () => {
    const g = gridworld({ noise: 0.2 })
    const policy = run(valueIteration(g), undefined, 500).policy
    const record = { g: (s: { return: number }) => s.return }
    for (const alg of [
      qLearning(cliff),
      sarsa(cliff),
      expectedSarsa(cliff),
      nStepSarsa(cliff, { n: 3 }),
      monteCarloControl(cliff),
      reinforce(cliff),
    ])
      expectProtocol(alg, undefined, { n: 10, record })
    expectProtocol(tdPrediction(g, policy), undefined, { n: 10 })
  })
})
