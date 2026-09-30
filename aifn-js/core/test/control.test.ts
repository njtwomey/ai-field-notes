import { describe, expect, test } from 'vitest'
import { ackermann, dlqr, lqr } from 'aifn/dynamics/control'
import {
  bode,
  controllability,
  controllabilityGramian,
  discretise,
  feedback,
  impulseResponse,
  margins,
  observability,
  poles,
  simulate,
  stateFeedback,
  stateSpace,
  stateSpaceToTf,
  stepResponse,
  tfToStateSpace,
  transferFunction,
} from 'aifn/systems'
import { doubling, hamiltonianSign, kleinman, lyapunov, riccatiRecursion } from 'aifn/numerics/linalg'
import { pid } from 'aifn-applied/dynamics/control'
import { eig } from 'aifn/numerics/linalg'
import { toFlat, toRows, type Tensor } from 'aifn/foundation/tensor'
import { extend, run, seek, trace, type Algorithm } from 'aifn/foundation/trace'
import { fixture } from './fixtures'

type Mat = number[][]
type Riccati = { A: Mat; B: Mat; Q: Mat; R: Mat; P: Mat }
type SS = { A: Mat; B: Mat; C: Mat; D: Mat }
const F = fixture<{
  care: Riccati
  cartpole: Riccati
  dare: Riccati
  dare_stable: Riccati
  lyapunov: { A: Mat; Q: Mat; X: Mat; Ad: Mat; Xd: Mat }
  discretise: SS & { dt: number; zoh: SS; tustin: SS }
  tf2ss: SS & { num: number[]; den: number[] }
  ss2tf: { num: number[]; den: number[] }
  bode: { num: number[]; den: number[]; w: number[]; mag: number[]; phase: number[] }
  margins: {
    num: number[]
    den: number[]
    gainCrossover: number
    phaseMargin: number
    phaseCrossover: number
    gainMargin: number
  }
}>('control')

const close = (a: Tensor | null, b: Mat | number[], tol: number) => {
  expect(a).not.toBeNull()
  const x = toFlat(a!)
  const y = (b as (number | number[])[]).flat()
  expect(x.length).toBe(y.length)
  const scale = Math.max(1, ...y.map(Math.abs))
  x.forEach((v, i) => expect(Math.abs(v - y[i])).toBeLessThan(tol * scale))
}

/** The trace protocol: same inputs → same trace; seek(i) = run(i); extend = a longer trace. */
function protocol<O, S>(alg: Algorithm<O, S>, opts: O, n: number) {
  const a = trace(alg, opts, n, { checkpointEvery: 2 })
  const b = trace(alg, opts, n)
  expect(JSON.stringify(a.steps)).toBe(JSON.stringify(b.steps))
  const i = Math.min(3, a.meta.steps)
  expect(JSON.stringify(seek(alg, opts, i, { checkpoints: a }))).toBe(JSON.stringify(run(alg, opts, i)))
  const short = trace(alg, opts, Math.min(2, n))
  const longer = extend(short, alg, opts, n - Math.min(2, n))
  expect(JSON.stringify(longer.steps.at(-1))).toBe(JSON.stringify(b.steps.at(-1)))
}

describe('Riccati equations and LQR', () => {
  test('CARE by Kleinman and by the sign function matches scipy', () => {
    for (const c of [F.care, F.cartpole]) {
      const k = lqr({ A: c.A, B: c.B }, c.Q, c.R)
      expect(k.converged).toBe(true)
      close(k.P, c.P, 1e-9)
      const s = lqr({ A: c.A, B: c.B }, c.Q, c.R, { method: 'sign' })
      expect(s.converged).toBe(true)
      close(s.P, c.P, 1e-8)
      expect(Math.max(...toFlat(k.closedLoop.real))).toBeLessThan(0)
    }
  })

  test('DARE by doubling and by the recursion matches scipy', () => {
    for (const c of [F.dare, F.dare_stable]) {
      const d = dlqr({ A: c.A, B: c.B }, c.Q, c.R)
      expect(d.converged).toBe(true)
      close(d.P, c.P, 1e-9)
    }
    const r = dlqr({ A: F.dare.A, B: F.dare.B }, F.dare.Q, F.dare.R, { method: 'recursion' })
    expect(r.converged).toBe(true)
    close(r.P, F.dare.P, 1e-8)
    const cl = r.closedLoop
    toFlat(cl.real).forEach((v, i) => expect(Math.hypot(v, toFlat(cl.imag)[i])).toBeLessThan(1))
  })

  test('Riccati algorithms follow the trace protocol', () => {
    const c = F.care
    protocol(kleinman(c), {}, 6)
    protocol(hamiltonianSign(c), {}, 6)
    protocol(riccatiRecursion(F.dare), {}, 6)
    protocol(doubling(F.dare), {}, 5)
  })

  test('an uncontrollable unstable mode is reported, not hidden', () => {
    const res = lqr(
      {
        A: [
          [1, 0],
          [0, -1],
        ],
        B: [[0], [1]],
      },
      [
        [1, 0],
        [0, 1],
      ],
      [[1]],
    )
    expect(res.converged).toBe(false)
    expect(res.failure).toBe('not stabilisable')
  })
})

describe('structure', () => {
  test('controllability and observability ranks', () => {
    const di = stateSpace({
      A: [
        [0, 1],
        [0, 0],
      ],
      B: [0, 1],
      C: [1, 0],
    })
    expect(controllability(di).full).toBe(true)
    expect(observability(di).full).toBe(true)
    // Two decoupled identical modes driven by one input: rank 1.
    const bad = stateSpace({
      A: [
        [-1, 0],
        [0, -1],
      ],
      B: [1, 1],
      C: [[0, 1]],
    })
    expect(controllability(bad).rank).toBe(1)
    expect(
      observability(
        stateSpace({
          A: [
            [-1, 0],
            [0, -2],
          ],
          B: [1, 1],
          C: [[1, 0]],
        }),
      ).rank,
    ).toBe(1)
  })

  test('Lyapunov equations match scipy', () => {
    close(lyapunov(F.lyapunov.A, F.lyapunov.Q).X, F.lyapunov.X, 1e-10)
    close(lyapunov(F.lyapunov.Ad, F.lyapunov.Q, { discrete: true }).X, F.lyapunov.Xd, 1e-10)
    const g = controllabilityGramian(stateSpace({ A: [[-1]], B: [1] }))
    close(g.W, [[0.5]], 1e-12)
    expect(controllabilityGramian(stateSpace({ A: [[1]], B: [1] })).W).toBeNull()
  })

  test('Ackermann places the requested poles', () => {
    const sys = stateSpace({ A: F.cartpole.A, B: F.cartpole.B })
    const p = ackermann(sys, { real: [-1, -1, -2, -3], imag: [1, -1, 0, 0] })
    expect(p.controllable).toBe(true)
    const e = eig(toRows(stateFeedback(sys, p.K!).A))
    close(e.real, [-1, -1, -2, -3], 1e-8)
    close(e.imag, [1, -1, 0, 0], 1e-8)
    const bad = ackermann(
      stateSpace({
        A: [
          [-1, 0],
          [0, -1],
        ],
        B: [1, 1],
      }),
      { real: [-2, -3] },
    )
    expect(bad.K).toBeNull()
  })
})

describe('discretisation and simulation', () => {
  const c = F.discretise
  const sys = stateSpace({ A: c.A, B: c.B, C: c.C, D: c.D })
  test('ZOH and Tustin match scipy cont2discrete', () => {
    for (const method of ['zoh', 'tustin'] as const) {
      const d = discretise(sys, c.dt, method)
      const ref = c[method]
      close(d.A, ref.A, 1e-12)
      close(d.B, ref.B, 1e-12)
      close(d.C, ref.C, 1e-12)
      close(d.D, ref.D, 1e-12)
    }
  })

  test('step response of a first-order lag is exact at the samples', () => {
    const lag = stateSpace({ A: [[-2]], B: [2], C: [1] })
    const r = stepResponse(lag, { tEnd: 2, dt: 0.1 })
    toFlat(r.t).forEach((t, k) => expect(toFlat(r.y)[k]).toBeCloseTo(1 - Math.exp(-2 * t), 12))
    const imp = impulseResponse(lag, { tEnd: 1, dt: 0.1 })
    toFlat(imp.t).forEach((t, k) => expect(toFlat(imp.y)[k]).toBeCloseTo(2 * Math.exp(-2 * t), 12))
  })

  test('simulate follows the trace protocol and closes state feedback', () => {
    const di = stateSpace({
      A: [
        [0, 1],
        [0, 0],
      ],
      B: [0, 1],
      C: [1, 0],
    })
    const K = toFlat(
      lqr(
        di,
        [
          [1, 0],
          [0, 1],
        ],
        [[1]],
      ).K,
    )
    const alg = simulate(di, (_t, x) => -(K[0] * toFlat(x)[0] + K[1] * toFlat(x)[1]), { dt: 0.05, tEnd: 10 })
    protocol(alg, { x0: [1, 0] }, 20)
    const end = run(alg, { x0: [1, 0] }, 1000)
    expect(end.t).toBeCloseTo(10, 9)
    expect(Math.abs(toFlat(end.x)[0])).toBeLessThan(1e-3)
    expect(poles(stateFeedback(di, [K])).stable).toBe(true)
  })
})

describe('transfer functions', () => {
  test('tf ↔ ss conversions match scipy', () => {
    const t = F.tf2ss
    const s = tfToStateSpace(transferFunction(t.num, t.den))
    close(s.A, t.A, 1e-14)
    close(s.B, t.B, 1e-14)
    close(s.C, t.C, 1e-14)
    close(s.D, t.D, 1e-14)
    const g = stateSpaceToTf(stateSpace({ A: F.cartpole.A, B: F.cartpole.B, C: [[1, 0, 0, 0]] }))
    close(g.den, F.ss2tf.den, 1e-10)
    const num = [...new Array(F.ss2tf.num.length - toFlat(g.num).length).fill(0), ...toFlat(g.num)]
    num.forEach((v, i) => expect(v).toBeCloseTo(F.ss2tf.num[i], 10))
  })

  test('Bode data match scipy', () => {
    const b = bode(transferFunction(F.bode.num, F.bode.den), F.bode.w)
    close(b.magnitudeDb, F.bode.mag, 1e-10)
    close(b.phase, F.bode.phase, 1e-9)
  })

  test('margins of 2/(s(s+1)(s+2)) and a delay', () => {
    const L = transferFunction(F.margins.num, F.margins.den)
    const m = margins(L)
    expect(m.gainMargin).toBeCloseTo(F.margins.gainMargin, 8)
    expect(m.phaseCrossover).toBeCloseTo(F.margins.phaseCrossover, 8)
    expect(m.phaseMargin).toBeCloseTo(F.margins.phaseMargin, 7)
    expect(m.gainCrossover).toBeCloseTo(F.margins.gainCrossover, 8)
    // Adding exactly the delay margin brings the phase margin to zero.
    const delayed = transferFunction(F.margins.num, F.margins.den, { delay: m.delayMargin })
    expect(margins(delayed).phaseMargin).toBeCloseTo(0, 6)
    const cl = feedback(L)
    expect(poles(tfToStateSpace(cl)).stable).toBe(true)
  })
})

describe('PID', () => {
  const plant = stateSpace({
    A: [
      [0, 1],
      [-1, -2],
    ],
    B: [0, 1],
    C: [1, 0],
  }) // 1/(s + 1)²
  test('a PI loop tracks a step with zero steady-state error', () => {
    const s = run(pid(plant, { kp: 2, ki: 1 }, { dt: 0.01, tEnd: 30 }), {}, 1e5)
    expect(s.t).toBeCloseTo(30, 9)
    expect(Math.abs(s.e)).toBeLessThan(1e-4)
  })

  test('anti-windup reduces overshoot under saturation', () => {
    const peak = (antiWindup: 'none' | 'clamp' | 'back-calculation') => {
      const t = trace(
        pid(plant, { kp: 2, ki: 2 }, { dt: 0.01, tEnd: 20, uMax: 1.2, uMin: -1.2, antiWindup }),
        {},
        2000,
        {
          record: { y: (s) => s.y },
        },
      )
      expect(t.steps.some((s) => s.saturated)).toBe(true)
      return Math.max(...toFlat(t.series.y))
    }
    expect(peak('clamp')).toBeLessThan(peak('none'))
    expect(peak('back-calculation')).toBeLessThan(peak('none'))
  })

  test('PID follows the trace protocol, with a delay', () => {
    protocol(pid(plant, { kp: 1, ki: 0.5, kd: 0.2, filter: 0.05 }, { dt: 0.05, delay: 0.2 }), {}, 12)
  })
})
