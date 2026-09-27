import { useMemo } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, useParam, type XYSeries } from '@/components/viz'
import { rng } from '@/lib/math'

const T = 200
const ITERATIONS = 150
const TRUTH = { a: 0.9, q: 0.5, r: 1 }
const START = { a: 0.2, q: 2, r: 0.2 }
/** Prior on the initial state z_0, held fixed. */
const MU0 = 0
const P0 = 1

type Params = { a: number; q: number; r: number }

function simulate(seed: number) {
  const g = rng(seed)
  let z = 0
  const truth: number[] = []
  const x: number[] = []
  for (let t = 0; t < T; t++) {
    z = TRUTH.a * z + Math.sqrt(TRUTH.q) * g.normal()
    truth.push(z)
    x.push(z + Math.sqrt(TRUTH.r) * g.normal())
  }
  return { truth, x }
}

/**
 * E-step for z_t = a z_{t−1} + w_t, x_t = z_t + v_t: a Kalman filter (which also gives the log-likelihood) and an RTS
 * smoother, extended back to z_0. Returns the smoothed means, variances and lag-one covariances cov(z_t, z_{t−1}).
 */
function eStep(x: number[], { a, q, r }: Params) {
  const mPred: number[] = []
  const pPred: number[] = []
  const mFilt: number[] = []
  const pFilt: number[] = []
  let m = MU0
  let P = P0
  let logLik = 0
  for (const obs of x) {
    const mp = a * m
    const pp = a * a * P + q
    const S = pp + r
    const e = obs - mp
    logLik -= 0.5 * ((e * e) / S + Math.log(2 * Math.PI * S))
    const K = pp / S
    m = mp + K * e
    P = (1 - K) * pp
    mPred.push(mp)
    pPred.push(pp)
    mFilt.push(m)
    pFilt.push(P)
  }
  const ms = [...mFilt]
  const ps = [...pFilt]
  const cross = new Array<number>(T).fill(0)
  for (let t = T - 2; t >= 0; t--) {
    const G = (pFilt[t] * a) / pPred[t + 1]
    ms[t] = mFilt[t] + G * (ms[t + 1] - mPred[t + 1])
    ps[t] = pFilt[t] + G * G * (ps[t + 1] - pPred[t + 1])
    cross[t + 1] = G * ps[t + 1]
  }
  const G0 = (P0 * a) / pPred[0]
  const m0 = MU0 + G0 * (ms[0] - mPred[0])
  const p0 = P0 + G0 * G0 * (ps[0] - pPred[0])
  cross[0] = G0 * ps[0]
  return { logLik, ms, ps, cross, m0, p0 }
}

/** M-step: closed-form maximisers of the expected complete-data log-likelihood. */
function mStep(x: number[], s: ReturnType<typeof eStep>): Params {
  let zz = 0
  let zPrev = 0
  let prev = 0
  let resid = 0
  for (let t = 0; t < T; t++) {
    const mPrev = t === 0 ? s.m0 : s.ms[t - 1]
    const pPrev = t === 0 ? s.p0 : s.ps[t - 1]
    zz += s.ps[t] + s.ms[t] ** 2
    zPrev += s.cross[t] + s.ms[t] * mPrev
    prev += pPrev + mPrev ** 2
    resid += x[t] ** 2 - 2 * x[t] * s.ms[t] + s.ps[t] + s.ms[t] ** 2
  }
  const a = zPrev / prev
  return { a, q: (zz - a * zPrev) / T, r: resid / T }
}

export function EmFit() {
  const iteration = useParam(10, { min: 0, max: ITERATIONS, step: 1 })
  const seed = useParam(7, { min: 1, max: 20, step: 1 })

  const run = useMemo(() => {
    const data = simulate(seed.value)
    const history: (Params & { logLik: number; ms: number[] })[] = []
    let params: Params = START
    for (let k = 0; k <= ITERATIONS; k++) {
      const s = eStep(data.x, params)
      history.push({ ...params, logLik: s.logLik, ms: s.ms })
      params = mStep(data.x, s)
    }
    return { data, history }
  }, [seed.value])

  const k = iteration.value
  const now = run.history[k]
  const its = run.history.map((_, i) => i)
  const likelihood: XYSeries[] = [
    { name: 'log-likelihood', type: 'line', x: its, y: run.history.map((h) => h.logLik), slot: 0 },
    { name: 'iteration k', type: 'scatter', x: [k], y: [now.logLik], emphasis: true },
  ]
  const params: XYSeries[] = [
    { name: 'a', type: 'line', x: its, y: run.history.map((h) => h.a), slot: 0 },
    { name: 'q', type: 'line', x: its, y: run.history.map((h) => h.q), slot: 1 },
    { name: 'r', type: 'line', x: its, y: run.history.map((h) => h.r), slot: 2 },
    { name: 'true a', type: 'line', x: [0, ITERATIONS], y: [TRUTH.a, TRUTH.a], slot: 0, dashed: true },
    { name: 'true q', type: 'line', x: [0, ITERATIONS], y: [TRUTH.q, TRUTH.q], slot: 1, dashed: true },
    { name: 'true r', type: 'line', x: [0, ITERATIONS], y: [TRUTH.r, TRUTH.r], slot: 2, dashed: true },
  ]
  const shown = 100
  const time = Array.from({ length: shown }, (_, t) => t + 1)
  const series: XYSeries[] = [
    { name: 'observations', type: 'scatter', x: time, y: run.data.x.slice(0, shown), muted: true },
    { name: 'true state', type: 'line', x: time, y: run.data.truth.slice(0, shown), emphasis: true },
    { name: 'smoothed state at iteration k', type: 'line', x: time, y: now.ms.slice(0, shown), slot: 0 },
  ]

  return (
    <Interactive
      title="Fitting a linear-Gaussian model by EM"
      caption={`Data: 200 steps of z_t = a z_{t−1} + w_t, x_t = z_t + v_t with a = ${TRUTH.a}, q = ${TRUTH.q}, r = ${TRUTH.r}. EM starts from a = ${START.a}, q = ${START.q}, r = ${START.r}. Each iteration runs the Kalman smoother (E-step) and then updates a, q and r in closed form (M-step). The log-likelihood rises at every iteration. It rises fast at first, as a is corrected, then slowly, as EM trades process noise q against measurement noise r: many (q, r) pairs explain the data almost equally well. With 200 points the maximum-likelihood estimates differ from the true values by sampling error.`}
      controls={
        <>
          <ParamSlider label="EM iteration k" param={iteration} format={(v) => String(v)} withArrows />
          <ParamSlider label="seed" param={seed} format={(v) => String(v)} withArrows />
        </>
      }
      readout={
        <>
          <Readout label="a" value={formatNumber(now.a)} />
          <Readout label="q" value={formatNumber(now.q)} />
          <Readout label="r" value={formatNumber(now.r)} />
          <Readout label="log-likelihood" value={formatNumber(now.logLik)} />
        </>
      }
    >
      <div className="space-y-4">
        <XYChart series={series} xLabel="t" yLabel="state" height={220} />
        <div className="grid gap-4 sm:grid-cols-2">
          <XYChart series={likelihood} xLabel="iteration" yLabel="log-likelihood" height={200} />
          <XYChart series={params} xLabel="iteration" yLabel="value" yRange={[0, undefined]} height={200} />
        </div>
      </div>
    </Interactive>
  )
}
