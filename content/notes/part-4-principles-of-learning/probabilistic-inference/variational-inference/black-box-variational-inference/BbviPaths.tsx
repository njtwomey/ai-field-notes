import { useMemo } from 'react'
import {
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Plot,
  Points,
  Raster,
  Readout,
  setting,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { linspace, toFlat } from 'aifn/foundation/tensor'
import { normal, stream } from 'aifn/foundation/random'

const X = 1.5
const MS = toFlat(linspace(-2, 3, 71))
const WS = toFlat(linspace(-2.5, 1, 57))
const LOG_2PI = Math.log(2 * Math.PI)
const OMEGA_MIN = -4
const OMEGA_MAX = 2

/** ELBO of q = N(m, e^{2ω}) for z ~ N(0, 1), x | z ~ N(z, 1), one observation x = 1.5. */
const elbo = (m: number, w: number) => {
  const v = Math.exp(2 * w)
  return -0.5 * LOG_2PI - 0.5 * ((X - m) ** 2 + v) - 0.5 * (v + m * m - 1 - 2 * w)
}
const logJoint = (z: number) => -LOG_2PI - (z * z) / 2 - (X - z) ** 2 / 2

type Path = { m: number[]; w: number[] }

/**
 * Stochastic gradient ascent on (m, ω = log s) with S Monte Carlo samples per step, using either the score-function
 * gradient (optionally with a leave-one-out baseline) or the reparameterisation gradient. Both estimators use the same
 * noise, so differences come from the estimators alone.
 */
function ascend(start: [number, number], samples: number, lr: number, steps: number, seed: number, baseline: boolean) {
  const g = stream(seed)
  const score: Path = { m: [start[0]], w: [start[1]] }
  const reparam: Path = { m: [start[0]], w: [start[1]] }
  let [sm, sw] = start
  let [rm, rw] = start
  for (let t = 0; t < steps; t++) {
    const eps = Array.from({ length: samples }, () => normal(g))
    // Score function: ∇ log q(z) × (log p(x, z) − log q(z)).
    const s = Math.exp(sw)
    const f = eps.map((e) => {
      const z = sm + s * e
      return logJoint(z) + 0.5 * LOG_2PI + sw + (e * e) / 2
    })
    const total = f.reduce((a, b) => a + b, 0)
    let gm = 0
    let gw = 0
    eps.forEach((e, i) => {
      const b = baseline && samples > 1 ? (total - f[i]) / (samples - 1) : 0
      gm += ((e / s) * (f[i] - b)) / samples
      gw += ((e * e - 1) * (f[i] - b)) / samples
    })
    sm += lr * gm
    sw = Math.min(OMEGA_MAX, Math.max(OMEGA_MIN, sw + lr * gw))
    // Reparameterisation: z = m + s ε, gradient of log p(x, z) through z, plus the entropy's gradient 1 in ω.
    const r = Math.exp(rw)
    let hm = 0
    let hw = 0
    eps.forEach((e) => {
      const dz = X - 2 * (rm + r * e)
      hm += dz / samples
      hw += (dz * r * e) / samples
    })
    rm += lr * hm
    rw = Math.min(OMEGA_MAX, Math.max(OMEGA_MIN, rw + lr * (hw + 1)))
    score.m.push(sm)
    score.w.push(sw)
    reparam.m.push(rm)
    reparam.w.push(rw)
  }
  return { score, reparam }
}

export function BbviPaths() {
  const state = useFigureState({
    samples: int(1, { min: 1, max: 50, step: 1, label: 'samples per step S', format: (v) => String(v) }),
    lr: float(0.05, { min: 0.005, max: 0.2, step: 0.005, label: 'step size' }),
    steps: int(200, { min: 10, max: 1000, step: 10, label: 'steps', format: (v) => String(v) }),
    seed: int(1, { min: 1, max: 30, step: 1, label: 'random seed' }),
    baseline: setting(false, 'baseline for the score function'),
    m0: slider(-2, 3, -1, { step: 0.05, onChart: true }),
    w0: slider(-2.5, 1, 0.2, { step: 0.05, onChart: true }),
  })

  const z = useMemo(() => WS.map((w) => MS.map((m) => elbo(m, w))), [])
  const run = useMemo(
    () => ascend([state.m0, state.w0], state.samples, state.lr, state.steps, state.seed, state.baseline),
    [state.m0, state.w0, state.samples, state.lr, state.steps, state.seed, state.baseline],
  )

  const overlay = useMemo(
    () =>
      [
        { name: 'score-function gradient', x: run.score.m, y: run.score.w, slot: 1 },
        { name: 'reparameterisation gradient', x: run.reparam.m, y: run.reparam.w, slot: 2 },
        { name: 'optimum', x: [X / 2], y: [Math.log(Math.SQRT1_2)], emphasis: true },
      ] as const,
    [run],
  )

  const last = (p: Path) => [p.m[p.m.length - 1], Math.exp(p.w[p.w.length - 1])]
  const [smEnd, ssEnd] = last(run.score)
  const [rmEnd, rsEnd] = last(run.reparam)

  const xAxis = useAxis({ label: 'mean m' })
  const yAxis = useAxis({ label: 'ω = log s' })
  return (
    <Figure
      title="Stochastic gradient ascent on the ELBO with two gradient estimators"
      state={state}
      caption="The shading is the ELBO of q = N(m, s²) for the model z ~ N(0, 1), x | z ~ N(z, 1) with x = 1.5, over the mean m and ω = log s. The optimum (diamond) is the exact posterior N(0.75, 0.5). Both paths use the same random draws and the same step size. The reparameterisation gradient follows the slope closely even with one sample; the score-function gradient wanders, and needs more samples, a baseline or a smaller step to settle. Drag the start point."

      readouts={
        <>
          <Readout label="score function: m, s" value={`${formatNumber(smEnd)}, ${formatNumber(ssEnd)}`} />
          <Readout label="reparameterisation: m, s" value={`${formatNumber(rmEnd)}, ${formatNumber(rsEnd)}`} />
          <Readout label="exact" value="0.75, 0.707" />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={380}>
        <Raster x={MS} y={WS} z={z} range={[-8, -1.5]} valueLabel={'ELBO'} />
        <Curve {...overlay[0]} live />
        <Curve {...overlay[1]} live />
        <Points {...overlay[2]} live />
        <Handle
          kind="point"
          at={[state.m0, state.w0]}
          label="start"
          onDrag={([m, w]) => {
            state.set('m0', m)
            state.set('w0', w)
          }}
        />
      </Plot>
    </Figure>
  )
}
