import { useMemo, useState } from 'react'
import {
  Heatmap,
  Interactive,
  ParamSlider,
  ParamSwitch,
  Readout,
  formatNumber,
  useParam,
  type Handle,
  type HeatmapOverlay,
} from 'aifn-render'
import { linspace, rng } from '@/lib/math'

const X = 1.5
const MS = linspace(-2, 3, 71)
const WS = linspace(-2.5, 1, 57)
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
  const g = rng(seed)
  const score: Path = { m: [start[0]], w: [start[1]] }
  const reparam: Path = { m: [start[0]], w: [start[1]] }
  let [sm, sw] = start
  let [rm, rw] = start
  for (let t = 0; t < steps; t++) {
    const eps = Array.from({ length: samples }, () => g.normal())
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
  const samples = useParam(1, { min: 1, max: 50, step: 1 })
  const lr = useParam(0.05, { min: 0.005, max: 0.2, step: 0.005 })
  const steps = useParam(200, { min: 10, max: 1000, step: 10 })
  const seed = useParam(1, { min: 1, max: 30, step: 1 })
  const m0 = useParam(-1, { min: -2, max: 3, step: 0.05 })
  const w0 = useParam(0.2, { min: -2.5, max: 1, step: 0.05 })
  const [baseline, setBaseline] = useState(false)

  const z = useMemo(() => WS.map((w) => MS.map((m) => elbo(m, w))), [])
  const run = useMemo(
    () => ascend([m0.value, w0.value], samples.value, lr.value, steps.value, seed.value, baseline),
    [m0.value, w0.value, samples.value, lr.value, steps.value, seed.value, baseline],
  )

  const overlay: HeatmapOverlay[] = useMemo(
    () => [
      { name: 'score-function gradient', type: 'line', x: run.score.m, y: run.score.w, slot: 1 },
      { name: 'reparameterisation gradient', type: 'line', x: run.reparam.m, y: run.reparam.w, slot: 2 },
      { name: 'optimum', type: 'scatter', x: [X / 2], y: [Math.log(Math.SQRT1_2)], emphasis: true },
    ],
    [run],
  )

  const handles: Handle[] = [
    {
      kind: 'point',
      at: [m0.value, w0.value],
      label: 'start',
      onDrag: ([m, w]) => {
        m0.set(m)
        w0.set(w)
      },
    },
  ]
  const last = (p: Path) => [p.m[p.m.length - 1], Math.exp(p.w[p.w.length - 1])]
  const [smEnd, ssEnd] = last(run.score)
  const [rmEnd, rsEnd] = last(run.reparam)

  return (
    <Interactive
      title="Stochastic gradient ascent on the ELBO with two gradient estimators"
      caption="The shading is the ELBO of q = N(m, s²) for the model z ~ N(0, 1), x | z ~ N(z, 1) with x = 1.5, over the mean m and ω = log s. The optimum (diamond) is the exact posterior N(0.75, 0.5). Both paths use the same random draws and the same step size. The reparameterisation gradient follows the slope closely even with one sample; the score-function gradient wanders, and needs more samples, a baseline or a smaller step to settle. Drag the start point."
      controls={
        <>
          <ParamSlider label="samples per step S" param={samples} format={(v) => String(v)} />
          <ParamSlider label="step size" param={lr} />
          <ParamSlider label="steps" param={steps} format={(v) => String(v)} />
          <ParamSlider label="random seed" param={seed} withArrows />
          <ParamSwitch label="baseline for the score function" checked={baseline} onChange={setBaseline} />
        </>
      }
      readout={
        <>
          <Readout label="score function: m, s" value={`${formatNumber(smEnd)}, ${formatNumber(ssEnd)}`} />
          <Readout label="reparameterisation: m, s" value={`${formatNumber(rmEnd)}, ${formatNumber(rsEnd)}`} />
          <Readout label="exact" value="0.75, 0.707" />
        </>
      }
    >
      <Heatmap
        x={MS}
        y={WS}
        z={z}
        xLabel="mean m"
        yLabel="ω = log s"
        overlay={overlay}
        handles={handles}
        valueLabel="ELBO"
        range={[-8, -1.5]}
        height={380}
      />
    </Interactive>
  )
}
