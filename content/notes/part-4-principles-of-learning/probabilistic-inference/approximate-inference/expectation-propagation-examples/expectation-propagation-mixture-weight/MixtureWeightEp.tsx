import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamButton,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type XYSeries,
} from 'aifn-render'
import { grid, normalLogPdf, normalPdf, normaliseOnGrid } from '../_shared/ep'
import { betaLogPdf, mixtureEp, type BetaParams } from '../_shared/mixture'

const INITIAL = [-0.5, 0.3, 1, 2.4, 0.1]
const XS = grid(-4, 6, 201)
const X_RANGE: [number, number] = [-4, 6]
const WS = grid(0.0025, 0.9975, 399)
const W_RANGE: [number, number] = [0, 1]
const Y_RANGE: [number | undefined, number | undefined] = [0, undefined]
/** Fixed so that a Beta with a < 1 or b < 1, which is infinite at an end, does not flatten the rest. */
const W_Y_RANGE: [number | undefined, number | undefined] = [0, 4.5]
const SWEEPS = 5
const P1 = XS.map((x) => normalPdf(x, 0, 1))
const P2 = XS.map((x) => normalPdf(x, 2, 1))

const betaCurve = (q: BetaParams) => WS.map((w) => Math.exp(betaLogPdf(w, q)))
const summary = (q: BetaParams) => {
  const s = q.a + q.b
  return `Beta(${formatNumber(q.a)}, ${formatNumber(q.b)}), mean ${formatNumber(q.a / s)}`
}

/** Laplace on the logit scale u = log(w/(1 − w)), where the uniform prior on w becomes the density w(1 − w). */
function laplaceLogit(p1: number[], p2: number[]) {
  const logDensity = (u: number) => {
    const w = 1 / (1 + Math.exp(-u))
    return p1.reduce((s, a, i) => s + Math.log(w * a + (1 - w) * p2[i]), 0) + Math.log(w) + Math.log1p(-w)
  }
  let u = 0
  const h = 1e-4
  for (let it = 0; it < 60; it++) {
    const g = (logDensity(u + h) - logDensity(u - h)) / (2 * h)
    const c = (logDensity(u + h) - 2 * logDensity(u) + logDensity(u - h)) / (h * h)
    const step = c < 0 ? -g / c : Math.sign(g) * 0.5
    u += Math.max(-2, Math.min(2, step))
    if (Math.abs(step) < 1e-9) break
  }
  const c = (logDensity(u + h) - 2 * logDensity(u) + logDensity(u - h)) / (h * h)
  return { u, v: -1 / c }
}

/**
 * EP for the weight w of a two-component mixture with known components, with Beta sites. Compares the KL projection
 * (matching E log w and E log(1 − w)) with mean-and-variance matching, the exact posterior and a Laplace approximation
 * on the logit scale.
 */
export function MixtureWeightEp() {
  const [points, setPoints] = useState(INITIAL)
  const step = useParam(INITIAL.length, { min: 0, max: INITIAL.length * SWEEPS, step: 1 })

  const r = useMemo(() => {
    const p1 = points.map((x) => normalPdf(x, 0, 1))
    const p2 = points.map((x) => normalPdf(x, 2, 1))
    const exact = normaliseOnGrid(
      WS,
      WS.map((w) => p1.reduce((s, a, i) => s + Math.log(w * a + (1 - w) * p2[i]), 0)),
    )
    const lap = laplaceLogit(p1, p2)
    const laplace = WS.map((w) => Math.exp(normalLogPdf(Math.log(w / (1 - w)), lap.u, lap.v)) / (w * (1 - w)))
    return {
      exact,
      laplace,
      kl: mixtureEp(p1, p2, SWEEPS, 'kl'),
      moments: mixtureEp(p1, p2, SWEEPS, 'moments'),
    }
  }, [points])

  const prior: BetaParams = { a: 1, b: 1 }
  const qKl = step.value === 0 ? prior : r.kl[step.value - 1].q
  const qMom = step.value === 0 ? prior : r.moments[step.value - 1].q
  const data: XYSeries[] = [
    { name: 'p₁ = N(0, 1)', type: 'line', x: XS, y: P1, slot: 3 },
    { name: 'p₂ = N(2, 1)', type: 'line', x: XS, y: P2, slot: 4 },
  ]
  const posterior: XYSeries[] = [
    { name: 'exact', type: 'line', x: WS, y: r.exact.density, emphasis: true },
    { name: 'EP, KL projection', type: 'line', x: WS, y: betaCurve(qKl), slot: 0 },
    { name: 'EP, mean and variance', type: 'line', x: WS, y: betaCurve(qMom), slot: 1, dashed: true },
    { name: 'Laplace (logit scale)', type: 'line', x: WS, y: r.laplace, slot: 2, dashed: true },
  ]
  const handles: Handle[] = points.map((x, i) => ({
    kind: 'point',
    at: [x, 0],
    label: `x${i + 1}`,
    onDrag: ([nx]) =>
      setPoints((ps) => ps.map((p, j) => (j === i ? Math.round(Math.min(5.8, Math.max(-3.8, nx)) * 20) / 20 : p))),
  }))
  const s = step.value === 0 ? null : r.kl[step.value - 1]
  const where = s ? `sweep ${s.sweep + 1}, site ${s.site + 1}` : 'prior'

  return (
    <Interactive
      title="EP for a mixture weight, with a Beta approximation"
      caption="Top: the two known components and the data; drag the points. A point near 0 favours p₁ and raises w; a point near 2 lowers it; a point at 1 is equally likely under both and carries no information. Bottom: the posterior of the weight w of p₁ after the chosen number of site updates, projected onto the Beta family either by the KL projection (matching E log w and E log(1 − w)) or by matching the mean and variance, against the exact posterior and a Laplace approximation on the logit scale."
      controls={
        <>
          <ParamSlider label="site updates" param={step} withArrows format={(v) => String(v)} />
          <ParamButton onClick={() => setPoints(INITIAL)}>Reset points</ParamButton>
        </>
      }
      readout={
        <>
          <Readout label="position" value={where} />
          <Readout label="KL projection" value={summary(qKl)} />
          <Readout label="mean and variance" value={summary(qMom)} />
          <Readout label="exact mean" value={formatNumber(r.exact.mean)} />
        </>
      }
    >
      <XYChart
        series={data}
        xLabel="x"
        yLabel="density"
        xRange={X_RANGE}
        yRange={Y_RANGE}
        handles={handles}
        height={200}
      />
      <XYChart series={posterior} xLabel="w" yLabel="density" xRange={W_RANGE} yRange={W_Y_RANGE} height={280} />
    </Interactive>
  )
}
