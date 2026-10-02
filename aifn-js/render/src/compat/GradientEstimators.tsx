import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamChoice,
  ParamSlider,
  ParamSwitch,
  XYChart,
  type XYSeries,
} from './index'
import { Readout, formatNumber } from '../viz'

type FId = 'quadratic' | 'cosine' | 'step'

type Objective = {
  label: string
  f: (z: number) => number
  /** f′(z); zero almost everywhere for the step. */
  df: (z: number) => number
  /** E f(z) for z ~ N(μ, σ²), used as the baseline. */
  mean: (mu: number, sigma: number) => number
  /** The exact gradient d/dμ E f(z). */
  grad: (mu: number, sigma: number) => number
}

function rng(seed: number) {
  let a = seed >>> 0
  const uniform = () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
  const normal = () => {
    const u = Math.max(uniform(), 1e-12)
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * uniform())
  }
  return { uniform, normal }
}

const erf = (x: number): number => {
  const s = Math.sign(x)
  const a = Math.abs(x)
  const t = 1 / (1 + 0.3275911 * a)
  const y =
    1 -
    ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-a * a)
  return s * y
}

const normalPdf = (z: number) => Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI)
const normalCdf = (z: number) => 0.5 * (1 + erf(z / Math.SQRT2))

const OBJECTIVES: Record<FId, Objective> = {
  quadratic: {
    label: 'f(z) = z²',
    f: (z) => z * z,
    df: (z) => 2 * z,
    mean: (mu, s) => mu * mu + s * s,
    grad: (mu) => 2 * mu,
  },
  cosine: {
    label: 'f(z) = cos z',
    f: Math.cos,
    df: (z) => -Math.sin(z),
    mean: (mu, s) => Math.cos(mu) * Math.exp((-s * s) / 2),
    grad: (mu, s) => -Math.sin(mu) * Math.exp((-s * s) / 2),
  },
  step: {
    label: 'f(z) = 1[z > 0]',
    f: (z) => (z > 0 ? 1 : 0),
    df: () => 0,
    mean: (mu, s) => normalCdf(mu / s),
    grad: (mu, s) => normalPdf(mu / s) / s,
  },
}

const REPEATS = 2000
const BINS = 45

const quantile = (sorted: number[], q: number) => sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))]
const meanSd = (xs: number[]) => {
  const m = xs.reduce((a, b) => a + b, 0) / xs.length
  return { mean: m, sd: Math.sqrt(xs.reduce((a, b) => a + (b - m) ** 2, 0) / (xs.length - 1)) }
}

/** A histogram as a line through bin centres, so two estimators can share one chart without hiding each other. */
function histogramLine(values: number[], lo: number, hi: number, name: string, slot: number): XYSeries {
  const width = (hi - lo) / BINS
  const counts = new Array(BINS).fill(0)
  for (const v of values) {
    const b = Math.floor((v - lo) / width)
    if (b >= 0 && b < BINS) counts[b]++
  }
  return {
    name,
    type: 'line',
    x: counts.map((_, i) => lo + (i + 0.5) * width),
    y: counts.map((c) => c / (values.length * width)),
    slot,
  }
}

/**
 * The spread of two unbiased estimators of d/dμ E_{z ~ N(μ, σ²)} f(z): the reparameterisation (pathwise) estimator
 * f′(μ + σε) and the score-function estimator (f(z) − b)(z − μ)/σ². Each estimate averages K samples; the histogram is
 * over 2,000 independent estimates.
 */
export function GradientEstimators() {
  const [id, setId] = useState<FId>('quadratic')
  const [mu, setMu] = useState(1)
  const [sigma, setSigma] = useState(1)
  const [k, setK] = useState(1)
  const [baseline, setBaseline] = useState(false)
  const obj = OBJECTIVES[id]
  const truth = obj.grad(mu, sigma)

  const result = useMemo(() => {
    const r = rng(12)
    const b = baseline ? obj.mean(mu, sigma) : 0
    const pathwise: number[] = []
    const score: number[] = []
    for (let i = 0; i < REPEATS; i++) {
      let p = 0
      let s = 0
      for (let j = 0; j < k; j++) {
        const eps = r.normal()
        const z = mu + sigma * eps
        p += obj.df(z)
        s += ((obj.f(z) - b) * eps) / sigma
      }
      pathwise.push(p / k)
      score.push(s / k)
    }
    const all = [...pathwise, ...score].sort((a, c) => a - c)
    let lo = Math.min(quantile(all, 0.01), truth)
    let hi = Math.max(quantile(all, 0.99), truth)
    if (hi - lo < 1e-6) {
      lo -= 0.5
      hi += 0.5
    }
    const pad = (hi - lo) * 0.05
    lo -= pad
    hi += pad
    return {
      lo,
      hi,
      series: [
        histogramLine(pathwise, lo, hi, 'reparameterisation', 0),
        histogramLine(score, lo, hi, 'score function', 1),
      ],
      pathwise: meanSd(pathwise),
      score: meanSd(score),
    }
  }, [obj, mu, sigma, k, baseline, truth])

  // A zero-variance estimator is a single spike; scale the axis to the other curve so both stay readable.
  const [pathLine, scoreLine] = result.series
  const top = 1.15 * Math.max(...scoreLine.y, ...(result.pathwise.sd > 1e-9 ? pathLine.y : []))
  const series: XYSeries[] = [
    ...result.series,
    { name: 'true gradient', type: 'line', x: [truth, truth], y: [0, top], dashed: true, emphasis: true },
  ]
  const summary = (m: { mean: number; sd: number }) => `${formatNumber(m.mean)} ± ${formatNumber(m.sd)}`

  return (
    <Interactive
      title="Two unbiased gradients, very different noise"
      caption="Both estimators target the same number, the derivative of E f(z) with respect to μ for z ~ N(μ, σ²). Each curve is a histogram of 2,000 independent estimates, each averaging K samples. The reparameterisation estimator uses f′ and is usually far less noisy. The score-function estimator uses only values of f; a baseline b = E f(z) subtracted from f removes much of its noise without adding bias. For the step function f′ is zero almost everywhere, so the reparameterisation estimator returns 0 every time and is wrong, while the score function stays correct."
      controls={
        <>
          <ParamChoice
            label="objective"
            value={id}
            onChange={setId}
            options={(Object.keys(OBJECTIVES) as FId[]).map((v) => ({ value: v, label: OBJECTIVES[v].label }))}
          />
          <ParamSlider label="mean μ" value={mu} onChange={setMu} min={-2} max={2} step={0.1} />
          <ParamSlider label="standard deviation σ" value={sigma} onChange={setSigma} min={0.2} max={2} step={0.1} />
          <ParamSlider label="samples per estimate K" value={k} onChange={setK} min={1} max={50} step={1} />
          <ParamSwitch label="score-function baseline" checked={baseline} onChange={setBaseline} />
        </>
      }
      readout={
        <>
          <Readout label="true gradient" value={formatNumber(truth)} />
          <Readout label="reparameterisation, mean ± sd" value={summary(result.pathwise)} />
          <Readout label="score function, mean ± sd" value={summary(result.score)} />
        </>
      }
    >
      <XYChart
        height={320}
        series={series}
        xRange={[result.lo, result.hi]}
        yRange={[0, top]}
        xLabel="gradient estimate"
        yLabel="density"
      />
    </Interactive>
  )
}
