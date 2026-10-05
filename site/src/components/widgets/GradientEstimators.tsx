import { useMemo } from 'react'
import { normal, stream } from 'aifn-compute/foundation/random'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { normalCdf, normalPdf } from 'aifn-compute/numerics/special'
import {
  Curve,
  Figure,
  Plot,
  Readout,
  choice,
  formatNumber,
  int,
  setting,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'

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
const IDS = Object.keys(OBJECTIVES) as FId[]

const REPEATS = 2000
const BINS = 45

const quantile = (sorted: number[], q: number) => sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))]
const meanSd = (xs: number[]) => {
  const m = xs.reduce((a, b) => a + b, 0) / xs.length
  return { mean: m, sd: Math.sqrt(xs.reduce((a, b) => a + (b - m) ** 2, 0) / (xs.length - 1)) }
}

/** A histogram as a line through bin centres, so two estimators can share one chart without hiding each other. */
function histogramLine(values: number[], lo: number, hi: number) {
  const width = (hi - lo) / BINS
  const counts = new Array<number>(BINS).fill(0)
  for (const v of values) {
    const b = Math.floor((v - lo) / width)
    if (b >= 0 && b < BINS) counts[b]++
  }
  return { x: counts.map((_, i) => lo + (i + 0.5) * width), y: counts.map((c) => c / (values.length * width)) }
}

/**
 * The spread of two unbiased estimators of d/dμ E_{z ~ N(μ, σ²)} f(z): the reparameterisation (pathwise) estimator
 * f′(μ + σε) and the score-function estimator (f(z) − b)(z − μ)/σ². Each estimate averages K samples; the histogram is
 * over 2,000 independent estimates.
 */
export function GradientEstimators() {
  const state = useFigureState({
    objective: choice(
      IDS.map((v) => ({ value: v, label: OBJECTIVES[v].label })),
      'quadratic',
      { label: 'objective' },
    ),
    mu: slider(-2, 2, 1, { step: 0.1, label: 'mean μ' }),
    sigma: slider(0.2, 2, 1, { step: 0.1, label: 'standard deviation σ' }),
    k: int(1, { min: 1, max: 50, suggestions: [1, 5, 10, 50], label: 'samples per estimate K' }),
    baseline: setting(false, 'score-function baseline'),
  })
  const { mu, sigma, k, baseline } = state
  const obj = OBJECTIVES[state.objective]
  const truth = obj.grad(mu, sigma)

  const result = useMemo(() => {
    const eps = toFlat(normal(stream(12), 0, 1, { shape: [REPEATS * k] }))
    const b = baseline ? obj.mean(mu, sigma) : 0
    const pathwise: number[] = []
    const score: number[] = []
    for (let i = 0; i < REPEATS; i++) {
      let p = 0
      let s = 0
      for (let j = 0; j < k; j++) {
        const e = eps[i * k + j]
        const z = mu + sigma * e
        p += obj.df(z)
        s += ((obj.f(z) - b) * e) / sigma
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
      pathLine: histogramLine(pathwise, lo, hi),
      scoreLine: histogramLine(score, lo, hi),
      pathwise: meanSd(pathwise),
      score: meanSd(score),
    }
  }, [obj, mu, sigma, k, baseline, truth])

  // A zero-variance estimator is a single spike; scale the axis to the other curve so both stay readable.
  const { pathLine, scoreLine } = result
  const top = 1.15 * Math.max(...scoreLine.y, ...(result.pathwise.sd > 1e-9 ? pathLine.y : []))
  const x = useAxis({ label: 'gradient estimate', range: [result.lo, result.hi] })
  const y = useAxis({ label: 'density', range: [0, top] })
  const summary = (m: { mean: number; sd: number }) => `${formatNumber(m.mean)} ± ${formatNumber(m.sd)}`

  return (
    <Figure
      title="Two unbiased gradients, very different noise"
      state={state}
      caption="Both estimators target the same number, the derivative of E f(z) with respect to μ for z ~ N(μ, σ²). Each curve is a histogram of 2,000 independent estimates, each averaging K samples. The reparameterisation estimator uses f′ and is usually far less noisy. The score-function estimator uses only values of f; a baseline b = E f(z) subtracted from f removes much of its noise without adding bias. For the step function f′ is zero almost everywhere, so the reparameterisation estimator returns 0 every time and is wrong, while the score function stays correct."
      readouts={
        <>
          <Readout label="true gradient" value={formatNumber(truth)} />
          <Readout label="reparameterisation, mean ± sd" value={summary(result.pathwise)} />
          <Readout label="score function, mean ± sd" value={summary(result.score)} />
        </>
      }
    >
      <Plot x={x} y={y} height={320}>
        <Curve name="reparameterisation" x={pathLine.x} y={pathLine.y} slot={0} />
        <Curve name="score function" x={scoreLine.x} y={scoreLine.y} slot={1} />
        <Curve name="true gradient" x={[truth, truth]} y={[0, top]} dashed emphasis />
      </Plot>
    </Figure>
  )
}
