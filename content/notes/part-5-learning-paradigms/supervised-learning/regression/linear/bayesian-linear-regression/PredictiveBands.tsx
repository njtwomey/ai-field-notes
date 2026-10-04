import { useMemo } from 'react'
import {
  Button,
  Curve,
  Figure,
  float,
  formatNumber,
  int,
  Plot,
  Points,
  Readout,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { effectiveParameters, logEvidence, maximiseEvidence, parameterVariance, posterior } from './blr'
import { linspace, toFlat } from 'aifn/foundation/tensor'
import { normal, stream, uniform } from 'aifn/foundation/random'

const MAX_POINTS = 25
/** Noise sd of the generated data; the true noise precision is 1/0.2² = 25. */
const NOISE = 0.2
/** Width of each Gaussian basis function. */
const WIDTH = 0.1
const GRID = toFlat(linspace(0, 1, 121))

/** A constant plus `m` Gaussian bumps with centres spread evenly over [0, 1]. */
function features(x: number, m: number): number[] {
  const centres = m === 1 ? [0.5] : toFlat(linspace(0, 1, m))
  return [1, ...centres.map((c) => Math.exp(-((x - c) ** 2) / (2 * WIDTH * WIDTH)))]
}

const fmtLog = (v: number) => formatNumber(10 ** v)

/**
 * The posterior predictive of a Gaussian-basis model fitted to noisy samples of sin(2πx): its mean, a band of two
 * predictive standard deviations, and the narrower band the noise alone would give. The button sets α and β to the
 * values that maximise the evidence.
 */
export function PredictiveBands() {
  const state = useFigureState({
    n: int(8, { min: 1, max: MAX_POINTS, step: 1, label: 'points N', format: (v) => String(v) }),
    m: int(9, { min: 1, max: 12, step: 1, label: 'bumps', format: (v) => String(v) }),
    logAlpha: float(0, { min: -3, max: 3, step: 0.05, label: 'prior precision α', format: fmtLog }),
    logBeta: float(1, { min: -1, max: 3, step: 0.05, label: 'noise precision β', format: fmtLog }),
    seed: int(3, { min: 1, max: 20, step: 1, label: 'seed' }),
  })
  const alpha = 10 ** state.logAlpha
  const beta = 10 ** state.logBeta

  const data = useMemo(() => {
    const g = stream(state.seed)
    return Array.from({ length: MAX_POINTS }, () => {
      const x = uniform(g)
      return { x, y: Math.sin(2 * Math.PI * x) + NOISE * normal(g) }
    }).slice(0, state.n)
  }, [state.seed, state.n])

  const r = useMemo(() => {
    const phi = data.map((d) => features(d.x, state.m))
    const y = data.map((d) => d.y)
    const post = posterior(phi, y, alpha, beta, state.m + 1)
    const mean: number[] = []
    const sd: number[] = []
    for (const x of GRID) {
      const f = features(x, state.m)
      mean.push(f.reduce((s, v, i) => s + v * post.mean[i], 0))
      sd.push(Math.sqrt(1 / beta + parameterVariance(post, f)))
    }
    return {
      phi,
      y,
      mean,
      sd,
      evidence: logEvidence(phi, y, alpha, beta, post),
      gamma: effectiveParameters(phi, y, alpha, beta),
    }
  }, [data, state.m, alpha, beta])

  const maximise = () => {
    const best = maximiseEvidence(r.phi, r.y, alpha, beta)
    state.set('logAlpha', Math.log10(best.alpha))
    state.set('logBeta', Math.log10(best.beta))
  }

  const noise = 2 / Math.sqrt(beta)
  const series = [
    { name: 'sin 2πx', x: GRID, y: GRID.map((x) => Math.sin(2 * Math.PI * x)), slot: 2, dashed: true },
    { name: 'noise only (±2/√β)', x: GRID, y: r.mean.map((v) => v + noise), muted: true },
    { name: 'noise only (±2/√β)', x: GRID, y: r.mean.map((v) => v - noise), muted: true },
    {
      name: 'predictive ±2 sd',
      x: GRID,
      y: r.mean.map((v, i) => v + 2 * r.sd[i]),
      slot: 0,
      dashed: true,
    },
    {
      name: 'predictive ±2 sd',
      x: GRID,
      y: r.mean.map((v, i) => v - 2 * r.sd[i]),
      slot: 0,
      dashed: true,
    },
    { name: 'predictive mean', x: GRID, y: r.mean, slot: 0 },
    { name: 'data', x: data.map((d) => d.x), y: data.map((d) => d.y), slot: 1 },
  ] as const

  const xAxis = useAxis({ label: 'x', range: [0, 1] })
  const yAxis = useAxis({ label: 'y', range: [-2, 2] })
  return (
    <Figure
      title="The posterior predictive"
      state={state}
      caption="Noisy samples of sin 2πx fitted with a constant plus Gaussian bumps of width 0.1. The dashed blue band is two predictive standard deviations, noise plus parameter uncertainty; the grey band is the noise alone. Away from the data the parameter term dominates and the band widens. Few points and a weak prior (small α) overfit; a strong prior flattens the fit. Maximise the evidence to let the data choose α and β."
      controls={
        <>
          <Button variant="outline" size="sm" onClick={maximise}>
            Maximise evidence
          </Button>
        </>
      }
      readouts={
        <>
          <Readout label="ln p(y | α, β)" value={formatNumber(r.evidence)} />
          <Readout label="effective parameters γ" value={formatNumber(r.gamma)} />
          <Readout label="α, β" value={`${formatNumber(alpha)}, ${formatNumber(beta)}`} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={380}>
        <Curve {...series[0]} />
        <Curve {...series[1]} />
        <Curve {...series[2]} />
        <Curve {...series[3]} />
        <Curve {...series[4]} />
        <Curve {...series[5]} />
        <Points {...series[6]} />
      </Plot>
    </Figure>
  )
}
