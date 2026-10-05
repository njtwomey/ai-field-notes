import { useMemo } from 'react'
import {
  choice,
  Figure,
  formatNumber,
  int,
  Plot,
  Readout,
  seriesLayers,
  type SeriesSpec,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { gaussPdf, histogram } from '../_shared/sde'
import { normal, stream } from 'aifn-compute/foundation/random'

type Integrand = 'constant' | 'steps' | 'ramp' | 'fade' | 'path'

const PATHS = 4000
const STEPS = 200
/** Paths whose traces are kept for drawing; the histogram always uses all PATHS. */
const MAX_DRAWN = 30

/** Each integrand: its horizon T, the function h(t) (unused for `path`, whose integrand is the path) and the exact variance. */
const INTEGRANDS: Record<Integrand, { label: string; T: number; h: (t: number) => number; variance: number }> = {
  constant: { label: 'constant 3', T: 2, h: () => 3, variance: 18 },
  steps: { label: '2, −1, 0.5', T: 2, h: (t) => (t < 0.5 ? 2 : t < 1.5 ? -1 : 0.5), variance: 3.125 },
  ramp: { label: 'ramp t', T: 1, h: (t) => t, variance: 1 / 3 },
  fade: { label: 'exponential e^−(1−t)', T: 1, h: (t) => Math.exp(-(1 - t)), variance: (1 - Math.exp(-2)) / 2 },
  path: { label: 'the path W', T: 1, h: () => 0, variance: 0.5 },
}

/** Density of ½(W² − T) for W ~ N(0, T): the exact law of ∫₀ᵀ W dW. */
function itoWdWDensity(y: number, T: number): number {
  const s = 2 * y + T
  if (s <= 0) return 0
  const x = Math.sqrt(s)
  return (2 * gaussPdf(x, 0, T)) / x
}

/**
 * Simulated Itô integrals ∫ h dW for a deterministic integrand (Gaussian with variance ∫ h² dt) or for the integrand h = W
 * (not Gaussian: ½(W_T² − T)). Each path uses left-point sums over 200 steps.
 */
export function IntegrandDistribution() {
  const state = useFigureState({
    integrand: choice<Integrand>(
      (Object.keys(INTEGRANDS) as Integrand[]).map((k) => ({ value: k, label: INTEGRANDS[k].label })),
      'steps',
      { label: 'integrand h(t)' },
    ),
    count: int(10, { min: 1, max: MAX_DRAWN, step: 1, label: 'paths', format: (v) => String(v) }),
    seed: int(1, { min: 1, max: 20, step: 1, label: 'seed' }),
  })
  const spec = INTEGRANDS[state.integrand]

  const sim = useMemo(() => {
    const rs = stream(state.seed)
    const dt = spec.T / STEPS
    const sd = Math.sqrt(dt)
    const values = new Float64Array(PATHS)
    // The first MAX_DRAWN paths keep their traces of W and of the running integral, for drawing.
    const wTraces: number[][] = []
    const sumTraces: number[][] = []
    for (let p = 0; p < PATHS; p++) {
      let w = 0
      let sum = 0
      const keep = p < MAX_DRAWN
      const wTrace = [0]
      const sumTrace = [0]
      for (let j = 0; j < STEPS; j++) {
        const t = j * dt
        const dW = sd * normal(rs)
        const h = state.integrand === 'path' ? w : spec.h(t)
        sum += h * dW
        w += dW
        if (keep) {
          wTrace.push(w)
          sumTrace.push(sum)
        }
      }
      values[p] = sum
      if (keep) {
        wTraces.push(wTrace)
        sumTraces.push(sumTrace)
      }
    }
    let mean = 0
    for (const v of values) mean += v / PATHS
    let variance = 0
    for (const v of values) variance += (v - mean) ** 2 / (PATHS - 1)
    return { values, mean, variance, wTraces, sumTraces }
  }, [state.integrand, state.seed, spec])

  const ts = useMemo(() => Array.from({ length: STEPS + 1 }, (_, j) => (j * spec.T) / STEPS), [spec])

  const { integrandSeries, runningSeries } = useMemo(() => {
    const many = state.count > 1
    const shown = sim.sumTraces.slice(0, state.count)
    const integrandSeries: SeriesSpec[] =
      state.integrand === 'path'
        ? sim.wTraces.slice(0, state.count).map((w) => ({
            name: many ? 'h(t) = W(t), sample paths' : 'h(t) = W(t), one sample path',
            type: 'line',
            x: ts,
            y: w,
            slot: 0,
            thin: many,
          }))
        : [{ name: 'h(t)', type: 'line', x: ts, y: ts.map((t) => spec.h(t)), slot: 0 }]
    // Variance of the running integral: t²/2 for h = W, else the left-point sum of h² dt (the isometry).
    const dt = spec.T / STEPS
    const v: number[] = [0]
    for (let j = 0; j < STEPS; j++)
      v.push(state.integrand === 'path' ? ts[j + 1] ** 2 / 2 : v[j] + spec.h(ts[j]) ** 2 * dt)
    const runningSeries: SeriesSpec[] = shown.map((y) => ({
      name: many ? 'running integrals, sample paths' : 'running integral, one sample path',
      type: 'line',
      x: ts,
      y,
      slot: 3,
      thin: many,
    }))
    for (const sgn of [1, -1])
      runningSeries.push({
        name: '± 2 sd, sd² = ∫₀ᵗ h² ds',
        type: 'line',
        x: ts,
        y: v.map((vv) => sgn * 2 * Math.sqrt(vv)),
        slot: 1,
        dashed: true,
      })
    return { integrandSeries, runningSeries }
  }, [state.integrand, sim, spec, ts, state.count])

  const { histSeries, below } = useMemo(() => {
    const sd = Math.sqrt(spec.variance)
    const lo = state.integrand === 'path' ? -0.75 : -4 * sd
    const hi = state.integrand === 'path' ? 3 : 4 * sd
    const hist = histogram(sim.values, lo, hi, 50)
    const grid = Array.from({ length: 241 }, (_, i) => lo + ((hi - lo) * i) / 240)
    const theory = grid.map((y) =>
      state.integrand === 'path' ? itoWdWDensity(y, spec.T) : gaussPdf(y, 0, spec.variance),
    )
    const histSeries: SeriesSpec[] = [
      { name: `${PATHS} simulated integrals`, type: 'bar', x: hist.x, y: hist.y, muted: true },
      {
        name: state.integrand === 'path' ? 'exact density of ½(W² − 1)' : `N(0, ${formatNumber(spec.variance)})`,
        type: 'line',
        x: grid,
        y: theory,
        slot: 1,
      },
    ]
    if (state.integrand === 'path')
      histSeries.push({
        name: 'Gaussian with the same variance',
        type: 'line',
        x: grid,
        y: grid.map((y) => gaussPdf(y, 0, spec.variance)),
        slot: 2,
        dashed: true,
      })
    let below = 0
    for (const v of sim.values) if (v < -0.4) below++
    return { histSeries, below: below / PATHS }
  }, [state.integrand, sim, spec])

  const xAxis = useAxis({ label: 't', hold: 'union' })
  const yAxis = useAxis({ label: 'h(t)', hold: 'union' })
  const xAxis2 = useAxis({ label: 't', hold: 'union' })
  const yAxis2 = useAxis({ label: '∫₀ᵗ h dW', hold: 'union' })
  const xAxis3 = useAxis({ label: '∫ h dW', hold: 'union' })
  const yAxis3 = useAxis({ label: 'density', hold: 'union' })
  return (
    <Figure
      title="The distribution of ∫ h dW for different integrands"
      state={state}
      caption="Left-point sums of ∫ h dW over 200 steps, on 4,000 simulated Brownian paths. Top: the integrand h(t); for h = W, the first few sampled paths of W. Middle: the running integral ∫₀ᵗ h dW on the first few paths, as light lines, inside a band of ± 2 standard deviations, sd² = ∫₀ᵗ h(s)² ds (for h = W, t²/2); the paths slider sets how many are drawn. Bottom: the histogram of all 4,000 values at the end of the interval against the theory. For a deterministic integrand the integral is Gaussian with variance ∫ h(t)² dt, whatever the integrand's shape. For the integrand h = W, which depends on the path, the integral is ½(W₁² − 1): mean 0 and variance ½ as the isometry says, but skewed, never below −½, and not Gaussian (dashed)."

      readouts={
        <>
          <Readout label="sample mean" value={formatNumber(sim.mean)} />
          <Readout label="sample variance" value={formatNumber(sim.variance)} />
          <Readout label="∫ h² dt (theory)" value={formatNumber(spec.variance)} />
          {state.integrand === 'path' && <Readout label="share below −0.4 (exact 0.345)" value={formatNumber(below)} />}
        </>
      }
    >
      <div className="space-y-4">
        <Plot x={xAxis} y={yAxis} height={180}>
          {seriesLayers(integrandSeries)}
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={220}>
          {seriesLayers(runningSeries)}
        </Plot>
        <Plot x={xAxis3} y={yAxis3} height={260}>
          {seriesLayers(histSeries)}
        </Plot>
      </div>
    </Figure>
  )
}
