import { biasVariance, biasVarianceSweep } from 'aifn-methods/theory/bias-variance'
import { stream } from 'aifn-compute/foundation/random'
import {
  Curve,
  Figure,
  formatNumber,
  Handle,
  int,
  Plot,
  Plots,
  Readout,
  slider,
  useAxis,
  useComputed,
  useFigureState,
} from 'aifn-render'

const fmt = (v: number) => (Number.isFinite(v) ? formatNumber(Number(v.toPrecision(3))) : '—')
const DEGREES = Array.from({ length: 13 }, (_, d) => d)
const SIZES = [10, 12, 16, 20, 25, 30, 40, 50, 60, 80, 100, 130, 160, 200]
const REPEATS = 200
const LOW = 1e-3
const HIGH = 10

/**
 * Training error and expected test error of least-squares polynomials on y = sin(πx) + noise, averaged over many
 * training sets: across the degree at a fixed sample size (left) and across the sample size at a fixed degree (right).
 * The vertical distance between the two curves is the expected generalisation gap.
 */
export function GapCurves() {
  const state = useFigureState({
    degree: int(5, { min: 0, max: 12, label: 'polynomial degree' }),
    m: int(30, { min: 6, max: 200, suggestions: [10, 30, 100], label: 'training points m' }),
    noise: slider(0.05, 1, 0.3, { step: 0.05, label: 'noise σ' }),
  })
  const sweep = useComputed(
    () =>
      biasVarianceSweep(stream('generalisation-gap/degree'), {
        target: 'sine',
        noise: state.noise,
        n: state.m,
        repeats: REPEATS,
        family: 'polynomial',
        values: DEGREES,
      }),
    [state.noise, state.m],
    { mode: 'release' },
  )
  // Below three points per coefficient the least-squares fit is close to interpolating, with errors far off the chart.
  const sizes = SIZES.filter((m) => m >= 3 * (state.degree + 1))
  const curve = useComputed(
    () =>
      sizes.map(
        (m) =>
          biasVariance(stream('generalisation-gap/size'), {
            target: 'sine',
            noise: state.noise,
            n: m,
            repeats: REPEATS,
            keep: 0,
            model: { kind: 'polynomial', degree: state.degree },
          }).totals,
      ),
    [state.noise, state.degree, sizes.length],
    { mode: 'release' },
  )
  const sw = sweep.value
  const at = Math.min(state.degree, sw.values.length - 1)
  const train = sw.trainError[at]
  const test = sw.error[at]
  const noiseVariance = state.noise ** 2
  const clip = (a: ArrayLike<number>) => Array.from(a, (v) => Math.max(LOW, Math.min(HIGH, v)))
  const shownSizes = sizes.slice(0, curve.value.length)

  const degreeAxis = useAxis({ label: 'polynomial degree', range: [0, 12] })
  const sizeAxis = useAxis({
    label: 'training points m',
    range: [10, 200],
    log: true,
    format: (v) => String(Math.round(v)),
  })
  const errorAxis = useAxis({ label: 'mean squared error', range: [LOW, HIGH], log: true })
  const errorAxis2 = useAxis({ label: 'mean squared error', range: [LOW, HIGH], log: true })
  return (
    <Figure
      title="Training error, test error and the gap between them"
      state={state}
      caption="Least-squares polynomials fitted to m points with x uniform on [−1, 1] and y = sin(πx) + σε, averaged over 200 training sets. The training error is measured on each fit's own points; the expected test error is its squared error on fresh data, so it includes the noise σ² (dashed line, the Bayes risk). The vertical distance between them is the expected generalisation gap. Left: across the degree at the chosen m; the gap widens as the class grows. Right: across m at the chosen degree; the two curves close in on σ² from either side as m grows, and the gap shrinks roughly in proportion to (degree + 1)/m. Drag the degree on the left chart. Errors are clipped to [0.001, 10]."
      readouts={
        <>
          <Readout label="training error" value={fmt(train)} />
          <Readout label="expected test error" value={fmt(test)} />
          <Readout label="gap" value={fmt(test - train)} />
          <Readout label="noise σ²" value={fmt(noiseVariance)} />
        </>
      }
    >
      <Plots cols={2}>
        <Plot x={degreeAxis} y={errorAxis} title={`m = ${state.m}`}>
          <Curve name="Bayes risk σ²" x={[0, 12]} y={[noiseVariance, noiseVariance]} muted dashed />
          <Curve name="training error" slot={1} x={sw.values} y={clip(sw.trainError)} showPoints stale={sweep.stale} />
          <Curve name="expected test error" slot={0} x={sw.values} y={clip(sw.error)} showPoints stale={sweep.stale} />
          <Handle {...state.handle('degree', { label: `degree ${state.degree}` })} />
        </Plot>
        <Plot x={sizeAxis} y={errorAxis2} title={`degree ${state.degree}`}>
          <Curve name="Bayes risk σ²" x={[10, 200]} y={[noiseVariance, noiseVariance]} muted dashed />
          <Curve
            name="training error"
            slot={1}
            x={shownSizes}
            y={clip(curve.value.map((t) => t.trainError))}
            showPoints
            stale={curve.stale}
          />
          <Curve
            name="expected test error"
            slot={0}
            x={shownSizes}
            y={clip(curve.value.map((t) => t.error))}
            showPoints
            stale={curve.stale}
          />
        </Plot>
      </Plots>
    </Figure>
  )
}
