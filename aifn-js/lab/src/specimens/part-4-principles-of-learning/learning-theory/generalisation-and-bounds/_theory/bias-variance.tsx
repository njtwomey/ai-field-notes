/**
 * The bias–variance decomposition by resampling (`aifn-applied/theory/bias-variance`): fits to many training sets,
 * their mean against the truth, the decomposition per x, and the totals across complexity with the complexity draggable.
 */
import { biasVariance, biasVarianceSweep } from 'aifn-applied/theory/bias-variance'
import { stream } from 'aifn/foundation/random'
import { Figure } from '@lab/layout'
import { choice, float, int, row, slider, useComputed, useFigureState, type AnyValues } from '@lab/state'
import { Area, Curve, formatNumber, Handle, Plot, Plots, Points, Readout, useAxis } from '@lab/viz'

const fmt = (v: number, digits = 3) => (Number.isFinite(v) ? formatNumber(Number(v.toPrecision(digits))) : '—')
const TARGETS = [
  { value: 'sine', label: 'sin πx' },
  { value: 'step', label: 'step' },
  { value: 'quadratic', label: 'quadratic' },
  { value: 'wiggly', label: 'wiggly' },
] as const
const isPoly = (v: AnyValues) => v.family === 'polynomial'
const DEGREES = Array.from({ length: 13 }, (_, i) => i)
const KS = [1, 2, 3, 5, 7, 10, 15, 20, 25, 30]

export function BiasVarianceSpecimen() {
  const state = useFigureState({
    problem: row('1 · problem', {
      target: choice(TARGETS, 'sine', { label: 'regression function' }),
      noise: slider(0, 1, 0.3, { step: 0.05, label: 'noise σ' }),
      n: int(30, { ge: 3, le: 500, suggestions: [10, 30, 100], label: 'points per training set' }),
      repeats: int(200, { ge: 10, le: 2000, suggestions: [50, 200, 500], label: 'training sets' }),
      seed: int(1, { ge: 0, le: 9999, label: 'seed' }),
    }),
    model: row('2 · model', {
      family: choice(
        [
          { value: 'polynomial', label: 'polynomial least squares' },
          { value: 'knn', label: 'k-nearest-neighbour regression' },
        ],
        'polynomial',
        { label: 'model' },
      ),
      degree: int(3, { ge: 0, le: 12, label: 'degree', when: isPoly }),
      ridge: float(0, { ge: 0, scale: 'log10', suggestions: [0, 1e-3, 1e-2, 1e-1], label: 'ridge λ', when: isPoly }),
      k: int(5, { ge: 1, le: 30, label: 'neighbours k', when: (v) => !isPoly(v) }),
    }),
  })
  const { target, noise, n, repeats, seed } = state.problem
  const { family, degree, ridge, k } = state.model
  const poly = family === 'polynomial'
  const one = useComputed(
    () =>
      biasVariance(stream(`bias-variance/${seed}`), {
        target: target as 'sine',
        noise,
        n,
        repeats,
        model: poly ? { kind: 'polynomial', degree, ridge } : { kind: 'knn', k: Math.min(k, n) },
      }),
    [target, noise, n, repeats, seed, poly, degree, ridge, k],
    { mode: 'release' },
  )
  const values = poly ? DEGREES : KS.filter((v) => v <= n)
  const sweep = useComputed(
    () =>
      biasVarianceSweep(stream(`bias-variance/${seed}`), {
        target: target as 'sine',
        noise,
        n,
        repeats: Math.min(repeats, 200),
        family: poly ? 'polynomial' : 'knn',
        values,
        ridge,
      }),
    [target, noise, n, repeats, seed, poly, ridge, values.length],
    { mode: 'release' },
  )
  const r = one.value
  const sw = sweep.value
  const xAxis = useAxis({ label: 'x', range: [-1, 1] })
  const yAxis = useAxis({ label: 'y', range: [-2, 2] })
  const partAxis = useAxis({
    label: 'contribution to E(y − ŷ)²',
    hold: 'union',
    key: `${target}/${noise}/${n}/${family}`,
  })
  const cAxis = useAxis({
    label: poly ? 'polynomial degree (complexity →)' : 'neighbours k (← complexity)',
    range: poly ? [0, 12] : [1, 30],
    key: family,
  })
  const errAxis = useAxis({ label: 'expected error', log: true, range: [1e-3, 10] })
  const grid = r.x.length
  const shown = Math.min(20, r.trainingSets.length)
  const complexity = poly ? degree : k
  const setComplexity = (v: number) =>
    poly
      ? state.set('model.degree', Math.max(0, Math.min(12, Math.round(v))))
      : state.set('model.k', Math.max(1, Math.min(30, Math.round(v))))
  const clip = (a: ArrayLike<number>) => Array.from(a, (v) => Math.max(1e-3, Math.min(10, v)))
  return (
    <Figure
      title="Bias and variance by resampling"
      purpose="Fitting one model to many training sets from the same problem separates its expected error into bias² (how far the average fit is from the truth), variance (how much the fits scatter around their average) and the irreducible noise σ²; more complex models trade the first for the second."
      state={state}
      defaultSize="L"
      readouts={{
        [poly ? `degree ${degree}` : `k = ${k}`]: (
          <>
            <Readout label="bias²" value={fmt(r.totals.bias2)} />
            <Readout label="variance" value={fmt(r.totals.variance)} />
            <Readout label="noise σ²" value={fmt(r.totals.noise)} />
            <Readout label="expected test error" value={fmt(r.totals.error)} />
            <Readout label="training error" value={fmt(r.totals.trainError)} />
          </>
        ),
      }}
      caption="aifn biasVariance draws training sets of n points with x uniform on [−1, 1] and y = f(x) + σε, fits each, and averages over x. Left: 20 of the fits (thin), their mean (slot colour) and f (dashed ink), with the first training set's points. Middle: bias² and variance at each x, stacked on σ². Right: biasVarianceSweep across complexity (the same training sets for every value), with the training error; drag the marker to change the model on the left. High-degree least-squares fits to 30 random points swing wildly near the ends of the interval, so their variance, and the bias of their mean, explode (values above 10 are clipped)."
    >
      <Plots cols={3}>
        <Plot x={xAxis} y={yAxis} title="fits to resampled training sets">
          {Array.from({ length: shown }, (_, i) => (
            <Curve
              key={i}
              name="fits"
              x={r.x}
              y={Array.from(r.fits.subarray(i * grid, (i + 1) * grid), (v) => Math.max(-2, Math.min(2, v)))}
              muted
              thin
            />
          ))}
          <Curve name="mean fit" slot={0} x={r.x} y={r.mean} />
          <Curve name="f" x={r.x} y={r.truth} emphasis dashed />
          {r.trainingSets[0] && (
            <Points name="one training set" x={r.trainingSets[0].x} y={r.trainingSets[0].y} slot={1} size={5} />
          )}
        </Plot>
        <Plot x={xAxis} y={partAxis} title="decomposition at each x">
          <Area name="noise σ²" x={r.x} y={r.x.map(() => r.totals.noise)} muted opacity={0.3} />
          <Area
            name="bias²"
            slot={2}
            x={r.x}
            y={r.x.map((_, i) => r.totals.noise + r.bias2[i])}
            base={r.totals.noise}
          />
          <Area
            name="variance"
            slot={3}
            x={r.x}
            y={r.x.map((_, i) => r.totals.noise + r.bias2[i] + r.variance[i])}
            base={Array.from(r.x, (_, i) => r.totals.noise + r.bias2[i])}
          />
        </Plot>
        <Plot x={cAxis} y={errAxis} title="across complexity">
          <Curve name="bias²" slot={2} x={sw.values} y={clip(sw.bias2)} showPoints stale={sweep.stale} />
          <Curve name="variance" slot={3} x={sw.values} y={clip(sw.variance)} showPoints stale={sweep.stale} />
          <Curve name="expected test error" slot={0} x={sw.values} y={clip(sw.error)} showPoints stale={sweep.stale} />
          <Curve name="training error" slot={1} x={sw.values} y={clip(sw.trainError)} dashed stale={sweep.stale} />
          <Handle kind="x" at={complexity} onDrag={setComplexity} label={poly ? `degree ${degree}` : `k = ${k}`} />
        </Plot>
      </Plots>
    </Figure>
  )
}
