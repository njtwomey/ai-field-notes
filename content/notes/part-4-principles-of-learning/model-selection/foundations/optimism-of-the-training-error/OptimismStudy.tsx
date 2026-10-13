import { useMemo, useState } from 'react'
import { type OptimismSweep } from 'aifn-methods/theory/optimism'
import {
  Annotation,
  call,
  choice,
  Curve,
  Figure,
  formatNumber,
  Handle,
  int,
  Plot,
  Plots,
  Readout,
  Segments,
  slider,
  useAxis,
  useComputed,
  useFigureState,
} from 'aifn-render'

type Family = 'polynomial' | 'ridge' | 'knn' | 'best-of-noise-features'

/** The ridge family penalises a degree-9 polynomial: 10 coefficients. */
const RIDGE_DEGREE = 9
const RIDGE = Array.from({ length: 19 }, (_, i) => 10 ** (-6 + i / 2))
const DEGREES = Array.from({ length: 15 }, (_, d) => d)
const NEIGHBOURS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 70, 100, 150, 200]
const CANDIDATES = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 70, 100, 150, 200]

/** Each family's swept values (k and M at most n), axis label, log axis and the value the handle opens at. */
const FAMILIES: Record<Family, { values: (n: number) => number[]; label: string; log: boolean; start: number }> = {
  polynomial: { values: () => DEGREES, label: 'polynomial degree d (p = d + 1 coefficients)', log: false, start: 9 },
  ridge: { values: () => RIDGE, label: `ridge penalty λ on a degree-${RIDGE_DEGREE} polynomial`, log: true, start: 1 },
  knn: { values: (n) => NEIGHBOURS.filter((k) => k <= n), label: 'neighbours k', log: true, start: 5 },
  'best-of-noise-features': {
    values: (n) => CANDIDATES.filter((m) => m <= n),
    label: 'candidate noise features M',
    log: true,
    start: 100,
  },
}

const fmt = (v: number | undefined) =>
  v !== undefined && Number.isFinite(v) ? formatNumber(Number(v.toPrecision(3))) : '—'
/** The index of the swept value nearest x (in log units on a log axis). */
const nearest = (values: number[], x: number, log: boolean) => {
  const t = (v: number) => (log ? Math.log(Math.max(v, 1e-300)) : v)
  let best = 0
  values.forEach((v, i) => {
    if (Math.abs(t(v) - t(x)) < Math.abs(t(values[best]) - t(x))) best = i
  })
  return best
}

/**
 * The optimism of the training error by simulation on a fixed design. Left: the mean training, in-sample and
 * extra-sample errors across a family of fits. Right: the optimism measured directly (in-sample minus training error,
 * ±1 Monte Carlo standard error) beside the covariance formula 2σ²·df/n with df from theory (trace S for the linear
 * smoothers, E max of M χ²₁ for the best of M noise features) and the covariance penalty estimated from the repeats.
 */
export function OptimismStudy() {
  const state = useFigureState({
    family: choice(
      [
        { value: 'polynomial', label: 'least-squares polynomial' },
        { value: 'ridge', label: 'ridge polynomial' },
        { value: 'knn', label: 'k-nearest neighbours' },
        { value: 'best-of-noise-features', label: 'best of M noise features' },
      ],
      'polynomial',
      { label: 'fitting procedure' },
    ),
    n: int(50, { ge: 10, le: 200, suggestions: [20, 50, 100, 200], label: 'training points n' }),
    noise: slider(0.1, 2, 1, { step: 0.1, label: 'noise σ' }),
    repeats: int(500, { ge: 20, le: 5000, suggestions: [100, 500, 2000], label: 'training sets per value' }),
    seed: int(1, { ge: 1, le: 999, label: 'seed' }),
  })
  const { family, n, noise, repeats, seed } = state
  const spec = FAMILIES[family as Family]
  const values = useMemo(() => spec.values(n), [spec, n])
  const run = useComputed(
    () =>
      call<OptimismSweep>(
        'theory/optimism/optimismSweep',
        call('foundation/random/stream', `optimism-of-the-training-error/${seed}`),
        {
          family,
          values,
          degree: RIDGE_DEGREE,
          target: family === 'best-of-noise-features' ? 'zero' : 'sine',
          noise,
          n,
          repeats,
        },
      ),
    [family, values, noise, n, repeats, seed],
    { mode: 'worker', initial: null as OptimismSweep | null },
  )
  const r = run.value
  const ready = r !== null && r.values.length === values.length && r.values.every((v, i) => v === values[i])
  const sigma2 = noise * noise

  const [picked, setPicked] = useState<{ family: string; value: number } | null>(null)
  const focusValue = picked && picked.family === family ? picked.value : spec.start
  const focus = nearest(values, focusValue, spec.log)

  const view = useMemo(() => {
    if (!ready || r === null) return null
    const formula = Array.from(r.referenceDf, (df) => (2 * sigma2 * df) / n)
    const bars = values.map((v, i) => ({
      from: [v, r.optimism[i] - r.optimismSe[i]] as const,
      to: [v, r.optimism[i] + r.optimismSe[i]] as const,
    }))
    return { formula, bars }
  }, [ready, r, values, sigma2, n])

  const xErr = useAxis({ label: spec.label, log: spec.log })
  const xOpt = useAxis({ label: spec.label, log: spec.log })
  // The extra-sample error of a flexible fit grows without bound near the ends of [−1, 1]; the axis keeps the training
  // and in-sample errors in view and clips it.
  const top = r === null ? undefined : 1.25 * Math.max(...r.inSampleError, ...r.trainError, 2 * sigma2)
  const yErr = useAxis({ label: 'mean squared error', range: [0, top] })
  const yOpt = useAxis({ label: 'optimism' })
  const at = values[focus]
  const isBest = family === 'best-of-noise-features'
  const unit = isBest ? 'M' : family === 'knn' ? 'k' : family === 'ridge' ? 'λ' : 'd'
  return (
    <Figure
      title="Optimism measured against the covariance formula"
      state={state}
      defaultSize="L"
      readouts={
        ready && r !== null && view !== null ? (
          <>
            <Readout
              label={`at ${unit} = ${fmt(at)}: training / in-sample error`}
              value={`${fmt(r.trainError[focus])} / ${fmt(r.inSampleError[focus])}`}
            />
            <Readout label="optimism measured" value={`${fmt(r.optimism[focus])} ± ${fmt(r.optimismSe[focus])}`} />
            <Readout label="2σ²·df/n, df from theory" value={fmt(view.formula[focus])} />
            <Readout label="covariance penalty" value={fmt(r.covariancePenalty[focus])} />
            <Readout
              label="df measured / from theory"
              value={`${fmt(r.df[focus])} ± ${fmt(r.dfSe[focus])} / ${fmt(r.referenceDf[focus])}`}
            />
            <Readout label="extra-sample error" value={fmt(r.extraSampleError[focus])} />
          </>
        ) : (
          <Readout label="status" value={run.error ?? 'computing…'} />
        )
      }
      caption={`Each value of the swept parameter is fitted to ${repeats} training sets of n = ${n} points, with the inputs drawn once on [−1, 1] and fresh noise of standard deviation σ in each set. The target is y = sin(πx) + σε, except for the best of M noise features, where y is pure noise and the procedure keeps the one of M orthogonal noise features most correlated with y. Left: the mean training error, the in-sample error (fresh targets at the same inputs) and the extra-sample error (fresh inputs as well), with the noise variance σ² dashed. Right: the optimism measured as in-sample minus training error, with ±1 Monte Carlo standard error, beside 2σ²·df/n with df from theory (trace S for least squares, ridge and nearest neighbours; the expected maximum of M χ²₁ variables for the best of M) and the covariance penalty (2/n)Σ cov(ŷᵢ, yᵢ) estimated from the same repeats. For the best of M, the grey dashed line counts one fitted coefficient, 2σ²/n. Drag the vertical line to read off one value. Computed in the background.`}
    >
      <Plots cols={2}>
        <Plot x={xErr} y={yErr} title="errors">
          {ready && r !== null && (
            <>
              <Curve name="noise σ²" x={[values[0], values[values.length - 1]]} y={[sigma2, sigma2]} muted dashed />
              <Curve name="training error" x={values} y={r.trainError} slot={0} showPoints stale={run.stale} />
              <Curve name="in-sample error" x={values} y={r.inSampleError} slot={1} showPoints stale={run.stale} />
              <Curve name="extra-sample error" x={values} y={r.extraSampleError} slot={2} stale={run.stale} />
              <Annotation x={at} text={`${unit} = ${fmt(at)}`} dashed />
            </>
          )}
        </Plot>
        <Plot x={xOpt} y={yOpt} title="optimism">
          {ready && r !== null && view !== null && (
            <>
              {isBest && (
                <Curve
                  name="one coefficient counted, 2σ²/n"
                  x={[values[0], values[values.length - 1]]}
                  y={[(2 * sigma2) / n, (2 * sigma2) / n]}
                  muted
                  dashed
                />
              )}
              <Curve name="2σ²·df/n, df from theory" x={values} y={view.formula} slot={0} dashed stale={run.stale} />
              <Curve name="covariance penalty" x={values} y={r.covariancePenalty} slot={1} stale={run.stale} />
              <Segments name="±1 Monte Carlo SE" segments={view.bars} />
              <Curve name="optimism measured" x={values} y={r.optimism} emphasis showPoints stale={run.stale} />
              <Handle
                kind="x"
                at={at}
                label={`${unit} = ${fmt(at)}`}
                onDrag={(x) => setPicked({ family, value: values[nearest(values, x, spec.log)] })}
              />
            </>
          )}
        </Plot>
      </Plots>
    </Figure>
  )
}
