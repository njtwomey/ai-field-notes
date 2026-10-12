import { useEffect, useMemo, useReducer, useState } from 'react'
import {
  Figure,
  Handle,
  int,
  Plot,
  Readout,
  seriesLayers,
  type SeriesSpec,
  setting,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { covarianceEllipse } from 'aifn-compute/numerics/geometry'
import { toRows } from 'aifn-compute/foundation/tensor'
import { blobData } from './data'
import { bestFits, fitNext, labelsOf, pending, restartsDone, type KFit } from './selection'

/** k means and k diagonal variances in 2-D, plus k − 1 free weights. */
const parameters = (k: number) => 4 * k + (k - 1)

const argmin = (xs: number[]) => xs.indexOf(Math.min(...xs))

/** Longest stretch of fitting per task before the page gets the main thread back, ms. */
const SLICE_MS = 30

type Shown = { n: number; restarts: number; fits: KFit[] }

/**
 * The best fits for n points and a restart budget, fitted a few at a time after paint so that no single task blocks
 * the page for long. Restart 0 of every k comes first, so the curves appear after one round and sharpen as the other
 * restarts finish. Until the first round for a new n is done, the last complete result stays on display.
 */
function useProgressiveFits(n: number, restarts: number): Shown | null {
  // Re-renders after each slice of work, so that `restartsDone` reads the fit cache again.
  const [, tick] = useReducer((v: number) => v + 1, 0)
  // The last n with at least one restart fitted for every k: what is shown while a new n has none yet.
  const [lastN, setLastN] = useState<number | null>(null)
  useEffect(() => {
    let timer: ReturnType<typeof setTimeout> | undefined
    const work = () => {
      const t0 = performance.now()
      do fitNext(n, restarts)
      while (pending(n, restarts) && performance.now() - t0 < SLICE_MS)
      if (restartsDone(n, restarts) > 0) setLastN(n)
      tick()
      if (pending(n, restarts)) timer = setTimeout(work, 0)
    }
    if (pending(n, restarts)) timer = setTimeout(work, 0)
    return () => clearTimeout(timer)
  }, [n, restarts])
  const shownN = restartsDone(n, restarts) > 0 ? n : lastN
  const done = shownN === null ? 0 : restartsDone(shownN, restarts)
  return useMemo(
    (): Shown | null =>
      shownN !== null && done > 0 ? { n: shownN, restarts: done, fits: bestFits(shownN, done) } : null,
    [shownN, done],
  )
}

/**
 * −2 log L, AIC and BIC against k. The best-fit deviances come from EM on the first n points (`selection.ts`), for the
 * current n and restart budget only; the penalties are added here.
 */
export function ModelSelection() {
  const state = useFigureState({
    k: int(3, { min: 1, max: 8, step: 1, label: 'k shown' }),
    n: int(400, { min: 30, max: 400, step: 10, label: 'points n' }),
    restarts: int(3, { min: 1, max: 5, step: 1, label: 'restarts per k' }),
    zoom: setting(true, 'zoom on k ≥ 3'),
  })
  const shown = useProgressiveFits(state.n, state.restarts)
  // The n the curves were fitted on, which lags behind the slider until the first restart at a new n is done.
  const n = shown?.n ?? state.n

  const result = useMemo(() => {
    if (!shown) return null
    const deviance = shown.fits.map((f) => f.deviance)
    const ks = shown.fits.map((f) => f.k)
    const aic = ks.map((k, i) => deviance[i] + 2 * parameters(k))
    const bic = ks.map((k, i) => deviance[i] + parameters(k) * Math.log(shown.n))
    return { ks, deviance, aic, bic, fits: shown.fits }
  }, [shown])

  const series = useMemo((): SeriesSpec[] => {
    if (!result) return []
    // A marker at each criterion's minimum, sharing the line's name so the legend toggles both.
    const minimum = (name: string, ys: number[], slot: number): SeriesSpec => ({
      name,
      type: 'scatter',
      x: [result.ks[argmin(ys)]],
      y: [Math.min(...ys)],
      slot,
    })
    return [
      { name: '−2 log L', type: 'line', x: result.ks, y: result.deviance, slot: 0, dashed: true },
      { name: 'AIC', type: 'line', x: result.ks, y: result.aic, slot: 1 },
      { name: 'BIC', type: 'line', x: result.ks, y: result.bic, slot: 2 },
      minimum('AIC', result.aic, 1),
      minimum('BIC', result.bic, 2),
    ]
  }, [result])

  // Zoomed: fit the y-axis to k ≥ 3, where the criteria differ. k = 1 and 2 run off the top.
  const yRange = useMemo((): [number, number] | undefined => {
    if (!result || !state.zoom) return undefined
    const tail = [...result.deviance, ...result.aic, ...result.bic].filter((_, i) => i % result.ks.length >= 2)
    const lo = Math.min(...tail)
    const hi = Math.max(...tail)
    const pad = (hi - lo) * 0.15
    // Round to multiples of 50 so the axis ends on tidy tick labels.
    return [Math.floor((lo - pad) / 50) * 50, Math.ceil((hi + pad) / 50) * 50]
  }, [result, state.zoom])

  // The best fit at the chosen k on the same first n points: colour by most likely component, 1σ and 2σ ellipses.
  const scatter = useMemo((): SeriesSpec[] => {
    const xs = blobData().x.slice(0, n)
    const ys = blobData().y.slice(0, n)
    if (!result) return [{ name: 'points', type: 'scatter', x: xs, y: ys }]
    const fit = result.fits[state.k - 1]
    const names = fit.means.map((_, j) => `component ${j + 1}`)
    return [
      { name: 'points', type: 'scatter', x: xs, y: ys, group: labelsOf(fit, n), groupNames: names },
      ...fit.means.flatMap((mean, j) =>
        [1, 2].map((radius): SeriesSpec => {
          const outline = toRows(covarianceEllipse(mean, fit.covariances[j], { k: radius, points: 64 }).points)
          return {
            name: names[j],
            type: 'line',
            x: outline.map((p) => p[0]),
            y: outline.map((p) => p[1]),
            slot: j,
            dashed: radius === 2,
          }
        }),
      ),
      {
        name: 'means',
        type: 'scatter',
        x: fit.means.map((m) => m[0]),
        y: fit.means.map((m) => m[1]),
        emphasis: true,
      },
    ]
  }, [result, n, state.k])

  const xAxis = useAxis({ label: 'k', hold: 'union' })
  const yAxis = useAxis({ label: 'criterion (lower is better)', range: yRange })
  const xAxis2 = useAxis({ label: 'x₁', hold: 'union' })
  const yAxis2 = useAxis({ label: 'x₂', hold: 'union' })
  const fitted = shown?.restarts ?? 0
  return (
    <Figure
      title="Choosing k: likelihood, AIC and BIC"
      state={state}
      caption="Each k is fitted by EM from several k-means++ starts, keeping the best. The dashed line is −2 log L. It only falls, so on its own it always prefers more components; its bend at k = 3 is the elbow. AIC adds 2 per parameter and BIC adds log n per parameter, turning the elbow into a minimum. Lower is better. Change n: BIC's penalty grows with n, AIC's does not. Drag the line labelled k, or use its slider, to see the fitted mixture at that k on the right: beyond k = 3, extra components split real clusters or cover a few stray points. The fits are made a few at a time after the page loads, one restart of every k before the next, and the curves update as each restart finishes."
      readouts={
        <>
          <Readout label="AIC picks k" value={result ? result.ks[argmin(result.aic)] : '—'} />
          <Readout label="BIC picks k" value={result ? result.ks[argmin(result.bic)] : '—'} />
          <Readout label="true k" value={3} />
          <Readout label="BIC penalty per parameter" value={Math.log(n).toFixed(2)} />
          <Readout
            label="restarts fitted"
            value={shown && shown.n === state.n ? `${fitted} of ${state.restarts}` : 'fitting…'}
          />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <Plot x={xAxis} y={yAxis} height={340}>
          {seriesLayers(series)}
          <Handle kind="x" at={state.k} label="k" onDrag={(x) => state.set('k', Math.round(x))} />
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={340}>
          {seriesLayers(scatter)}
        </Plot>
      </div>
    </Figure>
  )
}
