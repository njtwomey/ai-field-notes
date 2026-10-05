import { useMemo } from 'react'
import {
  Button,
  Figure,
  float,
  formatNumber,
  Handle,
  Plot,
  Readout,
  seriesLayers,
  type SeriesSpec,
  setting,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { metricsAt, negativeDensity, positiveDensity, posterior, scoreForPosterior } from '../_shared/binormal'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'

const S = toFlat(linspace(-4, 7, 301))

/**
 * Two class-conditional score densities, optionally scaled by the class priors, with a draggable decision threshold.
 * Shaded areas are the false positives (negatives right of the threshold) and false negatives (positives left of it).
 */
export function ThresholdExplorer() {
  const state = useFigureState({
    pi: slider(0.005, 0.5, 0.05, { step: 0.005, label: 'prevalence π' }),
    d: float(2, { min: 0.5, max: 4, step: 0.1, label: 'separation d' }),
    costRatio: slider(1, 100, 20, { step: 1, label: 'cost ratio c = c_FN / c_FP' }),
    t: slider(-4, 7, scoreForPosterior(0.5, 0.05, 2), { step: 0.01, label: 'threshold on the score' }),
    joint: setting(true, 'scale densities by prior'),
  })

  const pStar = 1 / (1 + state.costRatio)
  const sStar = scoreForPosterior(pStar, state.pi, state.d)
  const sHalf = scoreForPosterior(0.5, state.pi, state.d)

  const series = useMemo<SeriesSpec[]>(() => {
    const wNeg = state.joint ? 1 - state.pi : 1
    const wPos = state.joint ? state.pi : 1
    const neg = S.map((s) => wNeg * negativeDensity(s))
    const pos = S.map((s) => wPos * positiveDensity(s, state.d))
    const right = S.filter((s) => s >= state.t)
    const left = S.filter((s) => s <= state.t)
    return [
      { name: 'negatives', type: 'line', x: S, y: neg, slot: 0 },
      { name: 'positives', type: 'line', x: S, y: pos, slot: 1 },
      {
        name: 'false positives',
        type: 'line',
        x: right,
        y: right.map((s) => wNeg * negativeDensity(s)),
        slot: 0,
        area: true,
      },
      {
        name: 'false negatives',
        type: 'line',
        x: left,
        y: left.map((s) => wPos * positiveDensity(s, state.d)),
        slot: 1,
        area: true,
      },
    ]
  }, [state.joint, state.pi, state.d, state.t])

  const top = state.joint ? 0.42 * Math.max(1 - state.pi, state.pi) : 0.42
  const guides = useMemo<SeriesSpec[]>(
    () => [
      { name: 'cost-optimal threshold', type: 'line', x: [sStar, sStar], y: [0, top], dashed: true, slot: 2 },
      { name: 'posterior 0.5', type: 'line', x: [sHalf, sHalf], y: [0, top], dashed: true, muted: true },
    ],
    [sStar, sHalf, top],
  )

  const all = useMemo(() => [...series, ...guides] as const, [series, guides])

  const m = metricsAt(state.t, state.pi, state.d)
  const cost = m.fpRate + state.costRatio * m.fnRate
  const best = metricsAt(sStar, state.pi, state.d)
  const bestCost = best.fpRate + state.costRatio * best.fnRate

  const xAxis = useAxis({ label: 'score s', range: [-4, 7] })
  const yAxis = useAxis({ label: state.joint ? 'prior × density' : 'density', range: [0, top] })
  return (
    <Figure
      title="Moving the threshold"
      state={state}
      caption="Scores are N(0, 1) for negatives and N(d, 1) for positives. With the switch on, each density is scaled by its class prior, so the areas are the joint probabilities that the metrics count. Drag the threshold. The dashed coloured line is the cost-optimal threshold for a false negative costing c times a false positive; the grey dashed line is where the posterior is 0.5. Lower the prevalence and watch the positive hump shrink and the posterior-0.5 line run off to the right."
      controls={
        <>
          <Button variant="outline" size="sm" onClick={() => state.set('t', sStar)}>
            Snap to cost-optimal
          </Button>
        </>
      }
      readouts={
        <>
          <Readout label="posterior at threshold" value={formatNumber(posterior(state.t, state.pi, state.d))} />
          <Readout label="optimal posterior threshold" value={formatNumber(pStar)} />
          <Readout label="precision" value={formatNumber(m.precision)} />
          <Readout label="recall" value={formatNumber(m.recall)} />
          <Readout label="F₁" value={formatNumber(m.f1)} />
          <Readout label="balanced accuracy" value={formatNumber(m.balancedAccuracy)} />
          <Readout label="expected cost (c_FP = 1)" value={formatNumber(cost)} />
          <Readout label="minimum expected cost" value={formatNumber(bestCost)} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={320}>
        {seriesLayers(all)}
        <Handle {...state.handle('t', { label: 'threshold' })} />
      </Plot>
    </Figure>
  )
}
