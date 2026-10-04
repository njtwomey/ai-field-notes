import { useMemo } from 'react'
import {
  Figure,
  float,
  formatNumber,
  Plot,
  Readout,
  seriesLayers,
  type SeriesSpec,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { linspace, toFlat } from 'aifn/foundation/tensor'

const M = toFlat(linspace(-3, 3, 301))

type Surrogate = { name: string; phi: (m: number) => number; argmin: (eta: number) => number }

// Minimisers of the conditional risk ηφ(α) + (1 − η)φ(−α), derived in the note.
const SURROGATES: Surrogate[] = [
  { name: 'logistic (base 2)', phi: (m) => Math.log1p(Math.exp(-m)) / Math.LN2, argmin: (e) => Math.log(e / (1 - e)) },
  { name: 'hinge', phi: (m) => Math.max(0, 1 - m), argmin: (e) => (e > 0.5 ? 1 : e < 0.5 ? -1 : 0) },
  { name: 'exponential', phi: (m) => Math.exp(-m), argmin: (e) => 0.5 * Math.log(e / (1 - e)) },
  { name: 'squared hinge', phi: (m) => Math.max(0, 1 - m) ** 2, argmin: (e) => 2 * e - 1 },
]

/** The 0–1 loss and convex surrogates against the margin, and the conditional risk each one minimises. */
export function SurrogateLosses() {
  const state = useFigureState({
    eta: float(0.8, { min: 0.05, max: 0.95, step: 0.05, label: 'η = P(y = +1 | x)' }),
  })

  const losses = useMemo(
    (): SeriesSpec[] => [
      { name: '0–1', type: 'line', x: M, y: M.map((m) => (m <= 0 ? 1 : 0)), emphasis: true },
      ...SURROGATES.map((s, i): SeriesSpec => ({ name: s.name, type: 'line', x: M, y: M.map(s.phi), slot: i })),
    ],
    [],
  )
  const risks = useMemo((): SeriesSpec[] => {
    const e = state.eta
    return [
      {
        name: '0–1',
        type: 'line',
        x: M,
        y: M.map((a) => (a > 0 ? 1 - e : a < 0 ? e : 0.5)),
        emphasis: true,
      },
      ...SURROGATES.map((s, i): SeriesSpec => ({
        name: s.name,
        type: 'line',
        x: M,
        y: M.map((a) => e * s.phi(a) + (1 - e) * s.phi(-a)),
        slot: i,
      })),
    ]
  }, [state.eta])

  const xAxis = useAxis({ label: 'margin m = y f(x)', range: [-3, 3] })
  const yAxis = useAxis({ label: 'loss', range: [0, 4] })
  const xAxis2 = useAxis({ label: 'score α', range: [-3, 3] })
  const yAxis2 = useAxis({ label: 'conditional risk ηφ(α) + (1 − η)φ(−α)', range: [0, 4] })
  return (
    <Figure
      title="Surrogate losses and the conditional risk they minimise"
      state={state}
      caption="Left: each loss against the margin m = y·f(x); every surrogate is convex and lies on or above the 0–1 step. The logistic loss is drawn in base 2, so that it passes through 1 at m = 0. Right: the expected loss at a point where P(y = +1 | x) = η, as a function of the score α. Every surrogate's minimum lies on the same side of zero as the Bayes decision, sign(2η − 1). The logistic, exponential and squared-hinge minima also move with η, so they encode the probability; the hinge minimum sits at ±1 whatever η is."

      readouts={
        <>
          {SURROGATES.map((s) => (
            <Readout key={s.name} label={`${s.name} α*`} value={formatNumber(s.argmin(state.eta))} />
          ))}
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <Plot x={xAxis} y={yAxis} height={320}>
          {seriesLayers(losses)}
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={320}>
          {seriesLayers(risks)}
        </Plot>
      </div>
    </Figure>
  )
}
