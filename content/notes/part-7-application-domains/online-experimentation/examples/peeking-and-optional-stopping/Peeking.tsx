import { useMemo } from 'react'
import {
  choice,
  Curve,
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
import { BOUNDARIES } from './boundaries'
import { normal, stream } from 'aifn-compute/foundation/random'
import { normalQuantile } from 'aifn-compute/numerics/special'

type Design = 'naive' | 'pocock' | 'obf'

const EXPERIMENTS = 2000
const Y = 5
const ALPHAS = [
  { value: '0.01', label: '0.01' },
  { value: '0.05', label: '0.05' },
  { value: '0.1', label: '0.10' },
]
const DESIGNS: { value: Design; label: string }[] = [
  { value: 'naive', label: 'stop at first p < α' },
  { value: 'pocock', label: 'Pocock' },
  { value: 'obf', label: "O'Brien–Fleming" },
]

/** The critical value for |Z| at look j of K. */
function boundary(design: Design, alpha: string, K: number, j: number): number {
  const z = normalQuantile(1 - Number(alpha) / 2)
  if (design === 'naive' || K === 1) return z
  const table = BOUNDARIES[alpha]
  return design === 'pocock' ? table.pocock[K - 1] : table.obf[K - 1] * Math.sqrt(K / j)
}

/**
 * A/A experiments, where nothing differs between the arms, analysed after each of K equally sized batches. The
 * running z-statistic wanders; stopping the first time it crosses the fixed-sample threshold inflates the
 * false-positive rate. Group-sequential boundaries widen the threshold so that the overall rate stays at α.
 */
export function Peeking() {
  const state = useFigureState({
    looks: int(10, { min: 1, max: 100, step: 1, label: 'looks K' }),
    alpha: choice(ALPHAS, '0.05', { label: 'α' }),
    design: choice<Design>(DESIGNS, 'naive', { label: 'stopping rule' }),
    drawn: int(40, { min: 1, max: 50, step: 1, label: 'paths', format: (v) => String(v) }),
    seed: int(1, { min: 1, max: 20, step: 1, label: 'seed' }),
  })
  const K = state.looks

  // Z_j = S_j / √j, where S_j sums j independent N(0, 1) batch statistics: equal information per look.
  const paths = useMemo(() => {
    const g = stream(state.seed * 1000 + K)
    return Array.from({ length: EXPERIMENTS }, () => {
      let s = 0
      return Array.from({ length: K }, (_, i) => {
        s += normal(g)
        return s / Math.sqrt(i + 1)
      })
    })
  }, [K, state.seed])

  const r = useMemo(() => {
    const bounds = Array.from({ length: K }, (_, i) => boundary(state.design, state.alpha, K, i + 1))
    const firstCross = paths.map((z) => z.findIndex((v, i) => Math.abs(v) >= bounds[i]))
    const cumulative = bounds.map((_, j) => firstCross.filter((c) => c >= 0 && c <= j).length / EXPERIMENTS)
    return { bounds, firstCross, cumulative }
  }, [paths, state.design, state.alpha, K])

  // The first experiments of the simulated 2000 are drawn, so the count changes only the picture, never the rate.
  // Paths stop at the look where they cross. Each group is one series, with paths separated by NaN breaks.
  const drawnPaths = useMemo(() => {
    const kept = { x: [] as number[], y: [] as number[] }
    const crossed = { x: [] as number[], y: [] as number[] }
    paths.slice(0, state.drawn).forEach((z, e) => {
      const hit = r.firstCross[e] >= 0
      const stop = hit ? r.firstCross[e] : K - 1
      const target = hit ? crossed : kept
      // Every path starts from z = 0 before any data.
      target.x.push(0, ...z.slice(0, stop + 1).map((_, i) => i + 1), NaN)
      target.y.push(0, ...z.slice(0, stop + 1), NaN)
    })
    return { kept, crossed }
  }, [paths, r.firstCross, state.drawn, K])
  const lookAxis = Array.from({ length: K }, (_, i) => i + 1)
  const edge = (sign: number) => r.bounds.map((b) => sign * Math.min(b, Y + 1))
  const many = state.drawn > 1
  const pathSeries = [
    { name: 'ran to the end', x: drawnPaths.kept.x, y: drawnPaths.kept.y, slot: 0, thin: many },
    {
      name: 'stopped: false positive',
      x: drawnPaths.crossed.x,
      y: drawnPaths.crossed.y,
      slot: 1,
      thin: many,
    },
    {
      name: 'boundary',
      x: [...lookAxis, NaN, ...lookAxis],
      y: [...edge(1), NaN, ...edge(-1)],
      slot: 2,
      dashed: true,
    },
  ] as const

  const exactNaive = BOUNDARIES[state.alpha].naive
  const rateSeries: SeriesSpec[] = [
    { name: 'simulated', type: 'line', x: lookAxis, y: r.cumulative, slot: 1 },
    ...(state.design === 'naive'
      ? [{ name: 'exact', type: 'line' as const, x: lookAxis, y: exactNaive.slice(0, K), slot: 0, dashed: true }]
      : []),
    {
      name: 'α',
      type: 'line',
      x: [1, Math.max(K, 2)],
      y: [Number(state.alpha), Number(state.alpha)],
      slot: 2,
      dashed: true,
    },
  ]

  const final = r.cumulative[K - 1]
  const exact = state.design === 'naive' ? exactNaive[K - 1] : Number(state.alpha)

  const xAxis = useAxis({ label: 'look', range: [0, K] })
  const yAxis = useAxis({ label: 'z-statistic', range: [-Y, Y] })
  const xAxis2 = useAxis({ label: 'look', hold: 'union' })
  const yAxis2 = useAxis({ label: 'P(false positive by this look)', range: [0, undefined], hold: 'union' })
  return (
    <Figure
      title="Peeking at an A/A test"
      state={state}
      caption="Each light line is one A/A experiment: both arms are identical, so any significant result is a false positive. The z-statistic is recomputed after each of K equal batches, and the experiment stops at the first look where it crosses the boundary; stopped experiments are drawn in the second colour. The lines are the first of 2000 simulated experiments, and the paths slider sets how many are drawn, not how many are simulated. With the fixed-sample threshold at every look, the chance of stopping somewhere grows with K, far above α. Pocock's boundary raises the threshold equally at every look; O'Brien–Fleming's starts very high and falls to about the fixed-sample value at the end. Both hold the overall false-positive rate at α."

      readouts={
        <>
          <Readout label="false-positive rate (simulated)" value={formatNumber(final)} />
          <Readout label={state.design === 'naive' ? 'exact' : 'designed'} value={formatNumber(exact)} />
          <Readout label="threshold at first look" value={formatNumber(r.bounds[0])} />
          <Readout label="threshold at last look" value={formatNumber(r.bounds[K - 1])} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <Plot x={xAxis} y={yAxis} height={320}>
          <Curve {...pathSeries[0]} />
          <Curve {...pathSeries[1]} />
          <Curve {...pathSeries[2]} />
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={320}>
          {seriesLayers(rateSeries)}
        </Plot>
      </div>
    </Figure>
  )
}
