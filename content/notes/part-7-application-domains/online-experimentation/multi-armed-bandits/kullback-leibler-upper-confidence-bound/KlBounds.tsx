import { useMemo } from 'react'
import {
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Plot,
  Readout,
  seriesLayers,
  type SeriesSpec,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { klBernoulli, klUpper } from '../_shared/bandits'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'

const GRID = toFlat(linspace(0.0005, 0.9995, 400))
const COUNTS = Array.from({ length: 200 }, (_, i) => i + 1)

/** Hoeffding index p + √(ℓ / 2n): the q where the quadratic 2n(q − p)² reaches the level ℓ. */
const hoeffding = (p: number, n: number, level: number) => p + Math.sqrt(level / (2 * n))

/**
 * The two confidence bounds as level sets: n kl(p̂, q) against its Pinsker lower bound 2n(q − p̂)², and the level ℓ that
 * both must stay under. The sample mean and the level are draggable. A second chart shows both indices against n.
 */
export function KlBounds() {
  const state = useFigureState({
    p: int(0.1, { min: 0.01, max: 0.99, step: 0.01, label: 'sample mean p̂' }),
    n: int(20, { min: 1, max: 200, step: 1, label: 'pulls n', format: (v) => String(v) }),
    level: float(Math.round(Math.log(1000) * 100) / 100, {
      min: 0.5,
      max: 12,
      step: 0.01,
      label: 'level ℓ (ln t: ln 1000 ≈ 6.91)',
    }),
  })

  const pv = state.p
  const nv = state.n
  const lv = state.level
  const kl = klUpper(pv, nv, lv)
  const hf = hoeffding(pv, nv, lv)
  const yTop = 3 * lv

  const divergence = useMemo(
    (): SeriesSpec[] => [
      { name: 'n kl(p̂, q)', type: 'line', x: GRID, y: GRID.map((q) => nv * klBernoulli(pv, q)), slot: 2 },
      { name: '2n (q − p̂)², Hoeffding', type: 'line', x: GRID, y: GRID.map((q) => 2 * nv * (q - pv) ** 2), slot: 0 },
    ],
    [pv, nv],
  )
  const series: SeriesSpec[] = [
    ...divergence,
    { name: 'indices', type: 'scatter', x: [kl, Math.min(hf, 1.2)], y: [lv, lv], emphasis: true },
  ]

  const byCount = useMemo(
    () =>
      [
        { name: 'KL-UCB index', x: COUNTS, y: COUNTS.map((c) => klUpper(pv, c, lv)), slot: 2 },
        {
          name: 'Hoeffding index',
          x: COUNTS,
          y: COUNTS.map((c) => Math.min(hoeffding(pv, c, lv), 2)),
          slot: 0,
        },
        { name: 'sample mean p̂', x: [1, 200], y: [pv, pv], muted: true },
      ] as const,
    [pv, lv],
  )

  const xAxis = useAxis({ label: 'candidate mean q', range: [0, 1] })
  const yAxis = useAxis({ label: 'divergence', range: [0, yTop] })
  const xAxis2 = useAxis({ label: 'pulls n', range: [1, 200] })
  const yAxis2 = useAxis({ label: 'upper bound', range: [0, 1.2] })
  return (
    <Figure
      title="Hoeffding and KL confidence bounds"
      state={state}
      caption="Left: an arm with sample mean p̂ after n pulls. A candidate mean q is kept in the confidence set while the divergence n kl(p̂, q) is below the level ℓ (the horizontal line; drag it). The KL-UCB index is where the green curve crosses the line. Hoeffding's interval uses the parabola 2n(q − p̂)², which lies below the KL curve everywhere (Pinsker's inequality), so it crosses later and gives a wider interval. Drag p̂ (the vertical line) towards 0 or 1: the KL curve steepens and the KL bound becomes much tighter, while the parabola does not change shape. Right: both indices as the number of pulls grows."

      readouts={
        <>
          <Readout label="KL-UCB index" value={formatNumber(kl)} />
          <Readout label="Hoeffding index" value={formatNumber(hf)} />
          <Readout label="width ratio, Hoeffding / KL" value={formatNumber((hf - pv) / Math.max(kl - pv, 1e-9))} />
          <Readout label="level as rounds t = e^ℓ" value={formatNumber(Math.exp(lv))} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <Plot x={xAxis} y={yAxis}>
          {seriesLayers(series)}
          <Handle kind="x" at={pv} label="p̂" onDrag={(x) => state.set('p', x)} />
          <Handle kind="y" at={lv} label="level ℓ" onDrag={(y) => state.set('level', y)} />
        </Plot>
        <Plot x={xAxis2} y={yAxis2}>
          <Curve {...byCount[0]} />
          <Curve {...byCount[1]} />
          <Curve {...byCount[2]} />
        </Plot>
      </div>
    </Figure>
  )
}
