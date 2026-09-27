import { useMemo } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, useParam, type XYSeries } from '@/components/viz'
import { linspace } from '@/lib/math'
import { klBernoulli, klUpper } from '../_shared/bandits'

const GRID = linspace(0.0005, 0.9995, 400)
const COUNTS = Array.from({ length: 200 }, (_, i) => i + 1)

/** Hoeffding index p + √(ℓ / 2n): the q where the quadratic 2n(q − p)² reaches the level ℓ. */
const hoeffding = (p: number, n: number, level: number) => p + Math.sqrt(level / (2 * n))

/**
 * The two confidence bounds as level sets: n kl(p̂, q) against its Pinsker lower bound 2n(q − p̂)², and the level ℓ that
 * both must stay under. The sample mean and the level are draggable. A second chart shows both indices against n.
 */
export function KlBounds() {
  const p = useParam(0.1, { min: 0.01, max: 0.99, step: 0.01 })
  const n = useParam(20, { min: 1, max: 200, step: 1 })
  const level = useParam(Math.round(Math.log(1000) * 100) / 100, { min: 0.5, max: 12, step: 0.01 })

  const pv = p.value
  const nv = n.value
  const lv = level.value
  const kl = klUpper(pv, nv, lv)
  const hf = hoeffding(pv, nv, lv)
  const yTop = 3 * lv

  const divergence = useMemo(
    (): XYSeries[] => [
      { name: 'n kl(p̂, q)', type: 'line', x: GRID, y: GRID.map((q) => nv * klBernoulli(pv, q)), slot: 2 },
      { name: '2n (q − p̂)², Hoeffding', type: 'line', x: GRID, y: GRID.map((q) => 2 * nv * (q - pv) ** 2), slot: 0 },
    ],
    [pv, nv],
  )
  const series: XYSeries[] = [
    ...divergence,
    { name: 'indices', type: 'scatter', x: [kl, Math.min(hf, 1.2)], y: [lv, lv], emphasis: true },
  ]

  const byCount = useMemo(
    (): XYSeries[] => [
      { name: 'KL-UCB index', type: 'line', x: COUNTS, y: COUNTS.map((c) => klUpper(pv, c, lv)), slot: 2 },
      {
        name: 'Hoeffding index',
        type: 'line',
        x: COUNTS,
        y: COUNTS.map((c) => Math.min(hoeffding(pv, c, lv), 2)),
        slot: 0,
      },
      { name: 'sample mean p̂', type: 'line', x: [1, 200], y: [pv, pv], muted: true },
    ],
    [pv, lv],
  )

  return (
    <Interactive
      title="Hoeffding and KL confidence bounds"
      caption="Left: an arm with sample mean p̂ after n pulls. A candidate mean q is kept in the confidence set while the divergence n kl(p̂, q) is below the level ℓ (the horizontal line; drag it). The KL-UCB index is where the green curve crosses the line. Hoeffding's interval uses the parabola 2n(q − p̂)², which lies below the KL curve everywhere (Pinsker's inequality), so it crosses later and gives a wider interval. Drag p̂ (the vertical line) towards 0 or 1: the KL curve steepens and the KL bound becomes much tighter, while the parabola does not change shape. Right: both indices as the number of pulls grows."
      controls={
        <>
          <ParamSlider label="sample mean p̂" param={p} />
          <ParamSlider label="pulls n" param={n} format={(v) => String(v)} />
          <ParamSlider label="level ℓ (ln t: ln 1000 ≈ 6.91)" param={level} />
        </>
      }
      readout={
        <>
          <Readout label="KL-UCB index" value={formatNumber(kl)} />
          <Readout label="Hoeffding index" value={formatNumber(hf)} />
          <Readout label="width ratio, Hoeffding / KL" value={formatNumber((hf - pv) / Math.max(kl - pv, 1e-9))} />
          <Readout label="level as rounds t = e^ℓ" value={formatNumber(Math.exp(lv))} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <XYChart
          series={series}
          xLabel="candidate mean q"
          yLabel="divergence"
          xRange={[0, 1]}
          yRange={[0, yTop]}
          handles={[
            { kind: 'x', at: pv, label: 'p̂', onDrag: (x) => p.set(x) },
            { kind: 'y', at: lv, label: 'level ℓ', onDrag: (y) => level.set(y) },
          ]}
        />
        <XYChart series={byCount} xLabel="pulls n" yLabel="upper bound" xRange={[1, 200]} yRange={[0, 1.2]} />
      </div>
    </Interactive>
  )
}
