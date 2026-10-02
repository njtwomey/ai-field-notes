import { useMemo } from 'react'
import {
  Interactive,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type XYSeries,
} from 'aifn-render'
import { correctedArcCurve, selfJoin } from '../_shared/matrix-profile'
import { regimeChange } from '../_shared/synthetic'

const N = 600

/**
 * FLUSS on two regimes. Every subsequence points to its nearest neighbour; arcs rarely cross the boundary between
 * regimes, so the corrected arc curve (arc count over its expectation under no structure) dips there.
 */
export function ArcCurveFigure() {
  const change = useParam(300, { min: 120, max: 480, step: 1 })
  const m = useParam(40, { min: 16, max: 80, step: 4 })
  const seed = useParam(1, { min: 1, max: 20, step: 1 })

  const r = useMemo(() => {
    const x = regimeChange(N, change.value, seed.value)
    const { index } = selfJoin(x, m.value)
    const cac = correctedArcCurve(index, m.value)
    let best = 0
    cac.forEach((v, i) => {
      if (v < cac[best]) best = i
    })
    // Raw arc counts and the idealised parabola 2i(n − i)/n, for the middle panel.
    const n = index.length
    const marks = new Float64Array(n + 1)
    index.forEach((j, i) => {
      marks[Math.min(i, j) + 1] += 1
      marks[Math.max(i, j)] -= 1
    })
    const arcs: number[] = []
    let running = 0
    for (let i = 0; i < n; i++) arcs.push((running += marks[i]))
    const ideal = arcs.map((_, i) => (2 * i * (n - i)) / n)
    return { x, cac, best, arcs, ideal }
  }, [change.value, m.value, seed.value])

  const t = r.x.map((_, i) => i)
  const idx = r.arcs.map((_, i) => i)
  const top: XYSeries[] = [
    { name: 'regime A', type: 'line', x: t.slice(0, change.value), y: r.x.slice(0, change.value), slot: 0 },
    { name: 'regime B', type: 'line', x: t.slice(change.value), y: r.x.slice(change.value), slot: 1 },
  ]
  const middle: XYSeries[] = [
    { name: 'arc count AC', type: 'line', x: idx, y: r.arcs, slot: 0 },
    { name: 'idealised arc curve 2i(n − i)/n', type: 'line', x: idx, y: r.ideal, dashed: true, muted: true },
  ]
  const bottom: XYSeries[] = [
    { name: 'corrected arc curve CAC', type: 'line', x: idx, y: Array.from(r.cac), slot: 2 },
    { name: 'minimum', type: 'scatter', x: [r.best], y: [r.cac[r.best]], emphasis: true },
  ]
  const handles: Handle[] = [{ kind: 'x', at: change.value, label: 'change', onDrag: change.set }]

  return (
    <Interactive
      title="The arc curve dips at a regime change"
      caption="Top: normal heartbeat-like beats, then beats of a different shape from the change point on (drag it). Every length-m subsequence draws an arc to its nearest neighbour in the matrix profile index. Middle: the number of arcs passing over each position, against the parabola expected when neighbours are placed at random. Bottom: the corrected arc curve, the ratio of the two capped at 1, with the first and last m positions set to 1. Subsequences within one regime point inside that regime, so almost no arc crosses the boundary and the curve falls to about 0 there. The minimum lands within one subsequence length before the change, at the subsequences that straddle it."
      controls={
        <>
          <ParamSlider label="change point" param={change} format={(v) => String(v)} withArrows />
          <ParamSlider label="subsequence length m" param={m} format={(v) => String(v)} withArrows />
          <ParamSlider label="seed" param={seed} format={(v) => String(v)} withArrows />
        </>
      }
      readout={
        <>
          <Readout label="true change" value={change.value} />
          <Readout label="CAC minimum at" value={r.best} />
          <Readout label="CAC there" value={formatNumber(r.cac[r.best])} />
        </>
      }
    >
      <div className="space-y-4">
        <XYChart series={top} xLabel="time" yLabel="x" height={180} handles={handles} />
        <XYChart series={middle} xLabel="position i" yLabel="arcs over i" height={180} yRange={[0, undefined]} />
        <XYChart series={bottom} xLabel="position i" yLabel="CAC" height={180} yRange={[0, 1]} />
      </div>
    </Interactive>
  )
}
