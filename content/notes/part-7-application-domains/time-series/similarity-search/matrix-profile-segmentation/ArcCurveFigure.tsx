import { useMemo } from 'react'
import { Curve, Figure, formatNumber, Handle, int, Plot, Points, Readout, useAxis, useFigureState } from 'aifn-render'
import { correctedArcCurve, selfJoin } from '../_shared/matrix-profile'
import { regimeChange } from '../_shared/synthetic'

const N = 600

/**
 * FLUSS on two regimes. Every subsequence points to its nearest neighbour; arcs rarely cross the boundary between
 * regimes, so the corrected arc curve (arc count over its expectation under no structure) dips there.
 */
export function ArcCurveFigure() {
  const state = useFigureState({
    change: int(300, { min: 120, max: 480, step: 1, label: 'change point', format: (v) => String(v) }),
    m: int(40, { min: 16, max: 80, step: 4, label: 'subsequence length m', format: (v) => String(v) }),
    seed: int(1, { min: 1, max: 20, step: 1, label: 'seed', format: (v) => String(v) }),
  })

  const r = useMemo(() => {
    const x = regimeChange(N, state.change, state.seed)
    const { index } = selfJoin(x, state.m)
    const cac = correctedArcCurve(index, state.m)
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
  }, [state.change, state.m, state.seed])

  const t = r.x.map((_, i) => i)
  const idx = r.arcs.map((_, i) => i)
  const top = [
    { name: 'regime A', x: t.slice(0, state.change), y: r.x.slice(0, state.change), slot: 0 },
    { name: 'regime B', x: t.slice(state.change), y: r.x.slice(state.change), slot: 1 },
  ] as const
  const middle = [
    { name: 'arc count AC', x: idx, y: r.arcs, slot: 0 },
    { name: 'idealised arc curve 2i(n − i)/n', x: idx, y: r.ideal, dashed: true, muted: true },
  ] as const
  const bottom = [
    { name: 'corrected arc curve CAC', x: idx, y: Array.from(r.cac), slot: 2 },
    { name: 'minimum', x: [r.best], y: [r.cac[r.best]], emphasis: true },
  ] as const

  const xAxis = useAxis({ label: 'time', hold: 'union' })
  const yAxis = useAxis({ label: 'x', hold: 'union' })
  const xAxis2 = useAxis({ label: 'position i', hold: 'union' })
  const yAxis2 = useAxis({ label: 'arcs over i', range: [0, undefined], hold: 'union' })
  const xAxis3 = useAxis({ label: 'position i', hold: 'union' })
  const yAxis3 = useAxis({ label: 'CAC', range: [0, 1] })
  return (
    <Figure
      title="The arc curve dips at a regime change"
      state={state}
      caption="Top: normal heartbeat-like beats, then beats of a different shape from the change point on (drag it). Every length-m subsequence draws an arc to its nearest neighbour in the matrix profile index. Middle: the number of arcs passing over each position, against the parabola expected when neighbours are placed at random. Bottom: the corrected arc curve, the ratio of the two capped at 1, with the first and last m positions set to 1. Subsequences within one regime point inside that regime, so almost no arc crosses the boundary and the curve falls to about 0 there. The minimum lands within one subsequence length before the change, at the subsequences that straddle it."

      readouts={
        <>
          <Readout label="true change" value={state.change} />
          <Readout label="CAC minimum at" value={r.best} />
          <Readout label="CAC there" value={formatNumber(r.cac[r.best])} />
        </>
      }
    >
      <div className="space-y-4">
        <Plot x={xAxis} y={yAxis} height={180}>
          <Curve {...top[0]} />
          <Curve {...top[1]} />
          <Handle {...state.handle('change', { label: 'change' })} />
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={180}>
          <Curve {...middle[0]} />
          <Curve {...middle[1]} />
        </Plot>
        <Plot x={xAxis3} y={yAxis3} height={180}>
          <Curve {...bottom[0]} />
          <Points {...bottom[1]} />
        </Plot>
      </div>
    </Figure>
  )
}
