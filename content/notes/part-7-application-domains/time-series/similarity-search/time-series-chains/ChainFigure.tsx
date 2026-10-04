import { useMemo } from 'react'
import { Figure, int, Plot, Readout, seriesLayers, type SeriesSpec, useAxis, useFigureState } from 'aifn-render'
import { longestChain, movingStats, selfJoin } from '../_shared/matrix-profile'
import { driftingPattern } from '../_shared/synthetic'

const N = 560
const M = 40

/**
 * A time-series chain on a pattern that drifts. Each occurrence's right nearest neighbour is the next occurrence, and
 * that one's left nearest neighbour points back, so following the confirmed links recovers the occurrences in order.
 */
export function ChainFigure() {
  const state = useFigureState({
    seed: int(1, { min: 1, max: 20, step: 1, label: 'seed', format: (v) => String(v) }),
  })

  const r = useMemo(() => {
    const { x, starts } = driftingPattern(N, state.seed)
    const p = selfJoin(x, M)
    const chain = longestChain(p.rightIndex, p.leftIndex)
    const { mean, sd } = movingStats(x, M)
    // Chain members z-normalised, as the matrix profile compares them.
    const members = chain.map((s) => Array.from({ length: M }, (_, k) => (x[s + k] - mean[s]) / sd[s]))
    return { x, starts, chain, members }
  }, [state.seed])

  const t = r.x.map((_, i) => i)
  const span = (s: number) => Array.from({ length: M }, (_, k) => s + k)
  const top: SeriesSpec[] = [
    { name: 'series', type: 'line', x: t, y: r.x, muted: true },
    ...r.chain.map((s, k) => ({
      name: 'chain member',
      type: 'line' as const,
      x: span(s),
      y: span(s).map((i) => r.x[i]),
      slot: k === 0 ? 0 : k === r.chain.length - 1 ? 2 : 1,
    })),
  ]
  const offsets = Array.from({ length: M }, (_, k) => k)
  const last = r.members.length - 1
  const overlay: SeriesSpec[] = r.members.map((y, k) => ({
    name: k === 0 ? 'first member' : k === last ? 'last member' : 'intermediate members',
    type: 'line' as const,
    x: offsets,
    y,
    ...(k === 0 ? { slot: 0 } : k === last ? { slot: 2 } : { muted: true }),
  }))

  const xAxis = useAxis({ label: 'time', hold: 'union' })
  const yAxis = useAxis({ label: 'x', hold: 'union' })
  const xAxis2 = useAxis({ label: 'offset within subsequence', hold: 'union' })
  const yAxis2 = useAxis({ label: 'z-normalised', hold: 'union' })
  return (
    <Figure
      title="A chain follows a drifting pattern"
      state={state}
      caption={`Top: smooth noise with one pattern planted ${r.starts.length} times. Each copy has a second bump that grows from nothing to the height of the first, so each copy resembles its neighbours in time while the first and last copies differ. The highlighted windows (m = ${M}) are the longest time-series chain: each member's nearest neighbour to the right is the next member, and that member's nearest neighbour to the left points back. Bottom: the chain members z-normalised and overlaid; the shape evolves from the first member to the last. Chain members may start a few samples before each planted copy, because every window includes the same stretch of background before it.`}

      readouts={
        <>
          <Readout label="chain length" value={r.chain.length} />
          <Readout label="chain starts" value={r.chain.join(', ')} />
          <Readout label="patterns planted at" value={r.starts.join(', ')} />
        </>
      }
    >
      <div className="space-y-4">
        <Plot x={xAxis} y={yAxis} height={200}>
          {seriesLayers(top)}
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={200}>
          {seriesLayers(overlay)}
        </Plot>
      </div>
    </Figure>
  )
}
