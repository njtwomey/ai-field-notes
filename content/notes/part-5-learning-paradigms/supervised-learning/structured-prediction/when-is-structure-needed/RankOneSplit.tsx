import { useMemo } from 'react'
import {
  choice,
  Figure,
  formatNumber,
  Plot,
  Readout,
  seriesLayers,
  type SeriesSpec,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { forwardBackward, rankOne, type Mat, type Vec } from '../_shared/chain-crf'
import { normal, stream } from 'aifn/foundation/random'

const N = 10
const K = 3
/** The edge potential between positions 5 and 6 (0-based edge index 4) is the one made full rank or rank 1. */
const SPLIT = 4
const RESAMPLES = 40
const POSITIONS = Array.from({ length: N }, (_, i) => i + 1)

const nodes = (seed: number): Vec[] => {
  const g = stream(seed)
  return POSITIONS.map(() => Array.from({ length: K }, () => Math.exp(1.5 * normal(g))))
}
const edges = (seed: number): Mat[] => {
  const g = stream(seed)
  return Array.from({ length: N - 1 }, () =>
    Array.from({ length: K }, () => Array.from({ length: K }, () => Math.exp(1.5 * normal(g)))),
  )
}

type Rank = 'full' | 'one'
type Label = '1' | '2' | '3'

/**
 * The paper's experiment: marginals along a ten-position chain while every potential after position 5 is redrawn.
 * With a full-rank edge between positions 5 and 6, the redraws move every marginal; with a rank-1 edge, positions 1–5
 * do not move at all.
 */
export function RankOneSplit() {
  const state = useFigureState({
    rank: choice<Rank>(
      [
        { value: 'full', label: 'full rank' },
        { value: 'one', label: 'rank 1' },
      ],
      'one',
      { label: 'edge potential between positions 5 and 6' },
    ),
    label: choice<Label>(
      [
        { value: '1', label: '1' },
        { value: '2', label: '2' },
        { value: '3', label: '3' },
      ],
      '1',
      { label: 'label y' },
    ),
  })
  const y = Number(state.label) - 1

  const r = useMemo(() => {
    const baseNodes = nodes(11)
    const baseEdges = edges(12)
    if (state.rank === 'one') baseEdges[SPLIT] = rankOne(baseEdges[SPLIT])
    const traces = Array.from({ length: RESAMPLES }, (_, s) => {
      // Redraw every node potential after position 5 and every edge potential after the split.
      const redrawnNodes = nodes(1000 + s)
      const redrawnEdges = edges(2000 + s)
      const psi = baseNodes.map((p, n) => (n > SPLIT ? redrawnNodes[n] : p))
      const edge = baseEdges.map((e, n) => (n > SPLIT ? redrawnEdges[n] : e))
      return forwardBackward(psi, edge).marginals.map((m) => m[y])
    })
    const spread = (from: number, to: number) =>
      Math.max(
        ...POSITIONS.slice(from, to).map((_, i) => {
          const values = traces.map((t) => t[from + i])
          return Math.max(...values) - Math.min(...values)
        }),
      )
    return { traces, before: spread(0, SPLIT + 1), after: spread(SPLIT + 1, N) }
  }, [state.rank, y])

  const series: SeriesSpec[] = r.traces.map((t) => ({
    name: 'marginal under one redraw',
    type: 'line',
    x: POSITIONS,
    y: t,
    muted: true,
  }))

  const xAxis = useAxis({ label: 'position n', range: [1, 10] })
  const yAxis = useAxis({ label: 'P(Yₙ = y | x)', range: [0, 1] })
  return (
    <Figure
      title="A rank-1 edge splits the chain"
      state={state}
      caption="Each grey line is P(Yₙ = y | x) along a ten-position chain with three labels, for one of 40 redraws of every potential after position 5. With a full-rank edge potential between positions 5 and 6, the redraws change the marginals everywhere. With the rank-1 approximation of the same edge, the lines coincide at positions 1 to 5: the second half of the chain carries no information about the first."

      readouts={
        <>
          <Readout label="largest spread, positions 1–5" value={formatNumber(r.before)} />
          <Readout label="largest spread, positions 6–10" value={formatNumber(r.after)} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={300}>
        {seriesLayers(series)}
      </Plot>
    </Figure>
  )
}
