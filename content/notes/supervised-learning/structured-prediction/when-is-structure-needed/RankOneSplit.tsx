import { useMemo, useState } from 'react'
import { Interactive, ParamChoice, Readout, XYChart, formatNumber, type XYSeries } from 'aifn-render'
import { rng } from '@/lib/math'
import { forwardBackward, rankOne, type Mat, type Vec } from '../_shared/chain-crf'

const N = 10
const K = 3
/** The edge potential between positions 5 and 6 (0-based edge index 4) is the one made full rank or rank 1. */
const SPLIT = 4
const RESAMPLES = 40
const POSITIONS = Array.from({ length: N }, (_, i) => i + 1)

const nodes = (seed: number): Vec[] => {
  const g = rng(seed)
  return POSITIONS.map(() => Array.from({ length: K }, () => Math.exp(1.5 * g.normal())))
}
const edges = (seed: number): Mat[] => {
  const g = rng(seed)
  return Array.from({ length: N - 1 }, () =>
    Array.from({ length: K }, () => Array.from({ length: K }, () => Math.exp(1.5 * g.normal()))),
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
  const [rank, setRank] = useState<Rank>('one')
  const [label, setLabel] = useState<Label>('1')
  const y = Number(label) - 1

  const r = useMemo(() => {
    const baseNodes = nodes(11)
    const baseEdges = edges(12)
    if (rank === 'one') baseEdges[SPLIT] = rankOne(baseEdges[SPLIT])
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
  }, [rank, y])

  const series: XYSeries[] = r.traces.map((t) => ({
    name: 'marginal under one redraw',
    type: 'line',
    x: POSITIONS,
    y: t,
    muted: true,
  }))

  return (
    <Interactive
      title="A rank-1 edge splits the chain"
      caption="Each grey line is P(Yₙ = y | x) along a ten-position chain with three labels, for one of 40 redraws of every potential after position 5. With a full-rank edge potential between positions 5 and 6, the redraws change the marginals everywhere. With the rank-1 approximation of the same edge, the lines coincide at positions 1 to 5: the second half of the chain carries no information about the first."
      controls={
        <>
          <ParamChoice
            label="edge potential between positions 5 and 6"
            value={rank}
            onChange={setRank}
            options={[
              { value: 'full', label: 'full rank' },
              { value: 'one', label: 'rank 1' },
            ]}
          />
          <ParamChoice
            label="label y"
            value={label}
            onChange={setLabel}
            options={[
              { value: '1', label: '1' },
              { value: '2', label: '2' },
              { value: '3', label: '3' },
            ]}
          />
        </>
      }
      readout={
        <>
          <Readout label="largest spread, positions 1–5" value={formatNumber(r.before)} />
          <Readout label="largest spread, positions 6–10" value={formatNumber(r.after)} />
        </>
      }
    >
      <XYChart
        series={series}
        xLabel="position n"
        yLabel="P(Yₙ = y | x)"
        xRange={[1, 10]}
        yRange={[0, 1]}
        height={300}
      />
    </Interactive>
  )
}
