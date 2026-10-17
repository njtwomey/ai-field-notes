import { useMemo, useState } from 'react'
import { choice, Figure, Player, Plot, Plots, Readout, useAxis, useFigureState } from 'aifn-render'
import { weisfeilerLeman } from 'aifn-compute/graph/isomorphism'
import { drawingRange, graphLayers } from '../_shared/graph-layers'
import { pairById, PAIRS } from '../_shared/graph-pairs'

const histogramText = (h: readonly number[]) =>
  h
    .map((count, c) => (count > 0 ? `${count} × colour ${c + 1}` : null))
    .filter(Boolean)
    .join(', ')

/** Colour refinement on two graphs at once, with one shared palette, one round per step. */
export function ColourRefinement() {
  const state = useFigureState({
    pair: choice(
      PAIRS.map((p) => ({ value: p.id, label: p.label })),
      'path-star',
      { label: 'pair of graphs' },
    ),
  })
  const pair = pairById(state.pair)
  const trace = useMemo(() => weisfeilerLeman([pair.left.graph, pair.right.graph]), [pair])
  const [pick, setPick] = useState(0)
  const count = trace.stable + 1
  const round = Math.min(pick, count - 1)
  const colours = trace.colours[round]
  const classes = trace.classes[round]
  const names = Array.from({ length: classes }, (_, c) => `colour ${c + 1}`)
  const labels = (cs: Int32Array) => Array.from(cs, (c) => String(c + 1))
  const [hl, hr] = trace.histograms[round]
  const tied = hl.every((v, i) => v === hr[i])

  const range = drawingRange([pair.left, pair.right])
  const lx = useAxis({ range: range.x, nice: false, zoom: false })
  const ly = useAxis({ range: range.y, nice: false, zoom: false, equal: lx })
  const rx = useAxis({ range: range.x, nice: false, zoom: false })
  const ry = useAxis({ range: range.y, nice: false, zoom: false, equal: rx })
  const verdict =
    trace.distinguishedAt >= 0
      ? `histograms differ from round ${trace.distinguishedAt}: not isomorphic`
      : 'histograms equal at the stable round: the test is inconclusive'
  return (
    <Figure
      title="Colour refinement on two graphs"
      state={state}
      controls={
        <Player
          value={round}
          onChange={setPick}
          count={count}
          label="round"
          format={(t) => (t === trace.stable && t > 0 ? `${t} (stable)` : String(t))}
        />
      }
      readouts={
        <>
          <Readout label="round" value={round} />
          <Readout label="colours in use" value={classes} />
          <Readout label={`${pair.left.name}`} value={histogramText(hl)} />
          <Readout label={`${pair.right.name}`} value={histogramText(hr)} />
          <Readout label="this round" value={tied ? 'histograms equal' : 'histograms differ'} />
          <Readout label="verdict" value={verdict} />
        </>
      }
      caption={`${pair.note} Every node starts with colour 1. Each round gives a node a new colour for each distinct pair of its old colour and the multiset of its neighbours' old colours; the numbering is shared by both graphs, so equal numbers mean equal signatures. Step through the rounds with the player; the last round is the first one that splits no colour class.`}
    >
      <Plots cols={2}>
        <Plot x={lx} y={ly} bare title={pair.left.name} legend={false}>
          {graphLayers(pair.left, { classes: colours[0], classNames: names, labels: labels(colours[0]), id: 'left' })}
        </Plot>
        <Plot x={rx} y={ry} bare title={pair.right.name} legend={false}>
          {graphLayers(pair.right, { classes: colours[1], classNames: names, labels: labels(colours[1]), id: 'right' })}
        </Plot>
      </Plots>
    </Figure>
  )
}
