import { useMemo } from 'react'
import { toRows } from 'aifn-compute/foundation/tensor'
import { choice, Curve, Figure, int, Plot, Plots, Readout, useAxis, useFigureState } from 'aifn-render'
import { randomWalkEncoding } from 'aifn-compute/graph/matrices'
import { drawingRange, graphLayers } from '../_shared/graph-layers'
import { pairById, PAIRS } from '../_shared/graph-pairs'

/** Return probabilities of a random walk as node encodings, on two graphs that 1-WL may tie. */
export function ReturnProbabilities() {
  const state = useFigureState({
    pair: choice(
      PAIRS.map((p) => ({ value: p.id, label: p.label })),
      'hexagon-triangles',
      { label: 'pair of graphs' },
    ),
    steps: int(6, { ge: 1, le: 16, suggestions: [3, 6, 12, 16], label: 'walk lengths K' }),
  })
  const pair = pairById(state.pair)
  const r = useMemo(() => {
    const enc = [pair.left.graph, pair.right.graph].map((g) => toRows(randomWalkEncoding(g, state.steps)))
    // Nodes with equal encodings (to 1e-9) form one class, numbered over both graphs.
    const key = (row: number[]) => row.map((v) => v.toFixed(9)).join(',')
    const distinct = [...new Set(enc.flat().map(key))].sort()
    const ids = enc.map((rows) => rows.map((row) => distinct.indexOf(key(row))))
    // The first walk length at which the two graphs' multisets of encodings differ.
    let differ = -1
    for (let k = 0; k < state.steps && differ < 0; k++) {
      const sa = enc[0].map((row) => row[k].toFixed(9)).sort()
      const sb = enc[1].map((row) => row[k].toFixed(9)).sort()
      if (sa.length !== sb.length || sa.some((v, i) => v !== sb[i])) differ = k + 1
    }
    return { enc, ids, count: distinct.length, differ }
  }, [pair, state.steps])

  const range = drawingRange([pair.left, pair.right])
  const lx = useAxis({ range: range.x, nice: false, zoom: false })
  const ly = useAxis({ range: range.y, nice: false, zoom: false, equal: lx })
  const rx = useAxis({ range: range.x, nice: false, zoom: false })
  const ry = useAxis({ range: range.y, nice: false, zoom: false, equal: rx })
  const kx = useAxis({ label: 'walk length k', range: [0.5, state.steps + 0.5], integer: true })
  const py = useAxis({ label: 'return probability', range: [0, 1] })
  const names = Array.from({ length: r.count }, (_, c) => `encoding ${c + 1}`)
  const steps = Array.from({ length: state.steps }, (_, k) => k + 1)
  const curves = (g: 0 | 1) =>
    [...new Set(r.ids[g])]
      .sort((a, b) => a - b)
      .map((c) => {
        const v = r.ids[g].indexOf(c)
        const n = r.ids[g].filter((x) => x === c).length
        return (
          <Curve
            key={`${g}-${c}`}
            name={`encoding ${c + 1} (${n} node${n > 1 ? 's' : ''})`}
            x={steps}
            y={r.enc[g][v]}
            slot={c}
            showPoints
          />
        )
      })
  return (
    <Figure
      title="Random-walk return probabilities as node encodings"
      state={state}
      readouts={
        <>
          <Readout label="distinct encodings, both graphs" value={r.count} />
          <Readout
            label="graphs first differ at walk length"
            value={r.differ > 0 ? r.differ : `not within K = ${state.steps}`}
          />
        </>
      }
      caption={`${pair.note} Each node's encoding is the probability that a random walk started there is back after k = 1, …, K steps, the diagonal of (D⁻¹A)ᵏ. Nodes with equal encodings share a colour and a number in both graphs, and each curve below shows one encoding. Odd walk lengths return only through odd cycles: the triangles show up at k = 3 and the five-membered rings of bicyclopentyl at k = 5. The two strongly regular graphs give every node the same curve.`}
    >
      <Plots rows={2} cols={2} heights={[3, 2]}>
        <Plot x={lx} y={ly} bare title={pair.left.name} legend={false}>
          {graphLayers(pair.left, {
            classes: r.ids[0],
            classNames: names,
            labels: r.ids[0].map((c) => String(c + 1)),
            id: 'left',
          })}
        </Plot>
        <Plot x={rx} y={ry} bare title={pair.right.name} legend={false}>
          {graphLayers(pair.right, {
            classes: r.ids[1],
            classNames: names,
            labels: r.ids[1].map((c) => String(c + 1)),
            id: 'right',
          })}
        </Plot>
        <Plot x={kx} y={py}>
          {curves(0)}
        </Plot>
        <Plot x={kx} y={py}>
          {curves(1)}
        </Plot>
      </Plots>
    </Figure>
  )
}
