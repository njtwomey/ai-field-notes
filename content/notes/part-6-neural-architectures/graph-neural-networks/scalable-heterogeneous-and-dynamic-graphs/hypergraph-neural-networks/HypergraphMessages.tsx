import { useMemo, useState } from 'react'
import {
  choice,
  Figure,
  formatNumber,
  int,
  Player,
  Plot,
  Points,
  Readout,
  Segments,
  useAxis,
  useFigureState,
  useScaleColor,
} from 'aifn-render'
import { tensor, toFlat, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import { propagate } from 'aifn-compute/graph/propagation'
import { cliqueExpansion } from 'aifn-compute/graph/structures'
import { hypergraphMeanPass } from 'aifn-compute/nn/graph'
import { AUTHORS, EDGE_NAMES, edgePosition, FEATURES, HYPEREDGES, POSITIONS, type Structure } from './coauthors'

type View = 'hypergraph' | 'clique'
const X = tensor(FEATURES.map((f) => [f]))
const flat = (v: Value) => toFlat(v as Tensor)
const fmt = (v: number) => formatNumber(Number(v.toFixed(2)))

/**
 * Two-stage message passing on a small co-authorship hypergraph (`hypergraphMeanPass`), node to hyperedge then
 * hyperedge to node, against one round of mean aggregation on its clique expansion (`cliqueExpansion`).
 */
export function HypergraphMessages() {
  const state = useFigureState({
    structure: choice<Structure>(
      [
        { value: 'papers', label: 'three papers: {1,2,3}, {3,4}, {4,5,6}' },
        { value: 'split', label: 'first paper split into {1,2}, {2,3}, {1,3}' },
      ],
      'papers',
      { label: 'hypergraph' },
    ),
    view: choice<View>(
      [
        { value: 'hypergraph', label: 'two-stage (hypergraph)' },
        { value: 'clique', label: 'clique expansion' },
      ],
      'hypergraph',
      { label: 'message passing on' },
    ),
    author: int(3, { ge: 1, le: AUTHORS, label: 'inspect author' }),
  })
  const [step, setStep] = useState(0)
  const edges = HYPEREDGES[state.structure]
  const names = EDGE_NAMES[state.structure]

  const results = useMemo(() => {
    const pass = hypergraphMeanPass(X, edges)
    const clique = cliqueExpansion(AUTHORS, edges)
    return {
      edgeValues: flat(pass.edges),
      twoStage: flat(pass.nodes),
      cliqueGraph: clique,
      clique: flat(propagate(clique, X, { aggregate: 'mean', selfLoops: true })),
    }
  }, [edges])

  const hyper = state.view === 'hypergraph'
  const shownNodes = step < 2 ? FEATURES : hyper ? results.twoStage : results.clique
  const colour = useScaleColor('sequential')
  const nodeColours = shownNodes.map((v) => colour((v - 1) / 5))
  const v = state.author - 1

  const edgeAt = edges.map(edgePosition)
  const starSegments = edges.flatMap((e, k) => e.map((u) => ({ from: POSITIONS[u], to: edgeAt[k] })))
  const cliqueSegments = results.cliqueGraph.edges.map((e) => ({ from: POSITIONS[e.from], to: POSITIONS[e.to] }))
  const touching = edges.map((e) => e.includes(v))

  const xAxis = useAxis({ label: '', range: [-0.8, 5.6], nice: false })
  const yAxis = useAxis({ label: '', range: [-1.6, 1.8], nice: false, equal: xAxis })
  const stageText = hyper
    ? [
        'features',
        'node → hyperedge: each paper averages its authors',
        'hyperedge → node: each author averages its papers',
      ][step]
    : ['features', 'messages along the clique edges', 'each author averages itself and its clique neighbours'][step]

  return (
    <Figure
      title="Hyperedges against their clique expansion"
      state={state}
      hoverReadout={false}
      caption={`Six authors with features 1 to 6 (circles, labelled with their current value). In the hypergraph view each paper is a square joined to its authors, the star expansion: stage 1 averages each paper's authors into the square, stage 2 averages each author's papers back. In the clique view every two co-authors are joined by an edge and one round averages each author with its neighbours. Step with the player; ${stageText}. The papers of the inspected author carry a star. Switching to the split hypergraph replaces the paper {1,2,3} by three two-author papers: the clique expansion does not change, so neither does its output, while the two-stage output of authors 1 and 3 does.`}
      controls={
        <Player
          value={step}
          onChange={setStep}
          count={3}
          label="stage"
          format={(s) => ['start', 'stage 1', 'stage 2'][s]}
        />
      }
      readouts={
        <>
          <Readout label={`author ${state.author}: two-stage`} value={fmt(results.twoStage[v])} />
          <Readout label={`author ${state.author}: clique expansion`} value={fmt(results.clique[v])} />
          <Readout
            label="incidences / clique edges"
            value={`${edges.reduce((s, e) => s + e.length, 0)} / ${results.cliqueGraph.edges.length}`}
          />
        </>
      }
    >
      <div className="mx-auto w-full max-w-xl">
        <Plot x={xAxis} y={yAxis} height={300} bare>
          {hyper ? (
            <Segments name="membership" segments={starSegments} muted width={1.5} />
          ) : (
            <Segments name="clique edge" segments={cliqueSegments} muted width={1.5} />
          )}
          {hyper && (
            <Points
              name="paper (hyperedge)"
              x={edgeAt.map((p) => p[0])}
              y={edgeAt.map((p) => p[1])}
              shape={1}
              slot={0}
              size={step === 1 ? 22 : 14}
              labels={names.map((n, k) =>
                step >= 1 ? `${n}: ${fmt(results.edgeValues[k])}${touching[k] ? ' *' : ''}` : n,
              )}
            />
          )}
          <Points name={`author ${state.author}`} x={[POSITIONS[v][0]]} y={[POSITIONS[v][1]]} emphasis size={34} />
          <Points
            name="author"
            x={POSITIONS.map((p) => p[0])}
            y={POSITIONS.map((p) => p[1])}
            colors={nodeColours}
            size={24}
            labels={shownNodes.map((x, i) => `${i + 1}: ${fmt(x)}`)}
          />
        </Plot>
      </div>
    </Figure>
  )
}
