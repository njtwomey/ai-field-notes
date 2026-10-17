import { useMemo } from 'react'
import {
  choice,
  Curve,
  Figure,
  float,
  formatNumber,
  int,
  Plot,
  Readout,
  Segments,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { stream } from 'aifn-compute/foundation/random'
import { eye, ones, type Tensor } from 'aifn-compute/foundation/tensor'
import { auroc } from 'aifn-compute/learning/metrics'
import { pairHeuristics } from 'aifn-compute/graph/measures'
import { contextualBlockModel } from 'aifn-methods/data/synthetic'
import { linkPredictionRun, splitEdges, type LinkCheckpoint } from 'aifn-methods/neural/graph'
import {
  circleLayout,
  drawingRange,
  FixedHeight,
  graphLayers,
  useSlicedRun,
  type Drawing,
} from '../_shared/figure-helpers'

const PANEL = 280
const BLOCKS = 3
const SIZE = 20
const N = BLOCKS * SIZE
const SHOWN = 12

/** Three planted communities of 20 nodes; 15% of the edges held out, and as many non-edges. */
const DATA = (() => {
  const d = contextualBlockModel(stream('link/graph'), { sizes: [SIZE, SIZE, SIZE], pIn: 0.3, pOut: 0.02 })
  const split = splitEdges(stream('link/split'), d.graph, 0.15)
  const pairs = [...split.testPositive, ...split.testNegative]
  const labels = [...split.testPositive.map(() => 1), ...split.testNegative.map(() => 0)]
  const h = pairHeuristics(split.train, pairs)
  return {
    split,
    pairs,
    labels,
    commonNeighbours: auroc(labels, h.commonNeighbours),
    adamicAdar: auroc(labels, h.adamicAdar),
  }
})()

/** Each community on its own small circle, the three circles around a larger one. */
const DRAWING: Drawing = {
  pos: Array.from({ length: BLOCKS }, (_, b) => {
    const a = Math.PI / 2 - (2 * Math.PI * b) / BLOCKS
    return circleLayout(SIZE, 2.2 * Math.cos(a), 2.2 * Math.sin(a), 1)
  }).flat(),
  edges: DATA.split.train.edges.map((e) => [e.from, e.to] as const),
}

const FEATURES = [
  { value: 'identity', label: 'one-hot node identity' },
  { value: 'constant', label: 'the same feature on every node' },
] as const
type Features = (typeof FEATURES)[number]['value']
type Setup = { features: Features; steps: number; rate: number; seed: number }

function run(s: Setup): Generator<LinkCheckpoint> {
  const x: Tensor = s.features === 'identity' ? eye(N) : ones([N, 1])
  return linkPredictionRun({ split: DATA.split, x, steps: s.steps, stepSize: s.rate, every: 5, seed: s.seed })
}

/** A graph autoencoder predicting held-out edges of a three-community graph, against two neighbourhood heuristics. */
export function GraphAutoencoderRun() {
  const state = useFigureState({
    features: choice(FEATURES, 'identity', { label: 'node features' }),
    steps: int(200, { ge: 10, le: 3000, suggestions: [50, 200, 1000], label: 'Adam steps' }),
    rate: float(0.02, { gt: 0, le: 0.5, suggestions: [0.005, 0.02, 0.1], label: 'learning rate' }),
    seed: int(1, { ge: 1, le: 999, label: 'seed' }),
  })
  const { features, steps, rate, seed } = state
  const setup = useMemo<Setup>(() => ({ features, steps, rate, seed }), [features, steps, rate, seed])
  const result = useSlicedRun(setup, run)
  const cps = result.checkpoints
  const last = cps.at(-1)

  // The highest-scoring test pairs: held-out edges (hits) and non-edges (false alarms).
  const top = useMemo(() => {
    if (!last) return { hits: [], misses: [], correct: 0 }
    const order = last.scores.map((s, i) => [s, i] as const).sort((a, b) => b[0] - a[0])
    const chosen = order.slice(0, SHOWN).map(([, i]) => i)
    const seg = (i: number) => ({ from: DRAWING.pos[DATA.pairs[i][0]], to: DRAWING.pos[DATA.pairs[i][1]] })
    const hits = chosen.filter((i) => DATA.labels[i] === 1).map(seg)
    const misses = chosen.filter((i) => DATA.labels[i] === 0).map(seg)
    return { hits, misses, correct: hits.length }
  }, [last])

  const range = drawingRange([DRAWING], 0.3)
  const gx = useAxis({ range: range.x, nice: false, zoom: false })
  const gy = useAxis({ range: range.y, nice: false, zoom: false, equal: gx })
  const sx = useAxis({ label: 'Adam step', range: [0, steps], nice: false })
  const ay = useAxis({ label: 'test AUC', range: [0.4, 1], nice: false })

  return (
    <Figure
      title="A graph autoencoder predicts held-out edges"
      state={state}
      defaultSize="L"
      caption={`A stochastic block model with three communities of ${SIZE} nodes (edge probability 0.3 inside a community, 0.02 between). ${DATA.split.testPositive.length} edges (15%) are held out with as many non-edges; the encoder, two graph convolution layers of widths 16 and 8, passes messages over the remaining ${DATA.split.train.edges.length} edges only. The decoder scores a pair by the dot product of its two embeddings, and Adam minimises the binary cross-entropy of the training edges against a fresh draw of as many non-edges at every step. Right: the AUC on the held-out pairs during training, against the common-neighbour and Adamic–Adar heuristics computed on the training graph (dashed). Left: the training graph, with the ${SHOWN} highest-scoring held-out pairs at the end of training drawn as held-out edges found (solid) or non-edges predicted as edges (dashed). With the same feature on every node, the encoder can only tell nodes apart by their degrees and its embeddings collapse; one-hot identities give each node its own embedding.`}
      readouts={
        <>
          <Readout label="graph autoencoder AUC" value={last ? formatNumber(last.auc) : '…'} />
          <Readout label="common neighbours AUC" value={formatNumber(DATA.commonNeighbours)} />
          <Readout label="Adamic–Adar AUC" value={formatNumber(DATA.adamicAdar)} />
          <Readout label={`held-out edges among the top ${SHOWN}`} value={last ? `${top.correct} of ${SHOWN}` : '…'} />
          <Readout label="step" value={last ? `${last.step}${result.done ? '' : ', training…'}` : '…'} />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <FixedHeight height={PANEL}>
          <Plot x={gx} y={gy} bare title="training graph and top-scoring held-out pairs" legend>
            {graphLayers(DRAWING, {
              id: 'sbm',
              classes: Array.from({ length: N }, (_, v) => Math.floor(v / SIZE)),
              classNames: ['community 1', 'community 2', 'community 3'],
              size: 7,
              edgeWidth: 0.8,
            })}
            {top.hits.length > 0 && (
              <Segments id="hits" name="held-out edge found" segments={top.hits} width={2.5} tone="success" />
            )}
            {top.misses.length > 0 && (
              <Segments
                id="misses"
                name="non-edge predicted"
                segments={top.misses}
                width={2}
                dashed
                tone="destructive"
              />
            )}
          </Plot>
        </FixedHeight>
        <FixedHeight height={PANEL}>
          <Plot x={sx} y={ay} title="AUC on the held-out pairs">
            <Curve name="graph autoencoder" slot={0} x={cps.map((c) => c.step)} y={cps.map((c) => c.auc)} />
            <Curve
              name="common neighbours"
              slot={1}
              x={[0, steps]}
              y={[DATA.commonNeighbours, DATA.commonNeighbours]}
              dashed
            />
            <Curve name="Adamic–Adar" slot={2} x={[0, steps]} y={[DATA.adamicAdar, DATA.adamicAdar]} dashed />
          </Plot>
        </FixedHeight>
      </div>
    </Figure>
  )
}
