import {
  choice,
  Figure,
  formatNumber,
  Handle,
  int,
  Plot,
  Points,
  Readout,
  Segments,
  toggle,
  useAxis,
  useFigureState,
  Vectors,
  type Vector,
} from 'aifn-render'
import { add, matmul, tensor, toRows, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import { propagate } from 'aifn-compute/graph/propagation'
import type { Graph } from 'aifn-compute/graph'
import { rgcnTerms, type RgcnNormalisation } from 'aifn-compute/nn/graph'
import { FixedHeight } from '../_shared/FixedHeight'
import {
  AUTHORS,
  FEATURES,
  N,
  NAMES,
  PAPERS,
  paper,
  POSITIONS,
  RELATION_GRAPHS,
  RELATION_NAMES,
  RELATION_WEIGHTS,
  SELF_WEIGHT,
} from './bibliography'

const H = tensor(FEATURES)
const W0 = tensor(SELF_WEIGHT)
const rows = (v: Value) => toRows(v as Tensor) as number[][]
const fmt = (p: number[]) => `(${p.map((x) => formatNumber(Number(x.toFixed(3)))).join(', ')})`

/** The union of the relations as one homogeneous graph, for the single-weight comparison. */
const UNION: Graph = { kind: 'graph', nodes: N, edges: RELATION_GRAPHS.flatMap((g) => g.edges), directed: true }
// Single shared weight I: h_i + mean over all neighbours.
const HOMOGENEOUS = rows(add(H, propagate(UNION, H, { aggregate: 'mean' })))
const SELF_TERM = rows(matmul(H, W0))

const nearestPaper = ([x, y]: [number, number]) => {
  let best = 1
  let bestD = Infinity
  for (let p = 1; p <= PAPERS; p++) {
    const [px, py] = POSITIONS[paper(p)]
    const d = (px - x) ** 2 + (py - y) ** 2
    if (d < bestD) ((best = p), (bestD = d))
  }
  return best
}

/**
 * One relational graph convolution (`rgcnTerms`) into a chosen paper of a small bibliographic graph: the self term
 * and one term per relation, added tip to tail, against a single shared weight over all neighbours.
 */
export function RelationalMessages() {
  const state = useFigureState({
    target: int(1, { ge: 1, le: PAPERS, label: 'target paper' }),
    normalisation: choice<RgcnNormalisation>(
      [
        { value: 'relation', label: 'per relation, 1/|N_i^r|' },
        { value: 'node', label: 'across relations, 1/Σ_r |N_i^r|' },
      ],
      'relation',
      { label: 'normalisation' },
    ),
    writtenBy: toggle(true, { label: 'written by' }),
    cites: toggle(true, { label: 'cites' }),
    publishedIn: toggle(true, { label: 'published in' }),
  })
  const { target, normalisation } = state
  const active = [state.writtenBy, state.cites, state.publishedIn]
  const i = paper(target)

  // One term per relation (zero rows for the switched-off ones are not drawn): three sparse products on 20 nodes.
  const relations = RELATION_GRAPHS.map((graph, r) => ({ graph, weight: tensor(RELATION_WEIGHTS[r]) })).filter(
    (_, r) => active[r],
  )
  const computed = rgcnTerms(H, relations, normalisation).map((t) => rows(t))
  let k = 0
  const terms = active.map((on) => (on ? computed[k++] : null))

  // Tip-to-tail: the self term, then each active relation's term.
  const vectors: Vector[] = []
  let at: [number, number] = [0, 0]
  const push = (d: number[], slot: number | undefined, label: string) => {
    const to: [number, number] = [at[0] + d[0], at[1] + d[1]]
    vectors.push({ from: at, to, slot, label, labelAt: 'middle', width: 2 })
    at = to
  }
  push(SELF_TERM[i], undefined, 'self')
  terms.forEach((t, r) => {
    if (t && (t[i][0] !== 0 || t[i][1] !== 0)) push(t[i], r, RELATION_NAMES[r])
  })
  const pre = at
  const out = pre.map((x) => Math.max(0, x))

  // The graph panel: edges by relation, the target's incoming edges highlighted.
  const edgeSegments = (r: number, into: boolean) =>
    RELATION_GRAPHS[r].edges
      .filter((e) => (e.to === i) === into)
      .map((e) => ({ from: POSITIONS[e.from], to: POSITIONS[e.to] }))
  const gx = useAxis({ label: '', range: [-0.7, 5.7], nice: false })
  const gy = useAxis({ label: '', range: [-0.6, 7.6], nice: false })
  const vx = useAxis({ label: 'feature 1', range: [-0.5, 6], nice: false })
  const vy = useAxis({ label: 'feature 2', range: [-1, 4.5], nice: false, equal: vx })

  const names = POSITIONS.map((_, k) => NAMES[k])
  const typeShape = POSITIONS.map((_, k) => (k < AUTHORS ? 0 : k < AUTHORS + PAPERS ? 1 : 2))

  return (
    <Figure
      title="Relational messages into one paper"
      state={state}
      hoverReadout={false}
      caption={`Left: a bibliographic graph of 8 authors (circles), 8 papers (squares) and 4 venues (triangles), with three relations into papers: written by (author to paper), cites (cited paper to citing paper) and published in (venue to paper). Drag the marker to choose the paper that receives; its incoming edges are drawn solid. Right: the paper's new vector before the ReLU, built tip to tail from the self term W₀h and one term per relation, (1/c) Σ W_r h_j over that relation's neighbours. The weights are the worked example's: written by swaps the two features, cites halves them, published in keeps the first. The grey point is a homogeneous layer with one shared weight (the identity) averaging all neighbours together. Switching a relation off removes its arrow; with normalisation per relation, every relation weighs the same however many neighbours it has, while normalising across relations lets the relation with most neighbours dominate.`}
      readouts={
        <>
          {terms.map((t, r) => (
            <Readout key={RELATION_NAMES[r]} label={`${RELATION_NAMES[r]} term`} value={t ? fmt(t[i]) : 'off'} />
          ))}
          <Readout label="R-GCN output after ReLU" value={fmt(out)} />
          <Readout label="one shared weight" value={fmt(HOMOGENEOUS[i])} />
        </>
      }
    >
      <FixedHeight height={300}>
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          <Plot x={gx} y={gy} title={`receiving: P${target}`} bare>
            {RELATION_NAMES.map((name, r) => {
              const other = edgeSegments(r, false)
              return other.length > 0 ? (
                <Segments key={`bg-${name}`} name={name} slot={r} segments={other} width={1} dashed />
              ) : null
            })}
            {RELATION_NAMES.map((name, r) => {
              const incoming = edgeSegments(r, true)
              return active[r] && incoming.length > 0 ? (
                <Segments key={`in-${name}`} name={name} slot={r} segments={incoming} width={3} />
              ) : null
            })}
            <Points
              name="node"
              x={POSITIONS.map((p) => p[0])}
              y={POSITIONS.map((p) => p[1])}
              shape={typeShape}
              shapeNames={['author', 'paper', 'venue']}
              labels={names}
              emphasis
              size={12}
            />
            <Handle
              kind="point"
              at={POSITIONS[i]}
              label={`P${target}`}
              onDrag={(p) => state.set('target', nearestPaper(p))}
            />
          </Plot>
          <Plot x={vx} y={vy} title="pre-activation vector, tip to tail">
            <Vectors name="terms" vectors={vectors} />
            <Points name="R-GCN" x={[pre[0]]} y={[pre[1]]} emphasis size={10} />
            <Points name="one shared weight" x={[HOMOGENEOUS[i][0]]} y={[HOMOGENEOUS[i][1]]} muted size={10} />
          </Plot>
        </div>
      </FixedHeight>
    </Figure>
  )
}
