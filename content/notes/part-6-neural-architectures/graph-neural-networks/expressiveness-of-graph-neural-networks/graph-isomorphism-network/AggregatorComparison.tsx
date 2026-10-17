import { useMemo, useState } from 'react'
import { stream } from 'aifn-compute/foundation/random'
import { tensor, toRows, type Value } from 'aifn-compute/foundation/tensor'
import { fromEdges } from 'aifn-compute/graph'
import { Mlp } from 'aifn-compute/nn/layers'
import { choice, Figure, formatNumber, Player, Plot, Plots, Readout, useAxis, useFigureState } from 'aifn-render'
import { weisfeilerLeman } from 'aifn-compute/graph/isomorphism'
import { ginConv } from 'aifn-compute/nn/graph'
import { drawingRange, graphLayers } from '../_shared/graph-layers'
import type { Drawn } from '../_shared/graph-pairs'

type Side = Drawn & { features: number[]; focus: number }
type Case = { id: string; label: string; left: Side; right: Side; note: string }

function side(name: string, pos: [number, number][], edges: [number, number][], features: number[], focus: number) {
  return { name, graph: fromEdges(pos.length, edges, { directed: false }), pos, edges, features, focus }
}

/** A centre at the origin with leaves on a circle, carrying the given features. */
function star(name: string, leaves: number[], centreFeature = 1): Side {
  const k = leaves.length
  const pos: [number, number][] = [[0, 0]]
  for (let i = 0; i < k; i++) {
    const a = Math.PI / 2 + (2 * Math.PI * i) / k
    pos.push([1.2 * Math.cos(a), 1.2 * Math.sin(a)])
  }
  return side(
    name,
    pos,
    leaves.map((_, i) => [0, i + 1]),
    [centreFeature, ...leaves],
    0,
  )
}

const ring = (n: number, cx: number, r: number): [number, number][] =>
  Array.from({ length: n }, (_, i) => [
    cx + r * Math.cos(Math.PI / 2 + (2 * Math.PI * i) / n),
    r * Math.sin(Math.PI / 2 + (2 * Math.PI * i) / n),
  ])

const CASES: Case[] = [
  {
    id: 'path-star',
    label: 'path and star, equal features',
    left: side(
      'path',
      [
        [-1.5, 0],
        [-0.5, 0],
        [0.5, 0],
        [1.5, 0],
      ],
      [
        [0, 1],
        [1, 2],
        [2, 3],
      ],
      [1, 1, 1, 1],
      1,
    ),
    right: star('star', [1, 1, 1]),
    note: 'Every node carries feature 1. The marked nodes have two and three neighbours with identical vectors.',
  },
  {
    id: 'repeat',
    label: 'neighbours {2, 3} and {2, 3, 3}',
    left: star('two leaves', [2, 3]),
    right: star('three leaves', [2, 3, 3]),
    note: 'The marked centres see the same set of neighbour features, once with feature 3 repeated.',
  },
  {
    id: 'proportions',
    label: 'neighbours {2, 3} and {2, 2, 3, 3}',
    left: star('two leaves', [2, 3]),
    right: star('four leaves', [2, 2, 3, 3]),
    note: 'The marked centres see the same proportions of neighbour features, the second with every feature doubled.',
  },
  {
    id: 'hexagon-triangles',
    label: '6-cycle and two triangles',
    left: side(
      '6-cycle',
      ring(6, 0, 1),
      Array.from({ length: 6 }, (_, i) => [i, (i + 1) % 6] as [number, number]),
      [1, 1, 1, 1, 1, 1],
      0,
    ),
    right: side(
      'two triangles',
      [...ring(3, -1.1, 0.8), ...ring(3, 1.1, 0.8)],
      [
        [0, 1],
        [1, 2],
        [2, 0],
        [3, 4],
        [4, 5],
        [5, 3],
      ],
      [1, 1, 1, 1, 1, 1],
      0,
    ),
    note: 'Both graphs are 2-regular with equal features, so 1-WL ties and so does every aggregator.',
  },
]

const LAYERS = 3
const WIDTH = 8
type Aggregate = 'sum' | 'mean' | 'max'

/** Vectors equal to 1e-9 share a key. */
const keyOf = (row: number[]) => row.map((v) => (Math.abs(v) < 1e-12 ? 0 : v).toPrecision(9)).join(',')

/** Fixed random tanh MLPs, one per layer, shared by both graphs (as one trained network would be). */
function mlps(seed: number) {
  const s = stream(seed)
  return Array.from({ length: LAYERS }, (_, l) => {
    const mlp = Mlp([l === 0 ? 1 : WIDTH, 16, WIDTH], { activation: 'tanh', outputActivation: 'tanh' })
    const params = mlp.init(s)
    return (x: Value) => mlp.apply(params, x)
  })
}

/** Node vectors of both graphs after each layer, and their shared equality classes. */
function run(c: Case, aggregate: Aggregate) {
  const nets = mlps(7)
  const layers = [c.left, c.right].map((g) => {
    let h: Value = tensor(g.features.map((f) => [f]))
    const out: number[][][] = [toRows(h as never)]
    for (let l = 0; l < LAYERS; l++) {
      h = ginConv(g.graph, h, nets[l], { aggregate })
      out.push(toRows(h as never))
    }
    return out
  })
  // Classes per layer, numbered by sorted vector keys so that both graphs share the palette.
  const classes = Array.from({ length: LAYERS + 1 }, (_, l) => {
    const keys = layers.map((g) => g[l].map(keyOf))
    const distinct = [...new Set(keys.flat())].sort()
    const index = new Map(distinct.map((k, i) => [k, i]))
    return { ids: keys.map((ks) => ks.map((k) => index.get(k)!)), count: distinct.length }
  })
  // Graph readout up to layer l: the sums of node vectors of layers 0..l, concatenated.
  const readout = (g: number, l: number) =>
    layers[g].slice(0, l + 1).flatMap((rows) => rows[0].map((_, j) => rows.reduce((s, r) => s + r[j], 0)))
  const distance = (l: number) => {
    const a = readout(0, l)
    const b = readout(1, l)
    return Math.sqrt(a.reduce((s, v, j) => s + (v - b[j]) ** 2, 0))
  }
  return { layers, classes, distance }
}

/** GIN layers with sum, mean or max aggregation on pairs of graphs, one layer per step. */
export function AggregatorComparison() {
  const state = useFigureState({
    pair: choice(
      CASES.map((c) => ({ value: c.id, label: c.label })),
      'path-star',
      { label: 'pair' },
    ),
    aggregate: choice<Aggregate>(
      [
        { value: 'sum', label: 'sum (GIN)' },
        { value: 'mean', label: 'mean' },
        { value: 'max', label: 'max' },
      ],
      'sum',
      { label: 'aggregator' },
    ),
  })
  const c = CASES.find((x) => x.id === state.pair) ?? CASES[0]
  const result = useMemo(() => run(c, state.aggregate), [c, state.aggregate])
  const all = useMemo(
    () => (['sum', 'mean', 'max'] as const).map((a) => ({ a, r: a === state.aggregate ? result : run(c, a) })),
    [c, state.aggregate, result],
  )
  const wl = useMemo(
    () =>
      weisfeilerLeman([c.left.graph, c.right.graph], {
        initial: [c.left.features, c.right.features],
      }),
    [c],
  )
  const [pick, setPick] = useState(0)
  const layer = Math.min(pick, LAYERS)
  const ids = result.classes[layer].ids
  const names = Array.from({ length: result.classes[layer].count }, (_, k) => `vector ${k + 1}`)
  const focusEqual = ids[0][c.left.focus] === ids[1][c.right.focus]
  const firstSplit = (r: ReturnType<typeof run>) => {
    const l = r.classes.findIndex((cl) => cl.ids[0][c.left.focus] !== cl.ids[1][c.right.focus])
    return l < 0 ? 'never' : `layer ${l}`
  }
  const wlFocus = wl.colours.findIndex((cs) => cs[0][c.left.focus] !== cs[1][c.right.focus])

  const range = drawingRange([c.left, c.right], 0.5)
  const lx = useAxis({ range: range.x, nice: false, zoom: false })
  const ly = useAxis({ range: range.y, nice: false, zoom: false, equal: lx })
  const rx = useAxis({ range: range.x, nice: false, zoom: false })
  const ry = useAxis({ range: range.y, nice: false, zoom: false, equal: rx })
  const labels = (g: Side, k: number) => ids[k].map((id, v) => (layer === 0 ? `x = ${g.features[v]}` : String(id + 1)))
  return (
    <Figure
      title="Sum, mean and max aggregation in GIN layers"
      state={state}
      controls={<Player value={layer} onChange={setPick} count={LAYERS + 1} label="layer" />}
      readouts={{
        [`after layer ${layer}`]: (
          <>
            <Readout label="distinct node vectors" value={result.classes[layer].count} />
            <Readout label="marked nodes" value={focusEqual ? 'equal vectors' : 'different vectors'} />
            <Readout label="graph readout distance" value={formatNumber(result.distance(layer))} />
          </>
        ),
        'marked nodes first differ': (
          <>
            {all.map(({ a, r }) => (
              <Readout key={a} label={a} value={firstSplit(r)} />
            ))}
            <Readout label="1-WL colours" value={wlFocus < 0 ? 'never' : `round ${wlFocus}`} />
          </>
        ),
      }}
      caption={`${c.note} Each layer computes MLP(h_v + aggregate of the neighbours' h_u) with ε = 0 and fixed random tanh MLPs (width ${WIDTH}), the same weights for both graphs. Nodes with equal vectors share a colour and a number in both panels; the ringed nodes are the ones compared. Step through the layers, and switch the aggregator: sum separates whatever 1-WL separates, mean only multisets with different proportions, max only multisets with different sets of values. The graph readout sums each layer's node vectors and concatenates the layers.`}
    >
      <Plots cols={2}>
        <Plot x={lx} y={ly} bare title={c.left.name} legend={false}>
          {graphLayers(c.left, {
            classes: ids[0],
            classNames: names,
            labels: labels(c.left, 0),
            marked: [c.left.focus],
            id: 'left',
          })}
        </Plot>
        <Plot x={rx} y={ry} bare title={c.right.name} legend={false}>
          {graphLayers(c.right, {
            classes: ids[1],
            classNames: names,
            labels: labels(c.right, 1),
            marked: [c.right.focus],
            id: 'right',
          })}
        </Plot>
      </Plots>
    </Figure>
  )
}
