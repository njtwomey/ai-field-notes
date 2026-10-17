import { useMemo } from 'react'
import { stream, uniform } from 'aifn-compute/foundation/random'
import { tanh, tensor, toRows, type Value } from 'aifn-compute/foundation/tensor'
import { adjacency, type Graph } from 'aifn-compute/graph'
import { GraphConv } from 'aifn-compute/nn/graph'
import { Mlp } from 'aifn-compute/nn/layers'
import { choice, Figure, formatNumber, int, Plot, Plots, Readout, useAxis, useFigureState, when } from 'aifn-render'
import { weisfeilerLeman } from 'aifn-compute/graph/isomorphism'
import { randomWalkEncoding } from 'aifn-compute/graph/matrices'
import { ginConv } from 'aifn-compute/nn/graph'
import { drawingRange, graphLayers } from '../_shared/graph-layers'
import { pairById, PAIRS } from '../_shared/graph-pairs'

type Features = 'none' | 'random' | 'triangles' | 'random-walk'
const LAYERS = 3
const WIDTH = 8

/** Triangles through each node: the number of edges among its neighbours. */
function triangles(g: Graph): number[][] {
  const lists = adjacency(g).map((arcs) => new Set(arcs.map((a) => a.to)))
  return lists.map((nbrs) => {
    const ns = [...nbrs]
    let t = 0
    for (let i = 0; i < ns.length; i++) for (let j = i + 1; j < ns.length; j++) if (lists[ns[i]].has(ns[j])) t++
    return [t]
  })
}

function features(g: Graph, kind: Features, seed: number): number[][] {
  if (kind === 'none') return Array.from({ length: g.nodes }, () => [1])
  if (kind === 'triangles') return triangles(g).map(([t]) => [1 + t])
  if (kind === 'random-walk') return toRows(randomWalkEncoding(g, 6))
  const s = stream(seed)
  return Array.from({ length: g.nodes }, () => [uniform(s)])
}

/** Sum readout of each layer's node vectors, concatenated over the layers, for a GIN and a GCN with fixed weights. */
function readouts(g: Graph, x: number[][]) {
  const dim = x[0].length
  const s = stream(11)
  const gin = Array.from({ length: LAYERS }, (_, l) => {
    const mlp = Mlp([l === 0 ? dim : WIDTH, 16, WIDTH], { activation: 'tanh', outputActivation: 'tanh' })
    const p = mlp.init(s)
    return (z: Value) => mlp.apply(p, z)
  })
  const gcn = Array.from({ length: LAYERS }, (_, l) => {
    const layer = GraphConv(g, l === 0 ? dim : WIDTH, WIDTH)
    const p = layer.init(s)
    return (z: Value) => tanh(layer.apply(p, z))
  })
  const sum = (h: Value) => {
    const rows = toRows(h as never)
    return rows[0].map((_, j) => rows.reduce((t, r) => t + r[j], 0))
  }
  const run = (step: (l: number, h: Value) => Value) => {
    let h: Value = tensor(x)
    const out = [...sum(h)]
    for (let l = 0; l < LAYERS; l++) {
      h = step(l, h)
      out.push(...sum(h))
    }
    return out
  }
  return {
    gin: run((l, h) => ginConv(g, h, gin[l])),
    gcn: run((l, h) => gcn[l](h)),
  }
}

const distance = (a: number[], b: number[]) => {
  const d = Math.sqrt(a.reduce((s, v, j) => s + (v - b[j]) ** 2, 0))
  return d < 1e-9 ? 0 : d
}

/** 1-WL failure pairs, and what node features do to them. */
export function FailurePairs() {
  const state = useFigureState({
    pair: choice(
      PAIRS.map((p) => ({ value: p.id, label: p.label })),
      'hexagon-triangles',
      { label: 'pair of graphs' },
    ),
    features: choice<Features>(
      [
        { value: 'none', label: 'none (all equal)' },
        { value: 'triangles', label: 'triangle count' },
        { value: 'random-walk', label: 'random-walk return probabilities' },
        { value: 'random', label: 'random number per node' },
      ],
      'none',
      { label: 'added node feature' },
    ),
    seed: int(1, { ge: 0, le: 9999, label: 'seed', when: when('features', 'random') }),
  })
  const pair = pairById(state.pair)
  const r = useMemo(() => {
    const xl = features(pair.left.graph, state.features, state.seed)
    const xr = features(pair.right.graph, state.features, state.seed + 5000)
    // Discrete labels for colour refinement: equal feature vectors (to 1e-9) share a label in both graphs.
    const key = (row: number[]) => row.map((v) => v.toPrecision(9)).join(',')
    const distinct = [...new Set([...xl, ...xr].map(key))].sort()
    const label = (rows: number[][]) => rows.map((row) => distinct.indexOf(key(row)))
    const wl = weisfeilerLeman([pair.left.graph, pair.right.graph], { initial: [label(xl), label(xr)] })
    const left = readouts(pair.left.graph, xl)
    const right = readouts(pair.right.graph, xr)
    // The same graph again with a second draw of its features: zero unless the features are random.
    const again = readouts(pair.left.graph, features(pair.left.graph, state.features, state.seed + 9999))
    return { wl, left, right, again }
  }, [pair, state.features, state.seed])
  const last = r.wl.colours[r.wl.colours.length - 1]
  const classes = r.wl.classes[r.wl.classes.length - 1]
  const names = Array.from({ length: classes }, (_, k) => `colour ${k + 1}`)
  const shown = classes <= 8
  const range = drawingRange([pair.left, pair.right])
  const lx = useAxis({ range: range.x, nice: false, zoom: false })
  const ly = useAxis({ range: range.y, nice: false, zoom: false, equal: lx })
  const rx = useAxis({ range: range.x, nice: false, zoom: false })
  const ry = useAxis({ range: range.y, nice: false, zoom: false, equal: rx })
  const layers = (k: 0 | 1) =>
    graphLayers(k === 0 ? pair.left : pair.right, {
      classes: shown ? last[k] : undefined,
      classNames: names,
      labels: shown ? Array.from(last[k], (c) => String(c + 1)) : undefined,
      id: k === 0 ? 'left' : 'right',
    })
  return (
    <Figure
      title="Graph pairs that message passing cannot separate"
      state={state}
      readouts={{
        'colour refinement with these features': (
          <>
            <Readout label="stable colours" value={classes} />
            <Readout
              label="verdict"
              value={r.wl.distinguishedAt >= 0 ? `separated at round ${r.wl.distinguishedAt}` : 'tied'}
            />
          </>
        ),
        'graph readouts, 3 layers with fixed random weights': (
          <>
            <Readout label="GIN: distance between the graphs" value={formatNumber(distance(r.left.gin, r.right.gin))} />
            <Readout label="GCN: distance between the graphs" value={formatNumber(distance(r.left.gcn, r.right.gcn))} />
            <Readout
              label={`GIN: ${pair.left.name}, two feature draws`}
              value={formatNumber(distance(r.left.gin, r.again.gin))}
            />
          </>
        ),
      }}
      caption={`${pair.note} Nodes are coloured by their stable colour-refinement colour, started from the chosen feature (all nodes alike when none is added; with random numbers every node is its own colour and the colours are not drawn). The readouts sum each layer's node vectors of a GIN and a GCN with the same fixed random weights for both graphs: a distance of 0 means the network gives the two graphs identical outputs. Triangle counts separate the 6-cycle from two triangles; return probabilities also separate decalin from bicyclopentyl; neither separates the two strongly regular graphs. Random features separate everything, but the last readout shows that they also separate a graph from itself.`}
    >
      <Plots cols={2}>
        <Plot x={lx} y={ly} bare title={pair.left.name} legend={false}>
          {layers(0)}
        </Plot>
        <Plot x={rx} y={ry} bare title={pair.right.name} legend={false}>
          {layers(1)}
        </Plot>
      </Plots>
    </Figure>
  )
}
