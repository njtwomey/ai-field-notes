import { useMemo } from 'react'
import { Curve, Figure, formatNumber, Handle, int, Plot, Points, Readout, useAxis, useFigureState } from 'aifn-render'
import { normals, stream } from 'aifn-compute/foundation/random'
import { toFlat, type Tensor, type Value } from 'aifn-compute/foundation/tensor'
import { barabasiAlbertGraph } from 'aifn-compute/graph/structures'
import { computationGraph, sampledComputationGraph } from 'aifn-compute/nn/graph'
import { meanPropagation } from 'aifn-methods/neural/graph'
import { FixedHeight } from '../_shared/FixedHeight'

const N = 2000
const FANOUTS = [1, 2, 3, 4, 6, 8, 10, 12, 15, 20]
const DRAWS = 2

const flat = (v: Value): number[] => toFlat(v as Tensor)

/** Relative error of the sampled output at the targets against the exact one, sqrt(sum (a - e)^2 / sum e^2). */
function relativeError(approx: ArrayLike<number>, exact: ArrayLike<number>, targets: readonly number[]): number {
  let num = 0
  let den = 0
  for (const v of targets) {
    num += (approx[v] - exact[v]) ** 2
    den += exact[v] ** 2
  }
  return den > 0 ? Math.sqrt(num / den) : 0
}

/**
 * Nodes an L-layer minibatch needs on a Barabási–Albert graph, with full neighbourhoods and with GraphSAGE sampling
 * (`computationGraph`, `sampledComputationGraph`), and the error that sampling adds to a fixed mean-aggregation
 * network (`meanPropagation`).
 */
export function NeighbourhoodExplosion() {
  const state = useFigureState({
    edgesPerNode: int(3, { ge: 1, le: 8, label: 'edges per new node m', suggestions: [1, 2, 3, 5, 8] }),
    batch: int(16, { ge: 1, le: 512, label: 'targets per minibatch b', suggestions: [1, 16, 64, 256] }),
    layers: int(3, { ge: 1, le: 4, label: 'layers L' }),
    fanout: int(5, { ge: 1, le: 20, label: 'fan-out S per layer', suggestions: [2, 5, 10, 15] }),
    seed: int(1, { ge: 0, le: 999, label: 'seed' }),
  })
  const { edgesPerNode, batch, layers, fanout, seed } = state

  const graph = useMemo(() => barabasiAlbertGraph(stream(seed), N, edgesPerNode), [seed, edgesPerNode])
  const features = useMemo(() => normals(stream(seed + 1000), [N, 1]), [seed])
  // The targets: b nodes spread over the graph by a fixed stride (late, low-degree nodes as well as early hubs).
  const targets = useMemo(() => Array.from({ length: batch }, (_, i) => (i * 1237 + 11) % N), [batch])

  const full = useMemo(() => computationGraph(graph, targets, layers), [graph, targets, layers])
  const sampled = useMemo(
    () => sampledComputationGraph(graph, targets, new Array(layers).fill(fanout), stream(seed + 1)),
    [graph, targets, layers, fanout, seed],
  )

  // The sweep over fan-outs: nodes needed at depth L (as a fraction of the full computation graph) and the relative
  // error at the targets, each averaged over DRAWS samples.
  const sweep = useMemo(() => {
    const exact = flat(meanPropagation(graph, features, layers))
    const fraction: number[] = []
    const error: number[] = []
    for (const S of FANOUTS) {
      let f = 0
      let e = 0
      for (let d = 0; d < DRAWS; d++) {
        const run = sampledComputationGraph(
          graph,
          targets,
          new Array(layers).fill(S),
          stream(seed + 1 + 7919 * (d + 1)),
        )
        f += run.touched[layers] / full.touched[layers]
        e += relativeError(flat(meanPropagation(graph, features, layers, run.samples)), exact, targets)
      }
      fraction.push(f / DRAWS)
      error.push(e / DRAWS)
    }
    return { fraction, error }
  }, [graph, features, targets, layers, seed, full])

  const exactNow = useMemo(() => flat(meanPropagation(graph, features, layers)), [graph, features, layers])
  const errorNow = relativeError(flat(meanPropagation(graph, features, layers, sampled.samples)), exactNow, targets)

  const depths = Array.from({ length: layers + 1 }, (_, k) => k)
  const bound = depths.map((k) => Math.min(N, batch * depths.slice(0, k + 1).reduce((s, j) => s + fanout ** j, 0)))

  const dx = useAxis({ label: 'depth k (hops from the targets)', range: [0, layers], nice: false, integer: true })
  const dy = useAxis({ label: 'nodes needed', log: true, range: [1, N * 1.5], nice: false })
  const sx = useAxis({ label: 'fan-out S', range: [0, 21], nice: false, integer: true })
  const sy = useAxis({ label: 'fraction / relative error', range: [0, undefined] })
  const setFanout = (x: number) => state.set('fanout', Math.max(1, Math.min(20, Math.round(x))))

  return (
    <Figure
      title="Neighbourhood explosion and neighbour sampling"
      state={state}
      hoverReadout={false}
      caption={`A Barabási–Albert graph of ${N} nodes, each new node attached to m earlier ones, so a few early nodes become hubs with over a hundred neighbours. Left: the number of distinct nodes whose vectors an L-layer network needs to compute the outputs of b targets, by depth, with full neighbourhoods (ink) and with a sample of at most S neighbours per node and layer (coloured); the dashed line is the bound b(1 + S + … + S^k), and the top of the axis is the whole graph. Right: as S varies, the sampled computation graph as a fraction of the full one, and the relative error that sampling adds to the targets' outputs of a fixed network whose layer averages each node with the mean of its neighbours. Drag the vertical line to set S. Two or three full layers already reach most of the graph; a fan-out of 5 to 10 keeps a fraction of it and adds an error of about 10 per cent at three layers.`}
      readouts={
        <>
          <Readout label={`full: nodes needed at depth ${layers}`} value={`${full.touched[layers]} of ${N}`} />
          <Readout label={`sampled, S = ${fanout}`} value={`${sampled.touched[layers]}`} />
          <Readout label="bound b(1 + S + … + S^L)" value={formatNumber(bound[layers])} />
          <Readout label="relative error at the targets" value={formatNumber(Number(errorNow.toPrecision(2)))} />
        </>
      }
    >
      <FixedHeight height={260}>
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          <Plot x={dx} y={dy} title="nodes needed by depth">
            <Curve name="full neighbourhood" x={depths} y={full.touched} emphasis showPoints />
            <Curve name={`sampled, S = ${fanout}`} x={depths} y={sampled.touched} slot={0} showPoints />
            <Curve name="bound" x={depths} y={bound} muted dashed />
          </Plot>
          <Plot x={sx} y={sy} title={`trade-off at L = ${layers}`}>
            <Curve name="sampled / full nodes" x={FANOUTS} y={sweep.fraction} slot={0} showPoints />
            <Curve name="relative error" x={FANOUTS} y={sweep.error} slot={1} showPoints />
            <Points
              name="current S"
              x={[fanout, fanout]}
              y={[sampled.touched[layers] / full.touched[layers], errorNow]}
              emphasis
              size={9}
            />
            <Handle kind="x" at={fanout} label="S" onDrag={setFanout} />
          </Plot>
        </div>
      </FixedHeight>
    </Figure>
  )
}
