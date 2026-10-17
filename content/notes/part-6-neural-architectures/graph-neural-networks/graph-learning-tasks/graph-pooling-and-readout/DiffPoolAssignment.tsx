import { useMemo } from 'react'
import { Figure, float, formatNumber, int, Player, Plot, Raster, Readout, useAxis, useFigureState } from 'aifn-render'
import {
  drawingRange,
  FixedHeight,
  graphLayers,
  usePlayerPosition,
  useSlicedRun,
  type Drawing,
} from '../_shared/figure-helpers'
import { diffPoolRun, EDGES, EVERY, N, type Checkpoint, type Setup } from './diffpool-run'

const PANEL = 250

/** The three triangles placed around a ring, each node 3t + 2 facing the next triangle and 3t facing the previous. */
const DRAWING: Drawing = (() => {
  const centre = (t: number): [number, number] => {
    const a = Math.PI / 2 - (2 * Math.PI * t) / 3
    return [1.7 * Math.cos(a), 1.7 * Math.sin(a)]
  }
  const toward = (from: [number, number], to: [number, number], r: number): [number, number] => {
    const dx = to[0] - from[0]
    const dy = to[1] - from[1]
    const len = Math.hypot(dx, dy)
    return [from[0] + (r * dx) / len, from[1] + (r * dy) / len]
  }
  const pos: [number, number][] = []
  for (let t = 0; t < 3; t++) {
    const c = centre(t)
    pos.push(toward(c, centre((t + 2) % 3), 0.6)) // node 3t, facing the previous triangle
    pos.push(toward(c, [2 * c[0], 2 * c[1]], 0.6)) // node 3t + 1, outward
    pos.push(toward(c, centre((t + 1) % 3), 0.6)) // node 3t + 2, facing the next triangle
  }
  return { pos, edges: EDGES }
})()

const argmax = (r: readonly number[]) => r.indexOf(Math.max(...r))

/** DiffPool's soft assignment learned from its auxiliary losses alone, on a ring of three triangles. */
export function DiffPoolAssignment() {
  const state = useFigureState({
    clusters: int(3, { ge: 2, le: 4, label: 'clusters c' }),
    steps: int(300, { ge: 10, le: 3000, suggestions: [100, 300, 1000], label: 'Adam steps' }),
    rate: float(0.05, { gt: 0, le: 1, suggestions: [0.01, 0.05, 0.2], label: 'learning rate' }),
    entropyWeight: float(0.1, { ge: 0, le: 10, suggestions: [0, 0.1, 1], label: 'entropy weight λ' }),
    seed: int(1, { ge: 1, le: 999, label: 'seed' }),
  })
  const { clusters, steps, rate, entropyWeight, seed } = state
  const setup = useMemo<Setup>(
    () => ({ clusters, steps, rate, entropyWeight, seed }),
    [clusters, steps, rate, entropyWeight, seed],
  )
  const run = useSlicedRun(setup, diffPoolRun)
  const cps: Checkpoint[] = run.checkpoints
  const [pos, setPos] = usePlayerPosition(setup)
  const at = Math.min(pos, Math.max(0, cps.length - 1))
  const cp = cps[at]

  const range = drawingRange([DRAWING], 0.45)
  const gx = useAxis({ range: range.x, nice: false, zoom: false })
  const gy = useAxis({ range: range.y, nice: false, zoom: false, equal: gx })
  const cx = useAxis({ label: 'cluster', range: [-0.5, clusters - 0.5], nice: false, integer: true, zoom: false })
  const nodeY = useAxis({
    label: 'node',
    range: [-0.5, N - 0.5],
    nice: false,
    integer: true,
    inverse: true,
    zoom: false,
  })
  const kx = useAxis({ label: 'cluster', range: [-0.5, clusters - 0.5], nice: false, integer: true, zoom: false })
  const ky = useAxis({
    label: 'cluster',
    range: [-0.5, clusters - 0.5],
    nice: false,
    integer: true,
    inverse: true,
    zoom: false,
  })
  const ids = Array.from({ length: clusters }, (_, j) => j)

  return (
    <Figure
      title="DiffPool learns a soft clustering"
      state={state}
      defaultSize="L"
      caption={`A ring of three triangles (nodes 0–8, bridges 2–3, 5–6 and 8–0). One graph convolution maps each node's one-hot identity to c assignment logits; a row softmax gives the soft assignment S (middle, one row per node). Adam minimises DiffPool's link loss ‖A − SSᵀ‖²_F / n², which rewards putting linked nodes in the same cluster, plus λ times the mean entropy of the rows of S, which rewards confident assignments; there is no task loss. Left: each node coloured by its most probable cluster. Right: the coarsened adjacency SᵀAS, whose diagonal holds twice the edge weight inside each cluster and whose off-diagonal entries hold the weight between clusters. With c = 3 each triangle becomes a cluster and the bridge nodes keep part of their weight on the neighbouring cluster. With c = 4 the extra cluster stays empty or splits a triangle; with c = 2 one triangle is shared. Step through the checkpoints, one every ${EVERY} steps, with the player.`}
      controls={
        <Player
          value={at}
          onChange={setPos}
          count={Math.max(1, cps.length)}
          label="checkpoint"
          format={(a) => `step ${cps[a]?.step ?? 0}`}
        />
      }
      readouts={
        <>
          <Readout label="step" value={cp ? `${cp.step}${run.done ? '' : ', training…'}` : '…'} />
          <Readout label="link loss" value={cp ? formatNumber(cp.linkLoss) : '…'} />
          <Readout label="mean row entropy (nats)" value={cp ? formatNumber(cp.entropy) : '…'} />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        <FixedHeight height={PANEL}>
          <Plot x={gx} y={gy} bare title="most probable cluster" legend={false}>
            {graphLayers(DRAWING, {
              id: 'ring',
              classes: cp ? cp.s.map(argmax) : undefined,
              classNames: ids.map((j) => `cluster ${j}`),
              labels: DRAWING.pos.map((_, v) => String(v)),
            })}
          </Plot>
        </FixedHeight>
        <FixedHeight height={PANEL}>
          <Plot x={cx} y={nodeY} title="assignment S">
            {cp && (
              <Raster
                name="S"
                x={ids}
                y={Array.from({ length: N }, (_, v) => v)}
                z={cp.s}
                range={[0, 1]}
                valueLabel="S_vj"
              />
            )}
          </Plot>
        </FixedHeight>
        <FixedHeight height={PANEL}>
          <Plot x={kx} y={ky} title="coarsened adjacency SᵀAS">
            {cp && <Raster name="SᵀAS" x={ids} y={ids} z={cp.coarse} range={[0, 6]} valueLabel="A′_jk" />}
          </Plot>
        </FixedHeight>
      </div>
    </Figure>
  )
}
