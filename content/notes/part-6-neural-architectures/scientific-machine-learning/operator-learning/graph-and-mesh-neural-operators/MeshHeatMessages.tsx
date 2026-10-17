import { useMemo, useState } from 'react'
import { Curve, Figure, Handle, Player, Plot, Points, Readout, Segments, useAxis } from 'aifn-render'
import { meshHeatRollout } from 'aifn-methods/neural/operator'

/** Diffusivity of u_t = D u_xx on [0, 1] with u = 0 at both ends, and the time shown at the last frame. */
const D = 0.1
const TIME = 0.5
const FRAMES = 51
/** Where the mesh is drawn, below the curves. */
const MESH_Y = -1.15
const START = [0, 0.06, 0.13, 0.2, 0.34, 0.39, 0.52, 0.63, 0.7, 0.82, 0.92, 1]
const GAP = 0.015
const u0 = (x: number) => Math.sin(Math.PI * x) + 0.5 * Math.sin(3 * Math.PI * x)
const exact = (x: number, t: number) =>
  Math.exp(-(Math.PI ** 2) * D * t) * Math.sin(Math.PI * x) +
  0.5 * Math.exp(-9 * Math.PI ** 2 * D * t) * Math.sin(3 * Math.PI * x)
const fine = Array.from({ length: 201 }, (_, i) => i / 200)

/**
 * The heat equation on an irregular mesh as message passing: messages weighted by the edge lengths against messages
 * that ignore them.
 */
export function MeshHeatMessages() {
  const [mesh, setMesh] = useState(START)
  const [frame, setFrame] = useState(0)

  const runs = useMemo(() => {
    const values = mesh.map(u0)
    return {
      geometric: meshHeatRollout(mesh, values, { diffusivity: D, time: TIME, frames: FRAMES, weights: 'geometric' }),
      uniform: meshHeatRollout(mesh, values, { diffusivity: D, time: TIME, frames: FRAMES, weights: 'uniform' }),
    }
  }, [mesh])

  const t = runs.geometric.times[frame]
  const truth = mesh.map((x) => exact(x, t))
  const worst = (u: number[]) => Math.max(...u.map((v, i) => Math.abs(v - truth[i])))
  const edges = mesh.slice(1).map((x, i) => ({ from: [mesh[i], MESH_Y] as const, to: [x, MESH_Y] as const }))

  const move = (i: number, x: number) =>
    setMesh((m) => {
      const next = m.slice()
      next[i] = Math.min(m[i + 1] - GAP, Math.max(m[i - 1] + GAP, x))
      return next
    })

  const xAxis = useAxis({ label: 'x', range: [0, 1] })
  const uAxis = useAxis({ label: 'u(x, t)', range: [-1.3, 1.5] })

  return (
    <Figure
      title="Messages that know the edge lengths"
      defaultSize="L"
      hoverReadout={false}
      caption={`The heat equation u_t = ${D} u_xx on [0, 1] with both ends held at 0, from u(x, 0) = sin πx + ½ sin 3πx, on a mesh of ${START.length} nodes (the dots on the bottom line). Each interior node i adds Δt D Σⱼ wᵢⱼ (uⱼ − uᵢ) over its two neighbours at every step. Geometric weights wᵢⱼ = 2 / (hᵢⱼ (h₋ + h₊)) use the lengths h of the node's two edges; uniform weights 1/h̄² use the mean edge length for every edge, as a message function blind to relative positions would. The dashed curve is the exact solution. Drag the mesh nodes to move them; the player steps through time.`}
      controls={
        <Player
          value={frame}
          onChange={setFrame}
          count={FRAMES}
          label="time"
          format={(f) => `t = ${runs.geometric.times[f].toFixed(2)}`}
        />
      }
      readouts={
        <>
          <Readout label="time t" value={t.toFixed(2)} />
          <Readout
            label="largest error at the nodes, geometric weights"
            value={worst(runs.geometric.states[frame]).toFixed(3)}
          />
          <Readout
            label="largest error at the nodes, uniform weights"
            value={worst(runs.uniform.states[frame]).toFixed(3)}
          />
          <Readout label="explicit steps" value={String(runs.geometric.steps)} />
        </>
      }
    >
      <Plot x={xAxis} y={uAxis}>
        <Curve name="exact solution" x={fine} y={fine.map((x) => exact(x, t))} emphasis dashed />
        <Curve name="geometric weights" x={mesh} y={runs.geometric.states[frame]} slot={0} showPoints />
        <Curve name="uniform weights" x={mesh} y={runs.uniform.states[frame]} slot={1} showPoints />
        <Segments name="mesh edges" segments={edges} muted />
        <Points name="mesh nodes" x={mesh} y={mesh.map(() => MESH_Y)} muted size={6} />
        {mesh.slice(1, -1).map((x, k) => (
          <Handle key={k} kind="point" at={[x, MESH_Y]} label={`node ${k + 1}`} onDrag={([v]) => move(k + 1, v)} />
        ))}
      </Plot>
    </Figure>
  )
}
