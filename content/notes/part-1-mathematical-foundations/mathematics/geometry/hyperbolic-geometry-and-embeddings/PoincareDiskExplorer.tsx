import { useMemo, useState } from 'react'
import {
  ControlRow,
  Curve,
  Figure,
  Handle,
  Plot,
  Plots,
  Points,
  Readout,
  Select,
  Slider,
  formatNumber,
  useAxis,
} from 'aifn-render'

const fmt = (v: number, digits = 3) => (Number.isFinite(v) ? formatNumber(Number(v.toPrecision(digits))) : '—')

// Hyperbolic distance in Poincaré disk
function poincareDist(u: [number, number], v: [number, number]): number {
  const diffSq = (u[0] - v[0]) ** 2 + (u[1] - v[1]) ** 2
  const uNormSq = u[0] ** 2 + u[1] ** 2
  const vNormSq = v[0] ** 2 + v[1] ** 2
  const denom = (1 - uNormSq) * (1 - vNormSq)
  if (denom <= 1e-12) return 10
  const arg = 1 + (2 * diffSq) / Math.max(1e-12, denom)
  return Math.acosh(Math.max(1, arg))
}

// Möbius addition in Poincaré disk: u ⊕ v (translates -u to 0, or moves disk by translation vector)
function mobiusAdd(u: [number, number], v: [number, number]): [number, number] {
  const u2 = u[0] * u[0] + u[1] * u[1]
  const v2 = v[0] * v[0] + v[1] * v[1]
  const uv = u[0] * v[0] + u[1] * v[1]
  const denom = 1 + 2 * uv + u2 * v2
  if (Math.abs(denom) < 1e-9) return [v[0], v[1]]
  const x = ((1 + 2 * uv + v2) * u[0] + (1 - u2) * v[0]) / denom
  const y = ((1 + 2 * uv + v2) * u[1] + (1 - u2) * v[1]) / denom
  return [x, y]
}

// Generate circular arc geodesic connecting u and v orthogonal to unit circle
function poincareGeodesic(u: [number, number], v: [number, number], steps = 30): { xs: Float64Array; ys: Float64Array } {
  const d2U = u[0] ** 2 + u[1] ** 2
  const d2V = v[0] ** 2 + v[1] ** 2

  // If points are nearly collinear with origin, geodesic is a straight Euclidean line
  const cross = u[0] * v[1] - u[1] * v[0]
  if (Math.abs(cross) < 1e-4) {
    const xs = Float64Array.from({ length: steps }, (_, i) => u[0] + (v[0] - u[0]) * (i / (steps - 1)))
    const ys = Float64Array.from({ length: steps }, (_, i) => u[1] + (v[1] - u[1]) * (i / (steps - 1)))
    return { xs, ys }
  }

  // Circle passing through u, v and orthogonal to unit circle:
  // Orthogonal means r_c^2 = x_c^2 + y_c^2 - 1
  // (u_x - x_c)^2 + (u_y - y_c)^2 = r_c^2 => u_x^2 + u_y^2 - 2(u_x x_c + u_y y_c) = -1
  // 2 u . c = 1 + ||u||^2
  // 2 v . c = 1 + ||v||^2
  const b1 = (1 + d2U) / 2
  const b2 = (1 + d2V) / 2
  const det = u[0] * v[1] - u[1] * v[0]

  const cx = (b1 * v[1] - b2 * u[1]) / det
  const cy = (u[0] * b2 - v[0] * b1) / det
  const radius = Math.sqrt(cx * cx + cy * cy - 1)

  // Angular range along the circle
  let thU = Math.atan2(u[1] - cy, u[0] - cx)
  let thV = Math.atan2(v[1] - cy, v[0] - cx)

  // Choose the shorter arc connecting u and v
  let dTh = thV - thU
  while (dTh > Math.PI) dTh -= 2 * Math.PI
  while (dTh < -Math.PI) dTh += 2 * Math.PI

  const xs = Float64Array.from({ length: steps }, (_, i) => cx + radius * Math.cos(thU + dTh * (i / (steps - 1))))
  const ys = Float64Array.from({ length: steps }, (_, i) => cy + radius * Math.sin(thU + dTh * (i / (steps - 1))))
  return { xs, ys }
}

interface TreeNode {
  id: number
  label: string
  parent: number | null
  basePos: [number, number]
  depth: number
}

// Construct a hierarchical 3-level tree: 1 root -> 3 children -> 9 grandchildren
const RAW_TREE: TreeNode[] = (() => {
  const nodes: TreeNode[] = [{ id: 0, label: 'Root (organism)', parent: null, basePos: [0, 0], depth: 0 }]
  const cRadius = 0.55
  const gRadius = 0.85

  // Depth 1
  const categories = ['Mammal', 'Bird', 'Reptile']
  for (let i = 0; i < 3; i++) {
    const angle = (2 * Math.PI * i) / 3
    const pos: [number, number] = [cRadius * Math.cos(angle), cRadius * Math.sin(angle)]
    nodes.push({ id: nodes.length, label: categories[i], parent: 0, basePos: pos, depth: 1 })
  }

  // Depth 2
  const subcats = [
    ['Feline', 'Canine', 'Primate'],
    ['Raptor', 'Songbird', 'Waterfowl'],
    ['Serpent', 'Lizard', 'Chelonian'],
  ]

  for (let i = 0; i < 3; i++) {
    const parentId = i + 1
    const pAngle = (2 * Math.PI * i) / 3
    for (let j = 0; j < 3; j++) {
      const angle = pAngle + ((j - 1) * Math.PI) / 4.5
      const pos: [number, number] = [gRadius * Math.cos(angle), gRadius * Math.sin(angle)]
      nodes.push({
        id: nodes.length,
        label: subcats[i][j],
        parent: parentId,
        basePos: pos,
        depth: 2,
      })
    }
  }
  return nodes
})()

export function PoincareDiskExplorer() {
  const [panCenter, setPanCenter] = useState<[number, number]>([0.0, 0.0])
  const [focalNodeId, setFocalNodeId] = useState(1) // default focus on 'Mammal'
  const [showGeodesics, setShowGeodesics] = useState(true)

  // Boundary circle
  const boundaryCircle = useMemo(() => {
    const steps = 120
    const xs = Float64Array.from({ length: steps }, (_, i) => Math.cos((2 * Math.PI * i) / (steps - 1)))
    const ys = Float64Array.from({ length: steps }, (_, i) => Math.sin((2 * Math.PI * i) / (steps - 1)))
    return { xs, ys }
  }, [])

  // Translated tree node coordinates under Möbius translation by panCenter
  const translatedNodes = useMemo(() => {
    return RAW_TREE.map((node) => {
      // Isometry: u ⊕ (-panCenter)
      const negPan: [number, number] = [-panCenter[0], -panCenter[1]]
      const pos = mobiusAdd(node.basePos, negPan)
      return {
        ...node,
        pos,
      }
    })
  }, [panCenter])

  // Geodesic edges between parents and children
  const edges = useMemo(() => {
    if (!showGeodesics) return []
    const curves: { id: string; xs: Float64Array; ys: Float64Array }[] = []

    translatedNodes.forEach((node) => {
      if (node.parent !== null) {
        const parentNode = translatedNodes[node.parent]
        const geo = poincareGeodesic(parentNode.pos, node.pos, 25)
        curves.push({
          id: `${parentNode.id}-${node.id}`,
          xs: geo.xs,
          ys: geo.ys,
        })
      }
    })
    return curves
  }, [translatedNodes, showGeodesics])

  const nodeXs = useMemo(() => Float64Array.from(translatedNodes, (n) => n.pos[0]), [translatedNodes])
  const nodeYs = useMemo(() => Float64Array.from(translatedNodes, (n) => n.pos[1]), [translatedNodes])
  const nodeGroups = useMemo(() => translatedNodes.map((n) => n.depth), [translatedNodes])

  const focal = translatedNodes[focalNodeId]
  const distToOriginHyp = poincareDist([0, 0], focal.pos)
  const normEuclid = Math.sqrt(focal.pos[0] ** 2 + focal.pos[1] ** 2)

  // Sibling distance in hyperbolic vs Euclidean
  const siblingA = translatedNodes[4] // Feline
  const siblingB = translatedNodes[5] // Canine
  const siblingHypDist = poincareDist(siblingA.pos, siblingB.pos)
  const siblingEuclidDist = Math.sqrt((siblingA.pos[0] - siblingB.pos[0]) ** 2 + (siblingA.pos[1] - siblingB.pos[1]) ** 2)

  // Axes
  const diskAxisX = useAxis({ label: 'Poincaré u₁', range: [-1.15, 1.15] })
  const diskAxisY = useAxis({ label: 'Poincaré u₂', range: [-1.15, 1.15] })

  return (
    <Figure
      title="Hyperbolic geometry: Poincaré disk model and tree embeddings"
      purpose="Visualizes how the Poincaré disk model MH² equips continuous space with negative curvature, providing exponentially expanding capacity that embeds branching taxonomies and graph trees with negligible distortion compared to flat Euclidean space."
      controls={
        <>
          <ControlRow label="Hyperbolic navigation">
            <Slider
              label="Möbius horizontal translation"
              value={panCenter[0]}
              onChange={(x) => setPanCenter([x, panCenter[1]])}
              min={-0.65}
              max={0.65}
              step={0.05}
            />
            <Slider
              label="Möbius vertical translation"
              value={panCenter[1]}
              onChange={(y) => setPanCenter([panCenter[0], y])}
              min={-0.65}
              max={0.65}
              step={0.05}
            />
            <Select
              label="Inspect focal node"
              value={String(focalNodeId)}
              onChange={(v) => setFocalNodeId(Number(v))}
              options={RAW_TREE.map((n) => ({ value: String(n.id), label: `${n.label} (depth ${n.depth})` }))}
            />
          </ControlRow>
          <ControlRow label="Display options">
            <Select
              label="Circular arc geodesics"
              value={showGeodesics ? 'yes' : 'no'}
              onChange={(v) => setShowGeodesics(v === 'yes')}
              options={[
                { value: 'yes', label: 'Draw orthogonal circular geodesics' },
                { value: 'no', label: 'Nodes only' },
              ]}
            />
          </ControlRow>
        </>
      }
      readouts={{
        [`focal node: ${focal.label}`]: (
          <>
            <Readout label="tree depth" value={String(focal.depth)} />
            <Readout label="hyperbolic distance to center" value={`${fmt(distToOriginHyp, 3)} units`} />
            <Readout label="Euclidean radius ||u||" value={fmt(normEuclid, 4)} />
            <Readout
              label="metric expansion factor 2/(1−||u||²)"
              value={`${fmt(2 / Math.max(1e-4, 1 - normEuclid ** 2), 2)}×`}
            />
          </>
        ),
        'leaf separation (Feline vs Canine)': (
          <>
            <Readout label="intrinsic hyperbolic distance" value={`${fmt(siblingHypDist, 3)} units`} />
            <Readout label="Euclidean chord length" value={fmt(siblingEuclidDist, 3)} />
            <Readout
              label="Euclidean crowding distortion"
              value={`${fmt((1 - siblingEuclidDist / siblingHypDist) * 100, 1)}% contracted`}
            />
          </>
        ),
      }}
      caption="The Poincaré disk model D² = {u ∈ R² : ||u|| < 1}. The outer boundary ||u|| = 1 represents infinity; as points approach the perimeter, the metric tensor g_u = 4/(1−||u||²)² I diverges, giving the disk infinite volume. Tree hierarchy nodes (organisms → classes → families) are connected by circular arc geodesics that meet the boundary at right angles. Notice how leaf nodes appear crowded in Euclidean terms near the edge but remain far apart in intrinsic hyperbolic distance. Pan the translation sliders to apply Möbius isometries u ⊕ v: nodes moving toward the center dilate, while nodes near the edge contract, preserving all hyperbolic distances identically."
    >
      <Plots cols={1}>
        <Plot x={diskAxisX} y={diskAxisY} title="Poincaré disk model: tree hierarchy and geodesics">
          {/* Unit boundary circle representing infinity */}
          <Curve name="boundary at infinity (||u|| = 1)" x={boundaryCircle.xs} y={boundaryCircle.ys} emphasis />
          {/* Geodesic edges between tree nodes */}
          {edges.map((e) => (
            <Curve key={e.id} name="geodesic edge" x={e.xs} y={e.ys} muted thin />
          ))}
          {/* Hierarchy tree nodes */}
          <Points
            name="tree nodes"
            x={nodeXs}
            y={nodeYs}
            group={nodeGroups}
            groupNames={['Root (depth 0)', 'Classes (depth 1)', 'Families (depth 2)']}
            size={7}
          />
          {/* Highlighted focal node */}
          <Points
            name={`focal: ${focal.label}`}
            x={[focal.pos[0]]}
            y={[focal.pos[1]]}
            size={11}
            emphasis
          />
          <Handle
            kind="point"
            at={focal.pos}
            onDrag={([x, y]) => {
              // Adjust panCenter to move this focal node
              const r = Math.sqrt(x * x + y * y)
              if (r < 0.9) {
                setPanCenter([Math.max(-0.65, Math.min(0.65, -x * 0.4)), Math.max(-0.65, Math.min(0.65, -y * 0.4))])
              }
            }}
            label={`node: ${focal.label}`}
          />
        </Plot>
      </Plots>
    </Figure>
  )
}
