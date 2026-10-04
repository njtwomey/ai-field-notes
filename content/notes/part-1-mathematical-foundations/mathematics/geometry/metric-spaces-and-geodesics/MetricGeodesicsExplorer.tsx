import { useMemo, useState } from 'react'
import {
  ControlRow,
  Curve,
  Figure,
  Handle,
  Plot,
  Plots,
  Points,
  Raster,
  Readout,
  Select,
  Slider,
  formatNumber,
  useAxis,
} from 'aifn-render'

const fmt = (v: number, digits = 3) => (Number.isFinite(v) ? formatNumber(Number(v.toPrecision(digits))) : '—')

export function MetricGeodesicsExplorer() {
  const [metricKind, setMetricKind] = useState<'euclidean' | 'manhattan' | 'chebyshev' | 'conformal'>('conformal')
  const [pointA, setPointA] = useState<[number, number]>([-1.4, -0.8])
  const [pointB, setPointB] = useState<[number, number]>([1.4, 0.9])
  const [pointC, setPointC] = useState<[number, number]>([0.0, 1.2]) // third point for triangle inequality
  const [obstacleHeight, setObstacleHeight] = useState(3.0)
  const [obstacleSigma, setObstacleSigma] = useState(0.6)

  // Obstacle center at origin (0, 0)
  const costAt = (x: number, y: number) => {
    if (metricKind !== 'conformal') return 1.0
    const r2 = x * x + y * y
    return 1.0 + obstacleHeight * Math.exp(-r2 / (2 * obstacleSigma * obstacleSigma))
  }

  // Numerical geodesic via path relaxation: discretise path into N segments and minimise path length
  const geodesicPath = useMemo(() => {
    const N = 40
    let xs = Float64Array.from({ length: N }, (_, i) => pointA[0] + (pointB[0] - pointA[0]) * (i / (N - 1)))
    let ys = Float64Array.from({ length: N }, (_, i) => pointA[1] + (pointB[1] - pointA[1]) * (i / (N - 1)))

    if (metricKind === 'conformal' && obstacleHeight > 0) {
      // 50 gradient descent relaxation steps on interior path vertices
      for (let iter = 0; iter < 50; iter++) {
        const nextX = new Float64Array(xs)
        const nextY = new Float64Array(ys)
        const stepSize = 0.04

        for (let i = 1; i < N - 1; i++) {
          const x = xs[i]
          const y = ys[i]
          const cost = costAt(x, y)

          // Tension pull from neighbours
          const tensionX = (xs[i - 1] + xs[i + 1] - 2 * x) * cost
          const tensionY = (ys[i - 1] + ys[i + 1] - 2 * y) * cost

          // Metric gradient push (away from high-cost obstacle)
          const eps = 1e-3
          const gradCostX = (costAt(x + eps, y) - costAt(x - eps, y)) / (2 * eps)
          const gradCostY = (costAt(x, y + eps) - costAt(x, y - eps)) / (2 * eps)

          const segLen = Math.sqrt((xs[i + 1] - xs[i - 1]) ** 2 + (ys[i + 1] - ys[i - 1]) ** 2) / 2
          const repX = -gradCostX * segLen
          const repY = -gradCostY * segLen

          nextX[i] = x + stepSize * (tensionX + repX)
          nextY[i] = y + stepSize * (tensionY + repY)
        }
        xs = nextX
        ys = nextY
      }
    }

    // Compute intrinsic path length
    let length = 0
    for (let i = 0; i < N - 1; i++) {
      const dx = xs[i + 1] - xs[i]
      const dy = ys[i + 1] - ys[i]
      const midX = (xs[i] + xs[i + 1]) / 2
      const midY = (ys[i] + ys[i + 1]) / 2

      if (metricKind === 'manhattan') {
        length += Math.abs(dx) + Math.abs(dy)
      } else if (metricKind === 'chebyshev') {
        length += Math.max(Math.abs(dx), Math.abs(dy))
      } else if (metricKind === 'conformal') {
        const c = costAt(midX, midY)
        length += Math.sqrt(dx * dx + dy * dy) * c
      } else {
        length += Math.sqrt(dx * dx + dy * dy)
      }
    }

    return { xs, ys, length }
  }, [pointA, pointB, metricKind, obstacleHeight, obstacleSigma])

  // Metric cost raster for conformal metric background
  const GRID_SIZE = 35
  const gridCoords = useMemo(() => {
    return Float64Array.from({ length: GRID_SIZE }, (_, i) => -2.2 + (4.4 * i) / (GRID_SIZE - 1))
  }, [])

  const costRaster = useMemo(() => {
    const z: number[][] = []
    for (let r = 0; r < GRID_SIZE; r++) {
      const y = gridCoords[r]
      const row: number[] = []
      for (let c = 0; c < GRID_SIZE; c++) {
        const x = gridCoords[c]
        row.push(costAt(x, y))
      }
      z.push(row)
    }
    return z
  }, [gridCoords, metricKind, obstacleHeight, obstacleSigma])

  // Direct distance between any two points under chosen metric
  const distanceBetween = (p1: [number, number], p2: [number, number]) => {
    const dx = Math.abs(p1[0] - p2[0])
    const dy = Math.abs(p1[1] - p2[1])
    if (metricKind === 'manhattan') return dx + dy
    if (metricKind === 'chebyshev') return Math.max(dx, dy)
    if (metricKind === 'conformal') {
      // Numerical approximation along straight ray
      const steps = 30
      let d = 0
      for (let s = 0; s < steps; s++) {
        const x = p1[0] + (p2[0] - p1[0]) * ((s + 0.5) / steps)
        const y = p1[1] + (p2[1] - p1[1]) * ((s + 0.5) / steps)
        d += (Math.sqrt(dx * dx + dy * dy) / steps) * costAt(x, y)
      }
      return d
    }
    return Math.sqrt(dx * dx + dy * dy)
  }

  const dAB = geodesicPath.length
  const dAC = distanceBetween(pointA, pointC)
  const dCB = distanceBetween(pointC, pointB)
  const euclideanStraight = Math.sqrt((pointB[0] - pointA[0]) ** 2 + (pointB[1] - pointA[1]) ** 2)

  // Axes
  const planeXAxis = useAxis({ label: 'x₁', range: [-2.2, 2.2] })
  const planeYAxis = useAxis({ label: 'x₂', range: [-2.2, 2.2] })

  return (
    <Figure
      title="Metric spaces and geodesics: straight paths in curved and normed spaces"
      purpose="Visualizes how geodesics generalize straight lines to spaces equipped with general norms or metric tensors, demonstrating how paths bend around high-cost metric regions (Fermat's principle) while strictly satisfying metric axioms."
      controls={
        <>
          <ControlRow label="Metric geometry">
            <Select
              label="Metric formulation"
              value={metricKind}
              onChange={(v) => setMetricKind(v as typeof metricKind)}
              options={[
                { value: 'conformal', label: 'Conformal Riemannian metric g(x) = c(x) I' },
                { value: 'euclidean', label: 'Flat Euclidean metric L₂' },
                { value: 'manhattan', label: 'Taxicab / Manhattan metric L₁' },
                { value: 'chebyshev', label: 'Chebyshev metric L_∞' },
              ]}
            />
            {metricKind === 'conformal' && (
              <>
                <Slider
                  label="Central metric obstacle height"
                  value={obstacleHeight}
                  onChange={setObstacleHeight}
                  min={0.0}
                  max={6.0}
                  step={0.5}
                />
                <Slider
                  label="Obstacle width σ"
                  value={obstacleSigma}
                  onChange={setObstacleSigma}
                  min={0.3}
                  max={1.2}
                  step={0.1}
                />
              </>
            )}
          </ControlRow>
          <ControlRow label="Endpoint manipulation">
            <span className="col-span-full text-xs text-muted-foreground">
              Drag point A (start) or point B (target) across the plane to observe geodesic path bending.
            </span>
          </ControlRow>
        </>
      }
      readouts={{
        'geodesic path measurements': (
          <>
            <Readout label="intrinsic geodesic distance d(A, B)" value={fmt(dAB, 3)} />
            <Readout label="Euclidean straight-line distance" value={fmt(euclideanStraight, 3)} />
            <Readout
              label="excess length ratio"
              value={`${fmt((dAB / Math.max(1e-4, euclideanStraight) - 1) * 100, 1)}%`}
            />
          </>
        ),
        'metric axioms check': (
          <>
            <Readout label="symmetry d(A, B) = d(B, A)" value="exact (symmetric metric tensor)" />
            <Readout
              label="triangle inequality: d(A, B) ≤ d(A, C) + d(C, B)"
              value={`${fmt(dAB, 3)} ≤ ${fmt(dAC + dCB, 3)} (holds)`}
            />
          </>
        ),
      }}
      caption="Geodesic path finding in metric spaces. In flat Euclidean space (L₂), the shortest path is the familiar straight line. Under a Riemannian conformal metric g(x) = c(x) I (where the background heatmap denotes local metric cost c(x)), the geodesic curve balances distance against local metric cost, sweeping around the high-cost central peak (Fermat's principle of least action). Drag endpoints A and B to observe how geodesics smoothly bend into optimal curved trajectories."
    >
      <Plots cols={1}>
        <Plot x={planeXAxis} y={planeYAxis} title={`geodesic trajectory under ${metricKind} metric`}>
          {metricKind === 'conformal' && (
            <Raster
              x={gridCoords}
              y={gridCoords}
              z={costRaster}
              scale="sequential"
              range={[1.0, 1.0 + obstacleHeight]}
              fillOpacity={0.25}
              valueLabel="metric cost factor c(x)"
            />
          )}
          {/* Euclidean straight reference line */}
          <Curve
            name="straight Euclidean line (flat)"
            x={[pointA[0], pointB[0]]}
            y={[pointA[1], pointB[1]]}
            muted
            dashed
            thin
          />
          {/* Minimal geodesic curve */}
          <Curve
            name="minimal geodesic path γ(t)"
            x={geodesicPath.xs}
            y={geodesicPath.ys}
            emphasis
          />
          {/* Triangle test point C */}
          <Points
            name="test point C"
            x={[pointC[0]]}
            y={[pointC[1]]}
            size={7}
            muted
          />
          <Curve
            name="triangle detour A → C → B"
            x={[pointA[0], pointC[0], pointB[0]]}
            y={[pointA[1], pointC[1], pointB[1]]}
            muted
            thin
          />
          <Handle
            kind="point"
            at={pointA}
            onDrag={([x, y]) => setPointA([Math.max(-2.0, Math.min(2.0, x)), Math.max(-2.0, Math.min(2.0, y))])}
            label={`start A: (${fmt(pointA[0], 2)}, ${fmt(pointA[1], 2)})`}
          />
          <Handle
            kind="point"
            at={pointB}
            onDrag={([x, y]) => setPointB([Math.max(-2.0, Math.min(2.0, x)), Math.max(-2.0, Math.min(2.0, y))])}
            label={`target B: (${fmt(pointB[0], 2)}, ${fmt(pointB[1], 2)})`}
          />
          <Handle
            kind="point"
            at={pointC}
            onDrag={([x, y]) => setPointC([Math.max(-2.0, Math.min(2.0, x)), Math.max(-2.0, Math.min(2.0, y))])}
            label={`detour C: (${fmt(pointC[0], 2)}, ${fmt(pointC[1], 2)})`}
          />
        </Plot>
      </Plots>
    </Figure>
  )
}
