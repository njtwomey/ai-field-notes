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

const P_OPTIONS = [
  { value: '0.5', label: 'p = 0.5 (non-convex, sparse astroid)' },
  { value: '1.0', label: 'p = 1.0 (Manhattan / L1 diamond)' },
  { value: '1.5', label: 'p = 1.5 (intermediate superellipse)' },
  { value: '2.0', label: 'p = 2.0 (Euclidean / L2 circle)' },
  { value: '4.0', label: 'p = 4.0 (squircle)' },
  { value: '100', label: 'p = ∞ (Chebyshev / L∞ box)' },
]

const CANDIDATES: [number, number][] = [
  [1.4, 0.2],
  [1.1, 1.1],
  [0.3, 1.5],
  [1.8, 1.6],
]

const CANDIDATE_NAMES = ['A (mostly horizontal)', 'B (diagonal 45°)', 'C (mostly vertical)', 'D (same ray as B, farther)']

function minkowskiDist(x1: number, y1: number, x2: number, y2: number, p: number): number {
  const dx = Math.abs(x1 - x2)
  const dy = Math.abs(y1 - y2)
  if (p >= 90) return Math.max(dx, dy)
  return (dx ** p + dy ** p) ** (1 / p)
}

function cosineDist(x1: number, y1: number, x2: number, y2: number): number {
  const norm1 = Math.sqrt(x1 * x1 + y1 * y1)
  const norm2 = Math.sqrt(x2 * x2 + y2 * y2)
  if (norm1 < 1e-9 || norm2 < 1e-9) return 1
  const dot = x1 * x2 + y1 * y2
  const cosSim = Math.max(-1, Math.min(1, dot / (norm1 * norm2)))
  return 1 - cosSim
}

export function PairwiseDistanceExplorer() {
  const [pVal, setPVal] = useState(2.0)
  const [query, setQuery] = useState<[number, number]>([0.2, 0.3])
  const [contourDist, setContourDist] = useState(1.2)

  // Compute unit ball boundary for chosen p
  const unitBall = useMemo(() => {
    const steps = 180
    const xs: number[] = []
    const ys: number[] = []
    const p = pVal

    for (let i = 0; i <= steps; i++) {
      const angle = (2 * Math.PI * i) / steps
      const cosA = Math.cos(angle)
      const sinA = Math.sin(angle)
      const absCos = Math.abs(cosA)
      const absSin = Math.abs(sinA)

      let r = 1
      if (p >= 90) {
        r = 1 / Math.max(absCos, absSin)
      } else {
        r = (absCos ** p + absSin ** p) ** (-1 / p)
      }
      xs.push(r * cosA)
      ys.push(r * sinA)
    }
    return {
      x: Float64Array.from(xs),
      y: Float64Array.from(ys),
    }
  }, [pVal])

  // Expanding contour centered at query point
  const queryContour = useMemo(() => {
    const steps = 180
    const xs: number[] = []
    const ys: number[] = []
    const p = pVal
    const radius = contourDist

    for (let i = 0; i <= steps; i++) {
      const angle = (2 * Math.PI * i) / steps
      const cosA = Math.cos(angle)
      const sinA = Math.sin(angle)
      const absCos = Math.abs(cosA)
      const absSin = Math.abs(sinA)

      let r = 1
      if (p >= 90) {
        r = 1 / Math.max(absCos, absSin)
      } else {
        r = (absCos ** p + absSin ** p) ** (-1 / p)
      }
      xs.push(query[0] + radius * r * cosA)
      ys.push(query[1] + radius * r * sinA)
    }
    return {
      x: Float64Array.from(xs),
      y: Float64Array.from(ys),
    }
  }, [pVal, query, contourDist])

  // Distances to candidates
  const distances = useMemo(() => {
    return CANDIDATES.map(([cx, cy], i) => {
      const l1 = minkowskiDist(query[0], query[1], cx, cy, 1)
      const l2 = minkowskiDist(query[0], query[1], cx, cy, 2)
      const lInf = minkowskiDist(query[0], query[1], cx, cy, 100)
      const lp = minkowskiDist(query[0], query[1], cx, cy, pVal)
      const cosD = cosineDist(query[0], query[1], cx, cy)
      const cosSim = 1 - cosD
      return {
        name: CANDIDATE_NAMES[i],
        pt: [cx, cy],
        l1,
        l2,
        lInf,
        lp,
        cosD,
        cosSim,
      }
    })
  }, [query, pVal])

  // Nearest neighbour under selected metric
  const nearestIdx = useMemo(() => {
    let best = 0
    let bestD = Infinity
    distances.forEach((d, i) => {
      if (d.lp < bestD) {
        bestD = d.lp
        best = i
      }
    })
    return best
  }, [distances])

  const nearestCosIdx = useMemo(() => {
    let best = 0
    let bestD = Infinity
    distances.forEach((d, i) => {
      if (d.cosD < bestD) {
        bestD = d.cosD
        best = i
      }
    })
    return best
  }, [distances])

  const candX = useMemo(() => Float64Array.from(CANDIDATES, (c) => c[0]), [])
  const candY = useMemo(() => Float64Array.from(CANDIDATES, (c) => c[1]), [])

  // Triangle inequality check on Cosine vs Euclidean
  // Points: u = (1, 0) [0 deg], v = (1, 1) [45 deg], w = (0, 1) [90 deg]
  const triangleCosine = useMemo(() => {
    const dUV = 1 - Math.cos(Math.PI / 4) // 1 - 0.7071 = 0.2929
    const dVW = 1 - Math.cos(Math.PI / 4) // 0.2929
    const dUW = 1 - Math.cos(Math.PI / 2) // 1.0
    // d(u, w) = 1.0 > d(u, v) + d(v, w) = 0.5858 => violates triangle inequality!
    return { dUV, dVW, dUW, sumIntermediate: dUV + dVW, violates: dUW > dUV + dVW }
  }, [])

  // Axes
  const geomAxisX = useAxis({ label: 'x₁', range: [-2.2, 2.5] })
  const geomAxisY = useAxis({ label: 'x₂', range: [-2.2, 2.5] })

  const ballAxisX = useAxis({ label: 'u₁', range: [-1.4, 1.4] })
  const ballAxisY = useAxis({ label: 'u₂', range: [-1.4, 1.4] })

  return (
    <Figure
      title="Geometry of Minkowski distances and cosine similarity"
      purpose="Visualizes how the norm parameter p deforms the unit ball from the diamond L1 to the circle L2 and box L∞, demonstrating how metric choice flips nearest-neighbor rankings and why cosine distance violates the triangle inequality."
      controls={
        <>
          <ControlRow label="Distance metric">
            <Select
              label="Minkowski norm p"
              value={String(pVal)}
              onChange={(v) => setPVal(Number(v))}
              options={P_OPTIONS}
            />
            <Slider
              label="Contour radius"
              value={contourDist}
              onChange={setContourDist}
              min={0.2}
              max={2.5}
              step={0.05}
            />
          </ControlRow>
          <ControlRow label="Query manipulation">
            <Slider
              label="Query x₁"
              value={query[0]}
              onChange={(x) => setQuery([x, query[1]])}
              min={-1.5}
              max={1.5}
              step={0.05}
            />
            <Slider
              label="Query x₂"
              value={query[1]}
              onChange={(y) => setQuery([query[0], y])}
              min={-1.5}
              max={1.5}
              step={0.05}
            />
          </ControlRow>
        </>
      }
      readouts={{
        'nearest neighbour rankings': (
          <>
            <Readout label={`nearest under L_${pVal >= 90 ? '∞' : pVal}`} value={`${distances[nearestIdx].name} (${fmt(distances[nearestIdx].lp)})`} />
            <Readout label="nearest under cosine similarity" value={`${distances[nearestCosIdx].name} (cosθ = ${fmt(distances[nearestCosIdx].cosSim)})`} />
          </>
        ),
        'distances from query to candidate B': (
          <>
            <Readout label="Manhattan distance L₁" value={fmt(distances[1].l1)} />
            <Readout label="Euclidean distance L₂" value={fmt(distances[1].l2)} />
            <Readout label="Chebyshev distance L_∞" value={fmt(distances[1].lInf)} />
            <Readout label="cosine distance (1 − cosθ)" value={fmt(distances[1].cosD)} />
          </>
        ),
        'metric axiom: triangle inequality': (
          <>
            <Readout
              label="cosine distance d(0°, 90°)"
              value={`${fmt(triangleCosine.dUW)} > ${fmt(triangleCosine.sumIntermediate)} (violates!)`}
            />
            <Readout
              label="verdict"
              value="cosine distance is a semi-metric, not a true metric"
            />
          </>
        ),
      }}
      caption="Geometry of distance functions in R². Left: Candidate points A, B, C, D alongside draggable query point q. The red contour is the Lp level set centered at q; as it expands, whichever candidate it touches first is the nearest neighbour. Drag q or change p to observe nearest-neighbour inversions: L1 favours axis-aligned movements (penalising diagonals), L2 treats all rotations symmetrically, and L∞ measures only the single worst coordinate. Right: The Minkowski unit ball {u : ||u||_p ≤ 1}, showing how increasing p swells the geometry from the sharp L1 diamond (encouraging sparse coordinate selection) to the isotropic L2 sphere and the L∞ square."
    >
      <Plots cols={2}>
        <Plot x={geomAxisX} y={geomAxisY} title={`nearest neighbour search under L_${pVal >= 90 ? '∞' : pVal}`}>
          {/* Coordinate axes */}
          <Curve name="x axis" x={[-2.2, 2.5]} y={[0, 0]} muted dashed thin />
          <Curve name="y axis" x={[0, 0]} y={[-2.2, 2.5]} muted dashed thin />
          {/* Query ray from origin */}
          <Curve
            name="query direction ray"
            x={[0, query[0] * 2]}
            y={[0, query[1] * 2]}
            muted
            thin
          />
          {/* Level set contour around query */}
          <Curve
            name={`L_${pVal >= 90 ? '∞' : pVal} level contour`}
            x={queryContour.x}
            y={queryContour.y}
            emphasis
          />
          {/* Candidates */}
          <Points
            name="candidate points"
            x={candX}
            y={candY}
            size={8}
          />
          {/* Draggable query point */}
          <Handle
            kind="point"
            at={query}
            onDrag={([x, y]) => setQuery([x, y])}
            label={`query q: (${fmt(query[0], 2)}, ${fmt(query[1], 2)})`}
          />
        </Plot>
        <Plot x={ballAxisX} y={ballAxisY} title={`Minkowski unit ball ||u||_${pVal >= 90 ? '∞' : pVal} ≤ 1`}>
          <Curve name="u₁ axis" x={[-1.4, 1.4]} y={[0, 0]} muted dashed thin />
          <Curve name="u₂ axis" x={[0, 0]} y={[-1.4, 1.4]} muted dashed thin />
          <Curve
            name="L2 unit circle (reference)"
            x={Float64Array.from({ length: 90 }, (_, i) => Math.cos((2 * Math.PI * i) / 89))}
            y={Float64Array.from({ length: 90 }, (_, i) => Math.sin((2 * Math.PI * i) / 89))}
            muted
            dashed
            thin
          />
          <Curve
            name={`unit ball (p = ${pVal >= 90 ? '∞' : pVal})`}
            x={unitBall.x}
            y={unitBall.y}
            emphasis
          />
        </Plot>
      </Plots>
    </Figure>
  )
}
