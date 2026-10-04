import { useMemo, useState } from 'react'
import { Button, Figure, Handle, Plot, Readout, seriesLayers, type SeriesSpec, useAxis } from 'aifn-render'

type Pt = [number, number]
type Mat3 = number[][]

const SOURCE: Pt[] = [
  [0, 0],
  [1, 0],
  [1, 1],
  [0, 1],
]
const PERSPECTIVE: Pt[] = [
  [0, 0],
  [1, 0],
  [0.75, 0.75],
  [0.25, 0.75],
]

/** Solve the 8 × 8 DLT system with h₃₃ = 1 by Gaussian elimination with partial pivoting. Null if degenerate. */
function homography(src: Pt[], dst: Pt[]): Mat3 | null {
  const A: number[][] = []
  src.forEach(([x, y], i) => {
    const [u, v] = dst[i]
    A.push([x, y, 1, 0, 0, 0, -u * x, -u * y, u])
    A.push([0, 0, 0, x, y, 1, -v * x, -v * y, v])
  })
  for (let col = 0; col < 8; col++) {
    let pivot = col
    for (let r = col + 1; r < 8; r++) if (Math.abs(A[r][col]) > Math.abs(A[pivot][col])) pivot = r
    if (Math.abs(A[pivot][col]) < 1e-10) return null
    ;[A[col], A[pivot]] = [A[pivot], A[col]]
    for (let r = 0; r < 8; r++) {
      if (r === col) continue
      const f = A[r][col] / A[col][col]
      for (let c = col; c < 9; c++) A[r][c] -= f * A[col][c]
    }
  }
  const h = A.map((row, i) => row[8] / row[i])
  return [
    [h[0], h[1], h[2]],
    [h[3], h[4], h[5]],
    [h[6], h[7], 1],
  ]
}

const apply = (H: Mat3, [x, y]: Pt): [number, number, number] => [
  H[0][0] * x + H[0][1] * y + H[0][2],
  H[1][0] * x + H[1][1] * y + H[1][2],
  H[2][0] * x + H[2][1] * y + H[2][2],
]

/** Cross product sign of consecutive edges: all equal means the quadrilateral is convex and not folded. */
function convex(q: Pt[]): boolean {
  const signs = q.map((p, i) => {
    const a = q[(i + 1) % 4]
    const b = q[(i + 2) % 4]
    return Math.sign((a[0] - p[0]) * (b[1] - a[1]) - (a[1] - p[1]) * (b[0] - a[0]))
  })
  return signs.every((s) => s === signs[0] && s !== 0)
}

/** Sample a source path, map it through H and join the results into one polyline; NaN breaks between paths. */
function warpPaths(H: Mat3, paths: Pt[][]): { x: number[]; y: number[] } {
  const x: number[] = []
  const y: number[] = []
  for (const path of paths) {
    for (const p of path) {
      const [a, b, w] = apply(H, p)
      // A point mapped to (or across) the line at infinity cannot be drawn; break the line there.
      if (w <= 1e-6) {
        x.push(NaN)
        y.push(NaN)
      } else {
        x.push(a / w)
        y.push(b / w)
      }
    }
    x.push(NaN)
    y.push(NaN)
  }
  return { x, y }
}

const N_GRID = 8
const segment = (a: Pt, b: Pt, n = 24): Pt[] =>
  Array.from({ length: n + 1 }, (_, i) => [a[0] + ((b[0] - a[0]) * i) / n, a[1] + ((b[1] - a[1]) * i) / n])
const GRID: Pt[][] = Array.from({ length: N_GRID + 1 }, (_, i) => i / N_GRID).flatMap((t) => [
  segment([t, 0], [t, 1]),
  segment([0, t], [1, t]),
])
const CIRCLE: Pt[][] = [
  Array.from({ length: 97 }, (_, i) => {
    const a = (2 * Math.PI * i) / 96
    return [0.5 + 0.35 * Math.cos(a), 0.5 + 0.35 * Math.sin(a)] as Pt
  }),
]
const X_RANGE: [number, number] = [-1, 2.5]
const Y_RANGE: [number, number] = [-0.75, 2]

const fmt = (v: number) => (Math.abs(v) < 5e-4 ? '0' : v.toFixed(3))

/** Drag the four corners of the image of a unit square; the grid and circle inside it follow the homography. */
export function HomographyExplorer() {
  const [corners, setCorners] = useState<Pt[]>(PERSPECTIVE)
  const H = useMemo(() => homography(SOURCE, corners), [corners])
  const ok = H !== null && convex(corners)

  const series = useMemo<SeriesSpec[]>(() => {
    const square = { x: [0, 1, 1, 0, 0], y: [0, 0, 1, 1, 0] }
    const out: SeriesSpec[] = [{ name: 'unit square (source)', type: 'line', ...square, muted: true, dashed: true }]
    if (H) {
      out.push({ name: 'warped grid', type: 'line', ...warpPaths(H, GRID), slot: 0 })
      out.push({ name: 'warped circle', type: 'line', ...warpPaths(H, CIRCLE), slot: 1 })
      // Vanishing points of the two families of grid lines: images of the directions (1, 0, 0) and (0, 1, 0).
      const vp = [0, 1]
        .map((j) => [H[0][j], H[1][j], H[2][j]])
        .filter(([, , w]) => Math.abs(w) > 1e-6)
        .map(([a, b, w]) => [a / w, b / w])
        .filter(([a, b]) => a > X_RANGE[0] && a < X_RANGE[1] && b > Y_RANGE[0] && b < Y_RANGE[1])
      if (vp.length)
        out.push({
          name: 'vanishing point',
          type: 'scatter',
          x: vp.map((p) => p[0]),
          y: vp.map((p) => p[1]),
          emphasis: true,
        })
    }
    out.push({ name: 'corners', type: 'scatter', x: corners.map((c) => c[0]), y: corners.map((c) => c[1]), slot: 2 })
    return out
  }, [H, corners])

  const handles: Handle[] = corners.map((c, i) => ({
    kind: 'point',
    at: c,
    label: `corner ${i + 1}`,
    onDrag: ([x, y]) =>
      setCorners((prev) =>
        prev.map((p, j) =>
          j === i ? [Math.min(Math.max(x, X_RANGE[0]), X_RANGE[1]), Math.min(Math.max(y, Y_RANGE[0]), Y_RANGE[1])] : p,
        ),
      ),
  }))

  const affine = H !== null && Math.abs(H[2][0]) < 1e-6 && Math.abs(H[2][1]) < 1e-6

  const xAxis = useAxis({ label: 'x′', range: X_RANGE })
  const yAxis = useAxis({ label: 'y′', range: Y_RANGE, equal: xAxis })
  return (
    <Figure
      title="A homography maps a square to any quadrilateral"
      caption="Drag the four corners. The homography is fixed by where the corners of the unit square (dashed) go: 4 point pairs give the 8 equations for its 8 degrees of freedom. Straight lines stay straight, but parallel grid lines meet at vanishing points (diamonds) and equal steps along a line become unequal. The circle maps to an ellipse while the quadrilateral stays convex. If the corners stop forming a convex quadrilateral, the line that maps to infinity passes through the square and the warp folds the plane over it."
      controls={
        <>
          <Button variant="outline" size="sm" onClick={() => setCorners(PERSPECTIVE)}>
            perspective
          </Button>
          <Button variant="outline" size="sm" onClick={() => setCorners(SOURCE)}>
            identity
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() =>
              setCorners([
                [0.2, 0],
                [1.4, 0.3],
                [1.2, 1.3],
                [0, 1],
              ])
            }
          >
            affine
          </Button>
        </>
      }
      readouts={
        H ? (
          <>
            <Readout label="H row 1" value={H[0].map(fmt).join('  ')} />
            <Readout label="H row 2" value={H[1].map(fmt).join('  ')} />
            <Readout label="H row 3" value={H[2].map(fmt).join('  ')} />
            <Readout
              label="type"
              value={!ok ? 'folded (not convex)' : affine ? 'affine (h₃₁ = h₃₂ = 0)' : 'projective'}
            />
          </>
        ) : (
          <Readout label="H" value="degenerate: three corners are collinear" />
        )
      }
    >
      <Plot x={xAxis} y={yAxis} ariaLabel={'Grid and circle warped by a homography'}>
        {seriesLayers(series)}
        {(handles ?? []).map((h, i) => (
          <Handle key={i} {...h} />
        ))}
      </Plot>
    </Figure>
  )
}
