/**
 * The digital differential analyser of `castRay` (../_shared/world.ts), unrolled into a trace of its steps for the
 * walk-through figure. The arithmetic is the same, line for line, so the trace ends where `castRay` does.
 */
import { cellAt, MAX_DISTANCE, type Material, type World } from '../_shared/world'

/** The analyser's state after one step: it crossed a grid line at distance `t` and entered `cell`. */
export type DdaStep = {
  /** Distance along the ray to the crossing. */
  t: number
  /** The kind of grid line crossed: `vertical` (x = const) or `horizontal` (y = const). */
  crossed: 'vertical' | 'horizontal'
  /** The grid line crossed: its x for a vertical line, its y for a horizontal one. */
  line: number
  /** The cell entered, as integer (column, row). */
  cell: [number, number]
  /** The distances to the next vertical and horizontal grid line after the step. */
  nextX: number
  nextY: number
  /** The material of the cell entered (0 for empty). */
  material: Material
}

export type DdaTrace = {
  x: number
  y: number
  dx: number
  dy: number
  /** Distance along the ray between consecutive vertical (deltaX) and horizontal (deltaY) grid lines. */
  deltaX: number
  deltaY: number
  /** The camera's cell and the initial distances to the first vertical and horizontal grid lines. */
  cell: [number, number]
  nextX: number
  nextY: number
  /** True when the camera stands in a wall cell: there is nothing to trace. */
  inWall: boolean
  steps: DdaStep[]
}

/** Trace one ray from (x, y) along `angle` to the first wall, recording every step. */
export function traceRay(w: World, x: number, y: number, angle: number): DdaTrace {
  const dx = Math.cos(angle)
  const dy = Math.sin(angle)
  let cx = Math.floor(x)
  let cy = Math.floor(y)
  const stepX = dx < 0 ? -1 : 1
  const stepY = dy < 0 ? -1 : 1
  const deltaX = dx === 0 ? Infinity : Math.abs(1 / dx)
  const deltaY = dy === 0 ? Infinity : Math.abs(1 / dy)
  let nextX = dx < 0 ? (x - cx) * deltaX : (cx + 1 - x) * deltaX
  let nextY = dy < 0 ? (y - cy) * deltaY : (cy + 1 - y) * deltaY
  const out: DdaTrace = { x, y, dx, dy, deltaX, deltaY, cell: [cx, cy], nextX, nextY, inWall: false, steps: [] }
  if (cellAt(w, cx + 0.5, cy + 0.5) !== 0) return { ...out, inWall: true }
  let t = 0
  while (t < MAX_DISTANCE) {
    let crossed: DdaStep['crossed']
    let line: number
    if (nextX < nextY) {
      t = nextX
      nextX += deltaX
      line = stepX > 0 ? cx + 1 : cx
      cx += stepX
      crossed = 'vertical'
    } else {
      t = nextY
      nextY += deltaY
      line = stepY > 0 ? cy + 1 : cy
      cy += stepY
      crossed = 'horizontal'
    }
    const material = cellAt(w, cx + 0.5, cy + 0.5)
    out.steps.push({ t, crossed, line, cell: [cx, cy], nextX, nextY, material })
    if (material !== 0) break
  }
  return out
}
