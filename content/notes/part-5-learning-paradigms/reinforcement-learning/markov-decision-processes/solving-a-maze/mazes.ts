/**
 * The maze explorer's layouts: the preset mazes of aifn `MAZES` (rows of text, top row first: `#` wall, `.` open, `S`
 * start, `G` goal, `T` trap) and the walls a reader adds or removes by clicking, as cell indices s = y · width + x with
 * y = 0 at the bottom row.
 */
import { MAZES } from 'aifn-methods/gym/environments'

export type MazeName = 'room' | 'corridors' | 'cliff' | 'routes'

export const MAZE_OPTIONS: { value: MazeName; label: string }[] = [
  { value: 'routes', label: 'two routes' },
  { value: 'cliff', label: 'cliff edge' },
  { value: 'corridors', label: 'corridors' },
  { value: 'room', label: 'open room' },
]

const at = (rows: readonly string[], s: number) => {
  const width = rows[0].length
  return rows[rows.length - 1 - Math.floor(s / width)][s % width]
}

/** The preset's walls as sorted cell indices. */
export function presetWalls(name: MazeName): number[] {
  const rows = MAZES[name]
  const n = rows.length * rows[0].length
  return Array.from({ length: n }, (_, s) => s).filter((s) => at(rows, s) === '#')
}

/** The preset's rows with its walls replaced by `walls`. */
export function mazeRows(name: MazeName, walls: readonly number[]): string[] {
  const rows = MAZES[name]
  const width = rows[0].length
  const set = new Set(walls)
  return rows.map((r, i) =>
    [...r]
      .map((ch, x) => {
        const s = (rows.length - 1 - i) * width + x
        return set.has(s) ? '#' : ch === '#' ? '.' : ch
      })
      .join(''),
  )
}

/**
 * The walls after toggling cell (x, y), or null when the toggle is refused: outside the grid, on the start, the goal or
 * a trap, or a wall that would cut the start off from the goal (a trap returns the agent but can be crossed).
 */
export function wallsConnect(name: MazeName, walls: readonly number[], x: number, y: number): number[] | null {
  const rows = MAZES[name]
  const width = rows[0].length
  const height = rows.length
  if (x < 0 || y < 0 || x >= width || y >= height) return null
  const s = y * width + x
  if ('SGT'.includes(at(rows, s))) return null
  const next = new Set(walls)
  if (next.has(s)) next.delete(s)
  else next.add(s)
  const cells = mazeRows(name, [...next])
  const start = cells.join('').indexOf('S')
  const sx = start % width
  const sy = height - 1 - Math.floor(start / width)
  const seen = new Set([sy * width + sx])
  const queue = [sy * width + sx]
  while (queue.length) {
    const c = queue.pop()!
    const cx = c % width
    const cy = Math.floor(c / width)
    for (const [dx, dy] of [
      [0, 1],
      [1, 0],
      [0, -1],
      [-1, 0],
    ]) {
      const nx = cx + dx
      const ny = cy + dy
      const t = ny * width + nx
      if (nx < 0 || ny < 0 || nx >= width || ny >= height || seen.has(t) || at(cells, t) === '#') continue
      if (at(cells, t) === 'G') return [...next].sort((a, b) => a - b)
      seen.add(t)
      if (at(cells, t) !== 'T') queue.push(t)
    }
  }
  return null
}
