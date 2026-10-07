/**
 * Grid worlds and ray casting for the games notes. A world is a grid of square cells of side 1, each empty (0) or a wall
 * of one of four materials (1 to 4). Positions are continuous: x grows to the right and y grows downwards (the map is
 * drawn as it is stored), and a heading θ is measured from the +x axis towards +y. Rays are traced through the grid with
 * the digital differential analyser of Amanatides and Woo (1987), which visits exactly the cells a ray crosses.
 */

export type Material = 0 | 1 | 2 | 3 | 4

export type Pose = { x: number; y: number; theta: number }

export type World = {
  id: string
  label: string
  width: number
  height: number
  /** Row-major cells, `cells[y * width + x]`. */
  cells: Uint8Array
  start: Pose
}

/** Where a ray meets a wall. */
export type Hit = {
  /** Euclidean distance from the eye to the hit, along the ray. */
  distance: number
  x: number
  y: number
  /** 0 when the ray crossed a vertical grid line (an east or west face), 1 for a horizontal one (north or south). */
  side: 0 | 1
  material: Material
  /** Position along the face, in [0, 1): the texture column. */
  u: number
}

/** The longest ray traced; rays that reach it report this distance (the worlds are closed, so none should). */
export const MAX_DISTANCE = 40

const MAPS: { id: string; label: string; rows: string[]; theta: number }[] = [
  {
    id: 'room',
    label: 'Small room (6 × 6)',
    theta: -Math.PI / 4,
    rows: ['111111', '1....1', '1....1', '1....1', '1P...1', '111111'],
  },
  {
    id: 'pillar',
    label: 'One pillar (8 × 8)',
    theta: -Math.PI / 4,
    rows: ['11111111', '1......1', '1......1', '1..22..1', '1..22..1', '1......1', '1P.....1', '11111111'],
  },
  {
    id: 'tiny-maze',
    label: 'Tiny maze (7 × 7)',
    theta: 0,
    rows: ['1111111', '1P..2.1', '122.2.1', '1...2.1', '1.222.1', '1.....1', '1111111'],
  },
  {
    id: 'small-maze',
    label: 'Small maze (9 × 9)',
    theta: 0,
    rows: [
      '111111111',
      '1P....2.1',
      '12222.2.1',
      '1...2.2.1',
      '122.2.2.1',
      '1...2...1',
      '1.22222.1',
      '1.......1',
      '111111111',
    ],
  },
  {
    id: 'columns',
    label: 'Hall of columns',
    theta: 0,
    rows: [
      '1111111111111111',
      '1..............1',
      '1..2...2...2...1',
      '1..............1',
      '1..............1',
      '1..2...2...2...1',
      '1..............1',
      '1.P............1',
      '1..............1',
      '1..2...2...2...1',
      '1..............1',
      '1..............1',
      '1..2...2...2...1',
      '1..............1',
      '1..............1',
      '1111111111111111',
    ],
  },
  {
    id: 'courtyard',
    label: 'Courtyard',
    theta: -Math.PI / 4,
    rows: [
      '1111111111111111',
      '1..............1',
      '1.4...4..4...4.1',
      '1..............1',
      '1..222....222..1',
      '1..2........2..1',
      '1.4...3333...4.1',
      '1.....3333.....1',
      '1.....3333.....1',
      '1.4...3333...4.1',
      '1..2........2..1',
      '1..222....222..1',
      '1..............1',
      '1.4...4..4...4.1',
      '1P.............1',
      '1111111111111111',
    ],
  },
  {
    id: 'rooms',
    label: 'Rooms and doors',
    theta: -Math.PI / 2,
    rows: [
      '1111111111111111',
      '1....2......2..1',
      '1....2......2..1',
      '1....2..4...2..1',
      '1.........4....1',
      '122.22222.222221',
      '1......2.......1',
      '1..3...2...3...1',
      '1......2.......1',
      '1..3...........1',
      '12222.222222.221',
      '1....2......4..1',
      '1.P..2.........1',
      '1.......4...4..1',
      '1....2.........1',
      '1111111111111111',
    ],
  },
  {
    id: 'cloister',
    label: 'Cloister',
    theta: -Math.PI / 4,
    rows: [
      '1111111111111111',
      '1..............1',
      '1......44......1',
      '1...4......4...1',
      '1..............1',
      '1..4..3..3..4..1',
      '1..............1',
      '1.4...3333...4.1',
      '1.4...3333...4.1',
      '1..............1',
      '1..4..3..3..4..1',
      '1..............1',
      '1...4......4...1',
      '1......44......1',
      '1P.............1',
      '1111111111111111',
    ],
  },
  {
    id: 'maze',
    label: 'Maze',
    theta: 0,
    rows: [
      '111111111111111',
      '1P..2.......2.1',
      '122.2.2.222.2.1',
      '1...2.2.2.2.2.1',
      '1.222.2.2.2.2.1',
      '1.2...2.2.2.2.1',
      '1.24444.2.2.2.1',
      '1.2.....2...2.1',
      '1.2.22222.222.1',
      '1.2...4...2...1',
      '1.2.2.4.222.2.1',
      '1...2.4.....2.1',
      '122.2.4444444.1',
      '1.....4.......1',
      '111111111111111',
    ],
  },
]

function parse({ id, label, rows, theta }: (typeof MAPS)[number]): World {
  const height = rows.length
  const width = rows[0].length
  const cells = new Uint8Array(width * height)
  let start: Pose = { x: 1.5, y: 1.5, theta }
  rows.forEach((row, y) =>
    [...row].forEach((c, x) => {
      if (c === 'P') start = { x: x + 0.5, y: y + 0.5, theta }
      else if (c >= '1' && c <= '4') cells[y * width + x] = Number(c)
    }),
  )
  return { id, label, width, height, cells, start }
}

export const WORLDS: readonly World[] = MAPS.map(parse)

export const worldById = (id: string): World => WORLDS.find((w) => w.id === id) ?? WORLDS[0]

export function cellAt(w: World, x: number, y: number): Material {
  const i = Math.floor(x)
  const j = Math.floor(y)
  if (i < 0 || j < 0 || i >= w.width || j >= w.height) return 1
  return w.cells[j * w.width + i] as Material
}

/** True when a disc of radius r at (x, y) touches no wall. */
export function isFree(w: World, x: number, y: number, r = 0.2): boolean {
  for (const dx of [-r, r]) for (const dy of [-r, r]) if (cellAt(w, x + dx, y + dy) !== 0) return false
  return true
}

/** Trace one ray from (x, y) along angle `angle` to the first wall. */
export function castRay(w: World, x: number, y: number, angle: number): Hit {
  const dx = Math.cos(angle)
  const dy = Math.sin(angle)
  let cx = Math.floor(x)
  let cy = Math.floor(y)
  const stepX = dx < 0 ? -1 : 1
  const stepY = dy < 0 ? -1 : 1
  // Distance along the ray between successive vertical (x) and horizontal (y) grid lines.
  const deltaX = dx === 0 ? Infinity : Math.abs(1 / dx)
  const deltaY = dy === 0 ? Infinity : Math.abs(1 / dy)
  // Distance along the ray to the first vertical and horizontal grid line.
  let nextX = dx < 0 ? (x - cx) * deltaX : (cx + 1 - x) * deltaX
  let nextY = dy < 0 ? (y - cy) * deltaY : (cy + 1 - y) * deltaY
  let side: 0 | 1 = 0
  let t = 0
  while (t < MAX_DISTANCE) {
    if (nextX < nextY) {
      t = nextX
      nextX += deltaX
      cx += stepX
      side = 0
    } else {
      t = nextY
      nextY += deltaY
      cy += stepY
      side = 1
    }
    const m = cellAt(w, cx + 0.5, cy + 0.5)
    if (m !== 0) {
      const hx = x + t * dx
      const hy = y + t * dy
      let u = side === 0 ? hy - Math.floor(hy) : hx - Math.floor(hx)
      // Flip so the texture reads left to right on every face.
      if ((side === 0 && dx < 0) || (side === 1 && dy > 0)) u = 1 - u
      return { distance: t, x: hx, y: hy, side, material: m, u }
    }
  }
  return { distance: MAX_DISTANCE, x: x + MAX_DISTANCE * dx, y: y + MAX_DISTANCE * dy, side: 0, material: 1, u: 0 }
}

/**
 * The angle of each of R rays across a field of view `fov` (radians), relative to the heading. Rays pass through
 * equally spaced points of a flat projection plane, as screen columns do, so they are denser towards the edges in angle:
 * offset_i = atan((2 (i + ½) / R − 1) tan(fov / 2)).
 */
export function rayOffsets(R: number, fov: number): Float64Array {
  const half = Math.tan(fov / 2)
  return Float64Array.from({ length: R }, (_, i) => Math.atan(((2 * (i + 0.5)) / R - 1) * half))
}

/** Cast the fan of rays a camera at `pose` sees. */
export function castFan(w: World, pose: Pose, offsets: Float64Array): Hit[] {
  return Array.from(offsets, (o) => castRay(w, pose.x, pose.y, pose.theta + o))
}

/** Move by `forward` along the heading and `strafe` across it, sliding along walls one axis at a time. */
export function move(w: World, p: Pose, forward: number, strafe: number, turn: number): Pose {
  const theta = p.theta + turn
  const c = Math.cos(theta)
  const s = Math.sin(theta)
  const dx = forward * c - strafe * s
  const dy = forward * s + strafe * c
  let { x, y } = p
  if (isFree(w, x + dx, y)) x += dx
  if (isFree(w, x, y + dy)) y += dy
  return { x, y, theta }
}

/** The centre of every empty cell, for placing the camera. */
export function freeCells(w: World): { x: number; y: number }[] {
  const out: { x: number; y: number }[] = []
  for (let y = 0; y < w.height; y++)
    for (let x = 0; x < w.width; x++) if (w.cells[y * w.width + x] === 0) out.push({ x: x + 0.5, y: y + 0.5 })
  return out
}
