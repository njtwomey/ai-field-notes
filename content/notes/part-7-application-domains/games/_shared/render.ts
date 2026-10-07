/**
 * Canvas drawing for the games notes: the first-person view built from one wall slice per ray, and the top-down map
 * with the camera, its rays and their hits. Drawing is imperative (a game loop redraws every frame), so these take a
 * 2D context and plain data rather than being React components.
 */

import type { Hit, Pose, World } from './world'

const TEX = 64

/** Base colour of each wall material (1 stone, 2 brick, 3 wood, 4 metal), used on the map and for the textures. */
export const MATERIAL_COLOURS = ['', '#7d828c', '#9c4f3a', '#8a6a3c', '#46668c'] as const

/** A small integer hash in [0, 1), so the textures are the same on every load. */
function hash(x: number, y: number, k: number): number {
  let h = (x * 374761393 + y * 668265263 + k * 2147483647) | 0
  h = Math.imul(h ^ (h >>> 13), 1274126177)
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296
}

function shade(hex: string, f: number): [number, number, number] {
  const n = parseInt(hex.slice(1), 16)
  return [((n >> 16) & 255) * f, ((n >> 8) & 255) * f, (n & 255) * f]
}

/** One 64 × 64 texture per material: stone blocks, bricks, wooden planks and riveted metal panels. */
function makeTexture(material: number): HTMLCanvasElement {
  const c = document.createElement('canvas')
  c.width = TEX
  c.height = TEX
  const ctx = c.getContext('2d')!
  const img = ctx.createImageData(TEX, TEX)
  const base = MATERIAL_COLOURS[material]
  for (let y = 0; y < TEX; y++)
    for (let x = 0; x < TEX; x++) {
      let f = 0.85 + 0.25 * hash(x, y, material)
      if (material === 1) {
        // Stone: 32 × 21 blocks, offset every other row, with dark joints.
        const row = Math.floor(y / 21)
        const bx = (x + (row % 2) * 16) % 32
        if (y % 21 < 2 || bx < 2) f = 0.45
        else f *= 0.9 + 0.2 * hash(Math.floor((x + (row % 2) * 16) / 32), row, 9)
      } else if (material === 2) {
        // Brick: 16 × 8 bricks with light mortar.
        const row = Math.floor(y / 8)
        const bx = (x + (row % 2) * 8) % 16
        if (y % 8 === 0 || bx === 0) f = 1.35
        else f *= 0.85 + 0.3 * hash(Math.floor((x + (row % 2) * 8) / 16), row, 3)
      } else if (material === 3) {
        // Wood: vertical planks 16 wide with grain.
        if (x % 16 === 0) f = 0.5
        else f *= 0.9 + 0.12 * Math.sin(y * 0.35 + 3 * hash(Math.floor(x / 16), 0, 5) + x * 0.7)
      } else {
        // Metal: 32 × 32 panels with seams and rivets.
        const px = x % 32
        const py = y % 32
        if (px === 0 || py === 0) f = 0.55
        else if ((px - 4) ** 2 + (py - 4) ** 2 < 4 || (px - 28) ** 2 + (py - 28) ** 2 < 4) f = 1.4
        else f = 0.95 + 0.08 * hash(x, y, 4)
      }
      const [r, g, b] = shade(base, f)
      const i = 4 * (y * TEX + x)
      img.data[i] = Math.min(255, r)
      img.data[i + 1] = Math.min(255, g)
      img.data[i + 2] = Math.min(255, b)
      img.data[i + 3] = 255
    }
  ctx.putImageData(img, 0, 0)
  return c
}

let textures: HTMLCanvasElement[] | null = null
const getTextures = () =>
  (textures ??= [0, 1, 2, 3, 4].map((m) => (m === 0 ? document.createElement('canvas') : makeTexture(m))))

export type ViewOptions = {
  fov: number
  /** Ray angles relative to the heading (`rayOffsets`). */
  offsets: Float64Array
  /** 'texture' draws the walls' materials; 'depth' colours each slice by its distance alone. */
  mode: 'texture' | 'depth'
  /** Use the Euclidean distance for wall height, which bends straight walls (the fish-eye effect). */
  fisheye?: boolean
  dark: boolean
}

/** Colour for a distance in depth mode: near walls light, far ones dark, as a single-hue scale. */
export function depthColour(d: number, dark: boolean): string {
  const t = Math.min(1, Math.max(0, d / 14))
  const near = dark ? [205, 226, 251] : [42, 120, 214]
  const far = dark ? [13, 54, 107] : [205, 226, 251]
  const c = near.map((v, i) => Math.round(v + (far[i] - v) * t))
  return `rgb(${c[0]},${c[1]},${c[2]})`
}

/**
 * The first-person view: ceiling, floor and one vertical wall slice per ray. A slice at perpendicular distance p is
 * f / p pixels tall, where f = (W / 2) / tan(fov / 2) is the focal length in pixels and walls are one unit high.
 * `distances` gives each ray's Euclidean distance; hits (when given) supply material, side and texture column.
 */
export function drawView(
  ctx: CanvasRenderingContext2D,
  W: number,
  H: number,
  distances: ArrayLike<number>,
  hits: readonly Hit[] | null,
  o: ViewOptions,
) {
  const R = o.offsets.length
  const sky = ctx.createLinearGradient(0, 0, 0, H / 2)
  sky.addColorStop(0, o.dark ? '#26272b' : '#5d6470')
  sky.addColorStop(1, o.dark ? '#141416' : '#2f333a')
  ctx.fillStyle = sky
  ctx.fillRect(0, 0, W, H / 2)
  const floor = ctx.createLinearGradient(0, H / 2, 0, H)
  floor.addColorStop(0, o.dark ? '#121212' : '#3a3632')
  floor.addColorStop(1, o.dark ? '#3a3632' : '#77706a')
  ctx.fillStyle = floor
  ctx.fillRect(0, H / 2, W, H / 2)

  const focal = W / 2 / Math.tan(o.fov / 2)
  const tex = getTextures()
  ctx.imageSmoothingEnabled = false
  const colW = W / R
  for (let i = 0; i < R; i++) {
    const d = Math.max(1e-3, distances[i])
    const perp = o.fisheye ? d : d * Math.cos(o.offsets[i])
    const h = focal / Math.max(perp, 0.05)
    const top = H / 2 - h / 2
    const x0 = Math.floor(i * colW)
    const x1 = Math.floor((i + 1) * colW)
    const hit = hits?.[i]
    if (o.mode === 'texture' && hit) {
      const u = Math.min(TEX - 1, Math.floor(hit.u * TEX))
      ctx.drawImage(tex[hit.material], u, 0, 1, TEX, x0, top, x1 - x0, h)
      // Faces across horizontal grid lines are darker, as in Wolfenstein 3D; distant walls fade.
      const fog = Math.min(0.8, perp / 18) + (hit.side === 1 ? 0.25 : 0)
      ctx.fillStyle = `rgba(0,0,0,${Math.min(0.85, fog)})`
      ctx.fillRect(x0, top, x1 - x0, h)
    } else {
      ctx.fillStyle = depthColour(d, o.dark)
      ctx.fillRect(x0, top, x1 - x0, h)
    }
  }
}

export type RaySet = {
  /** Euclidean distance of each ray. */
  distances: ArrayLike<number>
  colour: string
  /** Draw the rays as lines (else only their end points). */
  lines?: boolean
}

export type MapOptions = {
  offsets: Float64Array
  dark: boolean
  ink: string
  grid: string
  surface: string
}

/** The map from above, scaled to fit `size` pixels, with each ray set drawn from the camera. */
export function drawMap(
  ctx: CanvasRenderingContext2D,
  size: number,
  w: World,
  pose: Pose,
  rays: readonly RaySet[],
  o: MapOptions,
) {
  const s = size / Math.max(w.width, w.height)
  ctx.fillStyle = o.surface
  ctx.fillRect(0, 0, size, size)
  for (let y = 0; y < w.height; y++)
    for (let x = 0; x < w.width; x++) {
      const m = w.cells[y * w.width + x]
      ctx.fillStyle = m === 0 ? o.surface : MATERIAL_COLOURS[m]
      ctx.fillRect(x * s, y * s, s + 0.5, s + 0.5)
    }
  ctx.strokeStyle = o.grid
  ctx.lineWidth = 1
  ctx.beginPath()
  for (let x = 0; x <= w.width; x++) {
    ctx.moveTo(x * s + 0.5, 0)
    ctx.lineTo(x * s + 0.5, w.height * s)
  }
  for (let y = 0; y <= w.height; y++) {
    ctx.moveTo(0, y * s + 0.5)
    ctx.lineTo(w.width * s, y * s + 0.5)
  }
  ctx.stroke()

  const px = pose.x * s
  const py = pose.y * s
  for (const set of rays) {
    const ends = Array.from(set.distances, (d, i) => {
      const a = pose.theta + o.offsets[i]
      return [px + d * s * Math.cos(a), py + d * s * Math.sin(a)]
    })
    if (set.lines) {
      ctx.strokeStyle = set.colour
      ctx.globalAlpha = 0.35
      ctx.beginPath()
      // At most about 60 lines, so a dense fan still shows its rays.
      const every = Math.max(1, Math.round(ends.length / 60))
      for (const [k, [ex, ey]] of ends.entries()) {
        if (k % every !== 0 && k !== ends.length - 1) continue
        ctx.moveTo(px, py)
        ctx.lineTo(ex, ey)
      }
      ctx.stroke()
      ctx.globalAlpha = 1
    }
    ctx.fillStyle = set.colour
    for (const [ex, ey] of ends) {
      ctx.beginPath()
      ctx.arc(ex, ey, Math.max(1.5, s * 0.09), 0, 2 * Math.PI)
      ctx.fill()
    }
  }

  // The camera: a dot with a heading arrow.
  ctx.fillStyle = o.ink
  ctx.strokeStyle = o.ink
  ctx.lineWidth = 2
  ctx.beginPath()
  ctx.arc(px, py, Math.max(3, s * 0.2), 0, 2 * Math.PI)
  ctx.fill()
  ctx.beginPath()
  ctx.moveTo(px, py)
  ctx.lineTo(px + 0.7 * s * Math.cos(pose.theta), py + 0.7 * s * Math.sin(pose.theta))
  ctx.stroke()
}

/** Size a canvas's backing store for the device pixel ratio and return its context scaled to CSS pixels. */
export function prepareCanvas(c: HTMLCanvasElement, w: number, h: number): CanvasRenderingContext2D {
  const r = window.devicePixelRatio || 1
  if (c.width !== Math.round(w * r) || c.height !== Math.round(h * r)) {
    c.width = Math.round(w * r)
    c.height = Math.round(h * r)
  }
  const ctx = c.getContext('2d')!
  ctx.setTransform(r, 0, 0, r, 0, 0)
  return ctx
}
