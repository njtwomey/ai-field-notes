/**
 * The Hough transform for lines on small synthetic images. Coordinates are pixels with the origin at the image centre
 * and y pointing up, so a line is x cos θ + y sin θ = ρ with θ in [0°, 180°) and |ρ| at most the half-diagonal.
 */
import { gradients, type Image } from '../_shared/image'
import { normal, stream, uniform } from 'aifn-compute/foundation/random'

export const N = 128
const HALF = (N - 1) / 2
/** Pixel-centre coordinates along either axis, from -63.5 to 63.5. */
export const AXIS = Array.from({ length: N }, (_, i) => i - HALF)
/** Half-diagonal of the image: the largest |ρ| of any line that meets it. */
export const R_MAX = Math.hypot(N / 2, N / 2)

export type Scene = 'grid' | 'corridor' | 'house' | 'circles' | 'points'

/** An edge point, with its gradient direction mod 180° in degrees when it came from an image. */
export type EdgePoint = { x: number; y: number; phi?: number }

/** The vanishing point of the corridor scene: every edge of that image lies on a line through it. */
export const VANISHING_POINT: [number, number] = [6, 12]

const DEG = Math.PI / 180

/** Intensity of each scene at (x, y), in [0, 1]. */
function shade(scene: Exclude<Scene, 'points'>, x: number, y: number): number {
  switch (scene) {
    case 'grid': {
      // A 6 × 6 chessboard of 16-pixel squares rotated by 15°, on mid-grey.
      const a = 15 * DEG
      const u = x * Math.cos(a) + y * Math.sin(a)
      const v = -x * Math.sin(a) + y * Math.cos(a)
      if (Math.abs(u) >= 48 || Math.abs(v) >= 48) return 0.5
      return (Math.floor((u + 48) / 16) + Math.floor((v + 48) / 16)) % 2 ? 0.85 : 0.15
    }
    case 'corridor': {
      // Wedges of ceiling, walls, floor and a runner, all bounded by rays from the vanishing point.
      const [vx, vy] = VANISHING_POINT
      const phi = (((Math.atan2(y - vy, x - vx) / DEG) % 360) + 360) % 360
      if (phi >= 30 && phi < 150) return 0.25
      if (phi >= 150 && phi < 235) return 0.55
      if (phi >= 260 && phi < 280) return 0.8
      if (phi >= 235 && phi < 305) return 0.4
      return 0.65
    }
    case 'house': {
      if (x > -8 && x < 8 && y > -42 && y < -14) return 0.2
      if (x > 14 && x < 26 && y > -10 && y < 2) return 0.9
      if (x > -34 && x < 34 && y > -42 && y < 8) return 0.6
      if (y >= 8 && y < 44 - (Math.abs(x) * 36) / 44) return 0.3
      return y < -42 ? 0.35 : 0.8
    }
    case 'circles': {
      if ((x - 28) ** 2 + (y - 10) ** 2 < 24 ** 2) return 0.85
      if ((x + 22) ** 2 + (y + 34) ** 2 < 18 ** 2) return 0.85
      // One straight edge: the half-plane beyond the line θ = 120°, ρ = 28.
      return x * Math.cos(120 * DEG) + y * Math.sin(120 * DEG) > 28 ? 0.55 : 0.2
    }
  }
}

/** The scene's image with seeded Gaussian pixel noise; row r holds y = r - 63.5, so row 0 is the bottom. */
export function sceneImage(scene: Exclude<Scene, 'points'>, noise: number, seed: number): Image {
  const g = stream(seed)
  return AXIS.map((y) => AXIS.map((x) => shade(scene, x, y) + noise * normal(g)))
}

/**
 * Edge points of an image: Sobel gradient after Gaussian smoothing (σ = 1), thinned to one pixel by non-maximum
 * suppression along the gradient, then kept where the magnitude is at least `threshold` times its maximum.
 */
export function edgePoints(img: Image, threshold: number): EdgePoint[] {
  const { gx, gy } = gradients(img, 1)
  const mag = gx.map((row, r) => row.map((v, c) => Math.hypot(v, gy[r][c])))
  let top = 0
  for (const row of mag) for (const m of row) top = Math.max(top, m)
  const at = (r: number, c: number) => (r >= 0 && r < N && c >= 0 && c < N ? mag[r][c] : 0)
  const out: EdgePoint[] = []
  for (let r = 0; r < N; r++)
    for (let c = 0; c < N; c++) {
      const m = mag[r][c]
      if (m < threshold * top || m === 0) continue
      const phi = (((Math.atan2(gy[r][c], gx[r][c]) / DEG) % 180) + 180) % 180
      // Neighbours along the gradient, quantised to 0°, 45°, 90° or 135°.
      const q = Math.round(phi / 45) % 4
      const [dr, dc] = [
        [0, 1],
        [1, 1],
        [1, 0],
        [1, -1],
      ][q]
      if (m >= at(r + dr, c + dc) && m > at(r - dr, c - dc)) out.push({ x: AXIS[c], y: AXIS[r], phi })
    }
  return out
}

/** Three noisy line segments and uniform clutter, seeded. `noise` is the perpendicular scatter in pixels. */
export function pointCloud(noise: number, clutter: number, seed: number): EdgePoint[] {
  const g = stream(seed)
  const out: EdgePoint[] = []
  const thetas: number[] = []
  while (thetas.length < 3) {
    const t = 180 * uniform(g)
    // Keep the three directions at least 25° apart (mod 180°), so the lines are distinct.
    if (thetas.every((s) => Math.min(Math.abs(t - s), 180 - Math.abs(t - s)) > 25)) thetas.push(t)
  }
  for (const t of thetas) {
    const rho = -35 + 70 * uniform(g)
    // |ρ| ≤ 35 always meets the square of half-width 58, so the chord exists.
    const [a, b] = clipLine(t, rho, 58)!
    const n = [Math.cos(t * DEG), Math.sin(t * DEG)]
    // A sub-segment covering 50–75% of the chord, with 40 points spread along it.
    const lo = 0.25 * uniform(g)
    const hi = lo + 0.5 + 0.25 * uniform(g)
    for (let i = 0; i < 40; i++) {
      const s = lo + (hi - lo) * uniform(g)
      const e = noise * normal(g)
      out.push({ x: a[0] + s * (b[0] - a[0]) + e * n[0], y: a[1] + s * (b[1] - a[1]) + e * n[1] })
    }
  }
  for (let i = 0; i < clutter; i++) out.push({ x: (uniform(g) - 0.5) * (N - 1), y: (uniform(g) - 0.5) * (N - 1) })
  return out.filter((p) => Math.abs(p.x) <= HALF && Math.abs(p.y) <= HALF)
}

/** The points as a binary image, for display. */
export function rasterise(points: EdgePoint[]): Image {
  const img: Image = AXIS.map(() => AXIS.map(() => 0))
  for (const p of points) img[Math.round(p.y + HALF)][Math.round(p.x + HALF)] = 1
  return img
}

/** The two ends of the line x cos θ + y sin θ = ρ inside the square [-h, h]², or null if it misses the square. */
export function clipLine(thetaDeg: number, rho: number, h = N / 2): [[number, number], [number, number]] | null {
  const c = Math.cos(thetaDeg * DEG)
  const s = Math.sin(thetaDeg * DEG)
  const hits: [number, number][] = []
  const eps = 1e-9
  if (Math.abs(s) > eps)
    for (const x of [-h, h]) {
      const y = (rho - x * c) / s
      if (Math.abs(y) <= h + eps) hits.push([x, y])
    }
  if (Math.abs(c) > eps)
    for (const y of [-h, h]) {
      const x = (rho - y * s) / c
      if (Math.abs(x) <= h + eps) hits.push([x, y])
    }
  if (hits.length < 2) return null
  // Corners are hit twice; the two points farthest apart are the chord's ends.
  let best: [[number, number], [number, number]] = [hits[0], hits[1]]
  let d = -1
  for (let i = 0; i < hits.length; i++)
    for (let j = i + 1; j < hits.length; j++) {
      const dd = Math.hypot(hits[i][0] - hits[j][0], hits[i][1] - hits[j][1])
      if (dd > d) {
        d = dd
        best = [hits[i], hits[j]]
      }
    }
  return d > 0 ? best : null
}

export type Grid = {
  /** θ of each column in degrees: 0, Δθ, 2Δθ, … */
  thetas: number[]
  /** ρ at the centre of each row, in pixels. */
  rhos: number[]
  dTheta: number
  dRho: number
  cos: number[]
  sin: number[]
}

export function grid(dTheta: number, dRho: number): Grid {
  const nT = Math.round(180 / dTheta)
  const nR = Math.ceil((2 * R_MAX) / dRho)
  const thetas = Array.from({ length: nT }, (_, j) => j * dTheta)
  const rhos = Array.from({ length: nR }, (_, k) => -(nR * dRho) / 2 + (k + 0.5) * dRho)
  return {
    thetas,
    rhos,
    dTheta,
    dRho,
    cos: thetas.map((t) => Math.cos(t * DEG)),
    sin: thetas.map((t) => Math.sin(t * DEG)),
  }
}

/** Row index of the ρ bin that holds `rho`. */
export const rhoBin = (g: Grid, rho: number) =>
  Math.min(g.rhos.length - 1, Math.max(0, Math.floor((rho + (g.rhos.length * g.dRho) / 2) / g.dRho)))

/** The θ columns a point votes in: all of them, or only those within `window` degrees of its gradient direction. */
export function votingColumns(g: Grid, p: EdgePoint, window: number | null): number[] {
  const nT = g.thetas.length
  if (window === null || p.phi === undefined) return g.thetas.map((_, j) => j)
  const centre = Math.round(p.phi / g.dTheta)
  const w = Math.ceil(window / g.dTheta)
  const cols: number[] = []
  for (let d = -w; d <= w; d++) cols.push((((centre + d) % nT) + nT) % nT)
  return [...new Set(cols)]
}

/** The accumulator, row-major as acc[ρ bin][θ column], and the number of votes cast. */
export function accumulate(g: Grid, points: EdgePoint[], window: number | null): { acc: number[][]; votes: number } {
  const acc = g.rhos.map(() => new Array<number>(g.thetas.length).fill(0))
  let votes = 0
  for (const p of points)
    for (const j of votingColumns(g, p, window)) {
      acc[rhoBin(g, p.x * g.cos[j] + p.y * g.sin[j])][j]++
      votes++
    }
  return { acc, votes }
}

export type Peak = { theta: number; rho: number; votes: number }

/**
 * Up to `k` peaks by greedy non-maximum suppression: take the highest cell, discard every cell within the window of
 * it, repeat. The window wraps at θ = 180°, where (θ, ρ) and (θ - 180°, -ρ) are the same line.
 */
export function findPeaks(g: Grid, acc: number[][], k: number, minVotes = 3): Peak[] {
  const cells: Peak[] = []
  acc.forEach((row, i) =>
    row.forEach((v, j) => {
      if (v >= minVotes) cells.push({ theta: g.thetas[j], rho: g.rhos[i], votes: v })
    }),
  )
  cells.sort((a, b) => b.votes - a.votes)
  const wT = Math.max(5, 2 * g.dTheta)
  const wR = Math.max(5, 2 * g.dRho)
  const near = (a: Peak, b: Peak) => {
    const dt = Math.abs(a.theta - b.theta)
    return (dt < wT && Math.abs(a.rho - b.rho) < wR) || (180 - dt < wT && Math.abs(a.rho + b.rho) < wR)
  }
  const peaks: Peak[] = []
  for (const c of cells) {
    if (peaks.length >= k) break
    if (!peaks.some((p) => near(p, c))) peaks.push(c)
  }
  return peaks
}
