/**
 * Light rays around a Schwarzschild (non-spinning) black hole, in units with G = c = M = 1: lengths in units of
 * GM/c², so the horizon is at r = 2.
 *
 * - A ray in its orbital plane obeys the orbit equation u'' = 3u² − u in u = 1/r, with the azimuth φ as the
 *   independent variable and state (u, w = du/dφ). The first integral w² = 1/b² − u² + 2u³ fixes the impact
 *   parameter b. The engine integrates (`rungeKutta`, `dormandPrince`) and locates the escape (u = 0), the horizon
 *   (u = 1/2) and the closest approach (w = 0) with `withEvents`.
 * - The lensed sky: a table of the azimuth Φ(b) a ray sweeps from an observer at r = R_OBS to infinity, splined in
 *   ln(b − b_c) with the engine's `cubicSpline`, so each pixel is one lookup.
 * - The thin disc: a table of u(φ) per b on a uniform φ grid (RK4 steps), so each pixel's crossing radius with the
 *   equatorial plane is a table lookup. Brightness from the Page–Thorne flux and the redshift factor of circular
 *   Keplerian emitters, both as given by Luminet (1979), eqs. (15) and (19).
 */
import {
  dormandPrince,
  rungeKutta,
  withEvents,
  type EventState,
  type OdeEvent,
  type OdeState,
  type Rhs,
} from 'aifn-compute/dynamics/ode'
import { child, stream, uniform } from 'aifn-compute/foundation/random'
import { tensor, toFlat } from 'aifn-compute/foundation/tensor'
import { run, trace } from 'aifn-compute/foundation/trace'
import { cubicSpline, evaluatePiecewise, type PiecewisePolynomial } from 'aifn-compute/numerics/interpolate'
import { findRoot } from 'aifn-compute/numerics/roots'

/** The event horizon, r = 2M. */
export const HORIZON = 2
/** The photon sphere, r = 3M: the unstable circular orbit of light. */
export const PHOTON_SPHERE = 3
/** The innermost stable circular orbit of a massive particle, r = 6M: the inner edge of the disc. */
export const ISCO = 6
/** The critical impact parameter b_c = 3√3 M ≈ 5.196: rays with b < b_c fall in. */
export const B_CRIT = 3 * Math.sqrt(3)
/** Luminet's (1979, eq. 7) strong-field constant: b − b_c ≈ 3.4823 e^(−α) for a deflection α. */
export const STRONG_FIELD_SCALE = 3.4823

/** The orbit equation as a first-order system in φ: (u, w)' = (w, 3u² − u). */
export const orbitRhs: Rhs = (_phi, x) => {
  const [u, w] = toFlat(x)
  return [w, 3 * u * u - u]
}

/** The first integral's residual w² − (1/b² − u² + 2u³); zero along an exact ray. */
export const firstIntegralResidual = (u: number, w: number, b: number) => w * w - (1 / (b * b) - u * u + 2 * u ** 3)

/** The weak-field deflection 4M/b. */
export const weakDeflection = (b: number) => 4 / b

/** The strong-field deflection ln(3.4823 / (b − b_c)) for b just above b_c (Luminet 1979, eq. 7). */
export const strongDeflection = (b: number) => (b > B_CRIT ? Math.log(STRONG_FIELD_SCALE / (b - B_CRIT)) : NaN)

const ESCAPE: OdeEvent = { name: 'escape', g: (_t, x) => toFlat(x)[0], direction: -1, terminal: true }
const CAPTURE: OdeEvent = { name: 'horizon', g: (_t, x) => toFlat(x)[0] - 1 / HORIZON, direction: 1, terminal: true }
const PERIAPSIS: OdeEvent = { name: 'periapsis', g: (_t, x) => toFlat(x)[1], direction: -1 }

export type Fate = 'escapes' | 'falls in' | 'still orbiting'

export type Ray = {
  b: number
  /** Azimuth φ from the incoming direction and u = 1/r at each step. */
  phi: number[]
  u: number[]
  fate: Fate
  /** The azimuth at the end: the escape or the horizon, or φ_max. */
  phiEnd: number
  /** The deflection α = φ_escape − π (NaN unless the ray escapes). */
  deflection: number
  /** The closest approach r₀ (NaN for a ray that falls in). */
  closest: number
  /** The largest |w² − (1/b² − u² + 2u³)| b² along the ray: the integrator's error in the first integral. */
  residual: number
}

function fateOf(events: { name: string }[]): Fate {
  if (events.some((e) => e.name === 'escape')) return 'escapes'
  if (events.some((e) => e.name === 'horizon')) return 'falls in'
  return 'still orbiting'
}

/**
 * A ray from infinity with impact parameter b > 0, integrated with RK4 at a fixed step in φ up to `phiMax` or until
 * it escapes or crosses the horizon.
 */
export function rayFromInfinity(b: number, step = 0.01, phiMax = 6 * Math.PI): Ray {
  const alg = withEvents(rungeKutta(orbitRhs, 'rk4', { stepSize: step, tEnd: phiMax }), orbitRhs, [
    ESCAPE,
    CAPTURE,
    PERIAPSIS,
  ])
  const tr = trace(alg, { x0: [0, 1 / b] }, Math.ceil(phiMax / step) + 2)
  const steps = tr.steps as EventState<OdeState>[]
  const final = steps[steps.length - 1]
  const phi: number[] = []
  const u: number[] = []
  let residual = 0
  for (const s of steps) {
    const [us, ws] = toFlat(s.x)
    phi.push(s.time)
    u.push(us)
    residual = Math.max(residual, Math.abs(firstIntegralResidual(us, ws, b)) * b * b)
  }
  const fate = fateOf(final.events)
  const peri = final.events.find((e) => e.name === 'periapsis')
  return {
    b,
    phi,
    u,
    fate,
    phiEnd: final.time,
    deflection: fate === 'escapes' ? final.time - Math.PI : NaN,
    closest: peri ? 1 / toFlat(peri.x)[0] : NaN,
    residual,
  }
}

/** The ray's points in the plane, arriving from x = −∞ at height `sign`·b, kept within r ≤ rMax. */
export function rayPoints(ray: Ray, rMax: number, sign = 1) {
  const x: number[] = []
  const y: number[] = []
  for (let i = 0; i < ray.phi.length; i++) {
    const u = ray.u[i]
    if (!(u > 1 / rMax)) continue
    x.push(-Math.cos(ray.phi[i]) / u)
    y.push((sign * Math.sin(ray.phi[i])) / u)
  }
  return { x, y }
}

/** The deflection of a ray from infinity by Dormand–Prince to a tight tolerance, with its fate and closest approach. */
export function deflection(b: number): { alpha: number; fate: Fate; closest: number; turns: number } {
  const alg = withEvents(dormandPrince(orbitRhs, { tEnd: 40, rtol: 1e-10, atol: 1e-12 }), orbitRhs, [
    ESCAPE,
    CAPTURE,
    PERIAPSIS,
  ])
  const s = run(alg, { x0: [0, 1 / b] }, 20000)
  const fate = fateOf(s.events)
  const peri = s.events.find((e) => e.name === 'periapsis')
  return {
    alpha: fate === 'escapes' ? s.time - Math.PI : NaN,
    fate,
    closest: peri ? 1 / toFlat(peri.x)[0] : NaN,
    turns: s.time / (2 * Math.PI),
  }
}

// ---------------------------------------------------------------------------------------------------------------
// The lensed sky: an observer at r = R_OBS, the background on a sphere at infinity.
// ---------------------------------------------------------------------------------------------------------------

/** The observer's distance from the hole. */
export const R_OBS = 60

/**
 * The azimuth Φ(b) that a ray sweeps from the observer at radius rObs, heading inwards, to infinity: Φ = π − θ in
 * flat space, θ the angle between the ray and the direction to the hole. NaN for a ray that falls in.
 */
export function sweepFromObserver(b: number, rObs = R_OBS): number {
  const u0 = 1 / rObs
  const w2 = 1 / (b * b) - u0 * u0 + 2 * u0 ** 3
  if (!(w2 >= 0)) return NaN
  const alg = withEvents(dormandPrince(orbitRhs, { tEnd: 60, rtol: 1e-10, atol: 1e-12 }), orbitRhs, [ESCAPE, CAPTURE])
  const s = run(alg, { x0: [u0, Math.sqrt(w2)] }, 20000)
  return fateOf(s.events) === 'escapes' ? s.time : NaN
}

export type SweepTable = {
  spline: PiecewisePolynomial
  rObs: number
  bMax: number
  /** The impact parameter of the Einstein ring of a source exactly behind the hole: Φ(b_E) = π. */
  einsteinB: number
  /** Milliseconds to build. */
  ms: number
}

/** Φ(b) on `nodes` values of s = ln(b − b_c) up to bMax, splined in s. */
export function sweepTable(rObs = R_OBS, bMax = 50, nodes = 240): SweepTable {
  const t0 = performance.now()
  const sMin = Math.log(1e-7)
  const sMax = Math.log(bMax - B_CRIT)
  const s = Array.from({ length: nodes }, (_, i) => sMin + ((sMax - sMin) * i) / (nodes - 1))
  const sweep = s.map((si) => sweepFromObserver(B_CRIT + Math.exp(si), rObs))
  const spline = cubicSpline(tensor(s), tensor(sweep))
  const root = findRoot((b) => sweepFromObserver(b, rObs) - Math.PI, [B_CRIT + 0.5, bMax])
  return { spline, rObs, bMax, einsteinB: root.x, ms: performance.now() - t0 }
}

/** The pixel coordinate (r tan θ) at which a ray of impact parameter b leaves the observer. */
export const imageRadius = (b: number, rObs = R_OBS) => {
  const sin = (b * Math.sqrt(1 - 2 / rObs)) / rObs
  return (rObs * sin) / Math.sqrt(1 - sin * sin)
}

/** A seeded star field on the celestial sphere, bucketed by longitude and latitude for lookup. */
export type StarField = {
  x: Float64Array
  y: Float64Array
  z: Float64Array
  light: Float64Array
  width: Float64Array
  /** Bucket b holds stars order[start[b]] … order[start[b + 1] − 1]. */
  start: Int32Array
  order: Int32Array
}

const CELL = 0.04
const N_LAT = Math.ceil(Math.PI / CELL)
const N_LON = Math.ceil((2 * Math.PI) / CELL)
/** Longitude about the view direction (0, 0, −1) and latitude above the x–z plane, of a unit vector. */
const lonOf = (x: number, z: number) => Math.atan2(x, -z)
const latOf = (y: number) => Math.asin(Math.max(-1, Math.min(1, y)))
const bucketOf = (lon: number, lat: number) => {
  const i = Math.min(N_LAT - 1, Math.floor((lat + Math.PI / 2) / CELL))
  const j = Math.min(N_LON - 1, Math.floor((lon + Math.PI) / CELL))
  return i * N_LON + j
}

/** The angular width of a field star's light and of the bright star at the centre of the view, in radians. */
export const STAR_WIDTH = 0.005
export const BRIGHT_WIDTH = 0.01

/**
 * `count` stars uniform on the sphere with mostly faint brightness (u⁴ for u uniform), plus one bright star in the
 * view direction (0, 0, −1), the target for the Einstein ring.
 */
export function starField(seed: number, count: number): StarField {
  const root = stream(seed)
  const cz = toFlat(uniform(child(root, 'z'), -1, 1, { shape: [count] }))
  const az = toFlat(uniform(child(root, 'azimuth'), 0, 2 * Math.PI, { shape: [count] }))
  const lu = toFlat(uniform(child(root, 'light'), 0, 1, { shape: [count] }))
  const n = count + 1
  const x = new Float64Array(n)
  const y = new Float64Array(n)
  const z = new Float64Array(n)
  const light = new Float64Array(n)
  const width = new Float64Array(n).fill(STAR_WIDTH)
  for (let k = 0; k < count; k++) {
    const s = Math.sqrt(1 - cz[k] * cz[k])
    x[k] = s * Math.cos(az[k])
    y[k] = cz[k]
    z[k] = s * Math.sin(az[k])
    light[k] = 0.15 + 0.85 * lu[k] ** 4
  }
  ;[x[count], y[count], z[count], light[count], width[count]] = [0, 0, -1, 1, BRIGHT_WIDTH]
  const bucket = new Int32Array(n)
  const start = new Int32Array(N_LAT * N_LON + 1)
  for (let k = 0; k < n; k++) {
    bucket[k] = bucketOf(lonOf(x[k], z[k]), latOf(y[k]))
    start[bucket[k] + 1]++
  }
  for (let c = 0; c < N_LAT * N_LON; c++) start[c + 1] += start[c]
  const fill = start.slice(0, -1)
  const order = new Int32Array(n)
  for (let k = 0; k < n; k++) order[fill[bucket[k]]++] = k
  return { x, y, z, light, width, start, order }
}

/** The light of the star field in the direction d (a unit vector): a Gaussian of each star's width, summed. */
export function starLight(f: StarField, dx: number, dy: number, dz: number): number {
  const lat = latOf(dy)
  const lon = lonOf(dx, dz)
  const i0 = Math.min(N_LAT - 1, Math.floor((lat + Math.PI / 2) / CELL))
  const j0 = Math.min(N_LON - 1, Math.floor((lon + Math.PI) / CELL))
  let total = 0
  for (let i = Math.max(0, i0 - 1); i <= Math.min(N_LAT - 1, i0 + 1); i++) {
    const edge = Math.max(Math.abs((i + 1) * CELL - Math.PI / 2), Math.abs(i * CELL - Math.PI / 2))
    const cos = Math.cos(Math.min(edge, Math.PI / 2))
    const span = cos > 1 / N_LON ? Math.min(Math.ceil(1 / cos), N_LON >> 1) : N_LON >> 1
    for (let dj = -span; dj <= span; dj++) {
      const j = (((j0 + dj) % N_LON) + N_LON) % N_LON
      const c = i * N_LON + j
      for (let m = f.start[c]; m < f.start[c + 1]; m++) {
        const k = f.order[m]
        const ex = dx - f.x[k]
        const ey = dy - f.y[k]
        const ez = dz - f.z[k]
        const d2 = ex * ex + ey * ey + ez * ez
        const w = f.width[k]
        if (d2 < 16 * w * w) total += f.light[k] * Math.exp(-d2 / (2 * w * w))
      }
    }
  }
  return Math.min(total, 1)
}

/** A checkerboard on the celestial sphere in longitude and latitude, squares of `cellDeg` degrees: 0.35 or 1. */
export function checkerLight(dx: number, dy: number, dz: number, cellDeg = 4): number {
  const c = (cellDeg * Math.PI) / 180
  const k = Math.floor(lonOf(dx, dz) / c) + Math.floor(latOf(dy) / c)
  return k % 2 === 0 ? 1 : 0.35
}

export type Image = { x: number[]; y: number[]; z: number[][] }

/** Evenly spaced pixel centres across [−half, half]. */
export const pixelCentres = (n: number, half: number) =>
  Array.from({ length: n }, (_, i) => -half + ((i + 0.5) * 2 * half) / n)

/**
 * The sky seen by the observer with the hole at pixel coordinates (hx, hy): each pixel's ray is bent in the plane of
 * the hole and the pixel direction, by the sweep Φ(b) from the table. Pixels of rays that fall in are 0.
 */
export function lensedSky(
  table: SweepTable,
  light: (dx: number, dy: number, dz: number) => number,
  hx: number,
  hy: number,
  n: number,
  half: number,
): Image {
  const D = table.rObs
  const xs = pixelCentres(n, half)
  const hn = Math.hypot(hx, hy, D)
  const [Hx, Hy, Hz] = [hx / hn, hy / hn, -D / hn]
  const scale = D / Math.sqrt(1 - 2 / D)
  const count = n * n
  const s = new Float64Array(count)
  const sinT = new Float64Array(count)
  const cosT = new Float64Array(count)
  const ex = new Float64Array(count)
  const ey = new Float64Array(count)
  const ez = new Float64Array(count)
  const inside = new Uint8Array(count)
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) {
      const p = i * n + j
      const nn = Math.hypot(xs[j], xs[i], D)
      const [Nx, Ny, Nz] = [xs[j] / nn, xs[i] / nn, -D / nn]
      const c = Nx * Hx + Ny * Hy + Nz * Hz
      const [Px, Py, Pz] = [Nx - c * Hx, Ny - c * Hy, Nz - c * Hz]
      const sn = Math.hypot(Px, Py, Pz)
      const b = scale * sn
      cosT[p] = c
      sinT[p] = sn
      if (b <= B_CRIT || sn === 0) {
        inside[p] = 1
        s[p] = 0
        continue
      }
      ;[ex[p], ey[p], ez[p]] = [Px / sn, Py / sn, Pz / sn]
      s[p] = Math.log(Math.min(b, table.bMax) - B_CRIT)
    }
  const sweep = toFlat(evaluatePiecewise(table.spline, tensor(s)))
  const z: number[][] = []
  for (let i = 0; i < n; i++) {
    const row: number[] = []
    for (let j = 0; j < n; j++) {
      const p = i * n + j
      if (inside[p]) {
        row.push(0)
        continue
      }
      const cp = Math.cos(sweep[p])
      const sp = Math.sin(sweep[p])
      // Direction at infinity: cos Φ e₁ + sin Φ e₂ with e₁ = −h (hole to observer) and e₂ the pixel's offset.
      row.push(light(-cp * Hx + sp * ex[p], -cp * Hy + sp * ey[p], -cp * Hz + sp * ez[p]))
    }
    z.push(row)
  }
  return { x: xs, y: xs, z }
}

// ---------------------------------------------------------------------------------------------------------------
// The thin disc.
// ---------------------------------------------------------------------------------------------------------------

export const DISC_INNER = ISCO
export const DISC_OUTER = 20

/** u(φ) per impact parameter on a uniform φ grid; 1 after a ray falls in, −1 after it escapes. */
export type RayTable = {
  u: Float32Array
  rows: number
  cols: number
  step: number
  /** Row j has b_j = b_c + scale·sinh(s0 + j·ds). */
  s0: number
  ds: number
  scale: number
  ms: number
}

const FELL = 1
const ESCAPED = -1

/**
 * Rays from infinity for `rows` impact parameters in [bMin, bMax], packed densely near b_c, each integrated by RK4
 * with a fixed step in φ to `phiMax` or to its escape or capture.
 */
export function rayTable({ rows = 360, step = 0.02, bMin = 0.2, bMax = 36, phiMax = 2 * Math.PI + 0.05 } = {}) {
  const t0 = performance.now()
  const scale = 0.02
  const s0 = Math.asinh((bMin - B_CRIT) / scale)
  const s1 = Math.asinh((bMax - B_CRIT) / scale)
  const ds = (s1 - s0) / (rows - 1)
  const cols = Math.floor(phiMax / step) + 1
  const u = new Float32Array(rows * cols)
  for (let j = 0; j < rows; j++) {
    const b = B_CRIT + scale * Math.sinh(s0 + j * ds)
    const alg = withEvents(rungeKutta(orbitRhs, 'rk4', { stepSize: step, tEnd: phiMax }), orbitRhs, [ESCAPE, CAPTURE])
    const tr = trace(alg, { x0: [0, 1 / b] }, cols + 2)
    const steps = tr.steps as EventState<OdeState>[]
    const final = steps[steps.length - 1]
    const fate = fateOf(final.events)
    let k = 0
    for (const st of steps) {
      if (k >= cols) break
      if (Math.abs(st.time - k * step) > 1e-9) break
      u[j * cols + k] = toFlat(st.x)[0]
      k++
    }
    const fill = fate === 'falls in' ? FELL : ESCAPED
    for (; k < cols; k++) u[j * cols + k] = fill
  }
  return { u, rows, cols, step, s0, ds, scale, ms: performance.now() - t0 } satisfies RayTable
}

/** u at impact parameter b and azimuth φ: bilinear, or the nearest entry next to a ray's end. */
export function lookupU(t: RayTable, b: number, phi: number): number {
  const fj = Math.min(Math.max((Math.asinh((b - B_CRIT) / t.scale) - t.s0) / t.ds, 0), t.rows - 1)
  const fk = phi / t.step
  if (fk >= t.cols - 1) return ESCAPED
  const j0 = Math.min(Math.floor(fj), t.rows - 2)
  const k0 = Math.floor(fk)
  const wj = fj - j0
  const wk = fk - k0
  const a = t.u[j0 * t.cols + k0]
  const bq = t.u[j0 * t.cols + k0 + 1]
  const c = t.u[(j0 + 1) * t.cols + k0]
  const d = t.u[(j0 + 1) * t.cols + k0 + 1]
  const ended = (v: number) => v === FELL || v === ESCAPED
  if (ended(a) || ended(bq) || ended(c) || ended(d)) {
    const jn = wj < 0.5 ? j0 : j0 + 1
    const kn = wk < 0.5 ? k0 : k0 + 1
    return t.u[jn * t.cols + kn]
  }
  return (1 - wj) * ((1 - wk) * a + wk * bq) + wj * ((1 - wk) * c + wk * d)
}

/**
 * The Page–Thorne flux from the disc surface at radius r (Luminet 1979, eq. 15, with M = Ṁ = 1): zero at the ISCO,
 * a peak of 1.146 × 10⁻⁴ · 3/(8π) at r = 9.55, and r⁻³ far out. The coefficient of the logarithm is √3/2, which
 * reproduces both of Luminet's quoted peak values and the Novikov–Thorne integral.
 */
export function pageThorneFlux(r: number): number {
  if (r <= ISCO) return 0
  const s = Math.sqrt(r)
  const r3 = Math.sqrt(3)
  const r6 = Math.sqrt(6)
  const log = Math.log(((s + r3) * (r6 - r3)) / ((s - r3) * (r6 + r3)))
  return (3 / (8 * Math.PI)) * (1 / ((r - 3) * r ** 2.5)) * (s - r6 + (r3 / 2) * log)
}

/**
 * The redshift factor 1 + z of light from a disc element on a circular Keplerian orbit at radius r, seen at image
 * coordinate X = b cos ψ by a distant observer at inclination i (Luminet 1979, eq. 19): the gravitational and
 * transverse-Doppler factor (1 − 3/r)^(−1/2) times the line-of-sight Doppler factor 1 + Ω X sin i, Ω = r^(−3/2).
 */
export const redshiftFactor = (r: number, X: number, inclination: number) =>
  (1 + r ** -1.5 * X * Math.sin(inclination)) / Math.sqrt(1 - 3 / r)

/** The peak of the Page–Thorne flux, for normalising. */
export const PEAK_FLUX = (() => {
  let best = 0
  for (let r = 6.01; r < 20; r += 0.01) best = Math.max(best, pageThorneFlux(r))
  return best
})()

export type DiscImage = Image & {
  /** The smallest and largest observed-to-emitted frequency ratio g = 1/(1 + z) over the disc's pixels. */
  gMin: number
  gMax: number
  /** The pixels showing the disc in each image order. */
  primary: number
  secondary: number
}

/**
 * The disc from r = 6 to r = 20 seen at inclination `inclinationDeg` (0 face-on), on an n × n grid across
 * [−half, half]. Each pixel's ray crosses the equatorial plane at φ₀ + kπ; the first crossing inside the disc gives its
 * light: the Page–Thorne flux, times g⁴ with `doppler`, relative to the peak flux.
 */
export function discImage(
  t: RayTable,
  inclinationDeg: number,
  n: number,
  half: number,
  { secondary = true, doppler = true } = {},
): DiscImage {
  const inc = (inclinationDeg * Math.PI) / 180
  const xs = pixelCentres(n, half)
  const z: number[][] = []
  const sinI = Math.sin(inc)
  const cosI = Math.cos(inc)
  const orders = secondary ? 2 : 1
  let [gMin, gMax, primary, second] = [Infinity, -Infinity, 0, 0]
  for (let i = 0; i < n; i++) {
    const row: number[] = []
    for (let j = 0; j < n; j++) {
      const X = xs[j]
      const Y = xs[i]
      const b = Math.hypot(X, Y)
      const sinPsi = b > 0 ? Y / b : 0
      // The first crossing of the plane: cos φ₀ = −sin ψ sin i / √(sin²ψ sin²i + cos²i), φ₀ in (0, π).
      const phi0 = Math.acos(-(sinPsi * sinI) / Math.sqrt(sinPsi * sinPsi * sinI * sinI + cosI * cosI))
      let value = 0
      for (let k = 0; k < orders; k++) {
        const u = lookupU(t, b, phi0 + k * Math.PI)
        if (u >= 1 / HORIZON || u <= 0) break
        const r = 1 / u
        if (r < DISC_INNER || r > DISC_OUTER) continue
        let light = pageThorneFlux(r) / PEAK_FLUX
        const g = 1 / redshiftFactor(r, X, inc)
        if (doppler) light *= g ** 4
        gMin = Math.min(gMin, g)
        gMax = Math.max(gMax, g)
        if (k === 0) primary++
        else second++
        value = light
        break
      }
      row.push(value)
    }
    z.push(row)
  }
  return { x: xs, y: xs, z, gMin, gMax, primary, secondary: second }
}
