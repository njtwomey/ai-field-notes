/**
 * Spin systems for the energy-based networks notes: the 2-D Ising lattice, Hopfield networks with Hebbian weights, and
 * small networks whose every state can be enumerated. Spins are ±1 throughout. Randomness comes in as a `uniform`
 * function (from `rng(seed)`), so every run is reproducible.
 */

export type Uniform = () => number

/* ----------------------------------------------------------------------------------------------------------------- */
/* 2-D Ising lattice                                                                                                  */
/* ----------------------------------------------------------------------------------------------------------------- */

/** Onsager's critical temperature of the square-lattice Ising model with J = 1: 2 / ln(1 + √2). */
export const T_CRITICAL = 2 / Math.log(1 + Math.SQRT2)

/** Yang's spontaneous magnetisation per spin of the infinite square lattice (J = 1, h = 0); zero above T_c. */
export function onsagerMagnetisation(T: number): number {
  if (T >= T_CRITICAL) return 0
  return Math.pow(1 - Math.pow(Math.sinh(2 / T), -4), 1 / 8)
}

export type Rule = 'metropolis' | 'glauber'

export type Lattice = { L: number; s: Int8Array }

export function randomLattice(L: number, uniform: Uniform): Lattice {
  const s = new Int8Array(L * L)
  for (let i = 0; i < s.length; i++) s[i] = uniform() < 0.5 ? -1 : 1
  return { L, s }
}

export function uniformLattice(L: number, value: 1 | -1): Lattice {
  return { L, s: new Int8Array(L * L).fill(value) }
}

/** Sum of the four neighbours of site i on a periodic L × L lattice. */
export function neighbourSum({ L, s }: Lattice, i: number): number {
  const r = Math.floor(i / L)
  const c = i - r * L
  return (
    s[((r + L - 1) % L) * L + c] + s[((r + 1) % L) * L + c] + s[r * L + ((c + L - 1) % L)] + s[r * L + ((c + 1) % L)]
  )
}

/** Energy change from flipping spin i: ΔE = 2 s_i (J Σ_nb s_j + h), with J = 1. */
export const flipCost = (lat: Lattice, i: number, h: number) => 2 * lat.s[i] * (neighbourSum(lat, i) + h)

/** Probability of flipping a spin whose flip costs ΔE, at temperature T, under each rule. */
export function flipProbability(dE: number, T: number, rule: Rule): number {
  if (rule === 'metropolis') return dE <= 0 ? 1 : Math.exp(-dE / T)
  return 1 / (1 + Math.exp(dE / T))
}

export type SpinUpdate = { site: number; dE: number; p: number; flipped: boolean }

/** One single-spin update at a uniformly chosen site. Mutates the lattice. */
export function updateSpin(lat: Lattice, T: number, h: number, rule: Rule, uniform: Uniform): SpinUpdate {
  const site = Math.floor(uniform() * lat.s.length)
  const dE = flipCost(lat, site, h)
  const p = flipProbability(dE, T, rule)
  const flipped = uniform() < p
  if (flipped) lat.s[site] = -lat.s[site] as 1 | -1
  return { site, dE, p, flipped }
}

/** One sweep: N = L² single-spin updates at random sites. */
export function sweepLattice(lat: Lattice, T: number, h: number, rule: Rule, uniform: Uniform): void {
  for (let n = 0; n < lat.s.length; n++) updateSpin(lat, T, h, rule, uniform)
}

/** Magnetisation per spin, m = (1/N) Σ s_i. */
export function magnetisation({ s }: Lattice): number {
  let m = 0
  for (let i = 0; i < s.length; i++) m += s[i]
  return m / s.length
}

/** Energy per spin, E/N with E = −Σ_<ij> s_i s_j − h Σ s_i (each bond counted once). */
export function latticeEnergy({ L, s }: Lattice, h: number): number {
  let e = 0
  for (let r = 0; r < L; r++)
    for (let c = 0; c < L; c++) {
      const v = s[r * L + c]
      e -= v * (s[r * L + ((c + 1) % L)] + s[((r + 1) % L) * L + c]) + h * v
    }
  return e / s.length
}

/* ----------------------------------------------------------------------------------------------------------------- */
/* Hopfield networks                                                                                                  */
/* ----------------------------------------------------------------------------------------------------------------- */

/** A symmetric weight matrix, row-major, with zero diagonal. */
export type Weights = { n: number; w: Float64Array }

/** Hebbian weights w_ij = (1/N) Σ_μ ξ_i^μ ξ_j^μ, w_ii = 0. */
export function hebbian(patterns: ArrayLike<number>[], n: number): Weights {
  const w = new Float64Array(n * n)
  for (const xi of patterns)
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) if (i !== j) w[i * n + j] += (xi[i] * xi[j]) / n
  return { n, w }
}

/** Local field h_i = Σ_j w_ij s_j. */
export function localField({ n, w }: Weights, s: ArrayLike<number>, i: number): number {
  let f = 0
  for (let j = 0; j < n; j++) f += w[i * n + j] * s[j]
  return f
}

/** Energy E = −½ Σ_ij w_ij s_i s_j. */
export function hopfieldEnergy(W: Weights, s: ArrayLike<number>): number {
  let e = 0
  for (let i = 0; i < W.n; i++) e -= 0.5 * s[i] * localField(W, s, i)
  return e
}

/** Overlap m = (1/N) ξ · s, in [−1, 1]. */
export function overlap(xi: ArrayLike<number>, s: ArrayLike<number>): number {
  let m = 0
  for (let i = 0; i < xi.length; i++) m += xi[i] * s[i]
  return m / xi.length
}

/** A permutation of 0…n−1 drawn with the Fisher–Yates shuffle. */
export function permutation(n: number, uniform: Uniform): number[] {
  const p = Array.from({ length: n }, (_, i) => i)
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(uniform() * (i + 1))
    ;[p[i], p[j]] = [p[j], p[i]]
  }
  return p
}

/**
 * The record of an asynchronous run: which neuron each update visited, whether it flipped, and the energy after it.
 * The state after t updates is the start state with the first t flips applied, so any step can be revisited.
 */
export type Trajectory = { start: Int8Array; sites: number[]; flipped: boolean[]; energy: number[]; converged: boolean }

/**
 * Asynchronous updates in a fresh random order each sweep. At T = 0 each neuron takes the sign of its local field (a
 * zero field leaves it unchanged) and the run stops after a sweep with no flip. At T > 0 each neuron is set to +1 with
 * probability 1/(1 + e^{−2h/T}) (Glauber dynamics) and the run lasts `maxSweeps`.
 */
export function runAsync(
  W: Weights,
  start: ArrayLike<number>,
  uniform: Uniform,
  { T = 0, maxSweeps = 20 }: { T?: number; maxSweeps?: number } = {},
): Trajectory {
  const s = Int8Array.from(start)
  const sites: number[] = []
  const flipped: boolean[] = []
  let e = hopfieldEnergy(W, s)
  const energy = [e]
  let converged = false
  for (let sweep = 0; sweep < maxSweeps && !converged; sweep++) {
    let changes = 0
    for (const i of permutation(W.n, uniform)) {
      const h = localField(W, s, i)
      let next = s[i]
      if (T > 0) next = uniform() < 1 / (1 + Math.exp((-2 * h) / T)) ? 1 : -1
      else if (h !== 0) next = h > 0 ? 1 : -1
      const flip = next !== s[i]
      if (flip) {
        // ΔE = 2 s_i h_i for a flip of s_i (w_ii = 0).
        e += 2 * s[i] * h
        s[i] = next
        changes++
      }
      sites.push(i)
      flipped.push(flip)
      energy.push(e)
    }
    if (T === 0 && changes === 0) converged = true
  }
  return { start: Int8Array.from(start), sites, flipped, energy, converged }
}

/** The state after the first t updates of a trajectory. */
export function stateAt(traj: Trajectory, t: number): Int8Array {
  const s = Int8Array.from(traj.start)
  for (let k = 0; k < t && k < traj.sites.length; k++) if (traj.flipped[k]) s[traj.sites[k]] = -s[traj.sites[k]]
  return s
}

/** Flip a fraction of the entries of a ±1 pattern, chosen without replacement. */
export function corrupt(xi: ArrayLike<number>, fraction: number, uniform: Uniform): Int8Array {
  const s = Int8Array.from(xi)
  const k = Math.round(fraction * s.length)
  for (const i of permutation(s.length, uniform).slice(0, k)) s[i] = -s[i]
  return s
}

export function randomPattern(n: number, uniform: Uniform): Int8Array {
  const s = new Int8Array(n)
  for (let i = 0; i < n; i++) s[i] = uniform() < 0.5 ? -1 : 1
  return s
}

/* ----------------------------------------------------------------------------------------------------------------- */
/* Small networks: every state enumerated                                                                             */
/* ----------------------------------------------------------------------------------------------------------------- */

/** Spin vector of an integer state code: bit k set means s_k = +1. */
export function spinsOf(code: number, n: number): Int8Array {
  const s = new Int8Array(n)
  for (let k = 0; k < n; k++) s[k] = (code >> k) & 1 ? 1 : -1
  return s
}

export function codeOf(s: ArrayLike<number>): number {
  let code = 0
  for (let k = 0; k < s.length; k++) if (s[k] > 0) code |= 1 << k
  return code
}

/** Reflected binary Gray code and its inverse: consecutive codes differ in one bit. */
export const gray = (k: number) => k ^ (k >> 1)
export function grayInverse(g: number): number {
  let k = g
  for (let shift = g >> 1; shift; shift >>= 1) k ^= shift
  return k
}

/**
 * A state whose every single flip leaves the energy unchanged or raises it: a fixed point of zero-temperature
 * asynchronous dynamics (a zero local field leaves the neuron as it is).
 */
export function isFixedPoint(W: Weights, s: ArrayLike<number>): boolean {
  for (let i = 0; i < W.n; i++) if (s[i] * localField(W, s, i) < 0) return false
  return true
}

/* ----------------------------------------------------------------------------------------------------------------- */
/* Image helpers                                                                                                      */
/* ----------------------------------------------------------------------------------------------------------------- */

/**
 * Tile small row-major images of size w × h into one raster, `cols` per row, separated by `gap` pixels of `fill`. Use
 * `fill` at the midpoint of a diverging scale so the gaps read as neutral.
 */
export function mosaic(
  images: ArrayLike<number>[],
  w: number,
  h: number,
  cols: number,
  gap = 1,
  fill = 0,
): { width: number; height: number; values: Float32Array } {
  const rows = Math.max(1, Math.ceil(images.length / cols))
  const width = cols * w + (cols - 1) * gap
  const height = rows * h + (rows - 1) * gap
  const values = new Float32Array(width * height).fill(fill)
  images.forEach((img, k) => {
    const r0 = Math.floor(k / cols) * (h + gap)
    const c0 = (k % cols) * (w + gap)
    for (let r = 0; r < h; r++) for (let c = 0; c < w; c++) values[(r0 + r) * width + c0 + c] = img[r * w + c]
  })
  return { width, height, values }
}

/** The outline of one pixel (column c, row r) as a closed path in pixel coordinates, for ImagePlot `lines`. */
export function pixelBox(c: number, r: number, size = 1): { x: number[]; y: number[] } {
  const lo = -0.5
  const hi = size - 0.5
  return { x: [c + lo, c + hi, c + hi, c + lo, c + lo], y: [r + lo, r + lo, r + hi, r + hi, r + lo] }
}
