/**
 * Vortex-sheet physics for the figure in this note: Krasny's vortex-blob (δ) equations for a sheet that is periodic in
 * x with period 1 and carries unit circulation per period, discretised by N blobs of circulation 1/N
 * (Krasny, 1986, J. Comput. Phys. 65, eq. for the δ-equations):
 *
 *   dx_j/dt = −(1/2N) Σ_k sinh 2π(y_j − y_k) / (cosh 2π(y_j − y_k) − cos 2π(x_j − x_k) + δ²),
 *   dy_j/dt =  (1/2N) Σ_k sin 2π(x_j − x_k)  / (cosh 2π(y_j − y_k) − cos 2π(x_j − x_k) + δ²).
 *
 * The kernel is the velocity of a periodic row of vortices, u − iv = (1/2i) cot π(z − z_k), with δ² added to its
 * denominator. δ = 0 gives Rosenhead's point-vortex approximation. The flow far above the sheet moves at −½ and far
 * below at +½, so the velocity jump across the sheet is 1. The state is [x₁, y₁, …, x_N, y_N].
 */
import type { Tensor } from 'aifn-compute/foundation/tensor'

const TWO_PI = 2 * Math.PI

/** The entries of a state vector as one contiguous array. */
const flat = (s: Tensor): ArrayLike<number> => {
  const d = s.data as Float64Array
  return s.offset === 0 && d.length === s.shape[0] ? d : d.subarray(s.offset, s.offset + s.shape[0])
}

/**
 * The right-hand side of the δ-equations. Each pair is evaluated once (the kernel is odd), with
 * cos 2π(x_j − x_k) and e^{2π(y_j − y_k)} formed from per-blob values, so one evaluation costs O(N) transcendental calls
 * and O(N²) arithmetic.
 *
 * @param n The number of blobs N.
 * @param delta The smoothing parameter δ ≥ 0 (dimensionless; δ = 0 is the point-vortex sheet).
 */
export function blobSheet(n: number, delta: number) {
  const d2 = delta * delta
  const c = new Float64Array(n)
  const s = new Float64Array(n)
  const e = new Float64Array(n)
  const w = 1 / (2 * n)
  return (_t: number, state: Tensor): Float64Array => {
    const z = flat(state)
    const out = new Float64Array(2 * n)
    for (let j = 0; j < n; j++) {
      const a = TWO_PI * z[2 * j]
      c[j] = Math.cos(a)
      s[j] = Math.sin(a)
      e[j] = Math.exp(TWO_PI * z[2 * j + 1])
    }
    for (let j = 0; j < n; j++) {
      for (let k = j + 1; k < n; k++) {
        // cos and sin of 2π(x_j − x_k); cosh and sinh of 2π(y_j − y_k) from e_j / e_k.
        const cos = c[j] * c[k] + s[j] * s[k]
        const sin = s[j] * c[k] - c[j] * s[k]
        const r = e[j] / e[k]
        const cosh = 0.5 * (r + 1 / r)
        const sinh = 0.5 * (r - 1 / r)
        const inv = w / (cosh - cos + d2)
        const u = -sinh * inv
        const v = sin * inv
        out[2 * j] += u
        out[2 * j + 1] += v
        out[2 * k] -= u
        out[2 * k + 1] -= v
      }
    }
    return out
  }
}

/**
 * Krasny's initial sheet: the Lagrangian parameter Γ_j = j/N (the circulation from the left end), at
 * x = Γ + ε sin 2πΓ, y = −ε sin 2πΓ, which is the growing mode of the linearised flat-sheet problem.
 *
 * @param n The number of blobs N.
 * @param eps The perturbation amplitude ε (Krasny uses 0.01).
 */
export function perturbedSheet(n: number, eps: number): number[] {
  const out: number[] = []
  for (let j = 0; j < n; j++) {
    const g = j / n
    const p = eps * Math.sin(TWO_PI * g)
    out.push(g + p, -p)
  }
  return out
}

/** The amplitude of the sheet's displacement, (max y − min y)/2. */
export function amplitude(z: ArrayLike<number>): number {
  let lo = Infinity
  let hi = -Infinity
  for (let j = 1; j < z.length; j += 2) {
    lo = Math.min(lo, z[j])
    hi = Math.max(hi, z[j])
  }
  return (hi - lo) / 2
}

/**
 * The exponential growth rate σ of mode m (wavelength 1/m) on the flat discretised δ-sheet, from the δ-equations
 * linearised about x_j = Γ_j, y_j = 0. With perturbations ξ_j, η_j ∝ e^{2πi m Γ_j}, θ_k = 2πk/N and
 * D_k = 1 − cos θ_k + δ²:
 *
 *   dξ/dt = −(2π/2N) A η,  dη/dt = (2π/2N) B ξ,
 *   A = Σ_k (1 − cos mθ_k)/D_k,  B = Σ_k (1 − cos mθ_k)(cos θ_k D_k − sin²θ_k)/D_k²,
 *
 * so σ² = −(π/N)² A B. At δ = 0, B = −A and σ = (π/N) A; for m = 1 that is π(N − 1)/N, which tends to the
 * continuous sheet's rate k/2 = π (wavenumber k = 2π, unit velocity jump). Used for the figure's readout.
 *
 * @param m The mode number.
 * @param n The number of blobs N.
 * @param delta The smoothing parameter δ.
 * @returns σ, or 0 when the mode does not grow.
 */
export function linearRate(m: number, n: number, delta: number): number {
  const d2 = delta * delta
  let a = 0
  let b = 0
  for (let k = 1; k < n; k++) {
    const th = (TWO_PI * k) / n
    const den = 1 - Math.cos(th) + d2
    const phase = 1 - Math.cos(m * th)
    a += phase / den
    b += (phase * (Math.cos(th) * den - Math.sin(th) ** 2)) / (den * den)
  }
  const s2 = -((Math.PI / n) ** 2) * a * b
  return s2 > 0 ? Math.sqrt(s2) : 0
}
