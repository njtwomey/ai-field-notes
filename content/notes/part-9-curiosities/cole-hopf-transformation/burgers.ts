/**
 * Burgers-equation physics for the figure in this note, on the periodic interval [0, 2π): the initial profiles, the
 * Cole–Hopf map between the Burgers velocity u and the heat-equation variable φ, the inviscid (characteristic)
 * solution, and the shock-width measurement (Hopf, 1950; Cole, 1951; Whitham, 1974, chapter 4).
 *
 * u_t + u u_x = ν u_xx is mapped to φ_t = ν φ_xx by u = −2ν φ_x/φ, with φ(x, 0) = exp(−F(x)/2ν), F(x) = ∫₀ˣ u₀.
 * Both profiles have zero mean over the period, so F and hence φ are periodic.
 */

/** The initial profiles the figure offers. */
export type ProfileName = 'sine' | 'step'

/** u₀(x): sin x, or +A on (0, π) and −A on (π, 2π). */
export function initialVelocity(name: ProfileName, amp: number): (x: number) => number {
  if (name === 'sine') return (x) => amp * Math.sin(x)
  return (x) => {
    const r = ((x % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI)
    return r < Math.PI ? amp : -amp
  }
}

/** F(x) = ∫₀ˣ u₀(s) ds, in closed form, for x in [0, 2π]. */
export function potential(name: ProfileName, amp: number): (x: number) => number {
  if (name === 'sine') return (x) => amp * (1 - Math.cos(x))
  return (x) => (x <= Math.PI ? amp * x : amp * (2 * Math.PI - x))
}

/**
 * φ(x, 0) = exp(−F(x)/2ν). Both profiles have F ≥ 0 with minimum 0, so φ runs from exp(−max F/2ν) up to 1, which
 * stays inside double precision for ν down to about max F/1400.
 *
 * @param name The initial profile.
 * @param amp The amplitude A of u₀.
 * @param nu The viscosity ν > 0.
 */
export function initialPhi(name: ProfileName, amp: number, nu: number): (x: number) => number {
  const f = potential(name, amp)
  return (x) => Math.exp(-f(x) / (2 * nu))
}

/**
 * u = −2ν (ln φ)_x on a periodic grid of n points (the last grid point repeats the first and is ignored), by the
 * central difference of ln φ, which stays accurate where φ is tiny.
 *
 * @param phi φ at the grid points x_i = i·Δx, i = 0 … n − 1 (length n, φ[n−1] = φ[0]).
 * @param dx The grid spacing Δx.
 * @param nu The viscosity ν.
 * @returns u at the same n points.
 */
export function velocityFromPhi(phi: ArrayLike<number>, dx: number, nu: number): Float64Array {
  const n = phi.length
  const m = n - 1
  const logs = new Float64Array(m)
  for (let i = 0; i < m; i++) logs[i] = Math.log(phi[i])
  const u = new Float64Array(n)
  for (let i = 0; i < m; i++) {
    const right = logs[(i + 1) % m]
    const left = logs[(i - 1 + m) % m]
    u[i] = (-nu * (right - left)) / dx
  }
  u[m] = u[0]
  return u
}

/** The jump across the steepest point and the measured width of a profile. */
export type ShockMeasure = {
  /** Δu = max u − min u. */
  jump: number
  /** The largest downward slope −u_x. */
  slope: number
  /** The width Δu / max(−u_x). */
  width: number
  /** Where −u_x is largest. */
  at: number
}

/**
 * Measures the shock on a periodic grid: the jump Δu = max u − min u, the steepest downward slope, and the width
 * Δu / max(−u_x), which equals 8ν/Δu for Taylor's steady profile u = −(Δu/2) tanh(Δu x/4ν).
 *
 * @param u The velocity at the grid points (length n, the last repeating the first).
 * @param dx The grid spacing.
 */
export function measureShock(u: ArrayLike<number>, dx: number): ShockMeasure {
  let lo = Infinity
  let hi = -Infinity
  let slope = 0
  let at = 0
  for (let i = 0; i < u.length - 1; i++) {
    lo = Math.min(lo, u[i])
    hi = Math.max(hi, u[i])
    const s = -(u[i + 1] - u[i]) / dx
    if (s > slope) {
      slope = s
      at = (i + 0.5) * dx
    }
  }
  const jump = hi - lo
  return { jump, slope, width: slope > 0 ? jump / slope : Infinity, at }
}

/**
 * The inviscid solution by characteristics: each point ξ carries u₀(ξ) to x = ξ + t u₀(ξ). After the breaking time
 * the curve folds over (three values of u at some x). Returned in order of ξ, wrapped into [0, 2π) with NaN breaks
 * where a characteristic wraps around.
 *
 * @param u0 The initial profile.
 * @param t The time.
 * @param count The number of characteristics.
 */
export function characteristics(u0: (x: number) => number, t: number, count: number): { x: number[]; u: number[] } {
  const L = 2 * Math.PI
  const x: number[] = []
  const u: number[] = []
  let prev = NaN
  for (let i = 0; i <= count; i++) {
    const xi = (i / count) * L
    const v = u0(xi)
    const p = (((xi + t * v) % L) + L) % L
    if (Number.isFinite(prev) && Math.abs(p - prev) > L / 2) {
      x.push(NaN)
      u.push(NaN)
    }
    x.push(p)
    u.push(v)
    prev = p
  }
  return { x, u }
}
