/**
 * Small numerical helpers for the control-theory figures: complex numbers, polynomials, frequency responses,
 * simulation of linear systems and the discrete Riccati equation. Everything is dense and naive, sized for systems of
 * order at most six.
 */

export type Vec = number[]
export type Mat = number[][]

// ---------------------------------------------------------------------------------------------------------------------
// Matrices

export const matmul = (a: Mat, b: Mat): Mat =>
  a.map((row) => b[0].map((_, j) => row.reduce((s, v, k) => s + v * b[k][j], 0)))
export const matvec = (a: Mat, x: Vec): Vec => a.map((row) => row.reduce((s, v, k) => s + v * x[k], 0))
export const transpose = (a: Mat): Mat => a[0].map((_, j) => a.map((row) => row[j]))
export const madd = (a: Mat, b: Mat): Mat => a.map((row, i) => row.map((v, j) => v + b[i][j]))
export const msub = (a: Mat, b: Mat): Mat => a.map((row, i) => row.map((v, j) => v - b[i][j]))
export const mscale = (a: Mat, s: number): Mat => a.map((row) => row.map((v) => v * s))
export const identity = (n: number): Mat =>
  Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => +(i === j)))
export const dot = (a: Vec, b: Vec) => a.reduce((s, v, i) => s + v * b[i], 0)

/** Inverse by Gauss–Jordan elimination with partial pivoting. */
export function inverse(a: Mat): Mat {
  const n = a.length
  const m = a.map((row, i) => [...row, ...identity(n)[i]])
  for (let c = 0; c < n; c++) {
    let p = c
    for (let r = c + 1; r < n; r++) if (Math.abs(m[r][c]) > Math.abs(m[p][c])) p = r
    ;[m[c], m[p]] = [m[p], m[c]]
    const d = m[c][c]
    for (let j = 0; j < 2 * n; j++) m[c][j] /= d
    for (let r = 0; r < n; r++) {
      if (r === c) continue
      const f = m[r][c]
      for (let j = 0; j < 2 * n; j++) m[r][j] -= f * m[c][j]
    }
  }
  return m.map((row) => row.slice(n))
}

// ---------------------------------------------------------------------------------------------------------------------
// Complex numbers

export type Complex = { re: number; im: number }
export const cx = (re: number, im = 0): Complex => ({ re, im })
export const cadd = (a: Complex, b: Complex): Complex => cx(a.re + b.re, a.im + b.im)
export const csub = (a: Complex, b: Complex): Complex => cx(a.re - b.re, a.im - b.im)
export const cmul = (a: Complex, b: Complex): Complex => cx(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re)
export function cdiv(a: Complex, b: Complex): Complex {
  const d = b.re * b.re + b.im * b.im
  return cx((a.re * b.re + a.im * b.im) / d, (a.im * b.re - a.re * b.im) / d)
}
export const cabs = (a: Complex) => Math.hypot(a.re, a.im)
export const carg = (a: Complex) => Math.atan2(a.im, a.re)

// ---------------------------------------------------------------------------------------------------------------------
// Polynomials, as coefficient arrays with the highest power first: [1, 3, 2] is s² + 3s + 2.

export function polyMul(a: Vec, b: Vec): Vec {
  const out = new Array(a.length + b.length - 1).fill(0)
  a.forEach((x, i) => b.forEach((y, j) => (out[i + j] += x * y)))
  return out
}
export function polyAdd(a: Vec, b: Vec): Vec {
  const n = Math.max(a.length, b.length)
  const pa = [...new Array(n - a.length).fill(0), ...a]
  const pb = [...new Array(n - b.length).fill(0), ...b]
  return pa.map((v, i) => v + pb[i])
}
/** The monic polynomial with the given real roots. */
export const polyFromRoots = (roots: number[]): Vec => roots.reduce((p, r) => polyMul(p, [1, -r]), [1])
export const polyEval = (p: Vec, s: Complex): Complex => p.reduce((acc, c) => cadd(cmul(acc, s), cx(c)), cx(0))

/** All complex roots by the Durand–Kerner iteration, warm-started from `guess` when given. */
export function roots(p: Vec, guess?: Complex[]): Complex[] {
  let q = [...p]
  while (q.length > 1 && Math.abs(q[0]) < 1e-14) q = q.slice(1)
  const n = q.length - 1
  if (n < 1) return []
  const monic = q.map((c) => c / q[0])
  let z: Complex[] =
    guess && guess.length === n
      ? guess.map((g) => cx(g.re + 1e-9, g.im + 1e-9))
      : Array.from({ length: n }, (_, k) => {
          const r = 1 + Math.max(...monic.slice(1).map(Math.abs))
          const t = (2 * Math.PI * k) / n + 0.4
          return cx(r * Math.cos(t), r * Math.sin(t))
        })
  for (let it = 0; it < 500; it++) {
    let delta = 0
    z = z.map((zi, i) => {
      let den = cx(1)
      z.forEach((zj, j) => {
        if (j !== i) den = cmul(den, csub(zi, zj))
      })
      const step = cdiv(polyEval(monic, zi), den)
      delta = Math.max(delta, cabs(step))
      return csub(zi, step)
    })
    if (delta < 1e-12) break
  }
  return z.map((r) => (Math.abs(r.im) < 1e-7 ? cx(r.re) : r))
}

// ---------------------------------------------------------------------------------------------------------------------
// Transfer functions G(s) = num(s) / den(s), optionally with a time delay e^{-sτ}.

export type TF = { num: Vec; den: Vec; delay?: number }

export function freqResponse(g: TF, w: number): Complex {
  const s = cx(0, w)
  const v = cdiv(polyEval(g.num, s), polyEval(g.den, s))
  const tau = g.delay ?? 0
  return tau ? cmul(v, cx(Math.cos(w * tau), -Math.sin(w * tau))) : v
}

/** Magnitude in dB and continuous (unwrapped) phase in degrees over a frequency grid. */
export function bode(g: TF, ws: number[]): { mag: number[]; phase: number[] } {
  const mag: number[] = []
  const phase: number[] = []
  let prev = 0
  ws.forEach((w, i) => {
    const h = freqResponse(g, w)
    mag.push(20 * Math.log10(cabs(h)))
    let p = (carg(h) * 180) / Math.PI
    if (i > 0) p += 360 * Math.round((prev - p) / 360)
    phase.push(p)
    prev = p
  })
  return { mag, phase }
}

export const logspace = (lo: number, hi: number, n: number) =>
  Array.from({ length: n }, (_, i) => 10 ** (lo + ((hi - lo) * i) / (n - 1)))

// ---------------------------------------------------------------------------------------------------------------------
// State space

export type SS = { A: Mat; B: Vec; C: Vec; D: number }

/** Controllable canonical form of a strictly proper or proper single-input single-output transfer function. */
export function tf2ss(num: Vec, den: Vec): SS {
  const a = den.map((c) => c / den[0])
  const n = a.length - 1
  const b = [...new Array(n + 1 - num.length).fill(0), ...num].map((c) => c / den[0])
  const D = b[0]
  const A = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i < n - 1 ? +(j === i + 1) : -a[n - j])),
  )
  const B = Array.from({ length: n }, (_, i) => +(i === n - 1))
  // y = (b_n - a_n D) x_1 + … : coefficients of the strictly proper part.
  const C = Array.from({ length: n }, (_, j) => b[n - j] - a[n - j] * D)
  return { A, B, C, D }
}

/** One classical Runge–Kutta step of dx/dt = f(x). */
export function rk4(f: (x: Vec) => Vec, x: Vec, dt: number): Vec {
  const k1 = f(x)
  const k2 = f(x.map((v, i) => v + (dt / 2) * k1[i]))
  const k3 = f(x.map((v, i) => v + (dt / 2) * k2[i]))
  const k4 = f(x.map((v, i) => v + dt * k3[i]))
  return x.map((v, i) => v + (dt / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]))
}

/**
 * Unit-step response of the negative-feedback loop y = L/(1+L) r, with L = K·G(s)·e^{-sτ}, simulated in the time
 * domain so that a delay needs no approximation.
 */
export function feedbackStep(g: TF, gain: number, tEnd: number, dt: number): { t: number[]; y: number[] } {
  const { A, B, C } = tf2ss(g.num, g.den)
  const lag = Math.round((g.delay ?? 0) / dt)
  const buffer: number[] = new Array(lag).fill(0)
  let x = new Array(A.length).fill(0)
  const t: number[] = []
  const y: number[] = []
  const steps = Math.round(tEnd / dt)
  for (let k = 0; k <= steps; k++) {
    const yk = dot(C, x)
    t.push(k * dt)
    y.push(yk)
    const e = 1 - yk
    buffer.push(gain * e)
    const u = buffer.shift()!
    x = rk4((z) => matvec(A, z).map((v, i) => v + B[i] * u), x, dt)
    if (!Number.isFinite(yk) || Math.abs(yk) > 1e6) break
  }
  return { t, y }
}

// ---------------------------------------------------------------------------------------------------------------------
// Discrete-time linear-quadratic regulator

/** Solve P = Q + AᵀPA − AᵀPB(R + BᵀPB)⁻¹BᵀPA by iterating the Riccati recursion; return P and K = (R+BᵀPB)⁻¹BᵀPA. */
export function dlqr(A: Mat, B: Mat, Q: Mat, R: Mat, iterations = 2000): { P: Mat; K: Mat } {
  let P = Q
  let K: Mat = []
  const At = transpose(A)
  const Bt = transpose(B)
  for (let i = 0; i < iterations; i++) {
    const BtP = matmul(Bt, P)
    K = matmul(inverse(madd(R, matmul(BtP, B))), matmul(BtP, A))
    const next = madd(Q, matmul(matmul(At, P), msub(A, matmul(B, K))))
    const change = Math.max(...next.flat().map((v, j) => Math.abs(v - P.flat()[j])))
    P = next.map((row, r) => row.map((v, c) => (v + next[c][r]) / 2))
    if (change < 1e-10 * (1 + Math.max(...P.flat().map(Math.abs)))) break
  }
  return { P, K }
}

/** Eigenvalues of a real 2 × 2 matrix. */
export function eig2(m: Mat): Complex[] {
  const tr = m[0][0] + m[1][1]
  const det = m[0][0] * m[1][1] - m[0][1] * m[1][0]
  const disc = (tr * tr) / 4 - det
  if (disc >= 0) return [cx(tr / 2 + Math.sqrt(disc)), cx(tr / 2 - Math.sqrt(disc))]
  return [cx(tr / 2, Math.sqrt(-disc)), cx(tr / 2, -Math.sqrt(-disc))]
}

/** Format a complex number compactly, e.g. "−1.2 ± 0.5i" is left to callers; this gives "−1.2 + 0.5i". */
export function formatComplex(z: Complex, digits = 2): string {
  const re = z.re.toFixed(digits)
  if (Math.abs(z.im) < 10 ** -digits) return re
  return `${re} ${z.im < 0 ? '−' : '+'} ${Math.abs(z.im).toFixed(digits)}i`
}
