/**
 * Spectral-estimation helpers for the spectral-estimation notes. A power spectral density here is the two-sided
 * S(ω) = Σ_m r[m] e^{−iωm} of a real, unit-spaced signal, evaluated for ω ∈ [0, π]; unit-variance white noise has
 * S(ω) = 1. The DFT and windows come from aifn.
 */
import { nextPowerOfTwo, rfft } from 'aifn/foundation/fourier'
import { imagPart, realPart, toFlat, type Tensor } from 'aifn/foundation/tensor'
import { getWindow, type WindowName } from 'aifn/signal/windows'
import { normal, stream } from 'aifn/foundation/random'

/** A periodic window of n samples, as plain numbers. */
export const windowOf = (win: WindowName, n: number): number[] => toFlat(getWindow(win, n, { periodic: true }))

/** Real and imaginary parts of X[k], k = 0..nfft/2, of a real signal zero-padded (or cut) to nfft samples. */
export function halfSpectrum(x: number[], nfft: number) {
  const z = rfft(Float64Array.from(x), { n: nfft }) as Tensor
  return { re: toFlat(realPart(z)), im: toFlat(imagPart(z)) }
}

/** Standard-normal white noise of length n. */
export function whiteNoise(n: number, seed: number): number[] {
  const g = stream(seed)
  return Array.from({ length: n }, () => normal(g))
}

/** An AR(p) process x[n] = Σ a_k x[n−k] + e[n], driven by unit-variance white noise, with a burn-in discarded. */
export function arProcess(a: number[], n: number, seed: number, burn = 500): number[] {
  const e = whiteNoise(n + burn, seed)
  const x = new Array<number>(n + burn).fill(0)
  for (let t = 0; t < n + burn; t++) {
    let v = e[t]
    for (let k = 0; k < a.length; k++) if (t - k - 1 >= 0) v += a[k] * x[t - k - 1]
    x[t] = v
  }
  return x.slice(burn)
}

/** The true PSD of an AR(p) process with driving variance σ²: σ² / |1 − Σ a_k e^{−iωk}|², at the given ω. */
export function arSpectrum(a: number[], sigma2: number, omega: number[]): number[] {
  return omega.map((w) => {
    let re = 1
    let im = 0
    a.forEach((ak, k) => {
      re -= ak * Math.cos(w * (k + 1))
      im += ak * Math.sin(w * (k + 1))
    })
    return sigma2 / (re * re + im * im)
  })
}

/**
 * Windowed periodogram of x: Ŝ(ω_k) = |Σ w[n] x[n] e^{−iω_k n}|² / Σ w[n]², at ω_k = 2πk/nfft for k = 0..nfft/2. With
 * a rectangular window this is the classical periodogram |X(e^{iω})|²/N, an estimate of the two-sided PSD S(ω), whose
 * integral over [−π, π) divided by 2π is the variance.
 */
export function periodogram(x: number[], win: WindowName = 'rectangular', nfft = nextPowerOfTwo(x.length)) {
  const w = windowOf(win, x.length)
  const u = w.reduce((s, v) => s + v * v, 0)
  const { re, im } = halfSpectrum(
    x.map((v, i) => v * w[i]),
    nfft,
  )
  const half = nfft / 2 + 1
  const omega = Array.from({ length: half }, (_, k) => (2 * Math.PI * k) / nfft)
  const psd = Array.from({ length: half }, (_, k) => (re[k] * re[k] + im[k] * im[k]) / u)
  return { omega, psd }
}

/** Welch's estimate: average of windowed periodograms of segments of length `seg`, stepping by `seg − overlap`. */
export function welch(x: number[], seg: number, overlap: number, win: WindowName = 'hann') {
  const step = Math.max(1, seg - overlap)
  let sum: number[] | null = null
  let omega: number[] = []
  let count = 0
  for (let start = 0; start + seg <= x.length; start += step) {
    const p = periodogram(x.slice(start, start + seg), win, seg)
    omega = p.omega
    sum = sum ? sum.map((v, i) => v + p.psd[i]) : [...p.psd]
    count++
  }
  return { omega, psd: (sum ?? []).map((v) => v / count), segments: count }
}

/** Sample autocovariance r̂[m] = (1/N) Σ_{n} (x[n] − x̄)(x[n+m] − x̄) for m = 0..maxLag (the biased estimator). */
export function autocovariance(x: number[], maxLag: number): number[] {
  const n = x.length
  const mean = x.reduce((s, v) => s + v, 0) / n
  return Array.from({ length: maxLag + 1 }, (_, m) => {
    let s = 0
    for (let t = 0; t + m < n; t++) s += (x[t] - mean) * (x[t + m] - mean)
    return s / n
  })
}

/**
 * Levinson–Durbin recursion: solves the Yule–Walker equations for AR coefficients a_1..a_p from autocovariances
 * r[0..p], returning the coefficients, the driving variance and the reflection coefficients.
 */
export function levinsonDurbin(r: number[], p: number) {
  let a: number[] = []
  let err = r[0]
  const reflection: number[] = []
  for (let m = 1; m <= p; m++) {
    let acc = r[m]
    for (let k = 1; k < m; k++) acc -= a[k - 1] * r[m - k]
    const kappa = acc / err
    reflection.push(kappa)
    const next = a.map((ak, k) => ak - kappa * a[m - 2 - k])
    next.push(kappa)
    a = next
    err *= 1 - kappa * kappa
  }
  return { a, sigma2: err, reflection }
}

/** Burg's method: reflection coefficients from forward and backward prediction errors, then the AR coefficients. */
export function burg(x: number[], p: number) {
  let f = [...x]
  let b = [...x]
  let a: number[] = []
  let err = x.reduce((s, v) => s + v * v, 0) / x.length
  for (let m = 1; m <= p; m++) {
    let num = 0
    let den = 0
    for (let t = m; t < x.length; t++) {
      num += f[t] * b[t - 1]
      den += f[t] * f[t] + b[t - 1] * b[t - 1]
    }
    const kappa = (2 * num) / den
    const nf = [...f]
    const nb = [...b]
    for (let t = m; t < x.length; t++) {
      nf[t] = f[t] - kappa * b[t - 1]
      nb[t] = b[t - 1] - kappa * f[t]
    }
    f = nf
    b = nb
    const next = a.map((ak, k) => ak - kappa * a[m - 2 - k])
    next.push(kappa)
    a = next
    err *= 1 - kappa * kappa
  }
  return { a, sigma2: err }
}

/**
 * Eigenvalues and eigenvectors of a real symmetric matrix by cyclic Jacobi rotations. Returns them sorted by
 * decreasing eigenvalue; vectors[k] is the unit eigenvector for values[k]. Fine for n up to a few hundred.
 */
export function symmetricEigen(input: number[][]): { values: number[]; vectors: number[][] } {
  const n = input.length
  const a = input.map((row) => [...row])
  const v = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0) as number))
  for (let sweep = 0; sweep < 60; sweep++) {
    let off = 0
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) off += a[i][j] * a[i][j]
    if (off < 1e-22) break
    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        if (Math.abs(a[p][q]) < 1e-15) continue
        const theta = (a[q][q] - a[p][p]) / (2 * a[p][q])
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1))
        const c = 1 / Math.sqrt(t * t + 1)
        const s = t * c
        for (let k = 0; k < n; k++) {
          const akp = a[k][p]
          const akq = a[k][q]
          a[k][p] = c * akp - s * akq
          a[k][q] = s * akp + c * akq
        }
        for (let k = 0; k < n; k++) {
          const apk = a[p][k]
          const aqk = a[q][k]
          a[p][k] = c * apk - s * aqk
          a[q][k] = s * apk + c * aqk
        }
        for (let k = 0; k < n; k++) {
          const vkp = v[k][p]
          const vkq = v[k][q]
          v[k][p] = c * vkp - s * vkq
          v[k][q] = s * vkp + c * vkq
        }
      }
    }
  }
  const order = Array.from({ length: n }, (_, i) => i).sort((i, j) => a[j][j] - a[i][i])
  return { values: order.map((i) => a[i][i]), vectors: order.map((i) => v.map((row) => row[i])) }
}

/**
 * The first k discrete prolate spheroidal sequences (Slepian tapers) of length n and half-bandwidth W = NW/n, as the
 * leading eigenvectors of the tridiagonal matrix that commutes with the concentration operator. Each has unit energy
 * and a positive sum (odd-order tapers have a positive leading slope).
 */
const dpssCache = new Map<string, number[][]>()

export function dpss(n: number, nw: number, k: number): number[][] {
  // The tapers depend only on N and NW, and the eigenproblem is the expensive part, so it is computed once per pair.
  const key = `${n}:${nw}`
  const cached = dpssCache.get(key)
  if (cached && cached.length >= k) return cached.slice(0, k)
  const all = computeDpss(n, nw, Math.max(k, Math.floor(2 * nw)))
  dpssCache.set(key, all)
  return all.slice(0, k)
}

function computeDpss(n: number, nw: number, k: number): number[][] {
  const w = nw / n
  const m = Array.from({ length: n }, () => new Array<number>(n).fill(0))
  for (let i = 0; i < n; i++) {
    m[i][i] = ((n - 1 - 2 * i) / 2) ** 2 * Math.cos(2 * Math.PI * w)
    if (i + 1 < n) {
      m[i][i + 1] = ((i + 1) * (n - i - 1)) / 2
      m[i + 1][i] = m[i][i + 1]
    }
  }
  const { vectors } = symmetricEigen(m)
  return vectors.slice(0, k).map((v, order) => {
    // Sign convention as in SciPy: even tapers sum positively, odd tapers start positively.
    const reference =
      order % 2 === 0 ? v.reduce((s, x) => s + x, 0) : v.slice(0, 4).reduce((s, x, i) => s + (4 - i) * x, 0)
    return reference < 0 ? v.map((x) => -x) : v
  })
}

/** Thomson's multitaper estimate: the average of the k eigenspectra |Σ v_k[n] x[n] e^{−iωn}|². */
export function multitaper(x: number[], nw: number, k: number, nfft = nextPowerOfTwo(x.length)) {
  const tapers = dpss(x.length, nw, k)
  const half = nfft / 2 + 1
  const sum = new Array<number>(half).fill(0)
  for (const v of tapers) {
    const { re, im } = halfSpectrum(
      x.map((val, i) => val * v[i]),
      nfft,
    )
    for (let j = 0; j < half; j++) sum[j] += re[j] * re[j] + im[j] * im[j]
  }
  const omega = Array.from({ length: half }, (_, j) => (2 * Math.PI * j) / nfft)
  return { omega, psd: sum.map((s) => s / k), tapers }
}

/** Decibels of a power, 10 log₁₀, floored. */
export const powerDb = (p: number, floor = -120) => Math.max(floor, 10 * Math.log10(Math.max(p, 1e-30)))
