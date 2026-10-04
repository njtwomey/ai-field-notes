/**
 * Orthogonal wavelet helpers shared by the wavelet notes: Daubechies filters, the periodic discrete wavelet transform
 * (Mallat's pyramid algorithm), the cascade algorithm and a Morlet continuous wavelet transform.
 */
import { fft, ifft, nextPowerOfTwo } from 'aifn/foundation/fourier'
import { complex, imagPart, realPart, tensor, toFlat } from 'aifn/foundation/tensor'

/** The complex FFT of (re, im), written back into the two arrays. The inverse is unscaled (no 1/N). */
function fftInPlace(re: Float64Array, im: Float64Array, inverse = false) {
  const z = complex(tensor(re), tensor(im))
  const out = inverse ? ifft(z, { norm: 'forward' }) : fft(z)
  re.set(toFlat(realPart(out)))
  im.set(toFlat(imagPart(out)))
}

export type Family = 'haar' | 'db2' | 'db3' | 'db4'

/**
 * Lowpass (scaling) filters h, normalised so Σ h = √2 and Σ h[n] h[n − 2k] = δ[k]. Computed by spectral factorisation
 * (minimum-phase root choice) and checked against those identities; dbN has N vanishing moments and 2N taps.
 */
export const FILTERS: Record<Family, number[]> = {
  haar: [Math.SQRT1_2, Math.SQRT1_2],
  db2: [0.4829629131445342, 0.8365163037378078, 0.2241438680420134, -0.1294095225512604],
  db3: [
    0.3326705529500827, 0.8068915093110928, 0.4598775021184915, -0.1350110200102549, -0.0854412738820267,
    0.0352262918857096,
  ],
  db4: [
    0.2303778133088973, 0.7148465705529179, 0.6308807679298601, -0.0279837694168616, -0.187034811719095,
    0.0308413818355603, 0.0328830116668852, -0.0105974017850691,
  ],
}

export const VANISHING_MOMENTS: Record<Family, number> = { haar: 1, db2: 2, db3: 3, db4: 4 }

/** Highpass (wavelet) filter g[n] = (−1)ⁿ h[L − 1 − n]. */
export const highpass = (h: number[]) => h.map((_, n) => (n % 2 === 0 ? 1 : -1) * h[h.length - 1 - n])

/** One analysis step with periodic extension: a[k] = Σ h[m] x[2k + m], d[k] = Σ g[m] x[2k + m]. */
export function analysisStep(x: ArrayLike<number>, h: number[]): { approx: Float64Array; detail: Float64Array } {
  const n = x.length
  const g = highpass(h)
  const half = n / 2
  const approx = new Float64Array(half)
  const detail = new Float64Array(half)
  for (let k = 0; k < half; k++) {
    let a = 0
    let d = 0
    for (let m = 0; m < h.length; m++) {
      const v = x[(2 * k + m) % n]
      a += h[m] * v
      d += g[m] * v
    }
    approx[k] = a
    detail[k] = d
  }
  return { approx, detail }
}

/** Inverse of analysisStep: x[2k + m] += h[m] a[k] + g[m] d[k]. Orthogonality makes this the adjoint and the inverse. */
export function synthesisStep(approx: ArrayLike<number>, detail: ArrayLike<number>, h: number[]): Float64Array {
  const n = approx.length * 2
  const g = highpass(h)
  const x = new Float64Array(n)
  for (let k = 0; k < approx.length; k++) {
    for (let m = 0; m < h.length; m++) {
      const idx = (2 * k + m) % n
      x[idx] += h[m] * approx[k] + g[m] * detail[k]
    }
  }
  return x
}

/** J-level decomposition: details from finest (level 1) to coarsest (level J), and the final approximation. */
export function wavedec(x: ArrayLike<number>, h: number[], levels: number) {
  let a: Float64Array = Float64Array.from(x)
  const details: Float64Array[] = []
  for (let j = 0; j < levels; j++) {
    const { approx, detail } = analysisStep(a, h)
    details.push(detail)
    a = approx
  }
  return { approx: a, details }
}

export function waverec(approx: ArrayLike<number>, details: Float64Array[], h: number[]): Float64Array {
  let a: Float64Array = Float64Array.from(approx)
  for (let j = details.length - 1; j >= 0; j--) a = synthesisStep(a, details[j], h)
  return a
}

/**
 * Cascade algorithm: iterate the two-scale equation φ(t) = √2 Σ h[n] φ(2t − n) from a unit impulse. After `iterations`
 * steps the samples approximate φ on a grid of spacing 2^{−iterations} over [0, L − 1]; ψ follows from g.
 */
export function cascade(h: number[], iterations = 8): { t: number[]; phi: number[]; psi: number[] } {
  const upsample = (c: number[]) => c.flatMap((v, i) => (i < c.length - 1 ? [v, 0] : [v]))
  const conv = (a: number[], b: number[]) => {
    const out = Array(a.length + b.length - 1).fill(0)
    a.forEach((av, i) => b.forEach((bv, j) => (out[i + j] += av * bv)))
    return out
  }
  const scaled = h.map((v) => v * Math.SQRT2)
  const g = highpass(h).map((v) => v * Math.SQRT2)
  let phi = [1]
  let psi = [1]
  for (let i = 0; i < iterations; i++) {
    phi = conv(upsample(phi), scaled)
    psi = i === 0 ? g.slice() : conv(upsample(psi), scaled)
  }
  // After the first step psi used g, then refined by h: ψ = Σ g φ(2t − n).
  const step = 2 ** -iterations
  const t = phi.map((_, i) => i * step)
  return { t, phi, psi: psi.slice(0, phi.length).concat(Array(Math.max(0, phi.length - psi.length)).fill(0)) }
}

/**
 * Morlet continuous wavelet transform |W(a, b)| at the given frequencies (Hz), computed per scale by FFT:
 * W(a, b) = (1/√a) ∫ x(t) ψ*((t − b)/a) dt with ψ(t) = π^{−1/4} e^{iω₀t} e^{−t²/2}, a = ω₀ / (2π f).
 * Returns rows (one per frequency) of magnitudes at every sample.
 */
export function morletCwt(x: ArrayLike<number>, fs: number, freqs: number[], omega0 = 6): Float64Array[] {
  const n = x.length
  const size = nextPowerOfTwo(2 * n)
  const xr = new Float64Array(size)
  const xi = new Float64Array(size)
  for (let i = 0; i < n; i++) xr[i] = x[i]
  fftInPlace(xr, xi)
  const rows: Float64Array[] = []
  for (const f of freqs) {
    const a = omega0 / (2 * Math.PI * f) // scale in seconds
    const re = new Float64Array(size)
    const im = new Float64Array(size)
    for (let k = 0; k < size; k++) {
      // Angular frequency of bin k in rad/s; the wavelet's spectrum is a Gaussian at ω₀/a, analytic (positive only).
      const w = (2 * Math.PI * fs * (k <= size / 2 ? k : k - size)) / size
      const psiHat = w > 0 ? Math.PI ** -0.25 * Math.sqrt(2 * Math.PI) * Math.exp(-0.5 * (a * w - omega0) ** 2) : 0
      const gain = Math.sqrt(a) * psiHat
      re[k] = xr[k] * gain
      im[k] = xi[k] * gain
    }
    fftInPlace(re, im, true)
    const row = new Float64Array(n)
    // 1/size from the inverse FFT; the result is in sample units, so scale by √fs for a rate-independent magnitude.
    for (let i = 0; i < n; i++) row[i] = Math.hypot(re[i], im[i]) / size / Math.sqrt(fs)
    rows.push(row)
  }
  return rows
}
