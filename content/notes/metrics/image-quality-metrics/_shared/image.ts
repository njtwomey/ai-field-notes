/**
 * Small greyscale images (row-major Float64Array, values nominally in [0, 255]) for image-quality figures: a synthetic
 * test image, distortions tuned to a target mean squared error, and PSNR and SSIM.
 */
import { rng } from '@/lib/math'

export const SIZE = 64
export const PEAK = 255

export type Image = Float64Array

/** A test image with a smooth gradient, a bright disc, fine stripes and a sharp-edged square. */
export function testImage(n = SIZE): Image {
  const img = new Float64Array(n * n)
  for (let r = 0; r < n; r++) {
    for (let c = 0; c < n; c++) {
      let v = 40 + 120 * (c / (n - 1))
      if ((r - 20) ** 2 + (c - 44) ** 2 < 120) v = 225
      if (r > 36 && c < 28) v = 60 + 90 * (Math.floor(c / 2) % 2)
      if (r > 42 && r < 58 && c > 36 && c < 56) v = 20
      img[r * n + c] = v
    }
  }
  return img
}

export const mse = (a: Image, b: Image) => a.reduce((s, v, i) => s + (v - b[i]) ** 2, 0) / a.length
export const psnr = (a: Image, b: Image) => 10 * Math.log10((PEAK * PEAK) / mse(a, b))

const mean = (x: Image) => x.reduce((s, v) => s + v, 0) / x.length
const variance = (x: Image) => {
  const m = mean(x)
  return x.reduce((s, v) => s + (v - m) ** 2, 0) / x.length
}

/** Separable Gaussian blur with standard deviation s (pixels), edges extended. */
export function blur(img: Image, s: number, n = SIZE): Image {
  if (s <= 0) return Float64Array.from(img)
  const radius = Math.ceil(3 * s)
  const k = Array.from({ length: 2 * radius + 1 }, (_, i) => Math.exp(-((i - radius) ** 2) / (2 * s * s)))
  const total = k.reduce((a, b) => a + b, 0)
  const pass = (src: Image, horizontal: boolean) => {
    const out = new Float64Array(n * n)
    for (let r = 0; r < n; r++)
      for (let c = 0; c < n; c++) {
        let acc = 0
        for (let i = -radius; i <= radius; i++) {
          const rr = horizontal ? r : Math.min(n - 1, Math.max(0, r + i))
          const cc = horizontal ? Math.min(n - 1, Math.max(0, c + i)) : c
          acc += k[i + radius] * src[rr * n + cc]
        }
        out[r * n + c] = acc / total
      }
    return out
  }
  return pass(pass(img, true), false)
}

/** Find x in [lo, hi] with f(x) ≈ target for increasing f, by bisection. */
function solve(f: (x: number) => number, target: number, lo: number, hi: number): number {
  for (let i = 0; i < 40; i++) {
    const mid = (lo + hi) / 2
    if (f(mid) < target) lo = mid
    else hi = mid
  }
  return (lo + hi) / 2
}

export type Distortion = 'noise' | 'shift' | 'contrast' | 'blur' | 'salt-and-pepper'

/** The reference image distorted so that its MSE against the reference is `target` (when reachable). */
export function distort(img: Image, kind: Distortion, target: number, seed = 1): Image {
  const n2 = img.length
  if (kind === 'shift') return img.map((v) => v + Math.sqrt(target))
  if (kind === 'contrast') {
    const m = mean(img)
    const s = 1 - Math.sqrt(target / variance(img))
    return img.map((v) => m + s * (v - m))
  }
  if (kind === 'noise') {
    const g = rng(seed)
    const z = Float64Array.from({ length: n2 }, () => g.normal())
    // Rescale the realised noise so the MSE is exactly the target.
    const scale = Math.sqrt(target / (z.reduce((s, v) => s + v * v, 0) / n2))
    return img.map((v, i) => v + scale * z[i])
  }
  if (kind === 'blur') {
    const s = solve((x) => mse(img, blur(img, x)), target, 0, 12)
    return blur(img, s)
  }
  // Salt and pepper: corrupt a growing prefix of a fixed random order of pixels.
  const g = rng(seed)
  const order = Array.from({ length: n2 }, (_, i) => i).sort(() => g.uniform() - 0.5)
  const values = order.map(() => (g.uniform() < 0.5 ? 0 : PEAK))
  const corrupt = (f: number) => {
    const out = Float64Array.from(img)
    const count = Math.round(f * n2)
    for (let i = 0; i < count; i++) out[order[i]] = values[i]
    return out
  }
  return corrupt(solve((f) => mse(img, corrupt(f)), target, 0, 1))
}

const K1 = 0.01
const K2 = 0.03

/**
 * The SSIM map with a 7 × 7 uniform window and sample covariances, as scikit-image's structural_similarity with its
 * defaults, and its mean over the pixels whose window lies inside the image.
 */
export function ssim(x: Image, y: Image, n = SIZE, win = 7): { map: Float64Array; mean: number } {
  const c1 = (K1 * PEAK) ** 2
  const c2 = (K2 * PEAK) ** 2
  const half = (win - 1) / 2
  const count = win * win
  const correction = count / (count - 1)
  const map = new Float64Array(n * n).fill(Number.NaN)
  let sum = 0
  let used = 0
  for (let r = half; r < n - half; r++) {
    for (let c = half; c < n - half; c++) {
      let sx = 0
      let sy = 0
      let sxx = 0
      let syy = 0
      let sxy = 0
      for (let i = -half; i <= half; i++)
        for (let j = -half; j <= half; j++) {
          const a = x[(r + i) * n + c + j]
          const b = y[(r + i) * n + c + j]
          sx += a
          sy += b
          sxx += a * a
          syy += b * b
          sxy += a * b
        }
      const mx = sx / count
      const my = sy / count
      const vx = (sxx / count - mx * mx) * correction
      const vy = (syy / count - my * my) * correction
      const cxy = (sxy / count - mx * my) * correction
      const s = ((2 * mx * my + c1) * (2 * cxy + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))
      map[r * n + c] = s
      sum += s
      used += 1
    }
  }
  return { map, mean: sum / used }
}
