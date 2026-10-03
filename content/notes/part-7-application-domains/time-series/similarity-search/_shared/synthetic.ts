/** Synthetic series for the matrix-profile figures. Seeded, so every figure is reproducible. */
import { rng } from '@/lib/math'

/** A heartbeat-like shape of the given length: small P bump, sharp R spike, S dip and a broad T wave. */
export function beat(length: number, shape: 'normal' | 'abnormal' = 'normal'): number[] {
  return Array.from({ length }, (_, k) => {
    const u = k / (length - 1)
    const p = 0.15 * Math.exp(-(((u - 0.2) / 0.05) ** 2))
    return shape === 'abnormal'
      ? p + 0.6 * Math.exp(-(((u - 0.4) / 0.06) ** 2)) - 0.3 * Math.exp(-(((u - 0.55) / 0.06) ** 2))
      : p +
          Math.exp(-(((u - 0.35) / 0.015) ** 2)) -
          0.25 * Math.exp(-(((u - 0.39) / 0.02) ** 2)) +
          0.3 * Math.exp(-(((u - 0.65) / 0.07) ** 2))
  })
}

/**
 * Concatenated beats with jittered lengths (base ± 3) and additive noise. `abnormalAt` lists beat numbers drawn with
 * the abnormal shape. Returns the series and the start index of every beat.
 */
export function beats(n: number, seed: number, noise = 0.03, abnormalAt: number[] = [], base = 40) {
  const g = rng(seed)
  const x: number[] = []
  const starts: number[] = []
  for (let b = 0; x.length < n; b++) {
    const length = base + Math.floor(g.uniform() * 7) - 3
    starts.push(x.length)
    x.push(...beat(length, abnormalAt.includes(b) ? 'abnormal' : 'normal'))
  }
  return { x: x.slice(0, n).map((v) => v + noise * g.normal()), starts: starts.filter((s) => s < n) }
}

/** Smooth background noise: white noise passed through a short moving average, so subsequences do not match by chance. */
export function smoothNoise(n: number, g: ReturnType<typeof rng>, scale = 0.3, width = 5): number[] {
  const white = Array.from({ length: n + width }, () => g.normal())
  return Array.from({ length: n }, (_, t) => {
    let s = 0
    for (let k = 0; k < width; k++) s += white[t + k]
    return (scale * s) / Math.sqrt(width)
  })
}

/** Normal beats up to `change`, then abnormal beats: two regimes with one boundary. */
export function regimeChange(n: number, change: number, seed: number) {
  const before = beats(change, seed, 0.04).x
  const after = beats(
    n - change,
    seed + 1000,
    0.04,
    Array.from({ length: n }, (_, b) => b),
  ).x
  return [...before, ...after]
}

/**
 * A pattern that drifts: `count` occurrences at jittered spacing in smooth noise. Occurrence k is a bump plus a second
 * bump whose height grows from 0 to 2, so neighbours in time look alike while the first and last differ.
 */
export function driftingPattern(n: number, seed: number, count = 8, length = 40) {
  const g = rng(seed)
  const x = smoothNoise(n, g, 0.06)
  const spacing = Math.floor(n / count)
  const starts: number[] = []
  for (let k = 0; k < count; k++) {
    const start = k * spacing + 10 + Math.floor(g.uniform() * (spacing - length - 20))
    starts.push(start)
    const second = (2 * k) / (count - 1)
    for (let t = 0; t < length; t++) {
      const u = t / (length - 1)
      x[start + t] += 1.5 * Math.exp(-(((u - 0.3) / 0.1) ** 2)) + second * Math.exp(-(((u - 0.7) / 0.1) ** 2))
    }
  }
  return { x, starts }
}

const bumpDip = (u: number) => 2 * Math.exp(-(((u - 0.3) / 0.1) ** 2)) - 1.5 * Math.exp(-(((u - 0.7) / 0.06) ** 2))
const sawtooth = (u: number) => 2 * ((3 * u) % 1) - 1

/** Two series in smooth noise. Both contain one shared pattern; only b contains a second, novel pattern. */
export function sharedAndNovel(n: number, seed: number, length = 40) {
  const g = rng(seed)
  const a = smoothNoise(n, g)
  const b = smoothNoise(n, g)
  const at = { a: 60 + Math.floor(g.uniform() * 200), b: 40 + Math.floor(g.uniform() * 100) }
  const novel = at.b + length + 60 + Math.floor(g.uniform() * (n - at.b - 2 * length - 80))
  for (let t = 0; t < length; t++) {
    const u = t / (length - 1)
    a[at.a + t] += bumpDip(u)
    b[at.b + t] += bumpDip(u)
    b[novel + t] += 1.5 * sawtooth(u)
  }
  return { a, b, shared: at, novel }
}

/** Smooth noise with a short pattern (length 20) planted twice and a long pattern (length 72) planted twice. */
export function twoScales(n: number, seed: number) {
  const g = rng(seed)
  const x = smoothNoise(n, g, 0.25)
  const jitter = (a: number) => a + Math.floor(g.uniform() * 30)
  const short = [jitter(40), jitter(300)]
  const long = [jitter(110), jitter(380)]
  for (const s of short) for (let t = 0; t < 20; t++) x[s + t] += 3 * sawtooth(t / 19)
  for (const s of long)
    for (let t = 0; t < 72; t++) {
      const u = t / 71
      x[s + t] += 1.2 * Math.sin(2 * Math.PI * u) + 0.6 * Math.sin(3 * Math.PI * u)
    }
  return { x, short, long }
}
