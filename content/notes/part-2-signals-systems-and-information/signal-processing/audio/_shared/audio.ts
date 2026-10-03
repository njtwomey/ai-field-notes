/** Audio feature helpers shared by the audio notes: mel filter banks, power spectrograms, the DCT, and synthesis. */
import { hzToMel, melToHz, stft } from '@/lib/dsp'
import { rng } from '@/lib/math'

export type MelNorm = 'htk' | 'slaney'

/**
 * Triangular mel filter bank: `count` filters with edges equally spaced in mel between fMin and fMax, evaluated at
 * the `nfft/2 + 1` bin frequencies. 'htk' filters peak at 1; 'slaney' filters are scaled to unit area in Hz (2 / band
 * width), so wide high-frequency filters do not dominate.
 */
export function melFilterBank(count: number, nfft: number, fs: number, fMin = 0, fMax = fs / 2, norm: MelNorm = 'htk') {
  const edgesMel = Array.from(
    { length: count + 2 },
    (_, i) => hzToMel(fMin) + ((hzToMel(fMax) - hzToMel(fMin)) * i) / (count + 1),
  )
  const edges = edgesMel.map(melToHz)
  const bins = Array.from({ length: nfft / 2 + 1 }, (_, k) => (k * fs) / nfft)
  const filters = Array.from({ length: count }, (_, m) => {
    const [lo, mid, hi] = [edges[m], edges[m + 1], edges[m + 2]]
    const scale = norm === 'slaney' ? 2 / (hi - lo) : 1
    return bins.map((f) => scale * Math.max(0, Math.min((f - lo) / (mid - lo), (hi - f) / (hi - mid))))
  })
  return { filters, edges, bins }
}

/** Power spectrogram |STFT|²: frames × bins. */
export function powerSpectrogram(x: ArrayLike<number>, size: number, hop: number, nfft = size) {
  const { frames, centres } = stft(x, size, hop, 'hann', nfft)
  return { power: frames.map((f) => Array.from(f, (v) => v * v)), centres }
}

/** Apply a filter bank to each power frame: frames × filters. */
export const applyBank = (power: number[][], filters: number[][]) =>
  power.map((frame) => filters.map((w) => w.reduce((s, wk, k) => s + wk * frame[k], 0)))

/** Orthonormal DCT-II, as scipy.fft.dct(norm='ortho'). */
export function dct(x: ArrayLike<number>): number[] {
  const n = x.length
  return Array.from({ length: n }, (_, k) => {
    let s = 0
    for (let i = 0; i < n; i++) s += x[i] * Math.cos((Math.PI * k * (2 * i + 1)) / (2 * n))
    return s * Math.sqrt((k === 0 ? 1 : 2) / n)
  })
}

/** A harmonic tone: Σ_h amp(h) cos(2π h f0 t), with optional white noise of standard deviation `noise`. */
export function harmonicTone(
  f0: number,
  fs: number,
  n: number,
  harmonics = 10,
  noise = 0,
  seed = 1,
  amp = (h: number) => 1 / h,
) {
  const g = rng(seed)
  return Array.from({ length: n }, (_, i) => {
    let s = 0
    for (let h = 1; h <= harmonics && h * f0 < fs / 2; h++) s += amp(h) * Math.cos((2 * Math.PI * h * f0 * i) / fs)
    return s + noise * g.normal()
  })
}

/** Linear chirp from f1 to f2 Hz over n samples. */
export const chirp = (f1: number, f2: number, fs: number, n: number) =>
  Array.from({ length: n }, (_, i) => {
    const t = i / fs
    const T = n / fs
    return Math.cos(2 * Math.PI * (f1 * t + ((f2 - f1) * t * t) / (2 * T)))
  })
