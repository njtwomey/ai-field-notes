import { dctMatrix } from 'aifn-compute/foundation/fourier'
import { child, normals, shuffle, stream, uniform } from 'aifn-compute/foundation/random'
import { toFlat, type Tensor } from 'aifn-compute/foundation/tensor'

/** Signal length. */
export const M = 64
export const I = Array.from({ length: M }, (_, i) => i)

export type DictId = 'spikes' | 'cosines' | 'both'

/** Atoms as columns: the cosines of the orthonormal DCT, and the spikes (the identity). */
export const COS: number[][] = (() => {
  const C = toFlat(dctMatrix(M))
  return I.map((i) => I.map((k) => C[k * M + i]))
})()
export const SPK: number[][] = I.map((i) => I.map((j) => (i === j ? 1 : 0)))

/**
 * A dictionary and where its two kinds of atom start: in the union the cosines are columns 0..63 and the spikes
 * 64..127; an offset of −1 means the kind is absent.
 */
export function dictionary(id: DictId): { D: number[][]; offset: { cos: number; spk: number } } {
  if (id === 'cosines') return { D: COS, offset: { cos: 0, spk: -1 } }
  if (id === 'spikes') return { D: SPK, offset: { cos: -1, spk: 0 } }
  return { D: I.map((i) => [...COS[i], ...SPK[i]]), offset: { cos: 0, spk: M } }
}

/** The cosine and spike coefficients held in a code over a dictionary, as two vectors of length M. */
export function split(x: number[], offset: { cos: number; spk: number }) {
  const part = (o: number) => (o < 0 ? new Array<number>(M).fill(0) : x.slice(o, o + M))
  return { cos: part(offset.cos), spk: part(offset.spk) }
}

/** The signal that cosine coefficients synthesise. */
export const cosineSignal = (cos: number[]) => I.map((i) => COS[i].reduce((acc, v, k) => acc + v * cos[k], 0))

/** A signal that is a few low-frequency cosines plus a few spikes, with random positions, signs and amplitudes. */
export function makeTruth(cosines: number, spikes: number, seed: number) {
  const s = stream(`sparse-approximation/mixture/${seed}`)
  // Frequencies 1 to 20 keep the cosine part smooth enough to see; any frequency works for the maths.
  const freqs = shuffle(
    child(s, 'freq'),
    Array.from({ length: 20 }, (_, k) => k + 1),
  ).slice(0, cosines)
  const where = shuffle(child(s, 'where'), [...I]).slice(0, spikes)
  const amp = (u: number, sign: number) => (sign < 0.5 ? -1 : 1) * (2 + 4 * u)
  const ua = toFlat(uniform(child(s, 'amp'), 0, 1, { shape: [2 * (cosines + spikes)] }) as Tensor)
  const cos = new Array<number>(M).fill(0)
  const spk = new Array<number>(M).fill(0)
  freqs.forEach((k, a) => (cos[k] = amp(ua[2 * a], ua[2 * a + 1])))
  where.forEach((i, a) => (spk[i] = 0.4 * amp(ua[2 * (cosines + a)], ua[2 * (cosines + a) + 1])))
  const smooth = cosineSignal(cos)
  const clean = I.map((i) => smooth[i] + spk[i])
  return { cos, spk, smooth, clean, noise: toFlat(normals(child(s, 'noise'), M)) }
}
