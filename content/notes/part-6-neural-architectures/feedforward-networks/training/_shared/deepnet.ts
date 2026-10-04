/**
 * A deep fully connected network at initialisation, for the signal-propagation figures in the training notes.
 *
 * W_l has entries drawn from N(0, 1) and is used as (gain / √n) W_l, so each weight has variance gain² / n. A plain
 * layer outputs h_l = φ((gain / √n) W_l h_{l-1}). A residual layer uses the pre-activation form
 * h_l = h_{l-1} + α (gain / √n) W_l φ(h_{l-1}), whose branch output has mean zero. The backward pass starts from a
 * random N(0, 1) gradient at the top and applies the chain rule down to the input. No biases.
 */
import { normal, stream } from 'aifn/foundation/random'

export const WIDTH = 128
export const BATCH = 32
export const MAX_DEPTH = 50

export type Activation = 'linear' | 'tanh' | 'sigmoid' | 'relu'
export type InitScheme = 'small' | 'xavier' | 'he'

const sigmoid = (a: number) => 1 / (1 + Math.exp(-a))

const PHI: Record<Activation, { f: (a: number) => number; df: (a: number) => number }> = {
  linear: { f: (a) => a, df: () => 1 },
  tanh: { f: Math.tanh, df: (a) => 1 - Math.tanh(a) ** 2 },
  sigmoid: { f: sigmoid, df: (a) => sigmoid(a) * (1 - sigmoid(a)) },
  relu: { f: (a) => (a > 0 ? a : 0), df: (a) => (a > 0 ? 1 : 0) },
}

/** Weight standard deviation times √n. Fan-in equals fan-out here, so Xavier's 2/(n_in + n_out) is 1/n. */
export const GAIN: Record<InitScheme, number> = {
  small: 0.01 * Math.sqrt(WIDTH),
  xavier: 1,
  he: Math.SQRT2,
}

export const INIT_LABEL: Record<InitScheme, string> = {
  small: 'N(0, 0.01²)',
  xavier: 'Xavier (1/n)',
  he: 'He (2/n)',
}

/** Standard-normal draws shared by every configuration with the same seed, so curves differ only by the setting. */
export type Draws = { weights: Float64Array[]; input: Float64Array; topGrad: Float64Array }

export function draws(seed: number): Draws {
  const g = stream(seed)
  const fill = (size: number) => Float64Array.from({ length: size }, () => normal(g))
  return {
    weights: Array.from({ length: MAX_DEPTH }, () => fill(WIDTH * WIDTH)),
    input: fill(WIDTH * BATCH),
    topGrad: fill(WIDTH * BATCH),
  }
}

const rms = (v: Float64Array) => {
  let s = 0
  for (let i = 0; i < v.length; i++) s += v[i] * v[i]
  return Math.max(Math.sqrt(s / v.length), 1e-300)
}

export type Profile = {
  /** RMS of h_l over units and batch, for l = 0 (the input) to depth. */
  activation: number[]
  /** RMS of ∂ℓ/∂h_l, for l = 0 to depth. */
  gradient: number[]
}

/** y = scale · W x for n × B arrays (row-major: entry (i, b) is at i * B + b). */
function matmul(W: Float64Array, x: Float64Array, scale: number): Float64Array {
  const n = WIDTH
  const B = BATCH
  const y = new Float64Array(n * B)
  for (let i = 0; i < n; i++) {
    const row = i * B
    for (let k = 0; k < n; k++) {
      const w = scale * W[i * n + k]
      const col = k * B
      for (let b = 0; b < B; b++) y[row + b] += w * x[col + b]
    }
  }
  return y
}

/** y = scale · Wᵀ x. */
function matmulT(W: Float64Array, x: Float64Array, scale: number): Float64Array {
  const n = WIDTH
  const B = BATCH
  const y = new Float64Array(n * B)
  for (let i = 0; i < n; i++) {
    const row = i * B
    for (let k = 0; k < n; k++) {
      const w = scale * W[i * n + k]
      const col = k * B
      for (let b = 0; b < B; b++) y[col + b] += w * x[row + b]
    }
  }
  return y
}

const map = (x: Float64Array, f: (v: number) => number) => x.map(f)

export function propagate(
  d: Draws,
  { activation, gain, depth, alpha = 0 }: { activation: Activation; gain: number; depth: number; alpha?: number },
): Profile {
  const { f, df } = PHI[activation]
  const scale = gain / Math.sqrt(WIDTH)
  const residual = alpha > 0
  // Plain layers store their pre-activations; residual layers store their inputs. Both are what φ′ is evaluated at.
  const at: Float64Array[] = []
  let h = d.input
  const hs = [rms(h)]
  for (let l = 0; l < depth; l++) {
    if (residual) {
      at.push(h)
      const branch = matmul(d.weights[l], map(h, f), scale)
      h = h.map((v, j) => v + alpha * branch[j])
    } else {
      const a = matmul(d.weights[l], h, scale)
      at.push(a)
      h = map(a, f)
    }
    hs.push(rms(h))
  }
  let g = d.topGrad
  const gs = [rms(g)]
  for (let l = depth - 1; l >= 0; l--) {
    const slope = map(at[l], df)
    if (residual) {
      // ∂ℓ/∂h_{l-1} = g + α φ′(h_{l-1}) ⊙ (scale · Wᵀ g): the identity path carries g down unchanged.
      const back = matmulT(d.weights[l], g, scale)
      g = g.map((v, j) => v + alpha * slope[j] * back[j])
    } else {
      g = matmulT(
        d.weights[l],
        g.map((v, j) => v * slope[j]),
        scale,
      )
    }
    gs.push(rms(g))
  }
  return { activation: hs, gradient: gs.reverse() }
}
