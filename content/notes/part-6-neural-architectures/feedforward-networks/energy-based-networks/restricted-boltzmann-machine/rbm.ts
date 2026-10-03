/**
 * A binary restricted Boltzmann machine: V visible and H hidden {0, 1} units, weights only between the layers. Trained
 * with CD-k or persistent contrastive divergence; the exact log-likelihood is computed by summing over the 2^H hidden
 * states, which is affordable for H ≤ 12.
 */

export type Uniform = () => number

/** W is V × H, row-major: W[i·H + j] joins visible i and hidden j. */
export type Rbm = { V: number; H: number; W: Float64Array; b: Float64Array; c: Float64Array }

const sigmoid = (x: number) => 1 / (1 + Math.exp(-x))
const softplus = (x: number) => (x > 30 ? x : Math.log1p(Math.exp(x)))

/** p(h_j = 1 | v) = σ((c_j + Σ_i W_ij v_i) / T). */
export function hiddenProbs(m: Rbm, v: ArrayLike<number>, T = 1): Float64Array {
  const p = new Float64Array(m.H)
  for (let j = 0; j < m.H; j++) {
    let a = m.c[j]
    for (let i = 0; i < m.V; i++) a += m.W[i * m.H + j] * v[i]
    p[j] = sigmoid(a / T)
  }
  return p
}

/** p(v_i = 1 | h) = σ((b_i + Σ_j W_ij h_j) / T). */
export function visibleProbs(m: Rbm, h: ArrayLike<number>, T = 1): Float64Array {
  const p = new Float64Array(m.V)
  for (let i = 0; i < m.V; i++) {
    let a = m.b[i]
    for (let j = 0; j < m.H; j++) a += m.W[i * m.H + j] * h[j]
    p[i] = sigmoid(a / T)
  }
  return p
}

export const sample = (p: ArrayLike<number>, uniform: Uniform) => Float64Array.from(p, (q) => (uniform() < q ? 1 : 0))

/** log Z = log Σ_h exp(c·h) Π_i (1 + exp(b_i + W_i·h)), summed over all 2^H hidden states. */
export function logPartition(m: Rbm): number {
  const terms: number[] = []
  const h = new Float64Array(m.H)
  for (let code = 0; code < 1 << m.H; code++) {
    let t = 0
    for (let j = 0; j < m.H; j++) {
      h[j] = (code >> j) & 1
      t += m.c[j] * h[j]
    }
    for (let i = 0; i < m.V; i++) {
      let a = m.b[i]
      for (let j = 0; j < m.H; j++) if (h[j]) a += m.W[i * m.H + j]
      t += softplus(a)
    }
    terms.push(t)
  }
  const max = Math.max(...terms)
  return max + Math.log(terms.reduce((s, t) => s + Math.exp(t - max), 0))
}

/** −F(v) = b·v + Σ_j log(1 + exp(c_j + W_j·v)): the log of the unnormalised marginal p(v). */
export function negFreeEnergy(m: Rbm, v: ArrayLike<number>): number {
  let f = 0
  for (let i = 0; i < m.V; i++) f += m.b[i] * v[i]
  for (let j = 0; j < m.H; j++) {
    let a = m.c[j]
    for (let i = 0; i < m.V; i++) a += m.W[i * m.H + j] * v[i]
    f += softplus(a)
  }
  return f
}

export function meanLogLikelihood(m: Rbm, data: ArrayLike<number>[]): number {
  const logZ = logPartition(m)
  return data.reduce((s, v) => s + negFreeEnergy(m, v) - logZ, 0) / data.length
}

export type Method = 'cd1' | 'cd10' | 'pcd'
export type RbmSnapshot = { epoch: number; W: Float64Array; b: Float64Array; c: Float64Array; logLik: number }

/**
 * Train with mini-batches. CD-k starts each negative chain at the data vector and runs k block-Gibbs steps; PCD keeps
 * one chain per batch element across updates and runs one step per update. The positive phase uses p(h | v) exactly.
 */
export function trainRbm(
  data: ArrayLike<number>[],
  H: number,
  {
    epochs,
    rate,
    method,
    batch,
    uniform,
    every = 5,
  }: {
    epochs: number
    rate: number
    method: Method
    batch: number
    uniform: Uniform
    /** Epochs between exact log-likelihood evaluations. */
    every?: number
  },
): RbmSnapshot[] {
  const V = data[0].length
  const m: Rbm = { V, H, W: new Float64Array(V * H), b: new Float64Array(V), c: new Float64Array(H) }
  for (let k = 0; k < m.W.length; k++) m.W[k] = (uniform() - 0.5) * 0.2
  const k = method === 'cd10' ? 10 : 1
  let persistent: Float64Array[] = Array.from({ length: batch }, () => sample(new Float64Array(V).fill(0.5), uniform))
  const snaps: RbmSnapshot[] = []
  let logLik = meanLogLikelihood(m, data)
  const snap = (epoch: number) => snaps.push({ epoch, W: m.W.slice(), b: m.b.slice(), c: m.c.slice(), logLik })
  snap(0)
  const order = data.map((_, i) => i)
  for (let epoch = 1; epoch <= epochs; epoch++) {
    for (let i = order.length - 1; i > 0; i--) {
      const j = Math.floor(uniform() * (i + 1))
      ;[order[i], order[j]] = [order[j], order[i]]
    }
    for (let start = 0; start < order.length; start += batch) {
      const rows = order.slice(start, start + batch).map((i) => data[i])
      const dW = new Float64Array(V * H)
      const db = new Float64Array(V)
      const dc = new Float64Array(H)
      const negStarts = method === 'pcd' ? persistent.slice(0, rows.length) : rows
      const ends: Float64Array[] = []
      rows.forEach((v, r) => {
        const ph = hiddenProbs(m, v)
        let vn = Float64Array.from(negStarts[r])
        let hn = sample(hiddenProbs(m, vn), uniform)
        for (let step = 0; step < k; step++) {
          vn = sample(visibleProbs(m, hn), uniform)
          const phn = hiddenProbs(m, vn)
          if (step === k - 1) {
            // Positive minus negative statistics, with probabilities in place of the last hidden sample.
            for (let i = 0; i < V; i++) {
              db[i] += v[i] - vn[i]
              for (let j = 0; j < H; j++) dW[i * H + j] += v[i] * ph[j] - vn[i] * phn[j]
            }
            for (let j = 0; j < H; j++) dc[j] += ph[j] - phn[j]
          } else hn = sample(phn, uniform)
        }
        ends.push(vn)
      })
      if (method === 'pcd') persistent = [...ends, ...persistent.slice(ends.length)]
      const scale = rate / rows.length
      for (let q = 0; q < dW.length; q++) m.W[q] += scale * dW[q]
      for (let i = 0; i < V; i++) m.b[i] += scale * db[i]
      for (let j = 0; j < H; j++) m.c[j] += scale * dc[j]
    }
    if (epoch % every === 0 || epoch === epochs) logLik = meanLogLikelihood(m, data)
    snap(epoch)
  }
  return snaps
}

/** The bars-and-stripes patterns on an s × s grid: every set of full rows, or every set of full columns (2^{s+1} − 2). */
export function barsAndStripes(s: number): Float64Array[] {
  const out = new Map<string, Float64Array>()
  for (let mask = 0; mask < 1 << s; mask++)
    for (const rowsOn of [true, false]) {
      const v = new Float64Array(s * s)
      for (let r = 0; r < s; r++) for (let c = 0; c < s; c++) v[r * s + c] = (mask >> (rowsOn ? r : c)) & 1
      out.set(v.join(''), v)
    }
  return [...out.values()]
}
