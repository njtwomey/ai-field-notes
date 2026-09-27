import { useMemo, useState } from 'react'
import { Interactive, ParamChoice, Readout, XYChart, formatNumber, type XYSeries } from '@/components/viz'
import { rng } from '@/lib/math'

// A stack of residual blocks with feed-forward branches only, at initialisation. Width D, hidden 4D, T tokens.
const D = 32
const H = 4 * D
const T = 16
const SEEDS = 2
const EPS = 1e-6

type Mode = 'post' | 'pre'

function matmul(a: Float64Array, b: Float64Array, n: number, k: number, m: number): Float64Array {
  // (n × k) · (k × m), row-major.
  const out = new Float64Array(n * m)
  for (let i = 0; i < n; i++)
    for (let p = 0; p < k; p++) {
      const v = a[i * k + p]
      if (v === 0) continue
      for (let j = 0; j < m; j++) out[i * m + j] += v * b[p * m + j]
    }
  return out
}

function matmulT(a: Float64Array, b: Float64Array, n: number, k: number, m: number): Float64Array {
  // (n × m) · (k × m)ᵀ → n × k.
  const out = new Float64Array(n * k)
  for (let i = 0; i < n; i++)
    for (let p = 0; p < k; p++) {
      let s = 0
      for (let j = 0; j < m; j++) s += a[i * m + j] * b[p * m + j]
      out[i * k + p] = s
    }
  return out
}

/** Frobenius norm of aᵀ g for a (T × H) and g (T × D): the gradient of the second branch matrix. */
function gradNorm(a: Float64Array, g: Float64Array): number {
  let s = 0
  for (let p = 0; p < H; p++)
    for (let j = 0; j < D; j++) {
      let v = 0
      for (let t = 0; t < T; t++) v += a[t * H + p] * g[t * D + j]
      s += v * v
    }
  return Math.sqrt(s)
}

function layerNorm(x: Float64Array): Float64Array {
  const y = new Float64Array(x.length)
  for (let t = 0; t < T; t++) {
    let mu = 0
    for (let j = 0; j < D; j++) mu += x[t * D + j]
    mu /= D
    let v = 0
    for (let j = 0; j < D; j++) v += (x[t * D + j] - mu) ** 2
    const s = Math.sqrt(v / D + EPS)
    for (let j = 0; j < D; j++) y[t * D + j] = (x[t * D + j] - mu) / s
  }
  return y
}

/** Gradient through LN at input x: (g − mean g − x̂ · mean(g x̂)) / s, row by row. */
function layerNormBack(x: Float64Array, g: Float64Array): Float64Array {
  const y = layerNorm(x)
  const out = new Float64Array(x.length)
  for (let t = 0; t < T; t++) {
    let mu = 0
    for (let j = 0; j < D; j++) mu += x[t * D + j]
    mu /= D
    let v = 0
    for (let j = 0; j < D; j++) v += (x[t * D + j] - mu) ** 2
    const s = Math.sqrt(v / D + EPS)
    let mg = 0
    let mgy = 0
    for (let j = 0; j < D; j++) {
      mg += g[t * D + j]
      mgy += g[t * D + j] * y[t * D + j]
    }
    mg /= D
    mgy /= D
    for (let j = 0; j < D; j++) out[t * D + j] = (g[t * D + j] - mg - y[t * D + j] * mgy) / s
  }
  return out
}

function gaussian(r: ReturnType<typeof rng>, n: number, sd: number): Float64Array {
  const out = new Float64Array(n)
  for (let i = 0; i < n; i++) out[i] = sd * r.normal()
  return out
}

/** Per-layer gradient norms of the branch output matrix and per-layer residual RMS, averaged over seeds. */
function simulate(depth: number, mode: Mode): { grad: number[]; rms: number[] } {
  const grad = new Array<number>(depth).fill(0)
  const rms = new Array<number>(depth).fill(0)
  for (let seed = 0; seed < SEEDS; seed++) {
    const r = rng(101 + seed)
    const W1 = Array.from({ length: depth }, () => gaussian(r, D * H, Math.sqrt(1 / D)))
    const W2 = Array.from({ length: depth }, () => gaussian(r, H * D, Math.sqrt(1 / H)))
    let x = gaussian(r, T * D, 1)
    const u = gaussian(r, T * D, 1)
    const cache: { x: Float64Array; h: Float64Array; a: Float64Array }[] = []
    for (let l = 0; l < depth; l++) {
      const inp = mode === 'post' ? x : layerNorm(x)
      const h = matmul(inp, W1[l], T, D, H)
      const a = h.map((v) => Math.max(v, 0))
      const f = matmul(a, W2[l], T, H, D)
      cache.push({ x, h, a })
      const s = x.map((v, i) => v + f[i])
      x = mode === 'post' ? layerNorm(s) : s
      rms[l] += Math.sqrt(x.reduce((acc, v) => acc + v * v, 0) / x.length) / SEEDS
    }
    // Loss = Σ_t u_tᵀ out_t, with a final normalisation in the pre-norm stack.
    let g = mode === 'post' ? u : layerNormBack(x, u)
    for (let l = depth - 1; l >= 0; l--) {
      const { x: x0, h, a } = cache[l]
      const f = matmul(a, W2[l], T, H, D)
      const gs =
        mode === 'post'
          ? layerNormBack(
              x0.map((v, i) => v + f[i]),
              g,
            )
          : g
      grad[l] += gradNorm(a, gs) / SEEDS
      const ga = matmulT(gs, W2[l], T, H, D)
      const gh = ga.map((v, i) => (h[i] > 0 ? v : 0))
      const gin = matmulT(gh, W1[l], T, D, H)
      const back = mode === 'post' ? gin : layerNormBack(x0, gin)
      g = (mode === 'post' ? gs : g).map((v, i) => v + back[i])
    }
  }
  return { grad, rms }
}

const RMS_RANGE: [number, undefined] = [0, undefined]

const DEPTHS = ['6', '12', '24', '48'] as const
type Depth = (typeof DEPTHS)[number]

/** Gradient norms and residual-stream scale per layer at initialisation, post-norm against pre-norm. */
export function NormGradients() {
  const [depth, setDepth] = useState<Depth>('12')
  const [view, setView] = useState<'grad' | 'rms'>('grad')
  const L = Number(depth)
  const sims = useMemo(() => ({ post: simulate(L, 'post'), pre: simulate(L, 'pre') }), [L])
  const layers = useMemo(() => Array.from({ length: L }, (_, i) => i + 1), [L])
  const series: XYSeries[] = useMemo(
    () => [
      { name: 'post-norm', type: 'line', x: layers, y: view === 'grad' ? sims.post.grad : sims.post.rms, slot: 0 },
      { name: 'pre-norm', type: 'line', x: layers, y: view === 'grad' ? sims.pre.grad : sims.pre.rms, slot: 1 },
    ],
    [layers, sims, view],
  )
  return (
    <Interactive
      title="Post-norm and pre-norm at initialisation"
      caption={`A stack of L residual blocks, each a ReLU feed-forward branch of width ${D} → ${H} → ${D}, run on ${T} random tokens with fan-in initialisation and a random linear loss on the output. "Gradient" is the Frobenius norm of the loss gradient with respect to each block's second weight matrix. In the post-norm stack the last layer's gradient does not depend on L. In the pre-norm stack the residual stream grows with depth, the final normalisation divides by its size, and the last layer's gradient shrinks roughly as 1/√L. Averaged over ${SEEDS} random draws.`}
      controls={
        <>
          <ParamChoice
            label="depth L"
            value={depth}
            onChange={setDepth}
            options={DEPTHS.map((d) => ({ value: d, label: d }))}
          />
          <ParamChoice
            label="show"
            value={view}
            onChange={setView}
            options={[
              { value: 'grad', label: 'gradient norm' },
              { value: 'rms', label: 'residual RMS' },
            ]}
          />
        </>
      }
      readout={
        <>
          <Readout
            label="last-layer gradient, post / pre"
            value={formatNumber(sims.post.grad[L - 1] / sims.pre.grad[L - 1])}
          />
          <Readout label="pre-norm residual RMS at layer L" value={formatNumber(sims.pre.rms[L - 1])} />
        </>
      }
    >
      <XYChart
        series={series}
        xLabel="layer"
        yLabel={view === 'grad' ? 'gradient norm' : 'residual-stream RMS'}
        yLog={view === 'grad'}
        yRange={view === 'rms' ? RMS_RANGE : undefined}
        height={300}
      />
    </Interactive>
  )
}
