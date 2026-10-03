import { useMemo, useState } from 'react'
import {
  Heatmap,
  Interactive,
  ParamChoice,
  ParamSlider,
  ParamSwitch,
  Readout,
  formatNumber,
  useParam,
  type Handle,
  type HeatmapOverlay,
} from 'aifn-render'
import { linspace, rng } from '@/lib/math'
import { splitRhat } from '../../_shared/mcmc'

/** Bayesian logistic regression without intercept: y ~ Bern(σ(wᵀx)), w ~ N(0, τ²I), N = 50 points. */
const N = 50
const TAU = 2
const TRUE_W: [number, number] = [1.5, -1]
/** Points drawn per burn-in path, and kept samples drawn in all; the readouts use every sample. */
const BURN_POINTS = 400
const SHOWN = 3000

const DATA = (() => {
  const g = rng(3)
  return Array.from({ length: N }, () => {
    const x: [number, number] = [g.normal(), g.normal()]
    const p = 1 / (1 + Math.exp(-(TRUE_W[0] * x[0] + TRUE_W[1] * x[1])))
    return { x, y: g.uniform() < p ? 1 : 0 }
  })
})()

const softplus = (z: number) => (z > 0 ? z + Math.log1p(Math.exp(-z)) : Math.log1p(Math.exp(z)))

function logPosterior(w0: number, w1: number) {
  let s = -(w0 * w0 + w1 * w1) / (2 * TAU * TAU)
  for (const { x, y } of DATA) {
    const z = w0 * x[0] + w1 * x[1]
    s += y * z - softplus(z)
  }
  return s
}

const W0 = linspace(-0.5, 5.5, 61)
const W1 = linspace(-3.5, 1.5, 61)

/** Exact posterior on the grid, normalised to sum to 1, with its mean and marginal standard deviations. */
const EXACT = (() => {
  const logs = W1.map((b) => W0.map((a) => logPosterior(a, b)))
  const top = Math.max(...logs.flat())
  const z = logs.map((row) => row.map((l) => Math.exp(l - top)))
  const total = z.flat().reduce((s, v) => s + v, 0)
  let m0 = 0
  let m1 = 0
  W1.forEach((b, i) => W0.forEach((a, j) => ((m0 += (a * z[i][j]) / total), (m1 += (b * z[i][j]) / total))))
  let v0 = 0
  W1.forEach((_, i) => W0.forEach((a, j) => (v0 += ((a - m0) ** 2 * z[i][j]) / total)))
  return { z, mean: [m0, m1], sd0: Math.sqrt(v0) }
})()

const BATCH_OPTIONS = [
  { value: '1', label: '1' },
  { value: '5', label: '5' },
  { value: '10', label: '10' },
  { value: '50', label: '50 (full)' },
] as const
type Batch = (typeof BATCH_OPTIONS)[number]['value']

/** Per-example gradient of log p(y | x, w). */
function exampleGrad(w: [number, number], i: number): [number, number] {
  const { x, y } = DATA[i]
  const r = y - 1 / (1 + Math.exp(-(w[0] * x[0] + w[1] * x[1])))
  return [r * x[0], r * x[1]]
}

/**
 * SGLD (Welling & Teh 2011): w ← w + (ε_t/2)(∇log p(w) + (N/n) Σ_batch ∇log p(y_i | x_i, w)) + N(0, ε_t I), with
 * ε_t = a (1 + t/100)^(−γ). Without injected noise it is minibatch SGD on the negative log posterior.
 */
function sgld(
  a: number,
  gamma: number,
  n: number,
  steps: number,
  noise: boolean,
  start: [number, number],
  seed: number,
) {
  const g = rng(seed)
  const w: [number, number] = [...start]
  const path = { x: [w[0]], y: [w[1]] }
  const order = Array.from({ length: N }, (_, i) => i)
  let eps = a
  for (let t = 0; t < steps; t++) {
    eps = a * (1 + t / 100) ** -gamma
    // Partial Fisher–Yates shuffle: the first n entries are a batch drawn without replacement.
    for (let i = 0; i < n; i++) {
      const j = i + Math.floor(g.uniform() * (N - i))
      ;[order[i], order[j]] = [order[j], order[i]]
    }
    let g0 = -w[0] / (TAU * TAU)
    let g1 = -w[1] / (TAU * TAU)
    for (let i = 0; i < n; i++) {
      const [d0, d1] = exampleGrad(w, order[i])
      g0 += (N / n) * d0
      g1 += (N / n) * d1
    }
    const sd = noise ? Math.sqrt(eps) : 0
    w[0] += (eps / 2) * g0 + sd * g.normal()
    w[1] += (eps / 2) * g1 + sd * g.normal()
    if (!Number.isFinite(w[0]) || Math.abs(w[0]) > 1e3 || Math.abs(w[1]) > 1e3) break
    path.x.push(w[0])
    path.y.push(w[1])
  }
  return { path, eps, end: w }
}

/**
 * Ratio of the trace of the minibatch-gradient noise covariance in one update, (ε²/4)(N²/n)(1 − (n−1)/(N−1)) tr V(w),
 * to the trace of the injected noise covariance, 2ε. V is the covariance of the per-example gradients at w.
 */
function noiseRatio(eps: number, n: number, w: [number, number]) {
  const grads = DATA.map((_, i) => exampleGrad(w, i))
  const m0 = grads.reduce((s, d) => s + d[0], 0) / N
  const m1 = grads.reduce((s, d) => s + d[1], 0) / N
  const trV = grads.reduce((s, d) => s + (d[0] - m0) ** 2 + (d[1] - m1) ** 2, 0) / N
  const fpc = 1 - (n - 1) / (N - 1)
  return (((eps * eps) / 4) * ((N * N) / n) * fpc * trV) / (2 * eps)
}

export function SgldLogistic() {
  const [batch, setBatch] = useState<Batch>('5')
  const [noise, setNoise] = useState(true)
  const chains = useParam(4, { min: 1, max: 10, step: 1 })
  const a = useParam(0.05, { min: 0.005, max: 0.3, step: 0.005 })
  const gamma = useParam(0.33, { min: 0, max: 1, step: 0.01 })
  const steps = useParam(2000, { min: 10, max: 5000, step: 10 })
  const seed = useParam(1, { min: 1, max: 20, step: 1 })
  const s0 = useParam(0, { min: -0.5, max: 5.5, step: 0.05 })
  const s1 = useParam(0, { min: -3.5, max: 1.5, step: 0.05 })
  const n = Number(batch)

  // Each chain has its own random stream and starts at the common start point.
  const runs = useMemo(
    () =>
      Array.from({ length: chains.value }, (_, k) =>
        sgld(a.value, gamma.value, n, steps.value, noise, [s0.value, s1.value], seed.value * 1000 + k),
      ),
    [chains.value, a.value, gamma.value, n, steps.value, noise, s0.value, s1.value, seed.value],
  )
  const run = runs[0]

  const { overlay, mean, sd0, rhat } = useMemo(() => {
    const many = runs.length > 1
    // Discard the first quarter of each chain as burn-in; pool and average the rest.
    const kept = runs.map((r) => {
      const from = Math.floor(r.path.x.length / 4)
      return { from, x: r.path.x.slice(from), y: r.path.y.slice(from) }
    })
    const xs = kept.flatMap((c) => c.x)
    const ys = kept.flatMap((c) => c.y)
    const k = xs.length || 1
    const mean = [xs.reduce((s, v) => s + v, 0) / k, ys.reduce((s, v) => s + v, 0) / k]
    const sd0 = Math.sqrt(xs.reduce((s, v) => s + (v - mean[0]) ** 2, 0) / k)
    // Split R̂ needs chains of equal length; a diverged chain stops early and has no R̂.
    const equal = kept.every((c) => c.x.length === kept[0].x.length) && kept[0].x.length >= 4
    const rhat = equal ? Math.max(splitRhat(kept.map((c) => c.x)), splitRhat(kept.map((c) => c.y))) : null
    // Drawing only: each burn-in path is thinned to about BURN_POINTS points, the kept samples to about SHOWN in all.
    const burn = runs.map((r, j): HeatmapOverlay => {
      const stride = Math.max(1, Math.ceil(kept[j].from / BURN_POINTS))
      const idx: number[] = []
      for (let i = 0; i <= kept[j].from; i += stride) idx.push(i)
      return {
        name: 'burn-in',
        type: 'line',
        x: idx.map((i) => r.path.x[i]),
        y: idx.map((i) => r.path.y[i]),
        slot: 2,
        thin: many,
      }
    })
    const stride = Math.max(1, Math.ceil(xs.length / SHOWN))
    const overlay: HeatmapOverlay[] = [
      ...burn,
      {
        name: 'kept samples',
        type: 'scatter',
        x: xs.filter((_, i) => i % stride === 0),
        y: ys.filter((_, i) => i % stride === 0),
        slot: 1,
      },
    ]
    return { overlay, mean, sd0, rhat }
  }, [runs])

  const ratio = noiseRatio(run.eps, n, run.end)
  const handles: Handle[] = [
    {
      kind: 'point',
      at: [s0.value, s1.value],
      label: 'start',
      onDrag: ([x, y]) => {
        s0.set(x)
        s1.set(y)
      },
    },
  ]

  return (
    <Interactive
      title="SGLD on a Bayesian logistic regression"
      caption="The shaded grid is the exact posterior of the two weights of a logistic regression on 50 points with prior N(0, 4I). SGLD takes minibatch gradient steps of size ε_t = a(1 + t/100)^(−γ) and adds N(0, ε_t I) noise. Several chains run from the same start, each with its own random stream; their burn-in paths (the first quarter of each run) are the light lines, and the chains slider sets how many run. Drag the start point. The kept samples, pooled over the chains, cover the posterior, and their mean and spread match the exact values. Split R̂ compares the kept samples of the chains: near 1 they agree. Switching the noise off turns SGLD into minibatch SGD: the chain settles near the posterior mode and its spread shrinks to the small jitter left by minibatch noise, far below the posterior spread. With γ = 0 and a large a the step stays large, the minibatch noise stays comparable to the injected noise, and the samples spread too wide. The last readout compares the two noise sources at the final step of the first chain."
      controls={
        <>
          <ParamChoice label="minibatch size n" value={batch} onChange={setBatch} options={BATCH_OPTIONS} />
          <ParamSlider label="chains" param={chains} format={(v) => String(v)} withArrows />
          <ParamSwitch label="inject Langevin noise" checked={noise} onChange={setNoise} />
          <ParamSlider label="initial step a" param={a} format={(v) => v.toFixed(3)} />
          <ParamSlider label="decay exponent γ" param={gamma} />
          <ParamSlider label="iterations" param={steps} format={(v) => String(v)} withArrows />
          <ParamSlider label="noise seed" param={seed} format={(v) => String(v)} withArrows />
        </>
      }
      readout={
        <>
          <Readout
            label="posterior mean (exact → SGLD)"
            value={`(${formatNumber(EXACT.mean[0])}, ${formatNumber(EXACT.mean[1])}) → (${formatNumber(mean[0])}, ${formatNumber(mean[1])})`}
          />
          <Readout label="sd of w₁ (exact → SGLD)" value={`${formatNumber(EXACT.sd0)} → ${formatNumber(sd0)}`} />
          <Readout label="split R̂ (worse weight)" value={rhat === null ? 'a chain diverged' : formatNumber(rhat)} />
          <Readout label="final step ε_t" value={run.eps.toExponential(2)} />
          <Readout
            label="minibatch / injected noise variance"
            value={noise ? formatNumber(ratio) : 'no injected noise'}
          />
        </>
      }
    >
      <Heatmap
        x={W0}
        y={W1}
        z={EXACT.z}
        xLabel="w₁"
        yLabel="w₂"
        valueLabel="posterior (relative)"
        overlay={overlay}
        handles={handles}
        height={420}
      />
    </Interactive>
  )
}
