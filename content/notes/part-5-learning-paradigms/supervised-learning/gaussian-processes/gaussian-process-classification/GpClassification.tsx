import { useMemo } from 'react'
import { Curve, Figure, float, formatNumber, int, Plot, Points, Readout, useAxis, useFigureState } from 'aifn-render'
import { cholesky, forward, backward, gram, makeKernel } from '../_shared/gp'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'
import { stream, uniform } from 'aifn-compute/foundation/random'
import { sigmoid } from 'aifn-compute/numerics/special'

const GRID = toFlat(linspace(-6, 6, 121))
const X_RANGE: [number, number] = [-6, 6]
const P_RANGE: [number | undefined, number | undefined] = [-0.05, 1.05]
const F_RANGE: [number | undefined, number | undefined] = [-7, 7]
/** Quadrature nodes on ±5 sd for averaging the sigmoid over the Gaussian latent posterior. */
const NODES = toFlat(linspace(-5, 5, 41))
const WEIGHTS = (() => {
  const w = NODES.map((z) => Math.exp(-0.5 * z * z))
  const total = w.reduce((s, v) => s + v, 0)
  return w.map((v) => v / total)
})()
const trueProb = (x: number) => sigmoid(3 * Math.sin(0.8 * x))

/** Laplace approximation for binary GP classification with the logistic likelihood (Rasmussen & Williams, alg. 3.1). */
function laplace(x: number[], t: number[], kernel: (a: number, b: number) => number) {
  const n = x.length
  const K = gram(kernel, x, x)
  let f = new Array<number>(n).fill(0)
  let iterations = 0
  let pi = f.map((v: number) => sigmoid(v))
  let sW = pi.map((p) => Math.sqrt(p * (1 - p)))
  let L = cholesky(K.map((row, i) => row.map((v, j) => sW[i] * v * sW[j] + (i === j ? 1 : 0))))
  for (; iterations < 50; iterations++) {
    pi = f.map((v: number) => sigmoid(v))
    sW = pi.map((p) => Math.sqrt(p * (1 - p)))
    L = cholesky(K.map((row, i) => row.map((v, j) => sW[i] * v * sW[j] + (i === j ? 1 : 0))))
    // Newton step f ← K a, with a = b − W^½ B⁻¹ W^½ K b and b = W f + ∇ log p(y | f).
    const b = f.map((fi, i) => pi[i] * (1 - pi[i]) * fi + (t[i] - pi[i]))
    const Kb = K.map((row) => row.reduce((s, v, j) => s + v * b[j], 0))
    const c = backward(
      L,
      forward(
        L,
        Kb.map((v, i) => sW[i] * v),
      ),
    )
    const a = b.map((bi, i) => bi - sW[i] * c[i])
    const next = K.map((row) => row.reduce((s, v, j) => s + v * a[j], 0))
    const step = Math.max(...next.map((v, i) => Math.abs(v - f[i])))
    f = next
    if (step < 1e-8) break
  }
  pi = f.map((v: number) => sigmoid(v))
  sW = pi.map((p) => Math.sqrt(p * (1 - p)))
  L = cholesky(K.map((row, i) => row.map((v, j) => sW[i] * v * sW[j] + (i === j ? 1 : 0))))
  const grad = t.map((ti, i) => ti - pi[i])
  return { f, grad, sW, L, iterations }
}

/** GP classification in one dimension: latent posterior and predictive probability under the Laplace approximation. */
export function GpClassification() {
  const state = useFigureState({
    n: int(30, { min: 5, max: 80, step: 1, label: 'training points N', format: (v) => String(v) }),
    logEll: float(0.1, {
      min: -0.7,
      max: 0.8,
      step: 0.01,
      label: 'length-scale ℓ',
      points_per_decade: 2,
      logTransform: 'value-is-log',
      format: (v) => formatNumber(10 ** v),
    }),
    sf: float(2, { min: 0.3, max: 5, step: 0.1, label: 'signal sd σ_f' }),
    seed: int(3, { min: 1, max: 20, step: 1, label: 'data seed', format: (v) => String(v) }),
  })
  const ell = 10 ** state.logEll

  const data = useMemo(() => {
    const g = stream(state.seed)
    const x = Array.from({ length: state.n }, () => -5 + 10 * uniform(g))
    return { x, t: x.map((v) => (uniform(g) < trueProb(v) ? 1 : 0)) }
  }, [state.n, state.seed])

  const r = useMemo(() => {
    const k = makeKernel('se', { ell, sf: state.sf })
    const fit = laplace(data.x, data.t, k)
    const mean: number[] = []
    const sd: number[] = []
    const averaged: number[] = []
    for (const s of GRID) {
      const ks = data.x.map((xi) => k(xi, s))
      const m = ks.reduce((acc, v, i) => acc + v * fit.grad[i], 0)
      const v = forward(
        fit.L,
        ks.map((val, i) => fit.sW[i] * val),
      )
      const sdv = Math.sqrt(Math.max(k(s, s) - v.reduce((acc, u) => acc + u * u, 0), 0))
      mean.push(m)
      sd.push(sdv)
      averaged.push(NODES.reduce((acc, z, i) => acc + WEIGHTS[i] * sigmoid(m + sdv * z), 0))
    }
    return { fit, mean, sd, averaged }
  }, [data, ell, state.sf])

  const latent = [
    { name: 'mean + 2 sd', x: GRID, y: r.mean.map((m, i) => m + 2 * r.sd[i]), slot: 0, dashed: true },
    { name: 'mean − 2 sd', x: GRID, y: r.mean.map((m, i) => m - 2 * r.sd[i]), slot: 0, dashed: true },
    { name: 'latent mean', x: GRID, y: r.mean, slot: 0 },
  ] as const
  const probability = [
    { name: 'true p(y = 1 | x)', x: GRID, y: GRID.map(trueProb), muted: true, dashed: true },
    { name: 'plug-in σ(mean)', x: GRID, y: r.mean.map((v: number) => sigmoid(v)), slot: 2 },
    { name: 'averaged probability', x: GRID, y: r.averaged, slot: 0 },
    { name: 'labels', x: data.x, y: data.t, slot: 1 },
  ] as const

  const xAxis = useAxis({ label: 'x', range: X_RANGE })
  const yAxis = useAxis({ label: 'latent f(x)', range: F_RANGE })
  const xAxis2 = useAxis({ label: 'x', range: X_RANGE })
  const yAxis2 = useAxis({ label: 'p(y = 1 | x)', range: P_RANGE })
  return (
    <Figure
      title="Gaussian process classification with the Laplace approximation"
      state={state}
      caption="Binary labels (0 or 1) drawn with probability given by the dashed curve. Left: the Laplace posterior over the latent function f, its mean and ±2 sd. Right: the predicted probability of class 1. The plug-in curve squashes the latent mean through the sigmoid; the averaged curve integrates the sigmoid over the latent posterior. Averaging pulls predictions toward 0.5 where the latent function is uncertain, far from the data and between classes. A large signal sd σ_f lets the latent function grow large and the plug-in probability saturate."

      readouts={<Readout label="Newton iterations" value={String(r.fit.iterations + 1)} />}
    >
      <div className="grid gap-4 md:grid-cols-2">
        <Plot x={xAxis} y={yAxis} height={320}>
          <Curve {...latent[0]} />
          <Curve {...latent[1]} />
          <Curve {...latent[2]} />
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={320}>
          <Curve {...probability[0]} />
          <Curve {...probability[1]} />
          <Curve {...probability[2]} />
          <Points {...probability[3]} />
        </Plot>
      </div>
    </Figure>
  )
}
