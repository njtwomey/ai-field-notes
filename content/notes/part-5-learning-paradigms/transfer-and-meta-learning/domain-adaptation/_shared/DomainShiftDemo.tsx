import { useMemo, useState } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, useParam, type XYSeries } from 'aifn-render'
import { linspace, rng, sigmoid } from '@/lib/math'

// Covariate shift in two dimensions. The labelling rule p(y = 1 | x) is the same in both domains and has a curved
// boundary, so a linear classifier is misspecified and its best line depends on where the inputs fall.
const N_SRC = 200
const N_TGT = 200
const N_TEST = 600
const SRC_MEAN: [number, number] = [-1, 0]
const SRC_SD = 0.9
const TGT_SD = 0.6
const boundary = (x1: number) => 0.35 * x1 * x1 - 0.6
const pTrue = (x1: number, x2: number) => sigmoid(5 * (x2 - boundary(x1)))
const GRID = linspace(-4, 4, 81)

const DATA = (() => {
  const r = rng(3)
  const pair = () => [r.normal(), r.normal()] as [number, number]
  const src = Array.from({ length: N_SRC }, () => {
    const [a, b] = pair()
    return [SRC_MEAN[0] + SRC_SD * a, SRC_MEAN[1] + SRC_SD * b] as [number, number]
  })
  const y = src.map(([a, b]) => (r.uniform() < pTrue(a, b) ? 1 : 0))
  // Standard normal draws for the target; the target mean moves them, so dragging never reshuffles the sample.
  const tgt = Array.from({ length: N_TGT }, pair)
  const test = Array.from({ length: N_TEST }, pair)
  return { src, y, tgt, test }
})()

/** Solve A x = b by Gaussian elimination with partial pivoting (A is small and symmetric positive definite). */
function solve(A: number[][], b: number[]): number[] {
  const n = b.length
  const M = A.map((row, i) => [...row, b[i]])
  for (let c = 0; c < n; c++) {
    let p = c
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r
    ;[M[c], M[p]] = [M[p], M[c]]
    for (let r = c + 1; r < n; r++) {
      const f = M[r][c] / M[c][c]
      for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k]
    }
  }
  const x = new Array(n).fill(0)
  for (let r = n - 1; r >= 0; r--) {
    let s = M[r][n]
    for (let k = r + 1; k < n; k++) s -= M[r][k] * x[k]
    x[r] = s / M[r][r]
  }
  return x
}

/** Weighted, L2-regularised logistic regression by Newton's method. Rows of F are feature vectors. */
function logistic(F: number[][], y: number[], w: number[], lambda: number): number[] {
  const d = F[0].length
  let theta = new Array(d).fill(0)
  const W = w.reduce((a, b) => a + b, 0)
  for (let it = 0; it < 25; it++) {
    const g = theta.map((t) => lambda * t)
    const H = Array.from({ length: d }, (_, i) => Array.from({ length: d }, (_, j) => (i === j ? lambda : 0)))
    F.forEach((f, n) => {
      const p = sigmoid(f.reduce((s, v, k) => s + v * theta[k], 0))
      const r = (w[n] * (p - y[n])) / W
      const c = (w[n] * p * (1 - p)) / W
      for (let i = 0; i < d; i++) {
        g[i] += r * f[i]
        for (let j = 0; j < d; j++) H[i][j] += c * f[i] * f[j]
      }
    })
    const step = solve(H, g)
    theta = theta.map((t, i) => t - step[i])
  }
  return theta
}

const linear = ([a, b]: [number, number]) => [1, a, b]
const quadratic = ([a, b]: [number, number]) => [1, a, b, a * a, b * b, a * b]
const dot = (u: number[], v: number[]) => u.reduce((s, x, i) => s + x * v[i], 0)

/** Expected error of a linear classifier under the true labelling rule, averaged over the given inputs. */
function expectedError(theta: number[], xs: [number, number][]) {
  let e = 0
  for (const x of xs) {
    const p = pTrue(x[0], x[1])
    e += dot(theta, linear(x)) > 0 ? 1 - p : p
  }
  return e / xs.length
}

const line = (theta: number[]) => GRID.map((x1) => -(theta[0] + theta[1] * x1) / theta[2])

/**
 * Source and target inputs, a linear classifier trained on the labelled source, the same classifier trained with
 * importance weights from a domain classifier, and the domain classifier's estimate of the divergence.
 */
export function DomainShiftDemo({ initial = [1, 0.3] }: { initial?: [number, number] }) {
  const mx = useParam(initial[0], { min: -3, max: 3, step: 0.05 })
  const my = useParam(initial[1], { min: -2, max: 2, step: 0.05 })
  const [lambda, setLambda] = useState(1)

  const result = useMemo(() => {
    const tgt = DATA.tgt.map(([a, b]) => [mx.value + TGT_SD * a, my.value + TGT_SD * b] as [number, number])
    const test = DATA.test.map(([a, b]) => [mx.value + TGT_SD * a, my.value + TGT_SD * b] as [number, number])

    // Domain classifier on half of each sample (0 = source, 1 = target), error on the other half.
    const half = N_SRC / 2
    const trainX = [...DATA.src.slice(0, half), ...tgt.slice(0, half)]
    const trainY = [...new Array(half).fill(0), ...new Array(half).fill(1)]
    const eta = logistic(trainX.map(quadratic), trainY, new Array(2 * half).fill(1), 1e-2)
    const heldX = [...DATA.src.slice(half), ...tgt.slice(half)]
    const heldY = [...new Array(N_SRC - half).fill(0), ...new Array(N_TGT - half).fill(1)]
    const domainError = heldX.filter((x, i) => (dot(eta, quadratic(x)) > 0 ? 1 : 0) !== heldY[i]).length / heldX.length

    // With equal sample sizes, p_t(x) / p_s(x) = P(target | x) / P(source | x) = exp(logit), flattened by λ.
    const raw = DATA.src.map((x) => Math.exp(lambda * dot(eta, quadratic(x))))
    const meanW = raw.reduce((a, b) => a + b, 0) / raw.length
    const w = raw.map((v) => v / meanW)
    const ess = w.reduce((a, b) => a + b, 0) ** 2 / w.reduce((a, b) => a + b * b, 0)

    const F = DATA.src.map(linear)
    const plain = logistic(F, DATA.y, new Array(N_SRC).fill(1), 1e-3)
    const weighted = logistic(F, DATA.y, w, 1e-3)

    const series: XYSeries[] = [
      {
        name: 'source',
        type: 'scatter',
        x: DATA.src.map((p) => p[0]),
        y: DATA.src.map((p) => p[1]),
        group: DATA.y,
        groupNames: ['source, y = 0', 'source, y = 1'],
      },
      { name: 'target (unlabelled)', type: 'scatter', x: tgt.map((p) => p[0]), y: tgt.map((p) => p[1]), muted: true },
      { name: 'true boundary', type: 'line', x: GRID, y: GRID.map(boundary), emphasis: true },
      { name: 'trained on source', type: 'line', x: GRID, y: line(plain), slot: 2 },
      { name: 'importance-weighted', type: 'line', x: GRID, y: line(weighted), slot: 3, dashed: true },
    ]
    return {
      series,
      sourceError: expectedError(plain, DATA.src),
      targetPlain: expectedError(plain, test),
      targetWeighted: expectedError(weighted, test),
      domainError,
      proxyA: 2 * (1 - 2 * domainError),
      ess,
    }
  }, [mx.value, my.value, lambda])

  return (
    <Interactive
      title="Covariate shift, a domain classifier and importance weights"
      caption="Drag the target mean, or use the sliders. The labelling rule is the same in both domains, but the true boundary is curved, so the best line for the source is not the best line for the target. A logistic domain classifier separates source from target inputs: its held-out error gives the proxy distance 2(1 − 2ε), and its odds give the importance weights. Far shifts leave few effective samples and no weighting can recover."
      controls={
        <>
          <ParamSlider label="target mean, x₁" param={mx} />
          <ParamSlider label="target mean, x₂" param={my} />
          <ParamSlider label="weight exponent λ" value={lambda} onChange={setLambda} min={0} max={1} step={0.05} />
        </>
      }
      readout={
        <>
          <Readout label="source error" value={formatNumber(result.sourceError)} />
          <Readout label="target error, source fit" value={formatNumber(result.targetPlain)} />
          <Readout label="target error, weighted fit" value={formatNumber(result.targetWeighted)} />
          <Readout label="domain classifier error ε" value={formatNumber(result.domainError)} />
          <Readout label="proxy distance 2(1 − 2ε)" value={formatNumber(result.proxyA)} />
          <Readout label="effective sample size" value={`${formatNumber(result.ess)} of ${N_SRC}`} />
        </>
      }
    >
      <XYChart
        series={result.series}
        xLabel="x₁"
        yLabel="x₂"
        xRange={[-4, 4]}
        yRange={[-3, 3]}
        equalAspect
        handles={[
          {
            kind: 'point',
            at: [mx.value, my.value],
            label: 'target mean',
            onDrag: ([a, b]) => {
              mx.set(a)
              my.set(b)
            },
          },
        ]}
      />
    </Interactive>
  )
}
