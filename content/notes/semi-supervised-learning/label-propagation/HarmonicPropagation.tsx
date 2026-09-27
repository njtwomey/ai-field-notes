import { useMemo } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, useParam, type XYSeries } from '@/components/viz'
import { rng } from '@/lib/math'

const PER_CLASS = 150
const NOISE = 0.1
const X_RANGE: [number, number] = [-1.5, 2.5]
const Y_RANGE: [number | undefined, number | undefined] = [-1, 1.5]

/** Two interleaved half-circles ("two moons"). */
function moons(seed: number) {
  const g = rng(seed)
  const x: number[] = []
  const y: number[] = []
  const label: number[] = []
  for (let c = 0; c < 2; c++) {
    for (let i = 0; i < PER_CLASS; i++) {
      const t = Math.PI * g.uniform()
      x.push((c === 0 ? Math.cos(t) : 1 - Math.cos(t)) + NOISE * g.normal())
      y.push((c === 0 ? Math.sin(t) : 0.5 - Math.sin(t)) + NOISE * g.normal())
      label.push(c)
    }
  }
  return { x, y, label }
}

/** Solves A z = b for a symmetric positive-definite A by Cholesky factorisation, in place on copies. */
function solveSpd(A: number[][], b: number[]): number[] {
  const n = b.length
  const L = A.map((row) => [...row])
  for (let j = 0; j < n; j++) {
    let s = L[j][j]
    for (let k = 0; k < j; k++) s -= L[j][k] * L[j][k]
    const d = Math.sqrt(Math.max(s, 1e-300))
    L[j][j] = d
    for (let i = j + 1; i < n; i++) {
      let t = L[i][j]
      for (let k = 0; k < j; k++) t -= L[i][k] * L[j][k]
      L[i][j] = t / d
    }
  }
  const z = [...b]
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < i; k++) z[i] -= L[i][k] * z[k]
    z[i] /= L[i][i]
  }
  for (let i = n - 1; i >= 0; i--) {
    for (let k = i + 1; k < n; k++) z[i] -= L[k][i] * z[k]
    z[i] /= L[i][i]
  }
  return z
}

/**
 * The harmonic solution of Zhu, Ghahramani and Lafferty on a Gaussian-weighted graph: each unlabelled point's score is
 * the weighted average of its neighbours' scores, with labelled points clamped to 0 or 1.
 */
export function HarmonicPropagation() {
  const perClass = useParam(2, { min: 1, max: 10, step: 1 })
  const sigma = useParam(0.15, { min: 0.01, max: 1, step: 0.01 })
  const seed = useParam(4, { min: 1, max: 20, step: 1 })

  const data = useMemo(() => moons(seed.value), [seed.value])
  const labelsPerClass = perClass.value
  const width = sigma.value

  const r = useMemo(() => {
    const n = data.x.length
    // The first perClass points of each class are labelled; points are generated in random positions, so this is a
    // random choice.
    const labelled = new Set<number>()
    for (let c = 0; c < 2; c++) for (let i = 0; i < labelsPerClass; i++) labelled.add(c * PER_CLASS + i)
    const L = [...labelled]
    const U = data.x.map((_, i) => i).filter((i) => !labelled.has(i))
    const w = (i: number, j: number) =>
      Math.exp(-((data.x[i] - data.x[j]) ** 2 + (data.y[i] - data.y[j]) ** 2) / (2 * width ** 2))
    // (D_UU − W_UU) f_U = W_UL f_L. A tiny ridge ties isolated points to the neutral score 0.5.
    const eps = 1e-8
    const A = U.map((i) => U.map((j) => (i === j ? 0 : -w(i, j))))
    const b = U.map(() => 0.5 * eps)
    U.forEach((i, a) => {
      let degree = 0
      for (let j = 0; j < n; j++) if (j !== i) degree += w(i, j)
      A[a][a] = degree + eps
      for (const j of L) b[a] += w(i, j) * data.label[j]
    })
    const fU = solveSpd(A, b)
    const f = new Array<number>(n).fill(0)
    L.forEach((i) => (f[i] = data.label[i]))
    U.forEach((i, a) => (f[i] = fU[a]))
    const pred = f.map((v) => (v > 0.5 ? 1 : 0))
    const accuracy = U.filter((i) => pred[i] === data.label[i]).length / U.length
    // Supervised baseline: 1-nearest neighbour among the labelled points only.
    const nn =
      U.filter((i) => {
        let best = L[0]
        let bestD = Infinity
        for (const j of L) {
          const d = (data.x[i] - data.x[j]) ** 2 + (data.y[i] - data.y[j]) ** 2
          if (d < bestD) {
            bestD = d
            best = j
          }
        }
        return data.label[best] === data.label[i]
      }).length / U.length
    return { L, U, pred, accuracy, nn, undecided: U.filter((i) => Math.abs(f[i] - 0.5) < 1e-3).length }
  }, [data, labelsPerClass, width])

  const before: XYSeries[] = [
    { name: 'unlabelled', type: 'scatter', x: r.U.map((i) => data.x[i]), y: r.U.map((i) => data.y[i]), muted: true },
    {
      name: 'labelled',
      type: 'scatter',
      x: r.L.map((i) => data.x[i]),
      y: r.L.map((i) => data.y[i]),
      group: r.L.map((i) => data.label[i]),
      groupNames: ['class 0', 'class 1'],
    },
  ]
  const after: XYSeries[] = [
    {
      name: 'predicted',
      type: 'scatter',
      x: data.x,
      y: data.y,
      group: r.pred,
      groupNames: ['predicted 0', 'predicted 1'],
    },
    { name: 'labelled', type: 'scatter', x: r.L.map((i) => data.x[i]), y: r.L.map((i) => data.y[i]), emphasis: true },
  ]

  return (
    <Interactive
      title="Label propagation on two moons"
      caption="Three hundred points on two interleaved half-circles, of which only a few per class are labelled (left). The graph joins every pair of points with weight exp(−d²/2σ²). The harmonic solution gives each unlabelled point the weighted average score of its neighbours, so labels flow along the dense moons rather than across the gap (right; diamonds are the labelled points). A supervised 1-nearest-neighbour rule using only the labelled points cuts straight across the moons. Too large a σ joins the moons and labels leak across; too small a σ disconnects the graph and leaves points undecided."
      controls={
        <>
          <ParamSlider label="labelled points per class" param={perClass} format={(v) => String(v)} />
          <ParamSlider label="kernel width σ" param={sigma} />
          <ParamSlider label="data seed" param={seed} format={(v) => String(v)} />
        </>
      }
      readout={
        <>
          <Readout label="propagation accuracy" value={formatNumber(r.accuracy)} />
          <Readout label="1-NN on labels only" value={formatNumber(r.nn)} />
          <Readout label="undecided points" value={String(r.undecided)} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <XYChart series={before} xLabel="x₁" yLabel="x₂" xRange={X_RANGE} yRange={Y_RANGE} height={320} />
        <XYChart series={after} xLabel="x₁" yLabel="x₂" xRange={X_RANGE} yRange={Y_RANGE} height={320} />
      </div>
    </Interactive>
  )
}
