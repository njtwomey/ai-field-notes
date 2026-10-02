import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamChoice,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Segment,
  type XYSeries,
} from 'aifn-render'
import { rng } from '@/lib/math'

/** Zachary's karate club: 34 members, 78 friendships (0-indexed, as in networkx's karate_club_graph). */
const EDGES: [number, number][] = [
  [0, 1],
  [0, 2],
  [0, 3],
  [0, 4],
  [0, 5],
  [0, 6],
  [0, 7],
  [0, 8],
  [0, 10],
  [0, 11],
  [0, 12],
  [0, 13],
  [0, 17],
  [0, 19],
  [0, 21],
  [0, 31],
  [1, 2],
  [1, 3],
  [1, 7],
  [1, 13],
  [1, 17],
  [1, 19],
  [1, 21],
  [1, 30],
  [2, 3],
  [2, 7],
  [2, 8],
  [2, 9],
  [2, 13],
  [2, 27],
  [2, 28],
  [2, 32],
  [3, 7],
  [3, 12],
  [3, 13],
  [4, 6],
  [4, 10],
  [5, 6],
  [5, 10],
  [5, 16],
  [6, 16],
  [8, 30],
  [8, 32],
  [8, 33],
  [9, 33],
  [13, 33],
  [14, 32],
  [14, 33],
  [15, 32],
  [15, 33],
  [18, 32],
  [18, 33],
  [19, 33],
  [20, 32],
  [20, 33],
  [22, 32],
  [22, 33],
  [23, 25],
  [23, 27],
  [23, 29],
  [23, 32],
  [23, 33],
  [24, 25],
  [24, 27],
  [24, 31],
  [25, 31],
  [26, 29],
  [26, 33],
  [27, 33],
  [28, 31],
  [28, 33],
  [29, 32],
  [29, 33],
  [30, 32],
  [30, 33],
  [31, 32],
  [31, 33],
  [32, 33],
]
/** Faction after the split: 0 = Mr. Hi (the instructor, node 0), 1 = the officer (node 33). */
const FACTION = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
const N = FACTION.length
const SEEDS = 50

type Matrix = number[][]

/** Â = D̃^{-1/2} (A + I) D̃^{-1/2}, the GCN propagation matrix with self-loops. */
const AHAT: Matrix = (() => {
  const A: number[][] = Array.from({ length: N }, (_, i) => Array.from({ length: N }, (_, j) => (i === j ? 1 : 0)))
  for (const [i, j] of EDGES) {
    A[i][j] = 1
    A[j][i] = 1
  }
  const d = A.map((row) => row.reduce((s, x) => s + x, 0))
  return A.map((row, i) => row.map((x, j) => x / Math.sqrt(d[i] * d[j])))
})()

const matmul = (X: Matrix, Y: Matrix): Matrix =>
  X.map((row) => Y[0].map((_, j) => row.reduce((s, x, k) => s + x * Y[k][j], 0)))

const gaussian = (rows: number, cols: number, scale: number, r: ReturnType<typeof rng>): Matrix =>
  Array.from({ length: rows }, () => Array.from({ length: cols }, () => scale * r.normal()))

type Mode = 'gcn' | 'linear'

/**
 * Untrained embeddings of the 34 nodes in 2-D. 'gcn': X = I and K layers H ← tanh(Â H W) with Gaussian (Glorot-scaled)
 * weights, widths 34 → 4 → … → 2. 'linear': Â^K R with a random Gaussian projection R ∈ ℝ^{34×2}. K = 0 gives a random
 * projection of the identity, i.e. random points.
 */
function embed(mode: Mode, K: number, seed: number): Matrix {
  const r = rng(seed)
  if (K === 0) return gaussian(N, 2, 1, r)
  if (mode === 'linear') {
    let H = gaussian(N, 2, 1, r)
    for (let k = 0; k < K; k++) H = matmul(AHAT, H)
    return H
  }
  const widths = [N, ...Array(K - 1).fill(4), 2]
  let H: Matrix = AHAT // Â · I
  for (let l = 0; l < K; l++) {
    const W = gaussian(widths[l], widths[l + 1], Math.sqrt(2 / (widths[l] + widths[l + 1])), r)
    H = matmul(l === 0 ? AHAT : matmul(AHAT, H), W).map((row) => row.map(Math.tanh))
  }
  return H
}

const standardise = (Z: Matrix): Matrix => {
  const out = Z.map((row) => [...row])
  for (let c = 0; c < 2; c++) {
    const m = Z.reduce((s, row) => s + row[c], 0) / N
    const sd = Math.sqrt(Z.reduce((s, row) => s + (row[c] - m) ** 2, 0) / N) || 1
    out.forEach((row) => (row[c] = (row[c] - m) / sd))
  }
  return out
}

/** Training accuracy of a logistic regression (the best linear separator found by gradient descent) on the 2-D points. */
function probeAccuracy(Z: Matrix): number {
  const X = standardise(Z)
  let w = [0, 0]
  let b = 0
  for (let it = 0; it < 400; it++) {
    const g = [0, 0]
    let gb = 0
    X.forEach((x, i) => {
      const p = 1 / (1 + Math.exp(-(w[0] * x[0] + w[1] * x[1] + b)))
      const e = p - FACTION[i]
      g[0] += e * x[0]
      g[1] += e * x[1]
      gb += e
    })
    w = [w[0] - (0.5 * g[0]) / N, w[1] - (0.5 * g[1]) / N]
    b -= (0.5 * gb) / N
  }
  const correct = X.filter((x, i) => (w[0] * x[0] + w[1] * x[1] + b > 0 ? 1 : 0) === FACTION[i]).length
  return correct / N
}

/** Mean silhouette of the two factions in the embedding (Euclidean). */
function silhouette(Z: Matrix): number {
  const dist = (a: number[], b: number[]) => Math.hypot(a[0] - b[0], a[1] - b[1])
  let total = 0
  Z.forEach((z, i) => {
    let same = 0
    let nSame = 0
    let other = 0
    let nOther = 0
    Z.forEach((w, j) => {
      if (i === j) return
      if (FACTION[j] === FACTION[i]) {
        same += dist(z, w)
        nSame++
      } else {
        other += dist(z, w)
        nOther++
      }
    })
    const a = same / nSame
    const b = other / nOther
    total += (b - a) / Math.max(a, b, 1e-12)
  })
  return total / N
}

/** A force-directed reference layout (Fruchterman–Reingold), computed once. */
const FORCE: Matrix = (() => {
  const r = rng(3)
  const P = Array.from({ length: N }, () => [r.uniform() - 0.5, r.uniform() - 0.5])
  const k = 1 / Math.sqrt(N)
  for (let it = 0; it < 300; it++) {
    const t = 0.1 * (1 - it / 300)
    const D = P.map(() => [0, 0])
    for (let i = 0; i < N; i++)
      for (let j = 0; j < N; j++) {
        if (i === j) continue
        const dx = P[i][0] - P[j][0]
        const dy = P[i][1] - P[j][1]
        const d = Math.max(Math.hypot(dx, dy), 1e-3)
        D[i][0] += ((dx / d) * k * k) / d
        D[i][1] += ((dy / d) * k * k) / d
      }
    for (const [i, j] of EDGES) {
      const dx = P[i][0] - P[j][0]
      const dy = P[i][1] - P[j][1]
      const d = Math.max(Math.hypot(dx, dy), 1e-3)
      const f = (d * d) / k
      D[i][0] -= (dx / d) * f
      D[i][1] -= (dy / d) * f
      D[j][0] += (dx / d) * f
      D[j][1] += (dy / d) * f
    }
    P.forEach((p, i) => {
      const len = Math.max(Math.hypot(D[i][0], D[i][1]), 1e-9)
      p[0] += (D[i][0] / len) * Math.min(len, t)
      p[1] += (D[i][1] / len) * Math.min(len, t)
    })
  }
  return standardise(P)
})()

const graphSeries = (Z: Matrix): { series: XYSeries[]; segments: Segment[] } => ({
  series: [
    {
      name: 'members',
      type: 'scatter',
      x: Z.map((z) => z[0]),
      y: Z.map((z) => z[1]),
      group: FACTION,
      groupNames: ["Mr. Hi's faction", "officer's faction"],
    },
  ],
  segments: EDGES.map(([i, j]) => ({ from: [Z[i][0], Z[i][1]], to: [Z[j][0], Z[j][1]] })),
})

/** Zachary's karate club embedded by an untrained GCN or by linear propagation, coloured by the real split. */
export function KarateEmbedding() {
  const layers = useParam(3, { min: 0, max: 4, step: 1 })
  const seed = useParam(1, { min: 1, max: 50, step: 1 })
  const [mode, setMode] = useState<Mode>('gcn')

  const Z = useMemo(() => standardise(embed(mode, layers.value, seed.value)), [mode, layers.value, seed.value])
  const view = useMemo(() => graphSeries(Z), [Z])
  const reference = useMemo(() => graphSeries(FORCE), [])
  const meanProbe = useMemo(() => {
    let s = 0
    for (let k = 1; k <= SEEDS; k++) s += probeAccuracy(embed(mode, layers.value, 1000 + k))
    return s / SEEDS
  }, [mode, layers.value])

  return (
    <Interactive
      title="An untrained GCN on Zachary's karate club"
      caption="Each member of the club is placed at the 2-D output of a graph network whose weights were never trained, and coloured by the faction they joined when the club split. Grey lines are friendships. With 0 layers the points are a random projection and the factions are mixed; each propagation layer averages every node with its neighbours, and after two or three layers the two factions separate. Linear propagation, Â^K R with a random projection R, does the same without any non-linearity. Step the layers and the random seed with the arrows; the right panel is a force-directed drawing of the same graph for reference."
      controls={
        <>
          <ParamSlider label="propagation layers K" param={layers} format={(v) => String(v)} withArrows />
          <ParamSlider label="random seed" param={seed} format={(v) => String(v)} withArrows />
          <ParamChoice
            label="propagation"
            value={mode}
            onChange={setMode}
            options={[
              { value: 'gcn', label: 'tanh GCN' },
              { value: 'linear', label: 'linear Â^K R' },
            ]}
          />
        </>
      }
      readout={
        <>
          <Readout label="linear-probe accuracy, this seed" value={formatNumber(probeAccuracy(Z))} />
          <Readout label={`mean over ${SEEDS} seeds`} value={formatNumber(meanProbe)} />
          <Readout label="silhouette" value={formatNumber(silhouette(Z))} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <XYChart series={view.series} segments={view.segments} xLabel="dimension 1" yLabel="dimension 2" height={320} />
        <XYChart series={reference.series} segments={reference.segments} bare height={320} />
      </div>
    </Interactive>
  )
}
