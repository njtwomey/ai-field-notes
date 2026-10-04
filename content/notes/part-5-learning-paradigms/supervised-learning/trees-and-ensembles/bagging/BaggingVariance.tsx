import { useMemo } from 'react'
import {
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Plot,
  Points,
  Readout,
  seriesLayers,
  type SeriesSpec,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { fitTree, predictTree, type TreeNode } from '../_shared/trees'
import { linspace, toFlat } from 'aifn/foundation/tensor'
import { normal, stream, uniform } from 'aifn/foundation/random'

const N = 50
const NOISE = 0.5
const REPLICATES = 40
const B_MAX = 50
const TEST = toFlat(linspace(0.02, 0.98, 25))
const GRID = toFlat(linspace(0, 1, 201))
const X_RANGE: [number, number] = [0, 1]
const Y_RANGE: [number | undefined, number | undefined] = [-2.2, 2.2]
const truth = (x: number) => Math.sin(2 * Math.PI * x)

/**
 * Bagged regression trees on 40 replicate training sets. The spread of the bagged prediction across replicates is its
 * variance, compared with ρσ² + (1 − ρ)σ²/B using σ² and ρ measured from the same trees.
 */
export function BaggingVariance() {
  const state = useFigureState({
    b: int(10, { min: 1, max: B_MAX, step: 1, label: 'trees B', format: (v) => String(v) }),
    minLeaf: int(1, { min: 1, max: 10, step: 1, label: 'minimum leaf size', format: (v) => String(v) }),
    fraction: float(1, { min: 0.2, max: 1, step: 0.05, label: 'resample size / n' }),
  })

  const sim = useMemo(() => {
    const g = stream(7)
    // preds[r][k][t]: tree k of replicate r at test point t.
    const preds: number[][][] = []
    let firstTrees: TreeNode[] = []
    let firstData = { x: [] as number[], y: [] as number[] }
    const m = Math.max(2, Math.round(state.fraction * N))
    for (let r = 0; r < REPLICATES; r++) {
      const x = Array.from({ length: N }, () => uniform(g))
      const y = x.map((v) => truth(v) + NOISE * normal(g))
      const trees: TreeNode[] = []
      for (let k = 0; k < B_MAX; k++) {
        const idx = Array.from({ length: m }, () => Math.floor(uniform(g) * N))
        trees.push(
          fitTree(
            idx.map((i) => [x[i]]),
            idx.map((i) => y[i]),
            { maxDepth: 30, minLeaf: state.minLeaf, criterion: 'squared' },
          ),
        )
      }
      preds.push(trees.map((t) => TEST.map((v) => predictTree(t, [v]))))
      if (r === 0) {
        firstTrees = trees
        firstData = { x, y }
      }
    }

    // Variance across replicates of the average of the first B trees, averaged over test points.
    const empirical = Array.from({ length: B_MAX }, (_, k) => {
      let total = 0
      TEST.forEach((_, t) => {
        const means = preds.map((p) => {
          let s = 0
          for (let j = 0; j <= k; j++) s += p[j][t]
          return s / (k + 1)
        })
        const mu = means.reduce((a, v) => a + v, 0) / REPLICATES
        total += means.reduce((a, v) => a + (v - mu) ** 2, 0) / (REPLICATES - 1)
      })
      return total / TEST.length
    })

    // σ²: variance of one tree. ρσ²: covariance of two different trees built from the same training set.
    let sigma2 = 0
    let cov = 0
    TEST.forEach((_, t) => {
      let s1 = 0
      let s2 = 0
      let pair = 0
      for (const p of preds) {
        let s = 0
        let q = 0
        for (let k = 0; k < B_MAX; k++) {
          s += p[k][t]
          q += p[k][t] ** 2
        }
        s1 += s
        s2 += q
        pair += (s * s - q) / (B_MAX * (B_MAX - 1))
      }
      const mu = s1 / (REPLICATES * B_MAX)
      sigma2 += s2 / (REPLICATES * B_MAX) - mu * mu
      cov += pair / REPLICATES - mu * mu
    })
    sigma2 /= TEST.length
    cov /= TEST.length
    return { empirical, sigma2, rho: cov / sigma2, firstTrees, firstData }
  }, [state.minLeaf, state.fraction])

  const bs = Array.from({ length: B_MAX }, (_, k) => k + 1)
  const theory = bs.map((k) => sim.rho * sim.sigma2 + ((1 - sim.rho) * sim.sigma2) / k)
  // Predictions of every tree fitted to the first training set; the bag averages the first B of them.
  const treeCurves = useMemo(() => sim.firstTrees.map((t) => GRID.map((v) => predictTree(t, [v]))), [sim])
  const fitSeries = useMemo((): SeriesSpec[] => {
    const shown = treeCurves.slice(0, state.b)
    const bagged = GRID.map((_, i) => shown.reduce((a, c) => a + c[i], 0) / state.b)
    const many = state.b > 1
    return [
      { name: 'true function', type: 'line', x: GRID, y: GRID.map(truth), muted: true, dashed: true },
      ...shown.map((y): SeriesSpec => ({
        name: many ? 'individual trees' : 'one tree',
        type: 'line',
        x: GRID,
        y,
        slot: 1,
        thin: many,
      })),
      { name: `bag of ${state.b}`, type: 'line', x: GRID, y: bagged, slot: 0 },
      { name: 'training data', type: 'scatter', x: sim.firstData.x, y: sim.firstData.y, muted: true },
    ]
  }, [treeCurves, state.b, sim])
  const varSeries = [
    { name: 'measured variance', x: bs, y: sim.empirical, slot: 0 },
    { name: 'ρσ² + (1 − ρ)σ²/B', x: bs, y: theory, slot: 1 },
    {
      name: 'floor ρσ²',
      x: [1, B_MAX],
      y: [sim.rho * sim.sigma2, sim.rho * sim.sigma2],
      muted: true,
      dashed: true,
    },
    { name: 'chosen B', x: [state.b], y: [sim.empirical[state.b - 1]], emphasis: true },
  ] as const

  const xAxis = useAxis({ label: 'x', range: X_RANGE })
  const yAxis = useAxis({ label: 'y', range: Y_RANGE })
  const xAxis2 = useAxis({ label: 'number of trees B', hold: 'union' })
  const yAxis2 = useAxis({ label: 'variance of prediction', range: [0, undefined], hold: 'union' })
  return (
    <Figure
      title="Variance of a bagged tree against the number of trees"
      state={state}
      caption="Forty training sets of 50 points each are drawn from y = sin(2πx) plus Gaussian noise of sd 0.5. On each, B unpruned regression trees are fitted to bootstrap samples and averaged. Left: the first training set, each of its B bootstrap trees as a light line, and the bag, their average. Right: the variance of the bagged prediction across the forty training sets, averaged over x, with the formula computed from the measured single-tree variance σ² and between-tree correlation ρ. The variance falls like 1/B and levels off at ρσ². Smaller resamples make the trees less alike (lower ρ) but individually noisier (higher σ²). Drag the line labelled B, or use the slider."

      readouts={
        <>
          <Readout label="σ² (one tree)" value={formatNumber(sim.sigma2)} />
          <Readout label="ρ" value={formatNumber(sim.rho)} />
          <Readout label="variance at B" value={formatNumber(sim.empirical[state.b - 1])} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <Plot x={xAxis} y={yAxis} height={320}>
          {seriesLayers(fitSeries)}
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={320}>
          <Points {...varSeries[0]} />
          <Curve {...varSeries[1]} />
          <Curve {...varSeries[2]} />
          <Points {...varSeries[3]} />
          <Handle kind="x" at={state.b} label="B" onDrag={(x) => state.set('b', Math.round(x))} />
        </Plot>
      </div>
    </Figure>
  )
}
