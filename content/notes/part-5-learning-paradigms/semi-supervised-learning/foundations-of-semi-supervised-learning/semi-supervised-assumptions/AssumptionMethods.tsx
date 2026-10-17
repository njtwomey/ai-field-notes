import { useMemo, useState } from 'react'
import {
  choice,
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Plot,
  Points,
  Raster,
  Readout,
  useAxis,
  useComputed,
  useFigureState,
  variants,
} from 'aifn-render'
import { blobs, circles, moons, spirals } from 'aifn-methods/data/synthetic'
import { gaussianMixture } from 'aifn-methods/unsupervised/clustering'
import { stream } from 'aifn-compute/foundation/random'
import { dataset } from 'aifn-compute/learning/estimators'
import { fromRows, toArray, toFlat } from 'aifn-compute/foundation/tensor'
import { trace } from 'aifn-compute/foundation/trace'
import { harmonicExtension, harmonicLabels, pointAffinity } from 'aifn-compute/graph/propagation'
import { assignNearest } from 'aifn-compute/numerics/neighbours'
import { clusterClasses, clusterThenLabelProbabilities, selfTrainingSteps } from 'aifn-methods/learning/semi-supervised'
import { centredField, predictionGrid, randomFeatureLogistic } from '../_shared/models'

const N = 200
const NEIGHBOURS = 8
const NX = 41
const NY = 31
const X_RANGE: [number, number] = [-1.8, 1.8]
const Y_RANGE: [number, number] = [-1.3, 1.3]
const GRID = predictionGrid(X_RANGE, Y_RANGE, NX, NY)
// Self-training: a smooth nonlinear classifier (logistic regression on 40 random Fourier features), adding the 10 most
// confident points of each class per round and keeping them (Blum and Mitchell's form), so the classes stay balanced.
const BASE = randomFeatureLogistic({ components: 40, lengthscale: 0.5, l2: 0.01, seed: 1 })
const ROUNDS = 12

type DataKey = 'moons' | 'blobs' | 'circles' | 'spirals'

/** Each data set's starting positions of the two labelled points (class 0, class 1). */
const START: Record<DataKey, [[number, number], [number, number]]> = {
  moons: [
    [-1.5, -0.25],
    [1.5, 0.25],
  ],
  blobs: [
    [-0.6, 0.6],
    [1.2, 0.3],
  ],
  circles: [
    [1, 0],
    [0.5, 0],
  ],
  spirals: [
    [-1, 0],
    [1, 0],
  ],
}

const dataCase = (label: string, noise: number, noiseMax: number, width: number) => ({
  label,
  params: {
    noise: float(noise, {
      ge: 0,
      le: noiseMax,
      label: 'noise sd',
      suggestions: [noise / 2, noise, noise * 2, noise * 3],
    }),
    width: float(width, { gt: 0, le: 1, label: 'graph kernel width σ', suggestions: [0.05, 0.1, 0.2, 0.4] }),
  },
})

const METHODS = [
  { value: 'none', label: 'none: nearest labelled point' },
  { value: 'cluster', label: 'cluster: Gaussian mixture, then label' },
  { value: 'low-density', label: 'low density: self-training' },
  { value: 'graph', label: 'manifold: label propagation' },
] as const
type Method = (typeof METHODS)[number]['value']

/** The data set, centred in the plotting box: inputs as rows and classes. */
function makeData(key: DataKey, noise: number, seed: number): { rows: number[][]; y: number[] } {
  const s = stream(seed)
  if (key === 'moons') {
    const d = moons(s, { n: N, noise, spacing: 'random' })
    return { rows: (toArray(d.x) as number[][]).map(([a, b]) => [a - 0.5, b - 0.25]), y: toFlat(d.y) }
  }
  const d =
    key === 'blobs'
      ? blobs(s, {
          n: N,
          centers: [
            [-0.8, 0],
            [0.8, 0],
          ],
          sd: Math.max(noise, 1e-3),
        })
      : key === 'circles'
        ? circles(s, { n: N, factor: 0.5, noise })
        : spirals(s, { n: N, turns: 1, noise })
  return { rows: toArray(d.x) as number[][], y: toFlat(d.y) }
}

/** Unlabelled accuracy of a row-major two-class probability matrix (a point with no class counts as wrong). */
function accuracy(probs: Float64Array | null, y: number[], unlabelled: number[]): number {
  if (!probs) return NaN
  let right = 0
  for (const i of unlabelled) {
    const p = probs[2 * i + 1]
    if (p !== 0.5 && (p > 0.5 ? 1 : 0) === y[i]) right++
  }
  return right / unlabelled.length
}

/** Two-class probabilities, row-major, from hard labels (class 1 → (0, 1); −1 → (½, ½)). */
function fromLabels(labels: ArrayLike<number>): Float64Array {
  const out = new Float64Array(2 * labels.length)
  for (let i = 0; i < labels.length; i++) {
    const c = labels[i]
    out[2 * i] = c < 0 ? 0.5 : 1 - c
    out[2 * i + 1] = c < 0 ? 0.5 : c
  }
  return out
}

/** Flatten an n × 2 score tensor to row-major probabilities. */
function flatScores(rows: number[][]): Float64Array {
  const out = new Float64Array(2 * rows.length)
  rows.forEach((r, i) => {
    const total = r[0] + r[1]
    out[2 * i] = total > 0 ? r[0] / total : 0.5
    out[2 * i + 1] = total > 0 ? r[1] / total : 0.5
  })
  return out
}

/**
 * Four data sets, two labelled points, and four methods that each rest on one semi-supervised assumption: where each
 * method puts the boundary, and how many unlabelled points it labels correctly.
 */
export function AssumptionMethods() {
  const state = useFigureState({
    data: variants(
      {
        moons: dataCase('two moons', 0.1, 0.5, 0.1),
        blobs: dataCase('two blobs', 0.3, 1.2, 0.25),
        circles: dataCase('two circles', 0.05, 0.3, 0.1),
        spirals: dataCase('two spirals', 0.03, 0.2, 0.1),
      },
      { choiceLabel: 'data set' },
    ),
    method: choice(METHODS, 'graph', { label: 'assumption used' }),
    seed: int(3, { ge: 1, le: 999, label: 'data seed' }),
  })
  const key = state.data.key as DataKey
  const { noise, width } = state.data.values
  const method = state.method as Method
  const seed = state.seed

  const [anchors, setAnchors] = useState(START)
  const anchor = anchors[key]

  const data = useMemo(() => {
    const d = makeData(key, noise, seed)
    return { ...d, X: fromRows(d.rows), px: d.rows.map((r) => r[0]), py: d.rows.map((r) => r[1]) }
  }, [key, noise, seed])

  // The labelled point of each class is the point of that class nearest its handle.
  const labelled = useMemo(
    () =>
      ([0, 1] as const).map((c) => {
        let best = -1
        let bestD = Infinity
        data.rows.forEach((r, i) => {
          if (data.y[i] !== c) return
          const d = (r[0] - anchor[c][0]) ** 2 + (r[1] - anchor[c][1]) ** 2
          if (d < bestD) {
            bestD = d
            best = i
          }
        })
        return best
      }),
    [data, anchor],
  )
  const labels = useMemo(() => {
    const l = data.y.map(() => -1)
    labelled.forEach((i, c) => (l[i] = c))
    return l
  }, [data, labelled])
  const unlabelled = useMemo(() => data.y.map((_, i) => i).filter((i) => labels[i] < 0), [data, labels])

  // The mixture is fitted to all inputs and never sees a label; only the naming of its components does.
  const mixture = useMemo(() => {
    const model = gaussianMixture({ k: 2 }).fit(dataset(data.X), { stream: stream(seed) })
    return { onData: model.responsibilities(data.X), onGrid: model.responsibilities(GRID.points) }
  }, [data, seed])

  const fast = useMemo(() => {
    const L = fromRows(labelled.map((i) => data.rows[i]))
    const none = {
      data: fromLabels(toFlat(assignNearest(data.X, L).labels)),
      grid: fromLabels(toFlat(assignNearest(GRID.points, L).labels)),
    }
    const map = clusterClasses(mixture.onData, labels, 2)
    const cluster = {
      data: clusterThenLabelProbabilities(mixture.onData, map, 2),
      grid: clusterThenLabelProbabilities(mixture.onGrid, map, 2),
    }
    return { none, cluster }
  }, [data, labelled, labels, mixture])

  // Harmonic label propagation on a k-NN graph, extended to the grid by kernel averaging of the node scores.
  const graph = useComputed(() => {
    try {
      const gamma = 1 / (2 * width * width)
      const W = pointAffinity(data.X, { gamma, neighbours: NEIGHBOURS })
      const scores = toArray(harmonicLabels(W, labels, { classes: 2 }).scores) as number[][]
      const ext = harmonicExtension(data.X, scores, GRID.points, { gamma, neighbours: NEIGHBOURS })
      return { data: flatScores(scores), grid: flatScores(toArray(ext.scores) as number[][]) }
    } catch {
      // The kernel is so narrow that the graph's linear system is singular to working precision.
      return null
    }
  }, [data, labels, width])

  const selfTrained = useComputed(
    () => {
      const run = trace(
        selfTrainingSteps(
          { x: data.X, labels, classes: 2, evaluate: GRID.points },
          { base: BASE, threshold: 0.5, perClass: 10, keep: 'accumulate' },
        ),
        undefined,
        ROUNDS,
      )
      return { data: run.final.probs, grid: run.final.evaluateProbs }
    },
    [data, labels],
    { mode: 'release' },
  )

  const results: Record<Method, { data: Float64Array; grid: Float64Array } | null> = {
    none: fast.none,
    cluster: fast.cluster,
    'low-density': selfTrained.value,
    graph: graph.value,
  }
  const shown = results[method]
  const acc = (m: Method) => accuracy(results[m]?.data ?? null, data.y, unlabelled)

  const xa = useAxis({ label: 'x₁', range: X_RANGE })
  const ya = useAxis({ label: 'x₂', range: Y_RANGE, equal: xa })
  const byClass = [0, 1].map((c) => unlabelled.filter((i) => data.y[i] === c))

  return (
    <Figure
      title="Four assumptions on four data sets"
      state={state}
      caption="Two hundred points, of which one per class is labelled (large markers). Small markers show the true class of the unlabelled points; the shading is the chosen method's P(class 1) − ½, with its boundary in ink. Drag either labelled point: it snaps to the nearest point of its class. With no assumption the boundary is the perpendicular bisector of the two labels. The Gaussian mixture is fitted to all points without labels, and the labels only name its two components; it succeeds when each class is one Gaussian-shaped cluster. Self-training grows the labelled set ten points per class per round and pulls the boundary into gaps. Label propagation spreads the labels along a graph that joins each point to its 8 nearest neighbours, so it follows any connected dense region. Raise the noise to close the gaps: every assumption then stops holding."
      readouts={
        <>
          <Readout label="no assumption" value={formatNumber(acc('none'))} />
          <Readout label="cluster" value={formatNumber(acc('cluster'))} />
          <Readout label="low density" value={selfTrained.stale ? '…' : formatNumber(acc('low-density'))} />
          <Readout label="manifold" value={graph.value ? formatNumber(acc('graph')) : 'graph singular'} />
        </>
      }
    >
      <Plot x={xa} y={ya} height={380}>
        {shown && (
          <Raster
            x={GRID.gx}
            y={GRID.gy}
            z={centredField(shown.grid, NX, NY)}
            scale="diverging"
            range={[-0.5, 0.5]}
            valueLabel="P(class 1) − ½"
            boundary
            fillOpacity={0.5}
            stale={(method === 'low-density' && selfTrained.stale) || (method === 'graph' && graph.stale)}
          />
        )}
        {byClass.map(
          (rows, c) =>
            rows.length > 0 && (
              <Points
                key={c}
                name={`class ${c}, unlabelled`}
                x={rows.map((i) => data.px[i])}
                y={rows.map((i) => data.py[i])}
                slot={c}
                thin
              />
            ),
        )}
        {labelled.map((i, c) => (
          <Points
            key={`l${c}`}
            name={`class ${c}, labelled`}
            x={[data.px[i]]}
            y={[data.py[i]]}
            slot={c}
            emphasis
            size={12}
          />
        ))}
        {labelled.map((i, c) => (
          <Handle
            key={`h${c}`}
            kind="point"
            at={[data.px[i], data.py[i]]}
            slot={c}
            label={`label ${c}`}
            onDrag={(p) => setAnchors((a) => ({ ...a, [key]: c === 0 ? [p, a[key][1]] : [a[key][0], p] }))}
          />
        ))}
      </Plot>
    </Figure>
  )
}
