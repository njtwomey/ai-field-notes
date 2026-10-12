import { useMemo } from 'react'
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
  setting,
  useAxis,
  useComputed,
  useFigureState,
} from 'aifn-render'
import { child, normals, stream, units } from 'aifn-compute/foundation/random'
import { fromData, toFlat, toRows, type Tensor } from 'aifn-compute/foundation/tensor'
import { classProbabilities, dataset, hasPredictive, type Distribution } from 'aifn-compute/learning/estimators'
import { rocCurve } from 'aifn-compute/learning/metrics'
import { grid2d } from 'aifn-compute/numerics/geometry'
import { multivariateKde } from 'aifn-compute/probability/stats'
import { blobs } from 'aifn-methods/data/synthetic'
import { quadraticDiscriminant } from 'aifn-methods/learning/generative-classifiers'
import { multinomialLogisticRegression } from 'aifn-methods/learning/generalised/glm'
import { kNearestNeighbours } from 'aifn-methods/learning/neighbours'
import { oneClassScore, oneClassSvm } from 'aifn-methods/unsupervised/anomaly'
import { backgroundCheck, decide, type Bias } from './background'

const K = 3
const N_TEST = 200
const N_OUTLIERS = 300
const NU = 0.1
const GRID = 70
const CLASS_NAMES = ['class 1', 'class 2', 'class 3']

type Estimator = 'density' | 'one-class' | 'discriminative'
type Classifier = 'qda' | 'knn' | 'logistic'

/** Stack rows of several n × 2 blocks into one tensor. */
function stack(blocks: readonly ArrayLike<number>[]): Tensor {
  const total = blocks.reduce((a, b) => a + b.length, 0)
  const out = new Float64Array(total)
  let at = 0
  for (const b of blocks) {
    out.set(b, at)
    at += b.length
  }
  return fromData(out, [total / 2, 2])
}

/** A fitted classifier's class probabilities at the rows of x (one-hot of its decision when it has no predictive). */
function probabilities(model: { decide(x: Tensor): Tensor }, x: Tensor): number[][] {
  if (hasPredictive(model)) {
    const d = (model as unknown as { predictive(x: Tensor): Distribution }).predictive(x)
    return toRows(classProbabilities(d)) as number[][]
  }
  return Array.from(toFlat(model.decide(x)), (c) => Array.from({ length: K }, (_, j) => (j === c ? 1 : 0)))
}

/**
 * Background Check in the plane with fitted models. Three Gaussian classes are the foreground; test outliers come
 * from a Gaussian twice as wide as the data. q_f comes from a kernel density estimate, a one-class SVM's score
 * through a sigmoid, or a classifier separating the data from uniform samples; a classifier trained on the
 * foreground alone gives p(c | f, x). The bias (μ0, μ1) then turns them into four probabilities everywhere.
 */
export function BackgroundCheckPlane() {
  const state = useFigureState({
    estimator: choice<Estimator>(
      [
        { value: 'density', label: 'familiarity: kernel density estimate' },
        { value: 'one-class', label: 'familiarity: one-class SVM score' },
        { value: 'discriminative', label: 'discriminative: data against uniform samples' },
      ],
      'density',
      { label: 'estimate of q_f' },
    ),
    classifier: choice<Classifier>(
      [
        { value: 'qda', label: 'quadratic discriminant' },
        { value: 'logistic', label: 'multinomial logistic regression' },
        { value: 'knn', label: '15 nearest neighbours' },
      ],
      'qda',
      { label: 'foreground classifier' },
    ),
    mu0: float(0.2, { min: 0, max: 1, step: 0.01, label: 'μ(0): background far from the foreground' }),
    mu1: float(0.1, { min: 0, max: 1, step: 0.01, label: 'μ(1): background at the foreground peak' }),
    separation: float(3, { min: 1, max: 6, step: 0.1, label: 'class separation (sd)' }),
    n: int(60, { min: 10, max: 150, label: 'training points per class', suggestions: [20, 60, 150] }),
    seed: int(1, { min: 0, max: 999, label: 'seed' }),
    view: choice(
      [
        { value: 'decision', label: 'decision' },
        { value: 'background', label: 'p(b | x)' },
      ],
      'decision',
      { label: 'map' },
    ),
    outliers: setting(true, { label: 'show test outliers' }),
    px: float(0, { min: -20, max: 20, step: 0.05, label: 'test point x₁', onChart: true }),
    py: float(-4, { min: -20, max: 20, step: 0.05, label: 'test point x₂', onChart: true }),
  })
  const b: Bias = useMemo(() => ({ mu0: state.mu0, mu1: state.mu1, gamma: 1 }), [state.mu0, state.mu1])

  const data = useMemo(() => {
    const s = stream(state.seed)
    const opts = { centers: K, layout: 'polygon' as const, separation: state.separation }
    const train = blobs(child(s, 'train'), { ...opts, n: [state.n, state.n, state.n] })
    const test = blobs(child(s, 'test'), { ...opts, n: [N_TEST, N_TEST, N_TEST] })
    const tx = toFlat(train.x)
    const mean = [0, 1].map((d) => tx.filter((_, i) => i % 2 === d).reduce((a, v) => a + v, 0) / (tx.length / 2))
    const sd = [0, 1].map((d) =>
      Math.sqrt(tx.filter((_, i) => i % 2 === d).reduce((a, v) => a + (v - mean[d]) ** 2, 0) / (tx.length / 2 - 1)),
    )
    // Outliers: a Gaussian with the data's mean and twice its standard deviation on each axis.
    const z = toFlat(normals(child(s, 'outliers'), [N_OUTLIERS, 2]))
    const outliers = Float64Array.from(z, (v, i) => mean[i % 2] + 2 * sd[i % 2] * v)
    const half = Math.max(...[0, 1].map((d) => 3 * sd[d])) + 1.5
    const box = [0, 1].map((d) => [mean[d] - half, mean[d] + half] as [number, number])
    return { train, test, outliers, box }
  }, [state.seed, state.n, state.separation])

  const fit = useComputed(
    () => {
      const { train, test, outliers, box } = data
      const g = grid2d(box[0], box[1], GRID)
      const queries = stack([toFlat(g.points), toFlat(test.x), outliers])
      const nGrid = g.shape[0] * g.shape[1]

      // log q_f, up to a constant fixed below so that its largest value over the grid is 0.
      let logQf: (q: Tensor) => Float64Array
      if (state.estimator === 'density') {
        logQf = (q) => Float64Array.from(toFlat(multivariateKde(train.x, q).logDensity))
      } else if (state.estimator === 'one-class') {
        const model = oneClassSvm(train.x, { nu: NU })
        const n = toFlat(train.y!).length
        // Scores scaled as scikit-learn's (dual weights times νn); the sigmoid's origin is the lowest training score.
        const score = (q: Tensor) => Float64Array.from(oneClassScore(model, q), (v) => -v * NU * n)
        const sStar = Math.min(...score(train.x))
        logQf = (q) => Float64Array.from(score(q), (s) => -Math.log1p(Math.exp(sStar - s)))
      } else {
        // Uniform samples in the box, twice as many as the data, and a classifier telling them apart.
        const nf = toFlat(train.y!).length
        const m = 2 * nf
        const u = units(child(stream(state.seed), 'uniform'), 2 * m)
        const uniform = Float64Array.from(u, (v, i) => box[i % 2][0] + v * (box[i % 2][1] - box[i % 2][0]))
        const x = stack([toFlat(train.x), uniform])
        const y = fromData(
          Float64Array.from({ length: nf + m }, (_, i) => (i < nf ? 0 : 1)),
          [nf + m],
        )
        const k = 25
        const model = kNearestNeighbours({ k }).fit(dataset(x, y))
        // The odds of data against uniform, smoothed by half a vote, are proportional to p(x | f).
        logQf = (q) =>
          Float64Array.from(probabilities(model, q), (p) => {
            const pf = (p[0] * k + 0.5) / (k + 1)
            return Math.log(pf / (1 - pf))
          })
      }
      const raw = logQf(queries)
      let top = -Infinity
      for (let i = 0; i < nGrid; i++) top = Math.max(top, raw[i])
      const lq = raw.map((v) => v - top)

      const classifier =
        state.classifier === 'qda'
          ? quadraticDiscriminant().fit(dataset(train.x, train.y!))
          : state.classifier === 'knn'
            ? kNearestNeighbours({ k: 15 }).fit(dataset(train.x, train.y!))
            : multinomialLogisticRegression().fit(dataset(train.x, train.y!))
      const within = probabilities(classifier, queries)
      const at = (p: [number, number]) => {
        const q = fromData(Float64Array.from(p), [1, 2])
        return { lq: logQf(q)[0] - top, within: probabilities(classifier, q)[0] }
      }
      return { g, nGrid, lq, within, at }
    },
    [data, state.estimator, state.classifier],
    { mode: 'release' },
  )

  const view = useMemo(() => {
    const f = fit.value
    const [ny, nx] = f.g.shape
    const labels: number[][] = []
    const pb: number[][] = []
    for (let i = 0; i < ny; i++) {
      const lr: number[] = []
      const br: number[] = []
      for (let j = 0; j < nx; j++) {
        const idx = i * nx + j
        const r = backgroundCheck(f.lq[idx], f.within[idx], b)
        const c = decide(r.probs)
        lr.push(c === K ? -1 : c)
        br.push(r.probs[K])
      }
      labels.push(lr)
      pb.push(br)
    }
    // Held-out foreground and outliers.
    const truth = toFlat(data.test.y!)
    let accepted = 0
    let correct = 0
    let fgFlagged = 0
    let outFlagged = 0
    const scores: number[] = []
    const kind: number[] = []
    const nTest = truth.length
    for (let t = 0; t < nTest + N_OUTLIERS; t++) {
      const idx = f.nGrid + t
      const r = backgroundCheck(f.lq[idx], f.within[idx], b)
      const c = decide(r.probs)
      scores.push(r.probs[K])
      kind.push(t < nTest ? 0 : 1)
      if (t < nTest) {
        if (c === K) fgFlagged++
        else {
          accepted++
          if (c === truth[t]) correct++
        }
      } else if (c === K) outFlagged++
    }
    const auroc = rocCurve(kind, scores, { positive: 1 }).area
    return {
      x: toFlat(f.g.x),
      y: toFlat(f.g.y),
      labels,
      pb,
      accuracy: accepted ? correct / accepted : NaN,
      fgFlagged: fgFlagged / nTest,
      outFlagged: outFlagged / N_OUTLIERS,
      auroc,
    }
  }, [fit.value, data, b])

  const probe = useMemo(() => {
    const p = fit.value.at([state.px, state.py])
    const r = backgroundCheck(p.lq, p.within, b)
    return { ...r, within: p.within, lq: p.lq }
  }, [fit.value, state.px, state.py, b])

  const pts = useMemo(() => {
    const tx = toFlat(data.train.x)
    return {
      x: tx.filter((_, i) => i % 2 === 0),
      y: tx.filter((_, i) => i % 2 === 1),
      group: toFlat(data.train.y!),
      ox: data.outliers.filter((_, i) => i % 2 === 0),
      oy: data.outliers.filter((_, i) => i % 2 === 1),
    }
  }, [data])

  const xAxis = useAxis({ label: 'x₁', range: data.box[0] })
  const yAxis = useAxis({ label: 'x₂', range: data.box[1], equal: xAxis })
  const clamp = (v: number, r: [number, number]) => Math.min(r[1], Math.max(r[0], v))
  const decision = decide(probe.probs)
  return (
    <Figure
      title="Background Check in the plane"
      state={state}
      caption="Three Gaussian classes are the foreground, drawn as coloured points. The map shows the decision of Background Check over the plane: one colour per class, and blank where the background is the most probable label. Grey points are held-out outliers from a Gaussian twice as wide as the data. Drag the black marker to read the four probabilities at any point. The readouts score held-out foreground points and the outliers. Set μ(0) = 0 to reject only ambiguous points between classes, μ(1) = 0 to reject only points far from the data."
      readouts={
        <>
          <Readout label="accuracy on accepted foreground" value={formatNumber(view.accuracy)} />
          <Readout label="foreground rejected" value={formatNumber(view.fgFlagged)} />
          <Readout label="outliers rejected" value={formatNumber(view.outFlagged)} />
          <Readout label="AUROC of p(b | x), outliers against foreground" value={formatNumber(view.auroc)} />
          <Readout label="test point: max p(c | f, x)" value={formatNumber(Math.max(...probe.within))} />
          <Readout label="test point: p(f | x)" value={formatNumber(probe.pf)} />
          {CLASS_NAMES.map((name, c) => (
            <Readout key={name} label={`test point: p(${name} | x)`} value={formatNumber(probe.probs[c])} />
          ))}
          <Readout label="test point: p(b | x)" value={formatNumber(probe.probs[K])} />
          <Readout label="test point: decision" value={decision === K ? 'background' : CLASS_NAMES[decision]} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={440}>
        {state.view === 'decision' ? (
          <Raster
            x={view.x}
            y={view.y}
            z={view.labels}
            scale="categorical"
            categoryNames={CLASS_NAMES}
            fillOpacity={0.45}
            boundary
            stale={fit.stale}
          />
        ) : (
          <Raster x={view.x} y={view.y} z={view.pb} range={[0, 1]} valueLabel="p(b | x)" boundary stale={fit.stale} />
        )}
        {state.outliers && <Points name="test outliers" x={pts.ox} y={pts.oy} muted />}
        <Points name="training data" x={pts.x} y={pts.y} group={pts.group} groupNames={CLASS_NAMES} />
        <Handle
          kind="point"
          at={[state.px, state.py]}
          label="test point"
          onDrag={([x, y]) => {
            state.set('px', clamp(x, data.box[0]))
            state.set('py', clamp(y, data.box[1]))
          }}
        />
      </Plot>
    </Figure>
  )
}
