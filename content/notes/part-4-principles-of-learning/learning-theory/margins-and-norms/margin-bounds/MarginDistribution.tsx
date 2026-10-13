import { ecdf, ecdfAt } from 'aifn-compute/probability/stats'
import { stream } from 'aifn-compute/foundation/random'
import { dataset } from 'aifn-compute/learning/estimators'
import { toFlat, toRows } from 'aifn-compute/foundation/tensor'
import { gaussians } from 'aifn-methods/data/synthetic'
import { linearSvm } from 'aifn-methods/learning/kernel-methods'
import {
  Annotation,
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Plot,
  Plots,
  Points,
  Readout,
  slider,
  useAxis,
  useComputed,
  useFigureState,
} from 'aifn-render'

const TEST = 4000
const BOX = 4
const MEANS = [
  [-1, -0.5],
  [1, 0.5],
]

const fmt = (v: number) => (Number.isFinite(v) ? formatNumber(Number(v.toPrecision(3))) : '—')

/** The steps of an empirical CDF as a polyline: a riser at each distinct value. */
function steps(values: number[], probabilities: number[]) {
  const x: number[] = []
  const y: number[] = []
  for (let i = 0; i < values.length; i++) {
    x.push(values[i], values[i])
    y.push(i === 0 ? 0 : probabilities[i - 1], probabilities[i])
  }
  return { x, y }
}

/** A line {x : u·x = c} with ‖u‖ = 1, as a segment long enough to cross the plotting box. */
function line(u: [number, number], c: number) {
  const L = 2 * BOX
  return {
    x: [c * u[0] - L * u[1], c * u[0] + L * u[1]],
    y: [c * u[1] + L * u[0], c * u[1] - L * u[0]],
  }
}

/**
 * A linear SVM without offset on two Gaussian classes. The normalised training margins y wᵀx / (‖w‖ X) form a
 * distribution; the margin bound reads it at a scale γ and pays a capacity term in 1/γ. The test error is the test
 * distribution at 0.
 */
export function MarginDistribution() {
  const state = useFigureState({
    separation: slider(1, 6, 5, { step: 0.1, label: 'class separation d′' }),
    C: float(10, {
      min: 0.001,
      max: 1000,
      scale: 'log10',
      suggestions: [0.001, 0.01, 0.1, 1, 10, 100],
      label: 'SVM penalty C',
    }),
    m: int(1000, { ge: 20, le: 2000, suggestions: [50, 200, 1000], label: 'training points m' }),
    gamma: slider(0.01, 0.5, 0.15, { step: 0.005, label: 'margin scale γ' }),
    seed: int(1, { ge: 0, le: 9999, label: 'seed' }),
  })
  const { separation, C, m, gamma, seed } = state
  const run = useComputed(
    () => {
      const options = { means: MEANS, separation }
      const train = gaussians(stream(`margin-bounds/${seed}/train`), { ...options, n: m })
      const test = gaussians(stream(`margin-bounds/${seed}/test`), { ...options, n: TEST })
      const model = linearSvm({ C, intercept: false }).fit(dataset(train.x, train.y!), {
        stream: stream(`margin-bounds/${seed}/fit`),
      })
      const w = toFlat(model.weights) as [number, number]
      const norm = Math.hypot(w[0], w[1])
      const rows = toRows(train.x) as number[][]
      const radius = Math.max(...rows.map((r) => Math.hypot(r[0], r[1])))
      const normalise = (scores: number[], labels: number[]) =>
        scores.map((s, i) => ((2 * labels[i] - 1) * s) / (norm * radius))
      const trainY = toFlat(train.y!)
      const testY = toFlat(test.y!)
      const trainMargins = normalise(toFlat(model.score(train.x)), trainY)
      const testMargins = normalise(toFlat(model.score(test.x)), testY)
      const trainCdf = ecdf(trainMargins)
      const testCdf = ecdf(testMargins)
      return {
        rows,
        trainY,
        trainMargins,
        testMargins,
        u: [w[0] / norm, w[1] / norm] as [number, number],
        radius,
        trainCurve: steps(toFlat(trainCdf.values), toFlat(trainCdf.probabilities)),
        testCurve: steps(toFlat(testCdf.values), toFlat(testCdf.probabilities)),
        testError: toFlat(ecdfAt(testMargins, [0]))[0],
        trainError: toFlat(ecdfAt(trainMargins, [0]))[0],
        smallest: Math.min(...trainMargins),
      }
    },
    [separation, C, m, seed],
    { mode: 'release' },
  )
  const r = run.value
  const inside = r.trainMargins.map((v) => v < gamma)
  const marginLoss = inside.filter(Boolean).length / r.trainMargins.length
  const pick = (keep: boolean) => r.rows.filter((_, i) => inside[i] === keep)
  const groups = (keep: boolean) => r.trainY.filter((_, i) => inside[i] === keep)
  const outsideRows = pick(false)
  const insideRows = pick(true)
  const boundary = line(r.u, 0)
  const upper = line(r.u, gamma * r.radius)
  const lower = line(r.u, -gamma * r.radius)

  const x1 = useAxis({ label: 'x₁', range: [-BOX, BOX] })
  const x2 = useAxis({ label: 'x₂', range: [-BOX, BOX], equal: x1 })
  const mAxis = useAxis({ label: 'normalised margin y wᵀx / (‖w‖ X)', range: [-0.6, 1] })
  const fAxis = useAxis({ label: 'fraction of points with margin ≤ value', range: [0, 1] })
  return (
    <Figure
      title="The training margin distribution"
      purpose="A margin bound reads the training margins at a scale γ: the points with margin below γ count as errors, and the capacity term falls like X/(γ√m). The test error is the test distribution at margin 0."
      state={state}
      defaultSize="L"
      readouts={
        <>
          <Readout label="margin loss R̂_γ at γ" value={fmt(marginLoss)} />
          <Readout label="training error" value={fmt(r.trainError)} />
          <Readout label="test error" value={fmt(r.testError)} />
          <Readout label="smallest training margin" value={fmt(r.smallest)} />
          <Readout label="data radius X" value={fmt(r.radius)} />
        </>
      }
      caption="A linear SVM without offset (aifn linearSvm, dual coordinate descent) on two Gaussian classes whose means are d′ standard deviations apart, tested on 4,000 fresh points. Left: the training points, the boundary wᵀx = 0 (ink) and the band of normalised margin γ either side of it (dashed); the points inside the band, marked in ink, are the margin errors. Right: the cumulative distribution of the normalised margins y wᵀx/(‖w‖X), with X the largest training input norm, for the training points (ink) and the test points. Drag the vertical line to move γ; its height on the ink curve is R̂_γ, and the test curve at 0 is the test error. At the opening values (d′ = 5, m = 1,000, γ = 0.15) R̂_γ is 0.046 and the test error 0.008. The margin bound adds 4X/(γ√m) on the normalised scale, 4/(0.15 × √1,000) ≈ 0.84, so it is close to vacuous: the largest input norm X ≈ 5.5 makes every normalised margin small. Reaching a capacity term of 0.1 at γ = 0.15 would take about 71,000 points. Lower d′ to 3 so that the classes overlap: the test error rises to 0.063, the margin loss at γ = 0.15 to 0.22, and even γ = 0.05 leaves a margin loss of 0.11. Changing C moves the boundary little here; it matters more when few points are near the boundary."
    >
      <Plots cols={2}>
        <Plot x={x1} y={x2} title="training points and the γ band">
          {outsideRows.length > 0 && (
            <Points
              name="outside the band"
              x={outsideRows.map((p) => p[0])}
              y={outsideRows.map((p) => p[1])}
              group={groups(false)}
              groupNames={['class −1', 'class +1']}
              thin
            />
          )}
          {insideRows.length > 0 && (
            <Points name="margin below γ" x={insideRows.map((p) => p[0])} y={insideRows.map((p) => p[1])} emphasis />
          )}
          <Curve name="boundary wᵀx = 0" x={boundary.x} y={boundary.y} emphasis stale={run.stale} />
          <Curve name="margin γX" x={upper.x} y={upper.y} dashed muted />
          <Curve name="margin γX" x={lower.x} y={lower.y} dashed muted />
        </Plot>
        <Plot x={mAxis} y={fAxis} title="margin distributions">
          <Annotation x={0} text="0" dashed />
          <Curve name="training margins" x={r.trainCurve.x} y={r.trainCurve.y} emphasis stale={run.stale} />
          <Curve name="test margins" x={r.testCurve.x} y={r.testCurve.y} slot={2} />
          <Handle kind="x" at={gamma} onDrag={(v) => state.set('gamma', Math.max(0.01, Math.min(0.5, v)))} label="γ" />
        </Plot>
      </Plots>
    </Figure>
  )
}
