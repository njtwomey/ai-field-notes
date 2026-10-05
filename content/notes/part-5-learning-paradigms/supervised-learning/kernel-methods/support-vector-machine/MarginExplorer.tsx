import { useMemo } from 'react'
import {
  Figure,
  float,
  formatNumber,
  Plot,
  Readout,
  seriesLayers,
  type SeriesSpec,
  int,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { linearKernel, trainSvm, type Point } from '../_shared/svm'
import { normal, stream } from 'aifn-compute/foundation/random'

const N = 50
const BOX = 4
const RANGE: [number, number] = [-BOX, BOX]
const Y_RANGE: [number | undefined, number | undefined] = [-BOX, BOX]

/** The part of the line w·x + b = c inside the plotting box, or nothing if the line misses it. */
function clipLine(w: Point, b: number, c: number): { x: number[]; y: number[] } {
  const pts: Point[] = []
  if (Math.abs(w[1]) > 1e-12) {
    for (const x of RANGE) {
      const y = (c - b - w[0] * x) / w[1]
      if (y >= -BOX && y <= BOX) pts.push([x, y])
    }
  }
  if (Math.abs(w[0]) > 1e-12) {
    for (const y of RANGE) {
      const x = (c - b - w[1] * y) / w[0]
      if (x >= -BOX && x <= BOX) pts.push([x, y])
    }
  }
  pts.sort((p, q) => p[0] - q[0] || p[1] - q[1])
  return pts.length >= 2 ? { x: [pts[0][0], pts.at(-1)![0]], y: [pts[0][1], pts.at(-1)![1]] } : { x: [], y: [] }
}

/** A linear soft-margin SVM on two Gaussian clouds: boundary, margins and support vectors as C varies. */
export function MarginExplorer() {
  const state = useFigureState({
    gap: float(1.2, { min: 0, max: 2.5, step: 0.05, label: 'class separation' }),
    C: float(1, {
      min: 0.01,
      max: 1000,
      scale: 'log10',
      suggestions: [0.01, 0.1, 1, 10, 100, 1000],
      label: 'C (penalty on slack)',
    }),
    seed: int(3, { min: 1, max: 100, label: 'data seed' }),
  })
  const { C } = state

  const data = useMemo(() => {
    const g = stream(state.seed)
    const x: Point[] = []
    const y: number[] = []
    for (let i = 0; i < N; i++) {
      const label = i < N / 2 ? -1 : 1
      x.push([label * state.gap * 0.7 + 0.9 * normal(g), label * state.gap * 0.7 + 0.9 * normal(g)])
      y.push(label)
    }
    return { x, y }
  }, [state.gap, state.seed])

  const r = useMemo(() => {
    const fit = trainSvm(data.x, data.y, C, linearKernel)
    const w: Point = [0, 0]
    fit.alpha.forEach((a, i) => {
      w[0] += a * data.y[i] * data.x[i][0]
      w[1] += a * data.y[i] * data.x[i][1]
    })
    const norm = Math.hypot(w[0], w[1])
    const margins = data.x.map((p, i) => data.y[i] * (w[0] * p[0] + w[1] * p[1] + fit.b))
    const sv = fit.alpha.map((a, i) => (a > 1e-6 ? i : -1)).filter((i) => i >= 0)
    const errors = margins.filter((m) => m < 0).length
    const hinge = margins.reduce((s, m) => s + Math.max(0, 1 - m), 0)
    return { fit, w, norm, sv, errors, hinge, atBound: sv.filter((i) => fit.alpha[i] > C * (1 - 1e-6)).length }
  }, [data, C])

  const boundary = clipLine(r.w, r.fit.b, 0)
  const plus = clipLine(r.w, r.fit.b, 1)
  const minus = clipLine(r.w, r.fit.b, -1)
  const series: SeriesSpec[] = [
    {
      name: 'points',
      type: 'scatter',
      x: data.x.map((p) => p[0]),
      y: data.x.map((p) => p[1]),
      group: data.y.map((v) => (v > 0 ? 1 : 0)),
      groupNames: ['class −1', 'class +1'],
    },
    { name: 'w·x + b = 0', type: 'line', ...boundary, emphasis: true },
    { name: 'w·x + b = +1', type: 'line', ...plus, slot: 1, dashed: true },
    { name: 'w·x + b = −1', type: 'line', ...minus, slot: 0, dashed: true },
    {
      name: 'support vectors',
      type: 'scatter',
      x: r.sv.map((i) => data.x[i][0]),
      y: r.sv.map((i) => data.x[i][1]),
      emphasis: true,
    },
  ]

  const xAxis = useAxis({ label: 'x₁', range: RANGE })
  const yAxis = useAxis({ label: 'x₂', range: Y_RANGE, equal: xAxis })
  return (
    <Figure
      title="The maximum-margin boundary and its support vectors"
      state={state}
      caption="Two classes of 25 points each. The solid line is the decision boundary w·x + b = 0 and the dashed lines are the margins w·x + b = ±1. Diamonds mark the support vectors, the points with αᵢ > 0: those on the margin, inside it, or misclassified. Only they determine the boundary. A small C tolerates margin violations cheaply, so the margin is wide and many points are support vectors. A large C penalises violations heavily, the margin narrows, and with separable data the solution approaches the hard-margin SVM."
      readouts={
        <>
          <Readout label="support vectors" value={`${r.sv.length} (${r.atBound} at αᵢ = C)`} />
          <Readout label="margin width 2/‖w‖" value={formatNumber(2 / r.norm)} />
          <Readout label="training errors" value={String(r.errors)} />
          <Readout label="total hinge loss Σξᵢ" value={formatNumber(r.hinge)} />
        </>
      }
    >
      <div className="mx-auto w-full max-w-lg">
        <Plot x={xAxis} y={yAxis}>
          {seriesLayers(series)}
        </Plot>
      </div>
    </Figure>
  )
}
