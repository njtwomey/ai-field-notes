import { useMemo } from 'react'
import { drawTrainingSet, unitGrid } from 'aifn-methods/theory'
import { fitAndPredict, type ResampledModel } from 'aifn-methods/theory/bias-variance'
import { stream } from 'aifn-compute/foundation/random'
import {
  choice,
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Plot,
  Points,
  Readout,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'

type LearnerName = 'least-squares' | 'ridge' | 'one-nn' | 'nine-nn'

const LEARNERS: { value: LearnerName; label: string }[] = [
  { value: 'least-squares', label: 'least squares, degree 9' },
  { value: 'ridge', label: 'ridge, degree 9' },
  { value: 'one-nn', label: '1-nearest neighbour' },
  { value: 'nine-nn', label: '9-nearest neighbours' },
]
const GRID = unitGrid(301)
const LIMIT = 2.5
const NOISE = 0.25

function modelOf(learner: LearnerName, lambda: number): ResampledModel {
  if (learner === 'least-squares') return { kind: 'polynomial', degree: 9 }
  if (learner === 'ridge') return { kind: 'polynomial', degree: 9, ridge: lambda }
  return { kind: 'knn', k: learner === 'one-nn' ? 1 : 9 }
}

const clip = (a: ArrayLike<number>) => Array.from(a, (v) => Math.max(-LIMIT, Math.min(LIMIT, v)))

/**
 * Replace-one stability made visible. A learner is fitted to a sample S and to the sample S⁽¹⁾ in which the first
 * point is replaced by a point the reader drags. The vertical distance between the two fits is the change that
 * uniform stability bounds.
 */
export function ReplaceOnePoint({ learner = 'least-squares' }: { learner?: LearnerName }) {
  const state = useFigureState({
    learner: choice(LEARNERS, learner, { label: 'learner' }),
    lambda: float(0.1, {
      gt: 0,
      le: 100,
      scale: 'log10',
      suggestions: [0.001, 0.01, 0.1, 1, 10],
      label: 'ridge penalty λ',
      when: (v) => v.learner === 'ridge',
    }),
    m: int(20, { ge: 10, le: 200, suggestions: [10, 20, 50, 200], label: 'training points m' }),
    zx: slider(-1, 1, 0.55, { step: 0.01, onChart: true }),
    zy: slider(-LIMIT, LIMIT, -1.6, { step: 0.01, onChart: true }),
    seed: int(4, { ge: 0, le: 999, label: 'seed' }),
  })
  const data = useMemo(
    () => drawTrainingSet(stream(`algorithmic-stability/replace/${state.seed}`), 'sine', state.m, NOISE),
    [state.m, state.seed],
  )
  const replaced = useMemo(() => {
    const x = Float64Array.from(data.x)
    const y = Float64Array.from(data.y)
    x[0] = state.zx
    y[0] = state.zy
    return { x, y }
  }, [data, state.zx, state.zy])
  const model = useMemo(() => modelOf(state.learner, state.lambda), [state.learner, state.lambda])
  const before = useMemo(() => fitAndPredict(model, data.x, data.y, GRID), [data, model])
  const after = useMemo(() => fitAndPredict(model, replaced.x, replaced.y, GRID), [replaced, model])
  // The largest vertical gap between the two fits over the plotted grid, for the readout.
  let largest = 0
  for (let j = 0; j < GRID.length; j++) largest = Math.max(largest, Math.abs(after[j] - before[j]))

  const xAxis = useAxis({ label: 'x', range: [-1, 1] })
  const yAxis = useAxis({ label: 'y', range: [-LIMIT, LIMIT] })
  return (
    <Figure
      title="Replace one training point, refit, compare"
      state={state}
      defaultSize="M"
      readouts={
        <Readout label="largest change in prediction on [−1, 1]" value={formatNumber(Number(largest.toPrecision(3)))} />
      }
      caption={`A sample S of m points drawn from y = sin(πx) + ε with x uniform on [−1, 1] and noise standard deviation ${NOISE}. The grey point is the first training point z₁; drag the large point to choose its replacement z₁′, which gives the sample S⁽¹⁾. The dashed curve is the learner's fit to S and the solid curve its fit to S⁽¹⁾. Uniform stability asks how far apart the two curves can be, over every sample, every replacement and every input. Least squares with a degree-9 polynomial swings across the whole interval when one point moves, and more so with few points. Ridge with a larger penalty λ (on the sum of squared residuals) moves less. The 1-nearest-neighbour rule changes only near the moved point, but there it changes by the full jump in the label; with 9 neighbours the jump is shared nine ways. Raising m shrinks the change for least squares and ridge, because each point carries less weight; for the nearest-neighbour rules it shrinks the region that changes but not the size of the jump. The penalty λ multiplies ‖w‖² against the sum of squared residuals, which is the per-example penalty (λ′/2)‖w‖² with λ′ = 2λ/m. Predictions are clipped to [−${LIMIT}, ${LIMIT}].`}
    >
      <Plot x={xAxis} y={yAxis} height={340}>
        <Curve name="fit to S" x={GRID} y={clip(before)} muted dashed />
        <Curve name="fit to S⁽¹⁾" x={GRID} y={clip(after)} slot={0} />
        <Points name="shared training points" x={data.x.subarray(1)} y={data.y.subarray(1)} slot={1} size={5} />
        <Points name="replaced point z₁" x={[data.x[0]]} y={[data.y[0]]} muted size={8} />
        <Points name="replacement z₁′" x={[state.zx]} y={[state.zy]} emphasis size={9} />
        <Handle {...state.handle(['zx', 'zy'], { label: 'z₁′' })} />
      </Plot>
    </Figure>
  )
}
