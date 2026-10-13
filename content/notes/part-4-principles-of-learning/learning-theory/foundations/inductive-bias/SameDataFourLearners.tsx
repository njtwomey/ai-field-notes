import { useMemo } from 'react'
import { drawTrainingSet, targetFunction, unitGrid } from 'aifn-methods/theory'
import { fitAndPredict, type ResampledModel } from 'aifn-methods/theory/bias-variance'
import { stream } from 'aifn-compute/foundation/random'
import { Curve, Figure, int, Plot, Points, slider, useAxis, useFigureState } from 'aifn-render'

const GRID = unitGrid(401)
const LEARNERS: { name: string; model: ResampledModel }[] = [
  { name: 'straight line (least squares)', model: { kind: 'polynomial', degree: 1 } },
  { name: 'cubic (least squares)', model: { kind: 'polynomial', degree: 3 } },
  { name: 'degree 20, minimum norm', model: { kind: 'polynomial', degree: 20 } },
  { name: '1-nearest neighbour', model: { kind: 'knn', k: 1 } },
]
const truth = targetFunction('sine')
const LIMIT = 2.5

/**
 * Four learners fitted to the same training points. Where there are data they roughly agree; between and beyond the
 * points each fills the gaps according to its own assumptions.
 */
export function SameDataFourLearners() {
  const state = useFigureState({
    m: int(8, { min: 3, max: 20, suggestions: [5, 8, 20], label: 'training points m' }),
    noise: slider(0, 0.6, 0.15, { step: 0.05, label: 'noise σ' }),
    seed: int(3, { min: 0, max: 999, label: 'seed' }),
  })
  const data = useMemo(
    () => drawTrainingSet(stream(`inductive-bias/${state.seed}`), 'sine', state.m, state.noise),
    [state.m, state.noise, state.seed],
  )
  const fits = useMemo(
    () =>
      LEARNERS.map(({ name, model }) => ({
        name,
        y: Array.from(fitAndPredict(model, data.x, data.y, GRID), (v) => Math.max(-LIMIT, Math.min(LIMIT, v))),
      })),
    [data],
  )
  const xAxis = useAxis({ label: 'x', range: [-1, 1] })
  const yAxis = useAxis({ label: 'y', range: [-LIMIT, LIMIT] })
  return (
    <Figure
      title="The same data, four inductive biases"
      state={state}
      caption="Training points (ink) drawn from y = sin(πx) + σε with x uniform on [−1, 1]; the dashed curve is sin(πx), which the learners never see. Each learner returns the hypothesis its bias prefers given the points: the least-squares line, the least-squares cubic, the smallest-norm polynomial of degree 20 that passes through every point, and the step function that copies the label of the nearest training point. The last two both have zero training error and still disagree everywhere between the points. With few points the choice of bias decides the prediction; with 20 points the cubic and the nearest-neighbour rule track the sine, the line still cannot, and the degree-20 interpolant still swings between the points. Predictions are clipped to [−2.5, 2.5]."
    >
      <Plot x={xAxis} y={yAxis} height={360}>
        <Curve name="sin(πx)" x={GRID} y={Array.from(GRID, truth)} muted dashed />
        {fits.map((f, i) => (
          <Curve key={f.name} name={f.name} slot={i} x={GRID} y={f.y} />
        ))}
        <Points name="training points" x={data.x} y={data.y} emphasis size={6} />
      </Plot>
    </Figure>
  )
}
