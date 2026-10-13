import { Curve, Figure, float, formatNumber, Plot, Readout, useAxis, useComputed, useFigureState } from 'aifn-render'
import { diabetes } from 'aifn-methods/data/real'
import { standardScaler } from 'aifn-methods/learning/preprocessing'
import { forwardStagewise, leastAngleRegression } from 'aifn-methods/learning/linear'
import { fromData, toFlat, toRows } from 'aifn-compute/foundation/tensor'

const MAX_STEPS = 40000

/** The diabetes data: ten standardised features, centred target. */
function diabetesData() {
  const raw = diabetes()
  const x = standardScaler().fit({ x: raw.x }).transform(raw.x)
  const target = toFlat(raw.y!)
  const mean = target.reduce((a, b) => a + b, 0) / target.length
  return { x, y: fromData(Float64Array.from(target, (v) => v - mean)) }
}

/**
 * Incremental forward stagewise regression with step ε against the exact lasso path from the LARS–lasso algorithm, on
 * the diabetes data, both plotted against the ℓ1 norm of the coefficients.
 */
export function StagewiseAgainstLasso() {
  const state = useFigureState({
    epsilon: float(0.2, { min: 0.01, max: 20, suggestions: [0.05, 0.2, 1, 5], label: 'step ε' }),
  })
  const data = useComputed(
    () => {
      const { x, y } = diabetesData()
      const lasso = leastAngleRegression(x, y, { method: 'lasso' })
      return { x, y, lasso: { l1: [...lasso.l1Norm], coef: toRows(lasso.coef) as number[][] } }
    },
    [],
    { mode: 'release' },
  ).value
  const stagewise = useComputed(
    () => {
      const path = forwardStagewise(data.x, data.y, { epsilon: state.epsilon, steps: MAX_STEPS })
      return {
        l1: [...path.l1Norm],
        coef: toRows(path.coef) as number[][],
        steps: path.features.length,
        converged: path.converged,
      }
    },
    [data, state.epsilon],
    { mode: 'release' },
  )
  const s = stagewise.value
  const xAxis = useAxis({ label: '‖w‖₁' })
  const yAxis = useAxis({ label: 'coefficient' })
  return (
    <Figure
      title="Forward stagewise approaches the lasso path"
      state={state}
      caption="Coefficients of the ten standardised diabetes features against the ℓ1 norm of the coefficient vector. Dashed: the exact lasso path, computed knot by knot by least-angle regression with the lasso modification. Solid: incremental forward stagewise regression, which moves the coefficient of the feature most correlated with the residual by ε at each step. With a large ε the stagewise path is jagged and overshoots; as ε falls it settles onto the lasso path, which it follows wherever the lasso coefficients move monotonically. Stagewise stops when no step of size ε lowers the residual sum of squares, short of the least-squares fit at the right-hand end of the lasso path; the smaller ε, the closer it gets."
      readouts={
        <>
          <Readout label="stagewise steps" value={`${s.steps}${s.converged ? '' : ' (step limit reached)'}`} />
          <Readout label="final ‖w‖₁" value={formatNumber(Number(s.l1[s.l1.length - 1].toPrecision(4)))} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis}>
        {data.lasso.coef.map((row, j) => (
          <Curve key={`l${j}`} id={`l${j}`} name="lasso path" x={data.lasso.l1} y={row} emphasis dashed width={1.5} />
        ))}
        {s.coef.map((row, j) => (
          <Curve
            key={`s${j}`}
            id={`s${j}`}
            name={`forward stagewise, ε = ${state.epsilon}`}
            x={s.l1}
            y={row}
            slot={0}
            stale={stagewise.stale}
          />
        ))}
      </Plot>
    </Figure>
  )
}
