import { useMemo } from 'react'
import { Curve, Figure, float, formatNumber, Handle, int, Plot, Readout, useAxis, useFigureState } from 'aifn-render'
import { history, makeRatings, rmse, type Factors } from '../_shared/mf'

const DATA = makeRatings()
const STEPS = 30
const SWEEPS = Array.from({ length: STEPS + 1 }, (_, t) => t)

/** Regularised objective: squared error on observed ratings plus λ times the squared Frobenius norms. */
function objective(f: Factors, lambda: number) {
  const fit = rmse(DATA, f, DATA.train) ** 2 * DATA.train.length
  const norm = [...f.W, ...f.V].reduce((s, x) => s + x.reduce((a, b) => a + b * b, 0), 0)
  return fit + lambda * norm
}

/** ALS against SGD on the same objective, sweep by sweep. */
export function AlsVsSgd() {
  const state = useFigureState({
    lambda: float(1, { min: 0.05, max: 4, step: 0.05, label: 'regularisation λ' }),
    lr: float(0.03, { min: 0.005, max: 0.3, step: 0.005, label: 'SGD step size η', format: (v) => v.toFixed(3) }),
    sweep: int(5, { min: 0, max: STEPS, step: 1, label: 'sweep', format: (v) => String(v) }),
  })

  const series = useMemo(() => {
    const als = history(DATA, 3, state.lambda, STEPS, 'als').map((f) => objective(f, state.lambda))
    const sgd = history(DATA, 3, state.lambda, STEPS, 'sgd', state.lr).map((f) => objective(f, state.lambda))
    return [
      { name: 'ALS', x: SWEEPS, y: als, slot: 0 },
      { name: 'SGD', x: SWEEPS, y: sgd, slot: 1 },
    ] as const
  }, [state.lambda, state.lr])

  const xAxis = useAxis({ label: 'sweep (ALS) or epoch (SGD)', range: [0, STEPS] })
  const yAxis = useAxis({ label: 'regularised objective', range: [0, undefined], hold: 'union' })
  return (
    <Figure
      title="ALS against SGD"
      state={state}
      caption="Both methods minimise the same regularised squared error for a rank-3 factorisation of a small ratings matrix. An ALS sweep solves every user's and then every item's ridge regression exactly, so the objective never increases. An SGD epoch takes one small step per rating; its progress depends on the step size. A small step converges slowly from the small random start; a large step gets close quickly but a fixed step keeps it bouncing slightly above the minimum. Step through sweeps with the arrows."

      readouts={
        <>
          <Readout label="ALS objective" value={formatNumber(series[0].y[state.sweep])} />
          <Readout label="SGD objective" value={formatNumber(series[1].y[state.sweep])} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis}>
        <Curve {...series[0]} />
        <Curve {...series[1]} />
        <Handle {...state.handle('sweep', { label: 'sweep' })} />
      </Plot>
    </Figure>
  )
}
