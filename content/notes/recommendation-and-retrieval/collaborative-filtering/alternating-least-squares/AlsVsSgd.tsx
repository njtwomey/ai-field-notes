import { useMemo } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, useParam, type XYSeries } from '@/components/viz'
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
  const lambda = useParam(1, { min: 0.05, max: 4, step: 0.05 })
  const lr = useParam(0.03, { min: 0.005, max: 0.3, step: 0.005 })
  const sweep = useParam(5, { min: 0, max: STEPS, step: 1 })

  const series = useMemo((): XYSeries[] => {
    const als = history(DATA, 3, lambda.value, STEPS, 'als').map((f) => objective(f, lambda.value))
    const sgd = history(DATA, 3, lambda.value, STEPS, 'sgd', lr.value).map((f) => objective(f, lambda.value))
    return [
      { name: 'ALS', type: 'line', x: SWEEPS, y: als, slot: 0 },
      { name: 'SGD', type: 'line', x: SWEEPS, y: sgd, slot: 1 },
    ]
  }, [lambda.value, lr.value])

  return (
    <Interactive
      title="ALS against SGD"
      caption="Both methods minimise the same regularised squared error for a rank-3 factorisation of a small ratings matrix. An ALS sweep solves every user's and then every item's ridge regression exactly, so the objective never increases. An SGD epoch takes one small step per rating; its progress depends on the step size. A small step converges slowly from the small random start; a large step gets close quickly but a fixed step keeps it bouncing slightly above the minimum. Step through sweeps with the arrows."
      controls={
        <>
          <ParamSlider label="regularisation λ" param={lambda} />
          <ParamSlider label="SGD step size η" param={lr} format={(v) => v.toFixed(3)} />
          <ParamSlider label="sweep" param={sweep} format={(v) => String(v)} withArrows />
        </>
      }
      readout={
        <>
          <Readout label="ALS objective" value={formatNumber(series[0].y[sweep.value])} />
          <Readout label="SGD objective" value={formatNumber(series[1].y[sweep.value])} />
        </>
      }
    >
      <XYChart
        series={series}
        xLabel="sweep (ALS) or epoch (SGD)"
        yLabel="regularised objective"
        xRange={[0, STEPS]}
        yRange={[0, undefined]}
        handles={[{ kind: 'x', at: sweep.value, label: 'sweep', onDrag: (x) => sweep.set(x) }]}
      />
    </Interactive>
  )
}
