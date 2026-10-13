import {
  Bars,
  Curve,
  Figure,
  Handle,
  int,
  Plot,
  Plots,
  Readout,
  slider,
  useAxis,
  useComputed,
  useFigureState,
} from 'aifn-render'
import { DEGREES, MAX_DEGREE, N_GRID, selectionStudy } from '../_shared/polynomialSelection'

const TRUE_DEGREE = 2
const nearest = (n: number) =>
  N_GRID.reduce((best, v, i) => (Math.abs(Math.log(v / n)) < Math.abs(Math.log(N_GRID[best] / n)) ? i : best), 0)
const pct = (v: number) => `${(100 * v).toFixed(0)}%`
const sum = (v: readonly number[]) => v.reduce((a, b) => a + b, 0)

/**
 * How often AIC, AICc and BIC choose the true degree of a quadratic regression function, against n. Left: the fraction
 * of samples in which each criterion picks degree 2, and the fraction in which it picks a larger degree. Right: the
 * distribution of the chosen degree at the n under the handle.
 */
export function SelectionFrequencies() {
  const state = useFigureState({
    n: int(250, { min: 30, max: 2000, scale: 'log10', label: 'n under the handle' }),
    noise: slider(0.1, 1, 0.5, { step: 0.05, label: 'noise σ' }),
    replicates: int(40, { min: 10, max: 400, suggestions: [20, 40, 100, 200], label: 'samples per n' }),
  })
  const study = useComputed(
    () => selectionStudy('selection-frequencies', 'quadratic', state.noise, state.replicates),
    [state.noise, state.replicates],
    { mode: 'release' },
  )
  const rows = study.value
  const at = rows[nearest(state.n)]
  const over = (f: number[]) => sum(f.slice(TRUE_DEGREE + 1))

  const nAxis = useAxis({ label: 'observations n', range: [N_GRID[0], N_GRID[N_GRID.length - 1]], log: true })
  const pAxis = useAxis({ label: 'fraction of samples', range: [0, 1] })
  const dAxis = useAxis({ label: 'chosen degree', range: [-0.5, MAX_DEGREE + 0.5] })
  const fAxis = useAxis({ label: 'fraction of samples', range: [0, 1] })
  return (
    <Figure
      title="How often each criterion finds the true degree"
      state={state}
      caption={`Samples of n points with x uniform on [−1, 1] and y = x² − 0.3 + σε, so the true degree is ${TRUE_DEGREE}. Polynomials of degree 0 to ${MAX_DEGREE} are fitted by maximum likelihood and each criterion picks one. Left: the fraction of samples in which AIC, AICc and BIC pick degree ${TRUE_DEGREE} (solid), and the fraction in which they pick a larger degree (dashed), against n. BIC's overfitting fraction falls towards zero as n grows; AIC's stays near a fixed level, because its penalty does not grow. At small n with large noise both can also underfit. Right: the distribution of the chosen degree at the n under the handle. Drag the handle on the left chart.`}
      readouts={
        <>
          <Readout label={`n = ${at.n}: AIC picks degree ${TRUE_DEGREE}`} value={pct(at.frequency.aic[TRUE_DEGREE])} />
          <Readout label="BIC picks it" value={pct(at.frequency.bic[TRUE_DEGREE])} />
          <Readout label="AIC overfits" value={pct(over(at.frequency.aic))} />
          <Readout label="BIC overfits" value={pct(over(at.frequency.bic))} />
        </>
      }
    >
      <Plots cols={2}>
        <Plot x={nAxis} y={pAxis} title={`true degree ${TRUE_DEGREE}`}>
          <Curve
            name="AIC: true degree"
            slot={0}
            x={N_GRID}
            y={rows.map((r) => r.frequency.aic[TRUE_DEGREE])}
            showPoints
            stale={study.stale}
          />
          <Curve
            name="AICc: true degree"
            slot={2}
            x={N_GRID}
            y={rows.map((r) => r.frequency.aicc[TRUE_DEGREE])}
            showPoints
            stale={study.stale}
          />
          <Curve
            name="BIC: true degree"
            slot={1}
            x={N_GRID}
            y={rows.map((r) => r.frequency.bic[TRUE_DEGREE])}
            showPoints
            stale={study.stale}
          />
          <Curve
            name="AIC: too large"
            slot={0}
            dashed
            x={N_GRID}
            y={rows.map((r) => over(r.frequency.aic))}
            stale={study.stale}
          />
          <Curve
            name="BIC: too large"
            slot={1}
            dashed
            x={N_GRID}
            y={rows.map((r) => over(r.frequency.bic))}
            stale={study.stale}
          />
          <Handle {...state.handle('n', { label: 'n' })} />
        </Plot>
        <Plot x={dAxis} y={fAxis} title={`n = ${at.n}`}>
          <Bars
            name="AIC"
            slot={0}
            x={DEGREES.map((d) => d - 0.2)}
            y={at.frequency.aic}
            width={0.38}
            stale={study.stale}
          />
          <Bars
            name="BIC"
            slot={1}
            x={DEGREES.map((d) => d + 0.2)}
            y={at.frequency.bic}
            width={0.38}
            stale={study.stale}
          />
        </Plot>
      </Plots>
    </Figure>
  )
}
