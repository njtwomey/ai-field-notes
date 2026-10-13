import { Curve, Figure, int, Plot, Plots, Readout, slider, useAxis, useComputed, useFigureState } from 'aifn-render'
import { MAX_DEGREE, N_GRID, selectionStudy, type SelectionSummary } from '../_shared/polynomialSelection'

const TRUE_DEGREE = 2
const N_RANGE: [number, number] = [N_GRID[0], N_GRID[N_GRID.length - 1]]
const fmt = (v: number) => (Number.isFinite(v) ? v.toFixed(2) : '—')
const last = (rows: SelectionSummary[]) => rows[rows.length - 1]

/**
 * Consistency and efficiency side by side. Left, a quadratic truth that the degree-2 candidate contains: the fraction of
 * samples in which AIC and BIC pick the true degree. Right, the truth sin(πx), which no polynomial contains: the mean
 * degree each criterion picks beside the mean degree of the best fit on each sample.
 */
export function ClosedAndOpen() {
  const state = useFigureState({
    noise: slider(0.2, 1.5, 0.8, { step: 0.05, label: 'noise σ' }),
    replicates: int(20, { min: 10, max: 400, suggestions: [20, 50, 100, 200], label: 'samples per n' }),
  })
  const closed = useComputed(
    () => selectionStudy('closed-and-open', 'quadratic', state.noise, state.replicates),
    [state.noise, state.replicates],
    { mode: 'release' },
  )
  const open = useComputed(
    () => selectionStudy('closed-and-open', 'sine', state.noise, state.replicates),
    [state.noise, state.replicates],
    { mode: 'release' },
  )

  const nAxis = useAxis({ label: 'observations n', range: N_RANGE, log: true })
  const pAxis = useAxis({ label: 'fraction choosing the true degree', range: [0, 1] })
  const n2Axis = useAxis({ label: 'observations n', range: N_RANGE, log: true })
  const dAxis = useAxis({ label: 'mean chosen degree', range: [0, MAX_DEGREE] })
  return (
    <Figure
      title="Consistency when a candidate is true, efficiency when none is"
      state={state}
      caption={`Polynomials of degree 0 to ${MAX_DEGREE} fitted by maximum likelihood to n points with x uniform on [−1, 1] and Gaussian noise σ; AIC and BIC each pick a degree. Left: the truth is x² − 0.3, which the degree-${TRUE_DEGREE} candidate contains. The fraction of samples in which BIC picks degree ${TRUE_DEGREE} rises towards one as n grows: BIC is consistent. AIC's stays below one, because it keeps spurious terms with a probability that does not shrink. Right: the truth is sin(πx), which no polynomial contains. The dashed curve is the mean degree of the best fit on each sample, the oracle's choice, measured by squared distance from the true function over [−1, 1]; it rises with n, because more data pay for more terms. AIC follows it. BIC's heavier penalty keeps it a step or two behind, so at large n its fits are too smooth: the readouts give the excess error of each criterion's choice divided by the oracle's at the largest n. At small n the lag can help BIC, which shows that efficiency is a large-sample property. Each point averages the samples per n.`}
      readouts={
        <>
          <Readout
            label={`closed, n = ${last(closed.value).n}: AIC picks degree ${TRUE_DEGREE}`}
            value={fmt(last(closed.value).frequency.aic[TRUE_DEGREE])}
          />
          <Readout label="BIC picks it" value={fmt(last(closed.value).frequency.bic[TRUE_DEGREE])} />
          <Readout
            label={`open, n = ${last(open.value).n}: AIC excess error ÷ oracle's`}
            value={fmt(last(open.value).lossRatio.aic)}
          />
          <Readout label="BIC excess error ÷ oracle's" value={fmt(last(open.value).lossRatio.bic)} />
        </>
      }
    >
      <Plots cols={2}>
        <Plot x={nAxis} y={pAxis} title="a candidate is true">
          <Curve
            name="AIC"
            slot={0}
            x={N_GRID}
            y={closed.value.map((r) => r.frequency.aic[TRUE_DEGREE])}
            showPoints
            stale={closed.stale}
          />
          <Curve
            name="BIC"
            slot={1}
            x={N_GRID}
            y={closed.value.map((r) => r.frequency.bic[TRUE_DEGREE])}
            showPoints
            stale={closed.stale}
          />
        </Plot>
        <Plot x={n2Axis} y={dAxis} title="no candidate is true">
          <Curve
            name="best degree (oracle)"
            emphasis
            dashed
            x={N_GRID}
            y={open.value.map((r) => r.bestDegree)}
            stale={open.stale}
          />
          <Curve
            name="AIC"
            slot={0}
            x={N_GRID}
            y={open.value.map((r) => r.meanDegree.aic)}
            showPoints
            stale={open.stale}
          />
          <Curve
            name="BIC"
            slot={1}
            x={N_GRID}
            y={open.value.map((r) => r.meanDegree.bic)}
            showPoints
            stale={open.stale}
          />
        </Plot>
      </Plots>
    </Figure>
  )
}
