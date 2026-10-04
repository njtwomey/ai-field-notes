import { useMemo } from 'react'
import { Curve, Figure, float, formatNumber, Handle, Plot, Readout, slider, useAxis, useFigureState } from 'aifn-render'
import { auroc, averagePrecision, metricsAt } from '../_shared/binormal'
import { linspace, toFlat } from 'aifn/foundation/tensor'

const LOG_PI = toFlat(linspace(-3, Math.log10(0.5), 70))

/**
 * Every common metric for one fixed classifier (fixed score distributions, fixed threshold) as the prevalence varies.
 * Metrics built from rates within each true class stay flat; metrics that mix the classes move.
 */
export function MetricsVsPrevalence() {
  const state = useFigureState({
    d: float(2, { min: 0.5, max: 4, step: 0.1, label: 'separation d' }),
    t: slider(-1, 5, 1, { step: 0.05, label: 'threshold t' }),
    logPi: slider(-3, -0.31, -2, { step: 0.01, label: 'log₁₀ prevalence' }),
  })

  const ap = useMemo(() => LOG_PI.map((lp) => averagePrecision(10 ** lp, state.d)), [state.d])

  const series = useMemo(() => {
    const ms = LOG_PI.map((lp) => metricsAt(state.t, 10 ** lp, state.d))
    return [
      { name: 'accuracy', x: LOG_PI, y: ms.map((m) => m.accuracy), slot: 0 },
      { name: 'balanced accuracy', x: LOG_PI, y: ms.map((m) => m.balancedAccuracy), slot: 1 },
      { name: 'F₁', x: LOG_PI, y: ms.map((m) => m.f1), slot: 2 },
      { name: 'MCC', x: LOG_PI, y: ms.map((m) => m.mcc), slot: 3 },
      { name: 'AUROC', x: LOG_PI, y: LOG_PI.map(() => auroc(state.d)), slot: 4 },
      { name: 'average precision', x: LOG_PI, y: ap, slot: 5 },
      {
        name: 'random ranking AP (= π)',
        x: LOG_PI,
        y: LOG_PI.map((lp) => 10 ** lp),
        dashed: true,
        muted: true,
      },
    ] as const
  }, [state.t, state.d, ap])

  const pi = 10 ** state.logPi
  const m = metricsAt(state.t, pi, state.d)

  const xAxis = useAxis({ label: 'log₁₀ prevalence', range: [-3, -0.3] })
  const yAxis = useAxis({ label: 'metric', range: [0, 1] })
  return (
    <Figure
      title="Which metrics move with prevalence"
      state={state}
      caption="One classifier, fixed: scores N(0, 1) for negatives and N(d, 1) for positives, positive when the score exceeds t. The horizontal axis is the prevalence on a log scale, from 1 in 1000 to 1 in 2. Balanced accuracy and AUROC are flat, because they use only rates within each true class. Accuracy rises as positives vanish, while F₁, MCC and average precision fall. Drag the vertical line to read the metrics at one prevalence."

      readouts={
        <>
          <Readout label="π" value={formatNumber(pi)} />
          <Readout label="TPR" value={formatNumber(m.tpr)} />
          <Readout label="FPR" value={formatNumber(m.fpr)} />
          <Readout label="precision" value={formatNumber(m.precision)} />
          <Readout label="accuracy" value={formatNumber(m.accuracy)} />
          <Readout label="balanced accuracy" value={formatNumber(m.balancedAccuracy)} />
          <Readout label="F₁" value={formatNumber(m.f1)} />
          <Readout label="MCC" value={formatNumber(m.mcc)} />
          <Readout label="AP" value={formatNumber(averagePrecision(pi, state.d))} />
          <Readout label="AUROC" value={formatNumber(auroc(state.d))} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={360}>
        <Curve {...series[0]} />
        <Curve {...series[1]} />
        <Curve {...series[2]} />
        <Curve {...series[3]} />
        <Curve {...series[4]} />
        <Curve {...series[5]} />
        <Curve {...series[6]} />
        <Handle {...state.handle('logPi', { label: 'prevalence' })} />
      </Plot>
    </Figure>
  )
}
