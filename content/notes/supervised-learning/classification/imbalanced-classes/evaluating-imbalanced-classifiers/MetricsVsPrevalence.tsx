import { useMemo } from 'react'
import {
  Interactive,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type XYSeries,
} from '@/components/viz'
import { linspace } from '@/lib/math'
import { auroc, averagePrecision, metricsAt } from '../_shared/binormal'

const LOG_PI = linspace(-3, Math.log10(0.5), 70)

/**
 * Every common metric for one fixed classifier (fixed score distributions, fixed threshold) as the prevalence varies.
 * Metrics built from rates within each true class stay flat; metrics that mix the classes move.
 */
export function MetricsVsPrevalence() {
  const d = useParam(2, { min: 0.5, max: 4, step: 0.1 })
  const t = useParam(1, { min: -1, max: 5, step: 0.05 })
  const logPi = useParam(-2, { min: -3, max: -0.31, step: 0.01 })

  const ap = useMemo(() => LOG_PI.map((lp) => averagePrecision(10 ** lp, d.value)), [d.value])

  const series = useMemo<XYSeries[]>(() => {
    const ms = LOG_PI.map((lp) => metricsAt(t.value, 10 ** lp, d.value))
    return [
      { name: 'accuracy', type: 'line', x: LOG_PI, y: ms.map((m) => m.accuracy), slot: 0 },
      { name: 'balanced accuracy', type: 'line', x: LOG_PI, y: ms.map((m) => m.balancedAccuracy), slot: 1 },
      { name: 'F₁', type: 'line', x: LOG_PI, y: ms.map((m) => m.f1), slot: 2 },
      { name: 'MCC', type: 'line', x: LOG_PI, y: ms.map((m) => m.mcc), slot: 3 },
      { name: 'AUROC', type: 'line', x: LOG_PI, y: LOG_PI.map(() => auroc(d.value)), slot: 4 },
      { name: 'average precision', type: 'line', x: LOG_PI, y: ap, slot: 5 },
      {
        name: 'random ranking AP (= π)',
        type: 'line',
        x: LOG_PI,
        y: LOG_PI.map((lp) => 10 ** lp),
        dashed: true,
        muted: true,
      },
    ]
  }, [t.value, d.value, ap])

  const pi = 10 ** logPi.value
  const m = metricsAt(t.value, pi, d.value)
  const handles: Handle[] = [{ kind: 'x', at: logPi.value, label: 'prevalence', onDrag: (x) => logPi.set(x) }]

  return (
    <Interactive
      title="Which metrics move with prevalence"
      caption="One classifier, fixed: scores N(0, 1) for negatives and N(d, 1) for positives, positive when the score exceeds t. The horizontal axis is the prevalence on a log scale, from 1 in 1000 to 1 in 2. Balanced accuracy and AUROC are flat, because they use only rates within each true class. Accuracy rises as positives vanish, while F₁, MCC and average precision fall. Drag the vertical line to read the metrics at one prevalence."
      controls={
        <>
          <ParamSlider label="separation d" param={d} />
          <ParamSlider label="threshold t" param={t} />
          <ParamSlider label="log₁₀ prevalence" param={logPi} />
        </>
      }
      readout={
        <>
          <Readout label="π" value={formatNumber(pi)} />
          <Readout label="TPR" value={formatNumber(m.tpr)} />
          <Readout label="FPR" value={formatNumber(m.fpr)} />
          <Readout label="precision" value={formatNumber(m.precision)} />
          <Readout label="accuracy" value={formatNumber(m.accuracy)} />
          <Readout label="balanced accuracy" value={formatNumber(m.balancedAccuracy)} />
          <Readout label="F₁" value={formatNumber(m.f1)} />
          <Readout label="MCC" value={formatNumber(m.mcc)} />
          <Readout label="AP" value={formatNumber(averagePrecision(pi, d.value))} />
          <Readout label="AUROC" value={formatNumber(auroc(d.value))} />
        </>
      }
    >
      <XYChart
        height={360}
        xLabel="log₁₀ prevalence"
        yLabel="metric"
        xRange={[-3, -0.3]}
        yRange={[0, 1]}
        handles={handles}
        series={series}
      />
    </Interactive>
  )
}
