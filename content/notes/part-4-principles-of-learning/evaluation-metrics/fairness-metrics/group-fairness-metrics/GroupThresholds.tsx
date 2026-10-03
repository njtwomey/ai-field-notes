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
} from 'aifn-render'
import { curves, expectedCounts, metricsFrom, nearestIndex } from '../../_shared/binormal'

/**
 * Two groups scored by the same kind of classifier, each with its own base rate and separation, and each with its own
 * threshold. The ROC plot shows both groups' curves and operating points; the readouts show the group-fairness gaps.
 */
export function GroupThresholds() {
  const baseA = useParam(0.4, { min: 0.05, max: 0.7, step: 0.01 })
  const baseB = useParam(0.2, { min: 0.05, max: 0.7, step: 0.01 })
  const dA = useParam(2, { min: 0.5, max: 3.5, step: 0.1 })
  const dB = useParam(1.5, { min: 0.5, max: 3.5, step: 0.1 })
  const tA = useParam(1, { min: -3, max: 5, step: 0.01 })
  const tB = useParam(1, { min: -3, max: 5, step: 0.01 })

  const cA = useMemo(() => curves(dA.value, baseA.value), [dA.value, baseA.value])
  const cB = useMemo(() => curves(dB.value, baseB.value), [dB.value, baseB.value])
  const mA = metricsFrom(expectedCounts(tA.value, dA.value, baseA.value, 1000))
  const mB = metricsFrom(expectedCounts(tB.value, dB.value, baseB.value, 1000))

  const series: XYSeries[] = [
    { name: 'chance', type: 'line', x: [0, 1], y: [0, 1], dashed: true, muted: true },
    { name: 'group A', type: 'line', x: cA.fpr, y: cA.tpr, slot: 0 },
    { name: 'group B', type: 'line', x: cB.fpr, y: cB.tpr, slot: 1 },
  ]
  // Each operating point is a location on its own group's ROC curve; dragging it along the curve sets that threshold.
  const handles: Handle[] = [
    {
      kind: 'point',
      at: [mA.fpr, mA.recall],
      label: 'group A',
      onDrag: (p) => tA.set(cA.thresholds[nearestIndex(cA.fpr, cA.tpr, p)]),
    },
    {
      kind: 'point',
      at: [mB.fpr, mB.recall],
      label: 'group B',
      onDrag: (p) => tB.set(cB.thresholds[nearestIndex(cB.fpr, cB.tpr, p)]),
    },
  ]
  const gap = (a: number, b: number) => formatNumber(Math.abs(a - b))

  return (
    <Interactive
      title="One classifier, two groups"
      caption="Each group has its own base rate and its own score separation, and each gets its own threshold (drag the operating points along the ROC curves). Demographic parity asks for equal positive-prediction rates, equal opportunity for equal true-positive rates, equalised odds for equal true- and false-positive rates, and predictive parity for equal precision. With different base rates, try to make every gap zero at once: you cannot, unless the classifier is perfect."
      controls={
        <>
          <ParamSlider label="base rate, group A" param={baseA} />
          <ParamSlider label="base rate, group B" param={baseB} />
          <ParamSlider label="separation, group A" param={dA} />
          <ParamSlider label="separation, group B" param={dB} />
          <ParamSlider label="threshold, group A" param={tA} />
          <ParamSlider label="threshold, group B" param={tB} />
        </>
      }
      readout={
        <>
          <Readout
            label="positive rate A / B"
            value={`${formatNumber(mA.selectionRate)} / ${formatNumber(mB.selectionRate)}`}
          />
          <Readout label="demographic parity gap" value={gap(mA.selectionRate, mB.selectionRate)} />
          <Readout label="TPR gap (equal opportunity)" value={gap(mA.recall, mB.recall)} />
          <Readout label="FPR gap" value={gap(mA.fpr, mB.fpr)} />
          <Readout label="precision gap (predictive parity)" value={gap(mA.precision, mB.precision)} />
        </>
      }
    >
      <div className="mx-auto w-full max-w-xl">
        <XYChart
          series={series}
          xLabel="false-positive rate"
          yLabel="true-positive rate"
          xRange={[0, 1]}
          yRange={[0, 1]}
          equalAspect
          handles={handles}
        />
      </div>
    </Interactive>
  )
}
