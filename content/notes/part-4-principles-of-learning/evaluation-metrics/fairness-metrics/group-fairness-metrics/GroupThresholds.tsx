import { useMemo } from 'react'
import { Curve, Figure, float, formatNumber, Handle, Plot, Readout, slider, useAxis, useFigureState } from 'aifn-render'
import { curves, expectedCounts, metricsFrom, nearestIndex } from '../../_shared/binormal'

/**
 * Two groups scored by the same kind of classifier, each with its own base rate and separation, and each with its own
 * threshold. The ROC plot shows both groups' curves and operating points; the readouts show the group-fairness gaps.
 */
export function GroupThresholds() {
  const state = useFigureState({
    baseA: float(0.4, { min: 0.05, max: 0.7, step: 0.01, label: 'base rate, group A' }),
    baseB: float(0.2, { min: 0.05, max: 0.7, step: 0.01, label: 'base rate, group B' }),
    dA: float(2, { min: 0.5, max: 3.5, step: 0.1, label: 'separation, group A' }),
    dB: float(1.5, { min: 0.5, max: 3.5, step: 0.1, label: 'separation, group B' }),
    tA: slider(-3, 5, 1, { step: 0.01, label: 'threshold, group A' }),
    tB: slider(-3, 5, 1, { step: 0.01, label: 'threshold, group B' }),
  })

  const cA = useMemo(() => curves(state.dA, state.baseA), [state.dA, state.baseA])
  const cB = useMemo(() => curves(state.dB, state.baseB), [state.dB, state.baseB])
  const mA = metricsFrom(expectedCounts(state.tA, state.dA, state.baseA, 1000))
  const mB = metricsFrom(expectedCounts(state.tB, state.dB, state.baseB, 1000))

  const series = [
    { name: 'chance', x: [0, 1], y: [0, 1], dashed: true, muted: true },
    { name: 'group A', x: cA.fpr, y: cA.tpr, slot: 0 },
    { name: 'group B', x: cB.fpr, y: cB.tpr, slot: 1 },
  ] as const
  // Each operating point is a location on its own group's ROC curve; dragging it along the curve sets that threshold.
  const gap = (a: number, b: number) => formatNumber(Math.abs(a - b))

  const xAxis = useAxis({ label: 'false-positive rate', range: [0, 1] })
  const yAxis = useAxis({ label: 'true-positive rate', range: [0, 1], equal: xAxis })
  return (
    <Figure
      title="One classifier, two groups"
      state={state}
      caption="Each group has its own base rate and its own score separation, and each gets its own threshold (drag the operating points along the ROC curves). Demographic parity asks for equal positive-prediction rates, equal opportunity for equal true-positive rates, equalised odds for equal true- and false-positive rates, and predictive parity for equal precision. With different base rates, try to make every gap zero at once: you cannot, unless the classifier is perfect."

      readouts={
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
        <Plot x={xAxis} y={yAxis}>
          <Curve {...series[0]} />
          <Curve {...series[1]} />
          <Curve {...series[2]} />
          <Handle
            kind="point"
            at={[mA.fpr, mA.recall]}
            label="group A"
            onDrag={(p) => state.set('tA', cA.thresholds[nearestIndex(cA.fpr, cA.tpr, p)])}
          />
          <Handle
            kind="point"
            at={[mB.fpr, mB.recall]}
            label="group B"
            onDrag={(p) => state.set('tB', cB.thresholds[nearestIndex(cB.fpr, cB.tpr, p)])}
          />
        </Plot>
      </div>
    </Figure>
  )
}
