import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function UserGeneratedContentVisualizer() {
  const state = useFigureState({
    analysisView: choice<'queue_volume' | 'dialect_disparity'>(
      [
        { value: 'queue_volume', label: 'Moderation Queue Load & Precision (Prevalence = 0.5%)' },
        { value: 'dialect_disparity', label: 'False Flag Disparity (Standard vs Colloquial Dialect)' },
      ],
      'queue_volume',
      { label: 'moderation metric' },
    ),
    threshold: slider(0.2, 0.8, 0.5, {
      step: 0.05,
      label: 'automated toxicity classifier operating threshold tau',
    }),
  })

  const results = useMemo(() => {
    // 100,000 daily posts, 0.5% true violation base rate (500 true violations, 99,500 clean)
    // 85% Standard, 15% Colloquial/AAVE
    const tau = state.threshold

    // Sweep across thresholds [0.15 .. 0.85]
    const tauGrid: number[] = []
    const queueVolumeCurve: number[] = []
    const precisionCurve: number[] = []
    const fprStdCurve: number[] = []
    const fprAaveCurve: number[] = []

    for (let t = 0.15; t <= 0.85; t += 0.05) {
      tauGrid.push(t)

      // Recall decays with higher threshold
      const recall = Math.max(0.1, 1.0 - Math.pow(t, 1.6))
      const truePositives = 500 * recall

      // False positive rates
      const fprStd = 0.06 * Math.exp(-3.5 * t)
      const fprAave = 0.15 * Math.exp(-3.2 * t) // 2.5x higher due to dialectal lexical overlap

      const falsePositives = 84575 * fprStd + 14925 * fprAave
      const totalQueue = truePositives + falsePositives
      const precision = totalQueue > 0 ? (truePositives / totalQueue) * 100.0 : 100.0

      queueVolumeCurve.push(totalQueue)
      precisionCurve.push(precision)
      fprStdCurve.push(fprStd * 100.0)
      fprAaveCurve.push(fprAave * 100.0)
    }

    // Current point calculations
    const currentRecall = Math.max(0.1, 1.0 - Math.pow(tau, 1.6))
    const currentTP = 500 * currentRecall
    const curFprStd = 0.06 * Math.exp(-3.5 * tau)
    const curFprAave = 0.15 * Math.exp(-3.2 * tau)
    const curFP = 84575 * curFprStd + 14925 * curFprAave
    const curQueue = currentTP + curFP
    const curPrecision = curQueue > 0 ? (currentTP / curQueue) * 100.0 : 100.0
    const disparityRatio = curFprAave / Math.max(0.0001, curFprStd)

    return {
      tauGrid,
      queueVolumeCurve,
      precisionCurve,
      fprStdCurve,
      fprAaveCurve,
      currentRecall: currentRecall * 100.0,
      curQueue: Math.round(curQueue),
      curPrecision,
      curFprStd: curFprStd * 100.0,
      curFprAave: curFprAave * 100.0,
      disparityRatio,
      currentY: state.analysisView === 'queue_volume' ? Math.round(curQueue) : curFprAave * 100.0,
    }
  }, [state.analysisView, state.threshold])

  const axisX = useAxis({
    label: 'Automated Moderation Filter Threshold (tau)',
    range: [0.15, 0.85],
  })

  const axisY = useAxis({
    label: state.analysisView === 'queue_volume' ? 'Daily Queue Volume (Posts/Day)' : 'False Positive Rate (%)',
    range: state.analysisView === 'queue_volume' ? [0, 4000] : [0, 10],
  })

  return (
    <Figure
      title="Content Moderation Ingestion: Queue Saturation and Dialectal False Flags"
      state={state}
      hoverReadout={false}
      caption="In user-generated content, extreme class imbalance (0.5% violation rate) means false positives dominate moderation queues at low thresholds. Furthermore, non-standard dialects and colloquial speech suffer 2.5x higher false-positive rates due to lexical overlap with toxicity keywords."
      readouts={
        <>
          <Readout label="Violation Recall" value={`${results.currentRecall.toFixed(1)}%`} />
          <Readout label="Queue Volume" value={`${results.curQueue} posts/day`} />
          <Readout label="Queue Precision" value={`${results.curPrecision.toFixed(1)}% true violations`} />
          <Readout
            label="Dialect Disparity"
            value={`${results.disparityRatio.toFixed(1)}x higher FPR in colloquial posts`}
          />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Operating Threshold Dynamics">
        {state.analysisView === 'queue_volume' ? (
          <Curve
            name="Human Review Queue Backlog"
            x={results.tauGrid}
            y={results.queueVolumeCurve}
            slot={0}
            width={2.5}
          />
        ) : (
          <>
            <Curve
              name="Colloquial / AAVE Dialect FPR"
              x={results.tauGrid}
              y={results.fprAaveCurve}
              slot={2}
              width={2.5}
            />
            <Curve name="Standard English FPR" x={results.tauGrid} y={results.fprStdCurve} slot={0} width={2} dashed />
          </>
        )}
        <Points name="Operating Point" x={[state.threshold]} y={[results.currentY]} slot={1} size={8} />
      </Plot>
    </Figure>
  )
}
