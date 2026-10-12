import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function BatchStreamProcessingVisualizer() {
  const state = useFigureState({
    processingMode: choice<'watermark' | 'late_data' | 'lambda_reconcile'>(
      [
        {
          value: 'watermark',
          label: 'Event-Time Watermark Slack (Akidau et al. 2015: Completeness vs trigger latency)',
        },
        { value: 'late_data', label: 'Late Event Dropping (Strict allowed lateness window)' },
        { value: 'lambda_reconcile', label: 'Batch vs Streaming Reconciliation (Lambda architecture dual-path)' },
      ],
      'watermark',
      { label: 'stream processing regime' },
    ),
    slackMinutes: slider(1, 60, 15, {
      step: 1,
      label: 'watermark allowed lateness slack tau (minutes)',
    }),
  })

  const results = useMemo(() => {
    // Generate completeness curves across slack tau in [1, 60] minutes
    const tauGrid: number[] = []
    const completenessGrid: number[] = []

    for (let t = 1; t <= 60; t += 2) {
      tauGrid.push(t)
      // Event arrival distribution follows heavy-tailed log-normal / exponential
      const comp = Math.min(99.9, 100.0 * (1.0 - 0.45 * Math.exp(-0.08 * t)))
      completenessGrid.push(comp)
    }

    const t = state.slackMinutes
    const currentCompleteness = Math.min(99.9, 100.0 * (1.0 - 0.45 * Math.exp(-0.08 * t)))
    const droppedLatePct = 100.0 - currentCompleteness
    const triggerDelay = t + 1.2

    let reconciliationStatus = ''
    if (state.processingMode === 'watermark') {
      reconciliationStatus =
        currentCompleteness >= 95.0
          ? 'High Completeness (Watermark bounds late events)'
          : 'Premature Watermark (High late event drop rate)'
    } else if (state.processingMode === 'late_data') {
      reconciliationStatus = `Dropped events: ${droppedLatePct.toFixed(1)}% permanently discarded`
    } else {
      reconciliationStatus = `Batch/Stream disagreement: ${(droppedLatePct * 0.8).toFixed(1)}% discrepancy until nightly batch reconciliation`
    }

    return {
      tauGrid,
      completenessGrid,
      currentTau: t,
      currentCompleteness,
      droppedLatePct,
      triggerDelay,
      reconciliationStatus,
    }
  }, [state.processingMode, state.slackMinutes])

  const axisX = useAxis({
    label: 'watermark allowed lateness slack tau (minutes)',
    range: [1, 60],
  })
  const axisY = useAxis({
    label: 'stream window event completeness (%)',
    range: [50, 102],
  })

  return (
    <Figure
      title="Batch and stream processing: event time, watermarks, and window completeness"
      state={state}
      hoverReadout={false}
      caption="Evaluating stream processing trade-offs under the Dataflow model (Akidau et al. 2015). In stream processing, events arrive out of order due to mobile offline queues and network retries. Watermarks advance event time, balancing trigger latency against completeness. Short watermark slack triggers window aggregations rapidly but discards late-arriving events, producing streaming feature values that systematically disagree with finalized offline batch computations."
      readouts={
        <>
          <Readout
            label="window completeness"
            value={`${results.currentCompleteness.toFixed(1)}% of events captured`}
          />
          <Readout
            label="window trigger latency"
            value={`${results.triggerDelay.toFixed(1)} minutes after event time`}
          />
          <Readout label="batch vs stream alignment" value={results.reconciliationStatus} />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Stream window completeness vs watermark delay slack">
        <Curve name="window completeness (%)" x={results.tauGrid} y={results.completenessGrid} slot={0} width={2.5} />
        <Points
          name="current watermark operating point"
          x={[results.currentTau]}
          y={[results.currentCompleteness]}
          slot={0}
          size={6}
        />
      </Plot>
    </Figure>
  )
}
