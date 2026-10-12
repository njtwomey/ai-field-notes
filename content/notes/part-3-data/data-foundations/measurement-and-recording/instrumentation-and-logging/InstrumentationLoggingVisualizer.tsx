import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function InstrumentationLoggingVisualizer() {
  const state = useFigureState({
    timeDomain: choice<'event_vs_proc' | 'temporal_inversion'>(
      [
        { value: 'event_vs_proc', label: 'Event Time vs Ingestion / Processing Time' },
        { value: 'temporal_inversion', label: 'Watermark Lag Buffer & Late Arrival Threshold' },
      ],
      'event_vs_proc',
      { label: 'telemetry perspective' },
    ),
    watermarkLag: slider(2, 12, 6, {
      step: 1,
      label: 'watermark allowable lateness buffer delta (minutes)',
    }),
  })

  const results = useMemo(() => {
    // 10 user interaction events
    const indices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    const eventTimes = [2, 4, 5, 8, 9, 11, 13, 15, 17, 20] // minutes
    // Variable network/offline delay for mobile telemetry
    const delays = [1, 8, 2, 1, 14, 2, 1, 9, 2, 3] // event 2 and 5 have severe mobile connection latency!
    const procTimes = eventTimes.map((et, idx) => et + delays[idx])

    const maxLag = state.watermarkLag
    let timelyCount = 0
    let lateDroppedCount = 0
    let inversionCount = 0

    // Watermark threshold relative to current processing time
    const watermarkBound = eventTimes.map((et) => et + maxLag)

    delays.forEach((d) => {
      if (d <= maxLag) timelyCount += 1
      else lateDroppedCount += 1
    })

    // Count order inversions: event j occurred after i, but was processed before i
    for (let i = 0; i < indices.length; i += 1) {
      for (let j = i + 1; j < indices.length; j += 1) {
        if (eventTimes[i] < eventTimes[j] && procTimes[i] > procTimes[j]) {
          inversionCount += 1
        }
      }
    }

    const maxSkew = Math.max(...delays)

    return {
      indices,
      eventTimes,
      procTimes,
      watermarkBound,
      maxSkew,
      timelyCount,
      lateDroppedCount,
      inversionCount,
    }
  }, [state.watermarkLag])

  const axisX = useAxis({
    label: 'Interaction Event Sequence (1 to 10)',
    range: [0.5, 10.5],
  })

  const axisY = useAxis({
    label: 'Timeline (Minutes)',
    range: [0, 32],
  })

  return (
    <Figure
      title="Event Time, Processing Time Skew and Out-of-Order Logging"
      state={state}
      hoverReadout={false}
      caption="Mobile and distributed telemetry introduces severe latency skew: events 2 and 5 occurred early but arrived late due to network buffering. Evaluating streaming features on processing time causes temporal inversions, leaking future context into historical inference windows."
      readouts={
        <>
          <Readout label="Max Network Latency" value={`${results.maxSkew} minutes`} />
          <Readout label="Temporal Inversions" value={`${results.inversionCount} out-of-order pairs`} />
          <Readout label="Watermark Captured" value={`${results.timelyCount} / 10 events captured`} />
          <Readout label="Dropped Late Data" value={`${results.lateDroppedCount} events exceed watermark`} />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Temporal Trajectories: Physical Occurrence vs Server Receipt">
        <Points name="True Event Time (Client Clock)" x={results.indices} y={results.eventTimes} slot={0} size={8} />
        <Points name="Processing Time (Server Arrival)" x={results.indices} y={results.procTimes} slot={1} size={8} />
        <Curve
          name="Watermark Threshold (Event + Lag)"
          x={results.indices}
          y={results.watermarkBound}
          slot={2}
          dashed
          width={2}
        />
      </Plot>
    </Figure>
  )
}
