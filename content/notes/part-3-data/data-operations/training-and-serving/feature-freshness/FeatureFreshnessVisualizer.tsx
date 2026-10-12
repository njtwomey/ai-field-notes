import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function FeatureFreshnessVisualizer() {
  const state = useFigureState({
    architecture: choice<'batch_precomputed' | 'streaming_near_realtime' | 'on_demand'>(
      [
        {
          value: 'streaming_near_realtime',
          label: 'Streaming Near-Real-Time (Kafka/Flink to KV Store: staleness < 1 min, latency ~8ms)',
        },
        {
          value: 'batch_precomputed',
          label: 'Batch Precomputed (Nightly ETL to KV Store: staleness ~12-24h, latency ~1.5ms)',
        },
        {
          value: 'on_demand',
          label: 'On-Demand Request-Time (Python/C++ compute in API: staleness 0s, latency ~65ms)',
        },
      ],
      'streaming_near_realtime',
      { label: 'feature computation architecture' },
    ),
    stalenessHours: slider(0.1, 48.0, 1.0, {
      step: 0.5,
      label: 'feature staleness / ingestion lag (hours)',
    }),
  })

  const results = useMemo(() => {
    // Generate AUC degradation curve as staleness tau increases from 0 to 48h
    const tauGrid: number[] = []
    const aucGrid: number[] = []

    for (let t = 0.1; t <= 48.0; t += 1.0) {
      tauGrid.push(t)
      // Exponential decay of temporal feature value
      const auc = 0.5 + 0.38 * Math.exp(-0.06 * t)
      aucGrid.push(auc)
    }

    const t = state.stalenessHours
    let p99LatencyMs = 0
    let effectiveAuc = 0.5 + 0.38 * Math.exp(-0.06 * t)
    let slaStatus = ''

    if (state.architecture === 'batch_precomputed') {
      p99LatencyMs = 1.8
      slaStatus = 'Low Latency (<2ms), High Information Staleness'
    } else if (state.architecture === 'streaming_near_realtime') {
      p99LatencyMs = 7.5
      slaStatus = 'Balanced Architecture (SLA Compliant with Fresh Features)'
    } else {
      p99LatencyMs = 68.0
      slaStatus = 'High Compute Latency (Risk of Timeout on SLA < 50ms)'
    }

    return {
      tauGrid,
      aucGrid,
      currentTau: t,
      effectiveAuc,
      p99LatencyMs,
      slaStatus,
    }
  }, [state.architecture, state.stalenessHours])

  const axisX = useAxis({
    label: 'feature staleness / lag tau (hours)',
    range: [0, 48],
  })
  const axisY = useAxis({
    label: 'model predictive accuracy (AUC)',
    range: [0.5, 0.9],
  })

  return (
    <Figure
      title="Feature freshness, serving latency budgets, and model accuracy decay"
      state={state}
      hoverReadout={false}
      caption="Evaluating the trade-offs of feature freshness in production serving (Polyzotis et al. 2018, Zinkevich 2017). Highly dynamic behavioral features (e.g. click sequence, session intent, inventory count) decay in predictive power as staleness tau grows. Batch precomputation delivers ultra-low serving latency (<2ms) via key-value caches but serves stale features. On-demand compute eliminates staleness but risks breaching strict P99 latency budgets. Streaming near-real-time pipelines balance latency against freshness."
      readouts={
        <>
          <Readout label="P99 lookup latency" value={`${results.p99LatencyMs.toFixed(1)} ms`} />
          <Readout label="predictive performance" value={`${results.effectiveAuc.toFixed(3)} AUC`} />
          <Readout label="operational posture" value={results.slaStatus} />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Predictive utility decay vs feature staleness (hours)">
        <Curve name="AUC decay curve" x={results.tauGrid} y={results.aucGrid} slot={0} width={2.5} />
        <Points
          name="current operational freshness"
          x={[results.currentTau]}
          y={[results.effectiveAuc]}
          slot={0}
          size={6}
        />
      </Plot>
    </Figure>
  )
}
