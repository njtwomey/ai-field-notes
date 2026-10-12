import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function TrainingServingSkewVisualizer() {
  const state = useFigureState({
    skewMechanism: choice<'reimplemented_logic' | 'batch_vs_streaming' | 'stale_caches' | 'different_defaults'>(
      [
        { value: 'reimplemented_logic', label: 'Reimplemented Logic (Python training vs Java/C++ online serving)' },
        { value: 'batch_vs_streaming', label: 'Batch vs Streaming Aggregations (Complete batch vs partial watermark)' },
        { value: 'different_defaults', label: 'Different Default Handling (Imputed mean in train vs 0.0 online)' },
        { value: 'stale_caches', label: 'Stale Feature Store Caches (TTL lag between training logs and KV read)' },
      ],
      'reimplemented_logic',
      { label: 'skew root cause mechanism' },
    ),
    discrepancyPct: slider(0, 30, 10, {
      step: 2,
      label: 'feature value discrepancy magnitude delta (%)',
    }),
  })

  const results = useMemo(() => {
    // Generate accuracy degradation curves across discrepancy delta in [0, 30]%
    const deltaGrid: number[] = []
    const degradationGrid: number[] = []

    for (let d = 0; d <= 30; d += 2) {
      deltaGrid.push(d)
      let deg = 0

      if (state.skewMechanism === 'reimplemented_logic') {
        // Severe non-linear degradation as logic divergence shifts decision boundary
        deg = 45.0 * (1.0 - Math.exp(-0.08 * d))
      } else if (state.skewMechanism === 'batch_vs_streaming') {
        deg = 35.0 * (1.0 - Math.exp(-0.06 * d))
      } else if (state.skewMechanism === 'different_defaults') {
        deg = 28.0 * (1.0 - Math.exp(-0.07 * d))
      } else {
        deg = 22.0 * (1.0 - Math.exp(-0.05 * d))
      }

      degradationGrid.push(deg)
    }

    const currentD = state.discrepancyPct
    let currentDeg = 0
    let psiScore = 0
    let mitigation = ''

    if (state.skewMechanism === 'reimplemented_logic') {
      currentDeg = 45.0 * (1.0 - Math.exp(-0.08 * currentD))
      psiScore = 0.02 + (currentD / 30) * 0.45
      mitigation = 'Single Source of Truth: compile feature logic via shared C++/Rust or ONNX pipeline'
    } else if (state.skewMechanism === 'batch_vs_streaming') {
      currentDeg = 35.0 * (1.0 - Math.exp(-0.06 * currentD))
      psiScore = 0.01 + (currentD / 30) * 0.38
      mitigation = 'Unified Streaming Engine: compute both offline and online views via Apache Beam/Flink'
    } else if (state.skewMechanism === 'different_defaults') {
      currentDeg = 28.0 * (1.0 - Math.exp(-0.07 * currentD))
      psiScore = 0.03 + (currentD / 30) * 0.32
      mitigation = 'Serialized Imputer Artifacts: bundle Scikit-Learn/TFDV imputer parameters into model pipeline'
    } else {
      currentDeg = 22.0 * (1.0 - Math.exp(-0.05 * currentD))
      psiScore = 0.01 + (currentD / 30) * 0.25
      mitigation = 'Point-in-Time Joins & TTL Alignment: synchronise feature store snapshots with inference logs'
    }

    return {
      deltaGrid,
      degradationGrid,
      currentD,
      currentDeg,
      psiScore,
      mitigation,
    }
  }, [state.skewMechanism, state.discrepancyPct])

  const axisX = useAxis({
    label: 'feature discrepancy magnitude delta (%)',
    range: [0, 30],
  })
  const axisY = useAxis({
    label: 'model accuracy loss (% degradation)',
    range: [0, 50],
  })

  return (
    <Figure
      title="Training-serving skew: feature divergence mechanisms and model performance degradation"
      state={state}
      hoverReadout={false}
      caption="Evaluating the impact of training-serving skew (Polyzotis et al. 2018, Breck et al. 2017). When feature transformations are reimplemented across training (Python/Spark) and serving (Java/C++), subtle bugs—such as different string tokenization rules, timezone offsets, or disparate default imputations—introduce numerical discrepancy delta. As discrepancy widens, the model suffers sharp degradation on production traffic while offline validation metrics remain deceptively high."
      readouts={
        <>
          <Readout label="performance degradation" value={`${results.currentDeg.toFixed(1)}% loss`} />
          <Readout
            label="distributional drift (PSI)"
            value={`${results.psiScore.toFixed(3)} (${results.psiScore > 0.25 ? 'Critical Drift' : 'Acceptable'})`}
          />
          <Readout label="recommended remedy" value={results.mitigation} />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Model performance loss vs feature discrepancy delta">
        <Curve
          name="accuracy degradation curve"
          x={results.deltaGrid}
          y={results.degradationGrid}
          slot={results.psiScore > 0.25 ? 1 : 0}
          width={2.5}
        />
        <Points
          name="current operating discrepancy"
          x={[results.currentD]}
          y={[results.currentDeg]}
          slot={results.psiScore > 0.25 ? 1 : 0}
          size={6}
        />
      </Plot>
    </Figure>
  )
}
