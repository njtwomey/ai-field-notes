import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function HeterogeneousAndTemporalGraphsVisualizer() {
  const state = useFigureState({
    graphMode: choice<'temporal_splits' | 'snapshot_tradeoff' | 'metapath_attention'>(
      [
        {
          value: 'temporal_splits',
          label: 'Temporal Splits: Random vs Chronological Leakage Inflation (Rossi 2020)',
        },
        {
          value: 'snapshot_tradeoff',
          label: 'Snapshot Granularity: Window Delta t vs Graph Sparsity (Pareja 2020)',
        },
        {
          value: 'metapath_attention',
          label: 'Heterogeneous Metapaths: Semantic Path Attention Weights (Wang 2019)',
        },
      ],
      'temporal_splits',
      { label: 'temporal/heterogeneous challenge' },
    ),
    param1: slider(0, 50, 15, {
      step: 5,
      label: 'temporal leakage fraction (%) / window delta t / path hops',
    }),
    param2: slider(1, 10, 4, {
      step: 1,
      label: 'historical window / event density / attention temperature',
    }),
  })

  const { graphMode, param1, param2 } = state

  const results = useMemo(() => {
    if (graphMode === 'temporal_splits') {
      // Future leakage fraction (%) from 0% to 50%
      const leakageFractions = [0, 5, 10, 15, 20, 25, 30, 35, 40, 50]
      // Offline evaluated test AUC (%) inflated by leaking future edges into past aggregations
      const leakedTestAuc = leakageFractions.map((l) => 72.0 + Math.pow(l / 50, 1.2) * 24.5)
      // True real-world forward-in-time deployment AUC (%) strictly without future leaks
      const deployAuc = leakageFractions.map((l) => Math.max(60, 72.0 - l * 0.18))

      const currLeak = param1
      const currEvalAuc = 72.0 + Math.pow(currLeak / 50, 1.2) * 24.5
      const currTrueAuc = Math.max(60, 72.0 - currLeak * 0.18)
      const inflationGap = currEvalAuc - currTrueAuc

      return {
        x: leakageFractions,
        y: leakedTestAuc,
        baseline: deployAuc,
        stat1: `Future Edge Leakage: ${currLeak}%`,
        stat2: `Inflated Eval AUC: ${currEvalAuc.toFixed(1)}%`,
        stat3: `True Deploy AUC: ${currTrueAuc.toFixed(1)}% (Gap: +${inflationGap.toFixed(1)}%)`,
        warning:
          currLeak > 10
            ? 'Random edge splitting allows future edges to participate in historical message passing; offline evaluation overestimates deployment performance by up to 25% AUC'
            : 'Chronological time-respecting split preserves the arrow of time; evaluation reflects realistic forward-in-time link prediction',
        axisXLabel: 'Future Edges Leaked into Historical Aggregation Graph (%)',
        axisYLabel: 'Dynamic Link Prediction Performance AUC (%)',
        xRange: [-2, 52] as [number, number],
        yRange: [55, 100] as [number, number],
      }
    } else if (graphMode === 'snapshot_tradeoff') {
      // Snapshot window size Delta t (days): 1 to 30 days
      const windowDays = [1, 2, 3, 5, 7, 10, 14, 21, 28, 30]
      // Graph sparsity / empty snapshot rate (% of snapshots with insufficient events)
      const emptySnapshotPct = windowDays.map((w) => Math.max(2, 85 * Math.exp(-0.25 * w)))
      // Temporal fine-grained dynamics blur / loss (% of sub-window velocity lost)
      const temporalBlurPct = windowDays.map((w) => Math.min(90, Math.pow(w / 30, 1.4) * 88))

      const currWin = Math.min(30, Math.max(1, Math.round((param1 / 50) * 29 + 1)))
      const currEmpty = Math.max(2, 85 * Math.exp(-0.25 * currWin))
      const currBlur = Math.min(90, Math.pow(currWin / 30, 1.4) * 88)

      return {
        x: windowDays,
        y: temporalBlurPct,
        baseline: emptySnapshotPct,
        stat1: `Snapshot Window: Delta t = ${currWin} days`,
        stat2: `Temporal Blur: ${currBlur.toFixed(1)}%`,
        stat3: `Empty Snapshots: ${currEmpty.toFixed(1)}%`,
        warning:
          currWin > 14
            ? 'Coarse snapshot window (Delta t > 14 days) collapses rapid temporal event orders into static aggregated edges, destroying event causality'
            : currWin < 3
              ? 'Ultra-fine snapshot window creates fragmented, empty graphs with high computational overhead; Continuous-Time TGN is preferable'
              : 'Balanced discrete snapshot pacing balances structural density with temporal evolution',
        axisXLabel: 'Discrete Graph Snapshot Window Interval Delta t (Days)',
        axisYLabel: 'Temporal Resolution Loss (%) / Empty Snapshot Fraction (%)',
        xRange: [0.5, 31] as [number, number],
        yRange: [0, 95] as [number, number],
      }
    } else {
      // Metapath candidate index: 1=APA, 2=APCPA, 3=APVPA, 4=APTPA, 5=APTPA-Extended
      const metapathIndices = [1, 2, 3, 4, 5]
      // Semantic Attention Weights (%) in HAN model across metapaths
      const attentionWeights = [48, 26, 15, 8, 3]
      // Feature divergence score (%)
      const featureDivergence = [10, 35, 60, 75, 90]

      const currIdx = Math.min(5, Math.max(1, Math.round((param1 / 50) * 4 + 1)))
      const names = [
        'Author-Paper-Author (APA)',
        'Author-Paper-Conference-Paper-Author (APCPA)',
        'Author-Paper-Venue-Paper-Author (APVPA)',
        'Author-Paper-Topic-Paper-Author (APTPA)',
        'Author-Paper-Year-Paper-Author (APYPA)',
      ]
      const currWeight = attentionWeights[currIdx - 1]
      const currDiv = featureDivergence[currIdx - 1]

      return {
        x: metapathIndices,
        y: attentionWeights,
        baseline: featureDivergence,
        stat1: `Metapath: ${names[currIdx - 1]}`,
        stat2: `Semantic Attention: ${currWeight}%`,
        stat3: `Semantic Divergence: ${currDiv}%`,
        warning:
          'Heterogeneous Graph Attention (HAN) automatically weights semantic metapaths via hierarchical attention; direct co-authorship (APA) dominates over noisy high-hop paths',
        axisXLabel: 'Heterogeneous Metapath Schema (1=APA, 2=APCPA, 3=APVPA, 4=APTPA, 5=APYPA)',
        axisYLabel: 'Learned Hierarchical Semantic Attention Weight (%)',
        xRange: [0.5, 5.5] as [number, number],
        yRange: [0, 100] as [number, number],
      }
    }
  }, [graphMode, param1, param2])

  const axisX = useAxis({ label: results.axisXLabel, range: results.xRange })
  const axisY = useAxis({ label: results.axisYLabel, range: results.yRange })

  return (
    <Figure
      title="Heterogeneous and Temporal Graphs: Time-Respecting Splits, Snapshots, and Metapaths"
      state={state}
      hoverReadout={false}
      readouts={
        <>
          <Readout label="Temporal Parameter" value={results.stat1} />
          <Readout label="Observed Benchmark Metric" value={results.stat2} />
          <Readout label="Real-World Deployment Delta" value={results.stat3} />
          <Readout label="Methodological Warning" value={results.warning} />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Temporal Generalization and Metapath Semantic Weights">
        <Curve
          name={
            graphMode === 'temporal_splits'
              ? 'True Forward-in-Time Deployment AUC (%)'
              : graphMode === 'snapshot_tradeoff'
                ? 'Empty Snapshot Fraction (%)'
                : 'Semantic Divergence Baseline (%)'
          }
          x={results.x}
          y={results.baseline}
          slot={2}
          dashed
          width={1.8}
        />
        <Curve
          name={
            graphMode === 'temporal_splits'
              ? 'Inflated Test Evaluation AUC under Leakage (%)'
              : graphMode === 'snapshot_tradeoff'
                ? 'Temporal Kinematic Resolution Loss (%)'
                : 'Learned Metapath Semantic Attention (%)'
          }
          x={results.x}
          y={results.y}
          slot={0}
          width={2.5}
        />
        <Points name="Evaluated Topologies" x={results.x} y={results.y} slot={0} size={7} />
      </Plot>
    </Figure>
  )
}
