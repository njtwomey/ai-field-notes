import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function KnowledgeGraphDataVisualizer() {
  const state = useFigureState({
    kgMode: choice<'inverse_leakage' | 'negative_sampling' | 'filtered_mrr'>(
      [
        {
          value: 'inverse_leakage',
          label: 'Inverse Leakage: FB15k vs FB15k-237 Benchmark Inflation (Dettmers 2018)',
        },
        {
          value: 'negative_sampling',
          label: 'Negative Sampling: Bernoulli Head/Tail Corruption vs 1-to-N Collisions',
        },
        {
          value: 'filtered_mrr',
          label: 'Open-World Evaluation: Raw vs Filtered MRR on Multi-Relational Triples',
        },
      ],
      'inverse_leakage',
      { label: 'knowledge graph challenge' },
    ),
    param1: slider(0, 80, 20, {
      step: 5,
      label: 'inverse leakage rate rho (%) / tails-per-head ratio / candidate rank',
    }),
    param2: slider(1, 10, 4, {
      step: 1,
      label: 'negative sample count / corruption weight / relation degree',
    }),
  })

  const { kgMode, param1, param2 } = state

  const results = useMemo(() => {
    if (kgMode === 'inverse_leakage') {
      // Leakage rate rho from 0% (clean FB15k-237) to 80% (contaminated FB15k)
      const leakageRates = [0, 10, 20, 30, 40, 50, 60, 70, 80]
      // Test MRR (%) artificially boosted by memorizing r_inv(t, h) <=> r(h, t)
      const testMrr = leakageRates.map((rho) => 31.5 + Math.pow(rho / 80, 1.3) * 52.0)
      // Hits@10 (%)
      const hitsAt10 = leakageRates.map((rho) => 49.0 + Math.pow(rho / 80, 1.2) * 39.5)

      const currRho = param1
      const currMrr = 31.5 + Math.pow(currRho / 80, 1.3) * 52.0
      const currHits = 49.0 + Math.pow(currRho / 80, 1.2) * 39.5

      return {
        x: leakageRates,
        y: testMrr,
        baseline: hitsAt10,
        stat1: `Leakage Share: rho = ${currRho}% of test triples`,
        stat2: `Empirical MRR: ${(currMrr / 100).toFixed(3)}`,
        stat3: `Hits@10: ${currHits.toFixed(1)}%`,
        warning:
          currRho > 30
            ? 'Severe inverse-relation leakage; the model trivially memorizes reciprocal rule patterns (e.g. parent/child), creating artificial performance gains without learning multi-hop reasoning'
            : 'Clean non-leaking benchmark (FB15k-237 standard); models must generalize across multi-hop relational path structures',
        axisXLabel: 'Reciprocal Inverse Relation Leakage into Test Set (%)',
        axisYLabel: 'Benchmark Test Link Prediction Metric (%)',
        xRange: [-2, 82] as [number, number],
        yRange: [20, 100] as [number, number],
      }
    } else if (kgMode === 'negative_sampling') {
      // Tails-per-head ratio tph from 0.1 (N-to-1 e.g. citizen_of) to 10.0 (1-to-N e.g. director_of)
      const tphRatios = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0]
      // Bernoulli corruption probability of replacing HEAD: p_head = tph / (tph + hpt)
      // where hpt = 1.0 baseline
      const headCorruptProb = tphRatios.map((tph) => (tph / (tph + 1.0)) * 100)
      // Uniform 50/50 baseline
      const uniformBaseline = tphRatios.map(() => 50)

      const currTph = 0.1 + (param1 / 80) * 9.9
      const currHeadProb = (currTph / (currTph + 1.0)) * 100
      const currTailProb = 100 - currHeadProb

      return {
        x: tphRatios,
        y: headCorruptProb,
        baseline: uniformBaseline,
        stat1: `Relation Ratio: ${currTph.toFixed(1)} tails per head`,
        stat2: `Head Corrupt Prob: ${currHeadProb.toFixed(1)}%`,
        stat3: `Tail Corrupt Prob: ${currTailProb.toFixed(1)}%`,
        warning:
          currTph > 3.0
            ? 'Strong 1-to-N relation: corrupting the head with higher probability prevents severe false-negative collisions when sampling tails'
            : currTph < 0.33
              ? 'Strong N-to-1 relation: corrupting the tail with higher probability preserves balanced negative discrimination'
              : 'Balanced 1-to-1 relation: Bernoulli sampling degenerates naturally to symmetric 50/50 corruption',
        axisXLabel: 'Relation Multiplicity: Tails per Head Ratio (TPH / HPT)',
        axisYLabel: 'Bernoulli Probability of Corrupting HEAD Entity (%)',
        xRange: [0, 10.5] as [number, number],
        yRange: [0, 100] as [number, number],
      }
    } else {
      // True target ranking position from 1 to 20
      const ranks = [1, 2, 3, 4, 5, 7, 9, 12, 15, 20]
      // In 1-to-N relations, there are k other true facts with higher scores
      const competingTrueFacts = param2 // 1 to 10
      // Raw rank includes all competing true facts: rank_raw = rank + competingTrueFacts
      const rawMrr = ranks.map((r) => (1 / (r + competingTrueFacts)) * 100)
      // Filtered rank strips all other true facts: rank_filtered = rank
      const filteredMrr = ranks.map((r) => (1 / r) * 100)

      const currRank = Math.min(20, Math.max(1, Math.round((param1 / 80) * 19 + 1)))
      const currFiltMrr = (1 / currRank) * 100
      const currRawMrr = (1 / (currRank + competingTrueFacts)) * 100
      const rawPenaltyRatio = (currFiltMrr / currRawMrr).toFixed(1)

      return {
        x: ranks,
        y: filteredMrr,
        baseline: rawMrr,
        stat1: `True Fact Rank: #${currRank}`,
        stat2: `Filtered MRR: ${(currFiltMrr / 100).toFixed(3)}`,
        stat3: `Raw MRR: ${(currRawMrr / 100).toFixed(3)} (${rawPenaltyRatio}x penalized)`,
        warning:
          'Under the Open-World Assumption, un-annotated or valid competing true triples must be filtered out during link prediction evaluation; raw MRR unfairly penalizes models for predicting valid alternative facts',
        axisXLabel: 'Ground Truth Target Entity Rank Position',
        axisYLabel: 'Reciprocal Rank (MRR x100)',
        xRange: [0.5, 20.5] as [number, number],
        yRange: [0, 105] as [number, number],
      }
    }
  }, [kgMode, param1, param2])

  const axisX = useAxis({ label: results.axisXLabel, range: results.xRange })
  const axisY = useAxis({ label: results.axisYLabel, range: results.yRange })

  return (
    <Figure
      title="Knowledge Graph Data: Inverse Relation Leakage, Negative Sampling, and Open-World MRR"
      state={state}
      hoverReadout={false}
      readouts={
        <>
          <Readout label="Evaluation Perspective" value={results.stat1} />
          <Readout label="Primary Link Metric" value={results.stat2} />
          <Readout label="Secondary Diagnostic" value={results.stat3} />
          <Readout label="Benchmarking Insight" value={results.warning} />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Knowledge Graph Benchmark Integrity and Ranking Curve">
        <Curve
          name={
            kgMode === 'inverse_leakage'
              ? 'Hits@10 Accuracy Baseline (%)'
              : kgMode === 'negative_sampling'
                ? 'Uniform 50/50 Corruption Baseline (%)'
                : 'Unfiltered Raw MRR Evaluation Curve'
          }
          x={results.x}
          y={results.baseline}
          slot={2}
          dashed
          width={1.8}
        />
        <Curve
          name={
            kgMode === 'inverse_leakage'
              ? 'Mean Reciprocal Rank MRR (%)'
              : kgMode === 'negative_sampling'
                ? 'Bernoulli Head Corruption Probability (%)'
                : 'Standard Filtered MRR Evaluation Curve'
          }
          x={results.x}
          y={results.y}
          slot={0}
          width={2.5}
        />
        <Points name="Benchmark Data Points" x={results.x} y={results.y} slot={0} size={7} />
      </Plot>
    </Figure>
  )
}
