import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function BiologicalSequenceDataVisualizer() {
  const state = useFigureState({
    mode: choice<'kmer_explosion' | 'homology_leakage' | 'alignment_substitution'>(
      [
        {
          value: 'kmer_explosion',
          label: 'k-mer Space: Vocabulary Explosion & Sparsity vs k (DNA vs Protein)',
        },
        {
          value: 'homology_leakage',
          label: 'Sequence Identity Split: Apparent Test AUC vs Homology Threshold (Rives 2021)',
        },
        {
          value: 'alignment_substitution',
          label: 'BLOSUM62 Alignment: Score Dynamics vs Sequence Divergence (Henikoff 1992)',
        },
      ],
      'kmer_explosion',
      { label: 'sequence analysis' },
    ),
    param1: slider(1, 6, 3, {
      step: 1,
      label: 'k-mer length k / sequence length scale / identity threshold (%)',
    }),
    param2: slider(100, 1500, 400, {
      step: 100,
      label: 'protein sequence length L / gap open penalty / test homology depth',
    }),
  })

  const { mode, param1, param2 } = state

  const results = useMemo(() => {
    if (mode === 'kmer_explosion') {
      const k = param1 // 1 to 6
      const seqLen = param2 // 100 to 1500

      const kVals = [1, 2, 3, 4, 5, 6]
      const dnaDoms: number[] = []
      const proteinDoms: number[] = []

      // For DNA: 4^k, For Protein: 20^k. We'll plot log10(vocabulary size)
      for (const curK of kVals) {
        dnaDoms.push(Math.log10(Math.pow(4, curK)))
        proteinDoms.push(Math.min(8, Math.log10(Math.pow(20, curK))))
      }

      const activeVocabDNA = Math.pow(4, k)
      const activeVocabProtein = Math.pow(20, k)
      const maxPossibleKmers = Math.max(1, seqLen - k + 1)
      const dnaSparsity = Math.max(0, 100 - (maxPossibleKmers / activeVocabDNA) * 100)
      const proteinSparsity = Math.max(0, 100 - (maxPossibleKmers / activeVocabProtein) * 100)

      return {
        x: kVals,
        y1: dnaDoms,
        y2: proteinDoms,
        stat1: `k = ${k}, L = ${seqLen} aa`,
        stat2: `DNA: ${activeVocabDNA.toLocaleString()} words (${dnaSparsity.toFixed(0)}% sparse)`,
        stat3: `Protein: ${activeVocabProtein.toLocaleString()} words`,
        insight: `Protein k-mer sparsity: ${proteinSparsity.toFixed(1)}% empty`,
        axisXLabel: 'k-mer Length k',
        axisYLabel: 'log10(Vocabulary Dimension |Σ|^k)',
        xRange: [0.8, 6.2] as [number, number],
        yRange: [0, 8] as [number, number],
      }
    } else if (mode === 'homology_leakage') {
      // Sequence identity cutoff tau from 20% to 100%
      const thresholds = [20, 30, 40, 50, 60, 70, 80, 90, 100]
      const randomSplitAUC: number[] = []
      const clusterSplitAUC: number[] = []

      // Random splits hold inflated performance around 95% due to homologous near-duplicates
      // Cluster splits drop to realistic performance at low identity (e.g. 72% at 30% identity)
      for (const tau of thresholds) {
        // Apparent test AUC with random leakage
        randomSplitAUC.push(95.5 - 0.02 * (100 - tau))
        // Rigorous cluster split test AUC
        const honestAUC = 68.0 + 26.0 * Math.pow(tau / 100, 1.6)
        clusterSplitAUC.push(honestAUC)
      }

      const activeTau = 20 + (param1 - 1) * 10 // 20% to 70%
      const activeHonest = 68.0 + 26.0 * Math.pow(activeTau / 100, 1.6)

      return {
        x: thresholds,
        y1: randomSplitAUC,
        y2: clusterSplitAUC,
        stat1: `Split cutoff: ≤${activeTau}% identity`,
        stat2: 'Random split: 95.2% AUC (leaked)',
        stat3: `Cluster split: ${activeHonest.toFixed(1)}% AUC`,
        insight: 'Homology leakage causes ~25% optimistic evaluation bias',
        axisXLabel: 'Max Training-Test Sequence Identity (%)',
        axisYLabel: 'Test Performance AUC (%)',
        xRange: [18, 102] as [number, number],
        yRange: [60, 100] as [number, number],
      }
    } else {
      // Alignment score vs divergence distance
      const divergences = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90]
      const blosumScores: number[] = []
      const identityPercentages: number[] = []

      for (const d of divergences) {
        // As divergence grows, BLOSUM alignment score decays rapidly
        const score = 100 * Math.exp(-0.035 * d) - 0.2 * d
        blosumScores.push(Math.max(-20, score))
        identityPercentages.push(100 - d)
      }

      return {
        x: divergences,
        y1: blosumScores,
        y2: identityPercentages,
        stat1: 'BLOSUM62 Log-Odds Scoring',
        stat2: 'Affine gap: γ(g) = -d - (g-1)e',
        stat3: 'PAM vs BLOSUM matrix scale',
        insight: 'Twilight zone at <25% identity: alignments lose statistical power',
        axisXLabel: 'Sequence Divergence (% mutations)',
        axisYLabel: 'Normalized Alignment Score & Identity (%)',
        xRange: [0, 95] as [number, number],
        yRange: [-20, 100] as [number, number],
      }
    }
  }, [mode, param1, param2])

  const axisX = useAxis({ label: results.axisXLabel, range: results.xRange })
  const axisY = useAxis({ label: results.axisYLabel, range: results.yRange })

  return (
    <Figure
      title="Biological Sequences: k-mer Dimensionality, Homology Leakage, and Alignment Dynamics"
      state={state}
      hoverReadout={false}
      readouts={
        <>
          <Readout label="Sequence Parameter" value={results.stat1} />
          <Readout label="Scale / Baseline" value={results.stat2} />
          <Readout label="Target Metric" value={results.stat3} />
          <Readout label="Biological Insight" value={results.insight} />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Sequence Modality Characteristics and Generalization Bounds">
        <Curve
          name={
            mode === 'kmer_explosion'
              ? 'DNA |Σ|^k Dimension (log10)'
              : mode === 'homology_leakage'
                ? 'Random Train-Test Split (Homology Leaked AUC %)'
                : 'BLOSUM62 Dynamic Alignment Score (%)'
          }
          x={results.x}
          y={results.y1}
          slot={0}
          width={2.5}
        />
        <Curve
          name={
            mode === 'kmer_explosion'
              ? 'Protein |Σ|^k Dimension (log10)'
              : mode === 'homology_leakage'
                ? 'Cluster Split by Identity (Unleaked AUC %)'
                : 'Observed Sequence Identity (%)'
          }
          x={results.x}
          y={results.y2}
          slot={1}
          dashed
          width={2.0}
        />
        <Points name="Evaluated Regimes" x={results.x} y={results.y1} slot={0} size={7} />
      </Plot>
    </Figure>
  )
}
