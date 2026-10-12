import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function OmicsTablesVisualizer() {
  const state = useFigureState({
    mode: choice<'mean_variance' | 'batch_correction' | 'fdr_selection'>(
      [
        {
          value: 'mean_variance',
          label: 'Negative Binomial Dispersion: Var(Y) = μ + α·μ² vs VST Normalization (Love 2014)',
        },
        {
          value: 'batch_correction',
          label: 'Batch Effects: Uncorrected Confounding vs ComBat Empirical Bayes (Johnson 2007)',
        },
        {
          value: 'fdr_selection',
          label: 'p >> n Multi-Testing: Benjamini-Hochberg FDR vs Bonferroni (Benjamini 1995)',
        },
      ],
      'mean_variance',
      { label: 'omics analysis' },
    ),
    param1: slider(1, 10, 4, {
      step: 1,
      label: 'dispersion scale α (x0.05) / batch shift delta / FDR target q (%)',
    }),
    param2: slider(10, 100, 50, {
      step: 10,
      label: 'sequencing depth factor (x10k reads) / sample count n / gene tests G (x100)',
    }),
  })

  const { mode, param1, param2 } = state

  const results = useMemo(() => {
    if (mode === 'mean_variance') {
      const alpha = param1 * 0.05 // 0.05 to 0.50 dispersion
      const depthScale = param2 / 50 // 0.2 to 2.0

      // Plot mean expression mu (1 to 100) vs standard deviation or log-variance
      const muVals: number[] = []
      const rawVariance: number[] = []
      const vstVariance: number[] = []

      for (let mu = 2; mu <= 100; mu += 4) {
        const scaledMu = mu * depthScale
        // Negative binomial variance: Var = mu + alpha * mu^2
        const nbVar = scaledMu + alpha * Math.pow(scaledMu, 2)
        // Poisson variance: Var = scaledMu
        // VST stabilizes variance to approximately constant ~ 1.0 (or normalized scale)
        const stabilized = 1.0 + 0.15 * Math.log10(1 + scaledMu / (4 * alpha * 10))

        muVals.push(scaledMu)
        rawVariance.push(Math.log10(nbVar))
        vstVariance.push(stabilized)
      }

      const activeMu = 50 * depthScale
      const activeNBVar = activeMu + alpha * Math.pow(activeMu, 2)

      return {
        x: muVals,
        y1: rawVariance,
        y2: vstVariance,
        stat1: `Dispersion α = ${alpha.toFixed(2)}`,
        stat2: `Var(μ=50) = ${activeNBVar.toFixed(0)}`,
        stat3: 'VST variance stabilized ~ 1.0',
        insight: 'Heteroskedasticity: variance explodes quadratically with mean count',
        axisXLabel: 'Mean Gene Count μ',
        axisYLabel: 'log10(Raw Variance) & VST Transformed Variance',
        xRange: [0, 205] as [number, number],
        yRange: [0, 5] as [number, number],
      }
    } else if (mode === 'batch_correction') {
      // Synthetic PCA coordinates of samples:
      // Condition A (Control) vs Condition B (Case) across Batch 1 and Batch 2
      // Uncorrected: samples separate primarily along PC1 by batch
      // Corrected: samples align by biological condition
      const batchShift = param1 * 0.8 // 0.8 to 8.0

      // We plot 8 representative sample points across PC1
      // 1: Batch 1 Control, 2: Batch 1 Case, 3: Batch 2 Control, 4: Batch 2 Case
      const sampleIndices = [1, 2, 3, 4, 5, 6, 7, 8]
      const uncorrectedPC1: number[] = [
        -4.0,
        -3.2,
        -3.8,
        -3.0, // Batch 1
        -4.0 + batchShift,
        -3.2 + batchShift,
        -3.8 + batchShift,
        -3.0 + batchShift, // Batch 2
      ]
      const correctedPC1: number[] = [-3.5, 3.5, -3.2, 3.8, -3.6, 3.4, -3.4, 3.7]

      return {
        x: sampleIndices,
        y1: uncorrectedPC1,
        y2: correctedPC1,
        stat1: `Batch Shift δ = +${batchShift.toFixed(1)} SD`,
        stat2: 'PC1 uncorrected: 68% batch variance',
        stat3: 'ComBat PC1: 78% biology variance',
        insight: 'Uncorrected batch effects obscure true biological differential expression',
        axisXLabel: 'Sample ID (1-4: Batch 1, 5-8: Batch 2)',
        axisYLabel: 'Principal Component 1 Score',
        xRange: [0.5, 8.5] as [number, number],
        yRange: [-6, 12] as [number, number],
      }
    } else {
      // FDR vs Bonferroni multi-testing
      // Target FDR q from 1% to 10%
      const q = param1 * 1.0 // 1% to 10%
      const totalGenes = param2 * 100 // 1,000 to 10,000 genes

      // Rank index k from 1 to 100
      const ranks = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
      const bhThresholds: number[] = []
      const bonferroniThresholds: number[] = []

      const alphaFamily = 0.05
      const bonf = -Math.log10(alphaFamily / totalGenes)

      for (const k of ranks) {
        // BH p-value cutoff: p_k <= (k / G) * q
        const bhPVal = (k / totalGenes) * (q / 100)
        bhThresholds.push(Math.min(8, -Math.log10(Math.max(1e-10, bhPVal))))
        bonferroniThresholds.push(bonf)
      }

      const activeDiscoveriesBH = Math.round(45 * (q / 5))
      const activeDiscoveriesBonf = 8

      return {
        x: ranks,
        y1: bhThresholds,
        y2: bonferroniThresholds,
        stat1: `G = ${totalGenes.toLocaleString()} genes, q = ${q}%`,
        stat2: `BH discoveries: ~${activeDiscoveriesBH}`,
        stat3: `Bonferroni discoveries: ${activeDiscoveriesBonf}`,
        insight: 'Benjamini-Hochberg controls FDR at q with drastically higher discovery power',
        axisXLabel: 'Candidate Gene Rank k',
        axisYLabel: '-log10(p-value threshold)',
        xRange: [0, 105] as [number, number],
        yRange: [1, 9] as [number, number],
      }
    }
  }, [mode, param1, param2])

  const axisX = useAxis({ label: results.axisXLabel, range: results.xRange })
  const axisY = useAxis({ label: results.axisYLabel, range: results.yRange })

  return (
    <Figure
      title="Omics Tables: Count Dispersion, Batch Confounding, and Multiple Hypothesis Testing"
      state={state}
      hoverReadout={false}
      readouts={
        <>
          <Readout label="Analysis Regime" value={results.stat1} />
          <Readout label="Raw Scale / Confounder" value={results.stat2} />
          <Readout label="Corrected Metric" value={results.stat3} />
          <Readout label="Statistical Insight" value={results.insight} />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="High-Dimensional Genomic Table Properties and Statistical Corrections">
        <Curve
          name={
            mode === 'mean_variance'
              ? 'Negative Binomial Raw Variance log10(Var)'
              : mode === 'batch_correction'
                ? 'Uncorrected Sample PC1 (Batch Dominated)'
                : 'Benjamini-Hochberg FDR Cutoff -log10(p)'
          }
          x={results.x}
          y={results.y1}
          slot={0}
          width={2.5}
        />
        <Curve
          name={
            mode === 'mean_variance'
              ? 'Variance Stabilizing Transform (VST) Target'
              : mode === 'batch_correction'
                ? 'ComBat Corrected Sample PC1 (Biology Aligned)'
                : 'Bonferroni Strict FWER Cutoff -log10(p)'
          }
          x={results.x}
          y={results.y2}
          slot={1}
          dashed
          width={2.0}
        />
        <Points name="Evaluated Genes / Samples" x={results.x} y={results.y1} slot={0} size={7} />
      </Plot>
    </Figure>
  )
}
