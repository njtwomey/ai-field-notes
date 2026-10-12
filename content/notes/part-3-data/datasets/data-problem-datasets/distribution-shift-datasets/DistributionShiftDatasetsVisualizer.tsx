import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function DistributionShiftDatasetsVisualizer() {
  const state = useFigureState({
    mode: choice<'id_vs_ood_gap' | 'imagenet_c_severity' | 'waterbirds_spurious'>(
      [
        {
          value: 'id_vs_ood_gap',
          label: 'In-Domain vs OOD Performance Drop Across Benchmarks (WILDS Koh 2021)',
        },
        {
          value: 'imagenet_c_severity',
          label: 'ImageNet-C: Corruption Severity Level 1-5 vs Mean Error mCE (Hendrycks 2019)',
        },
        {
          value: 'waterbirds_spurious',
          label: 'Waterbirds: Spurious Correlation Ratio vs Worst-Group Accuracy (Sagawa 2020)',
        },
      ],
      'id_vs_ood_gap',
      { label: 'distribution shift benchmark' },
    ),
    param1: slider(1, 5, 3, {
      step: 1,
      label: 'corruption severity / benchmark index / spurious correlation strength (%)',
    }),
    param2: slider(10, 90, 50, {
      step: 10,
      label: 'regularization strength / subpopulation minority ratio / model capacity',
    }),
  })

  const { mode, param1, param2 } = state

  const results = useMemo(() => {
    if (mode === 'id_vs_ood_gap') {
      // 5 representative benchmarks:
      // 1: ImageNetV2, 2: ImageNet-C, 3: WILDS Camelyon17, 4: WILDS iWildCam, 5: Waterbirds
      const benchmarks = [1, 2, 3, 4, 5]
      const idAccuracies = [82.5, 82.5, 98.2, 85.0, 97.5]
      const oodAccuracies = [70.2, 54.0, 70.3, 62.4, 66.8]

      return {
        x: benchmarks,
        y1: idAccuracies,
        y2: oodAccuracies,
        stat1: '5 Canonical Shift Benchmarks',
        stat2: 'In-Domain Avg: 89.1%',
        stat3: 'OOD / Shift Avg: 64.7%',
        insight: 'Real-world shifts induce a ~24.4% absolute accuracy degradation across benchmarks',
        axisXLabel: 'Benchmark (1=ImageNetV2, 2=IN-C, 3=Camelyon17, 4=iWildCam, 5=Waterbirds)',
        axisYLabel: 'Model Accuracy (%)',
        xRange: [0.5, 5.5] as [number, number],
        yRange: [40, 105] as [number, number],
      }
    } else if (mode === 'imagenet_c_severity') {
      const severityLevels = [1, 2, 3, 4, 5]
      const standardResNetError: number[] = []
      const robustAugmixError: number[] = []

      for (const s of severityLevels) {
        // Mean Corruption Error escalates with severity level 1-5
        const stdErr = 25.0 + 13.5 * s
        const robErr = 18.0 + 8.8 * s
        standardResNetError.push(stdErr)
        robustAugmixError.push(robErr)
      }

      const activeSeverity = param1
      const activeStd = 25.0 + 13.5 * activeSeverity
      const activeRob = 18.0 + 8.8 * activeSeverity

      return {
        x: severityLevels,
        y1: standardResNetError,
        y2: robustAugmixError,
        stat1: `Corruption Severity Level ${activeSeverity}`,
        stat2: `Standard ResNet50 mCE: ${activeStd.toFixed(1)}%`,
        stat3: `AugMix Robust mCE: ${activeRob.toFixed(1)}%`,
        insight: 'ImageNet-C evaluates 15 corruption types (Gaussian noise, blur, frost, fog, jpeg) at 5 severities',
        axisXLabel: 'Corruption Severity Level (1 to 5)',
        axisYLabel: 'Mean Corruption Error mCE (%)',
        xRange: [0.8, 5.2] as [number, number],
        yRange: [20, 100] as [number, number],
      }
    } else {
      // Waterbirds spurious correlation: correlation ratio from 50% (unbiased) to 99% (extreme spurious)
      const ratios = [50, 60, 70, 80, 85, 90, 95, 99]
      const averageAcc: number[] = []
      const worstGroupAcc: number[] = []

      for (const r of ratios) {
        // Average accuracy stays high because 95% of data follows the spurious correlation
        const avg = 98.0 - 0.05 * (r - 50)
        // Worst-group accuracy (waterbirds on land or landbirds on water) collapses severely
        const worst = Math.max(30, 95.0 - 1.4 * (r - 50))
        averageAcc.push(avg)
        worstGroupAcc.push(worst)
      }

      const activeRatio = 50 + (param1 - 1) * 12
      const activeWorst = Math.max(30, 95.0 - 1.4 * (activeRatio - 50))

      return {
        x: ratios,
        y1: averageAcc,
        y2: worstGroupAcc,
        stat1: `Spurious Ratio: ${activeRatio}% background alignment`,
        stat2: 'Average Acc: ~96% (misleading)',
        stat3: `Worst-Group Acc: ${activeWorst.toFixed(1)}%`,
        insight: 'Standard ERM learns the background shortcut; Group DRO optimizes worst-case subgroup risk',
        axisXLabel: 'Spurious Correlation Proportion in Training Set (%)',
        axisYLabel: 'Test Accuracy (%)',
        xRange: [45, 102] as [number, number],
        yRange: [25, 105] as [number, number],
      }
    }
  }, [mode, param1, param2])

  const axisX = useAxis({ label: results.axisXLabel, range: results.xRange })
  const axisY = useAxis({ label: results.axisYLabel, range: results.yRange })

  return (
    <Figure
      title="Distribution Shift Datasets: Robustness, Corruption Error, and Subpopulation Gaps"
      state={state}
      hoverReadout={false}
      readouts={
        <>
          <Readout label="Shift Regime" value={results.stat1} />
          <Readout label="Baseline / Average" value={results.stat2} />
          <Readout label="Shift / Worst-Group" value={results.stat3} />
          <Readout label="Robustness Insight" value={results.insight} />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Generalization Under Distribution Shift and Adversarial Corruptions">
        <Curve
          name={
            mode === 'id_vs_ood_gap'
              ? 'In-Domain Test Accuracy (%)'
              : mode === 'imagenet_c_severity'
                ? 'Standard Model Corruption Error mCE (%)'
                : 'Overall Average Test Accuracy (%)'
          }
          x={results.x}
          y={results.y1}
          slot={0}
          width={2.5}
        />
        <Curve
          name={
            mode === 'id_vs_ood_gap'
              ? 'Out-of-Distribution Shift Accuracy (%)'
              : mode === 'imagenet_c_severity'
                ? 'Robust Data-Augmented Model Error mCE (%)'
                : 'Worst-Group Minority Accuracy (%)'
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
