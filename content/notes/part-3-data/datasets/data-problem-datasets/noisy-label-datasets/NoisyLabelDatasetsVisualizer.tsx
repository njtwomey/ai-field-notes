import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function NoisyLabelDatasetsVisualizer() {
  const state = useFigureState({
    benchmark: choice<'synthetic_vs_real' | 'noise_rate_sweep' | 'confident_learning'>(
      [
        {
          value: 'noise_rate_sweep',
          label: 'Synthetic Noise Sweep: Standard ERM vs Forward Correction vs Co-Teaching',
        },
        {
          value: 'synthetic_vs_real',
          label: 'Real-World Noisy Benchmarks: CIFAR-10N, Clothing1M, WebVision',
        },
        {
          value: 'confident_learning',
          label: 'Confident Learning: Noise Filter Pruning Threshold vs Precision-Recall',
        },
      ],
      'noise_rate_sweep',
      { label: 'evaluation mode' },
    ),
    noiseLevel: slider(10, 80, 40, {
      step: 5,
      label: 'corruption rate / noise ratio η (%)',
    }),
    filterThreshold: slider(0.1, 0.9, 0.5, {
      step: 0.05,
      label: 'confident learning threshold / selection percentile',
    }),
  })

  const { benchmark, noiseLevel, filterThreshold } = state

  const results = useMemo(() => {
    if (benchmark === 'noise_rate_sweep') {
      const noiseGrid = [0, 10, 20, 30, 40, 50, 60, 70, 80]
      const standardLossAcc: number[] = []
      const forwardCorrectionAcc: number[] = []
      const coTeachingAcc: number[] = []

      for (const eta of noiseGrid) {
        // Under standard cross-entropy, memorization collapses accuracy severely as eta grows
        const stdAcc = Math.max(15, 94.0 - 0.95 * eta - 0.005 * eta * eta)
        // Forward correction uses transition matrix T to correct loss
        const fwdAcc = Math.max(25, 93.5 - 0.35 * eta - 0.003 * eta * eta)
        // Co-teaching exchanges small-loss samples between two networks
        const coAcc = Math.max(30, 93.8 - 0.28 * eta - 0.002 * eta * eta)

        standardLossAcc.push(stdAcc)
        forwardCorrectionAcc.push(fwdAcc)
        coTeachingAcc.push(coAcc)
      }

      const activeStd = Math.max(15, 94.0 - 0.95 * noiseLevel - 0.005 * noiseLevel * noiseLevel)
      const activeCo = Math.max(30, 93.8 - 0.28 * noiseLevel - 0.002 * noiseLevel * noiseLevel)
      const gap = activeCo - activeStd

      return {
        x: noiseGrid,
        y1: standardLossAcc,
        y2: coTeachingAcc,
        y3: forwardCorrectionAcc,
        stat1: `Noise Rate: η = ${noiseLevel}%`,
        stat2: `Standard ERM Acc: ${activeStd.toFixed(1)}%`,
        stat3: `Co-Teaching Acc: ${activeCo.toFixed(1)}% (gap: +${gap.toFixed(1)}%)`,
        insight: 'Standard deep networks memorize random labels; small-loss sample selection isolates clean data',
        axisXLabel: 'Label Noise Rate η (%)',
        axisYLabel: 'Clean Test Accuracy (%)',
        xRange: [-5, 85] as [number, number],
        yRange: [10, 100] as [number, number],
      }
    } else if (benchmark === 'synthetic_vs_real') {
      // 5 benchmarks: 1=CIFAR-10N (Random 1), 2=CIFAR-10N (Worst), 3=Clothing1M, 4=WebVision, 5=Animal-10N
      const indices = [1, 2, 3, 4, 5]
      const ermAccuracy = [85.2, 77.8, 68.9, 72.1, 79.5]
      const robustAccuracy = [91.4, 88.5, 74.3, 79.8, 87.2]

      return {
        x: indices,
        y1: ermAccuracy,
        y2: robustAccuracy,
        stat1: '5 Real-World Benchmarks',
        stat2: 'Standard ResNet Avg: 76.7%',
        stat3: 'Robust Noise-Resistant Avg: 84.2%',
        insight: 'Real noise is instance-dependent P(y|x) rather than uniform class-conditional T_ij',
        axisXLabel: 'Benchmark (1=CIFAR-10N Rand, 2=Worst, 3=Clothing1M, 4=WebVision, 5=Animal-10N)',
        axisYLabel: 'Test Accuracy on Clean Ground Truth (%)',
        xRange: [0.5, 5.5] as [number, number],
        yRange: [60, 98] as [number, number],
      }
    } else {
      // Confident learning threshold sweep
      const thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
      const precision: number[] = []
      const recall: number[] = []

      for (const t of thresholds) {
        // High threshold = confident errors detected with high precision, lower recall
        const p = Math.min(98, 55 + 45 * Math.sqrt(t))
        const r = Math.max(20, 95 - 75 * t * t)
        precision.push(p)
        recall.push(r)
      }

      const activeP = Math.min(98, 55 + 45 * Math.sqrt(filterThreshold))
      const activeR = Math.max(20, 95 - 75 * filterThreshold * filterThreshold)
      const f1 = (2 * activeP * activeR) / (activeP + activeR)

      return {
        x: thresholds,
        y1: precision,
        y2: recall,
        stat1: `Threshold: ${filterThreshold.toFixed(2)}`,
        stat2: `Error Detection Precision: ${activeP.toFixed(1)}%`,
        stat3: `Recall: ${activeR.toFixed(1)}% (F1: ${f1.toFixed(1)}%)`,
        insight: 'Confident Learning prunes samples where model confidence exceeds observed class threshold',
        axisXLabel: 'Confidence Pruning Threshold',
        axisYLabel: 'Noise Identification Rate (%)',
        xRange: [0.05, 0.95] as [number, number],
        yRange: [15, 105] as [number, number],
      }
    }
  }, [benchmark, noiseLevel, filterThreshold])

  const axisX = useAxis({ label: results.axisXLabel, range: results.xRange })
  const axisY = useAxis({ label: results.axisYLabel, range: results.yRange })

  return (
    <Figure
      title="Noisy-Label Datasets: Memorization Collapse, Transition Estimation, and Robust Training"
      state={state}
      hoverReadout={false}
      readouts={
        <>
          <Readout label="Regime" value={results.stat1} />
          <Readout label="Baseline / Precision" value={results.stat2} />
          <Readout label="Robust / Recall" value={results.stat3} />
          <Readout label="Methodology" value={results.insight} />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Learning Dynamics and Robustness Across Label Noise Regimes">
        <Curve
          name={
            benchmark === 'confident_learning'
              ? 'Error Detection Precision (%)'
              : 'Standard ERM / Baseline Test Accuracy (%)'
          }
          x={results.x}
          y={results.y1}
          slot={0}
          width={2.5}
        />
        <Curve
          name={
            benchmark === 'confident_learning'
              ? 'Error Detection Recall (%)'
              : benchmark === 'noise_rate_sweep'
                ? 'Co-Teaching Test Accuracy (%)'
                : 'Robust Algorithmic Accuracy (%)'
          }
          x={results.x}
          y={results.y2}
          slot={1}
          dashed
          width={2.2}
        />
        {benchmark === 'noise_rate_sweep' && results.y3 && (
          <Curve name="Forward Loss Correction Accuracy (%)" x={results.x} y={results.y3} slot={2} width={1.8} />
        )}
        <Points name="Evaluated Configurations" x={results.x} y={results.y1} slot={0} size={7} />
      </Plot>
    </Figure>
  )
}
