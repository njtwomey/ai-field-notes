import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function TargetPopulationVisualizer() {
  const state = useFigureState({
    weightingMode: choice<'unweighted' | 'importance_weighted'>(
      [
        { value: 'unweighted', label: 'Naive Internal Validation (Uniform Sample Weighting)' },
        {
          value: 'importance_weighted',
          label: 'Importance-Weighted Transport (Sugiyama et al. 2007: w(x) = P_t(x)/P_s(x))',
        },
      ],
      'unweighted',
      { label: 'evaluation regime' },
    ),
    targetMeanShift: slider(30, 65, 52, {
      step: 1,
      label: 'target population mean age (study sample mean = 32)',
    }),
  })

  const results = useMemo(() => {
    const xGrid: number[] = []
    const sourceDensity: number[] = []
    const targetDensity: number[] = []
    const errorProfile: number[] = []

    const muSource = 32.0
    const sigSource = 8.0

    const muTarget = state.targetMeanShift
    const sigTarget = 10.0

    // Gaussian density helper
    const normalPdf = (x: number, mu: number, sig: number) => {
      const z = (x - mu) / sig
      return (1.0 / (sig * Math.sqrt(2 * Math.PI))) * Math.exp(-0.5 * z * z)
    }

    let internalErrorAcc = 0
    let targetErrorAcc = 0
    let weightedErrorAcc = 0
    let totalSourceW = 0
    let totalTargetW = 0
    let totalWeightedW = 0

    for (let x = 15; x <= 80; x += 1) {
      xGrid.push(x)

      const ps = normalPdf(x, muSource, sigSource)
      const pt = normalPdf(x, muTarget, sigTarget)

      // Model error rate escalates outside training support: low error in 20-40, high in 50-80
      const err = 0.03 + 0.35 / (1.0 + Math.exp(-0.15 * (x - 48)))

      sourceDensity.push(ps * 100.0) // scaled for visualizer plot
      targetDensity.push(pt * 100.0)
      errorProfile.push(err * 100.0)

      internalErrorAcc += err * ps
      totalSourceW += ps

      targetErrorAcc += err * pt
      totalTargetW += pt

      // Importance weight: w = pt / max(ps, 1e-6)
      const w = Math.min(15.0, pt / Math.max(ps, 1e-4))
      weightedErrorAcc += err * ps * w
      totalWeightedW += ps * w
    }

    const internalErrorPct = (internalErrorAcc / totalSourceW) * 100.0
    const trueTargetErrorPct = (targetErrorAcc / totalTargetW) * 100.0
    const estimatedErrorPct =
      state.weightingMode === 'importance_weighted' ? (weightedErrorAcc / totalWeightedW) * 100.0 : internalErrorPct

    const gap = Math.abs(trueTargetErrorPct - estimatedErrorPct)

    return {
      xGrid,
      sourceDensity,
      targetDensity,
      errorProfile,
      internalErrorPct,
      trueTargetErrorPct,
      estimatedErrorPct,
      gap,
    }
  }, [state.targetMeanShift, state.weightingMode])

  const axisX = useAxis({
    label: 'Demographic Feature x (Age Bracket)',
    range: [15, 80],
  })

  const axisY = useAxis({
    label: 'Density / Error Rate (%)',
    range: [0, 42],
  })

  return (
    <Figure
      title="Target Population Shift and the Transportability Gap"
      state={state}
      hoverReadout={false}
      caption="A model trained on an accessible study cohort (blue, centered at age 32) achieves 5.2% internal validation error. When deployed on the older target population (orange), true error jumps to 22.8%. Naive evaluation blinds engineers to external validity failure; importance weighting correctly anticipates the true deployment loss."
      readouts={
        <>
          <Readout label="Internal Validation Loss" value={`${results.internalErrorPct.toFixed(1)}%`} />
          <Readout label="True Deployed Target Loss" value={`${results.trueTargetErrorPct.toFixed(1)}%`} />
          <Readout
            label="Estimated Evaluation"
            value={`${results.estimatedErrorPct.toFixed(1)}% (${results.gap < 1.5 ? 'Calibrated' : 'Severe Blindspot'})`}
          />
          <Readout
            label="Transport Gap"
            value={`+${(results.trueTargetErrorPct - results.internalErrorPct).toFixed(1)}% real-world degradation`}
          />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Study Sample Distribution vs Deployed Target Support">
        <Curve name="Study Sample P_source(x)" x={results.xGrid} y={results.sourceDensity} slot={0} width={2.5} />
        <Curve name="Target Population P_target(x)" x={results.xGrid} y={results.targetDensity} slot={2} width={2.5} />
        <Curve
          name="Model Pointwise Error Rate (%)"
          x={results.xGrid}
          y={results.errorProfile}
          slot={1}
          dashed
          width={1.8}
        />
      </Plot>
    </Figure>
  )
}
