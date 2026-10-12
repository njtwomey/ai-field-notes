import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function SurveyOfficialStatsVisualizer() {
  const state = useFigureState({
    estimationMethod: choice<'unweighted' | 'horvitz_thompson'>(
      [
        { value: 'unweighted', label: 'Naive Unweighted Sample Mean (IID Assumption)' },
        { value: 'horvitz_thompson', label: 'Horvitz-Thompson Survey-Weighted Estimator' },
      ],
      'unweighted',
      { label: 'estimator type' },
    ),
    sampleRuralShare: slider(0.1, 0.6, 0.45, {
      step: 0.05,
      label: 'sample share of oversampled stratum (true pop share = 20%)',
    }),
  })

  const results = useMemo(() => {
    // True population proportions:
    // Stratum A (Urban / Majority): 80% of pop, Mean income = 65,000 USD
    // Stratum B (Rural / Minority): 20% of pop, Mean income = 40,000 USD
    const popShareA = 0.8
    const popShareB = 0.2
    const meanA = 65.0
    const meanB = 40.0

    const truePopMean = popShareA * meanA + popShareB * meanB // 60.0k

    // Sample composition:
    const sB = state.sampleRuralShare
    const sA = 1.0 - sB

    // Inclusion probabilities / sampling weights:
    // wA ~ popShareA / sA, wB ~ popShareB / sB
    const wA = popShareA / sA
    const wB = popShareB / sB

    // Unweighted sample mean:
    const unweightedMean = sA * meanA + sB * meanB

    // Weighted Horvitz-Thompson mean:
    const htMean = (sA * wA * meanA + sB * wB * meanB) / (sA * wA + sB * wB)

    const estimatedMean = state.estimationMethod === 'unweighted' ? unweightedMean : htMean
    const biasError = Math.abs(estimatedMean - truePopMean)

    // Sweep across possible sample shares [0.1 .. 0.6] to plot estimated mean vs rural sample share
    const shareGrid: number[] = []
    const unweightedCurve: number[] = []
    const htCurve: number[] = []

    for (let s = 0.1; s <= 0.6; s += 0.02) {
      shareGrid.push(s * 100.0)
      const curSa = 1.0 - s
      unweightedCurve.push(curSa * meanA + s * meanB)
      htCurve.push(truePopMean) // HT is consistently unbiased across sample designs
    }

    return {
      truePopMean,
      estimatedMean,
      biasError,
      wA,
      wB,
      shareGrid,
      unweightedCurve,
      htCurve,
      currentSharePct: state.sampleRuralShare * 100.0,
    }
  }, [state.estimationMethod, state.sampleRuralShare])

  const axisX = useAxis({
    label: 'Oversampled Stratum Share in Sample (%) [True Pop = 20%]',
    range: [8, 62],
  })

  const axisY = useAxis({
    label: 'Estimated Metric Mean (k USD)',
    range: [45, 68],
  })

  return (
    <Figure
      title="Complex Survey Design Weights and Horvitz-Thompson Calibration"
      state={state}
      hoverReadout={false}
      caption="Official statistical agencies deliberately oversample small strata to ensure subgroup precision. Treating survey microdata as unweighted IID observations produces massive bias, whereas Horvitz-Thompson inverse-probability weighting recovers the true population mean."
      readouts={
        <>
          <Readout label="True Population Benchmark" value={`${results.truePopMean.toFixed(1)}k USD`} />
          <Readout
            label="Estimated Sample Mean"
            value={`${results.estimatedMean.toFixed(1)}k USD (${results.biasError > 0.1 ? 'Biased' : 'Unbiased'})`}
          />
          <Readout label="Estimation Bias Delta" value={`${results.biasError.toFixed(2)}k USD`} />
          <Readout label="Design Weight Ratio (w_U / w_R)" value={`${(results.wA / results.wB).toFixed(2)}x`} />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Estimator Sensitivity to Disproportionate Stratification">
        <Curve
          name="Horvitz-Thompson Design-Weighted Mean"
          x={results.shareGrid}
          y={results.htCurve}
          slot={0}
          width={2.5}
        />
        <Curve
          name="Naive Unweighted Mean"
          x={results.shareGrid}
          y={results.unweightedCurve}
          slot={2}
          dashed
          width={2}
        />
        <Points
          name="Current Sample Selection"
          x={[results.currentSharePct]}
          y={[results.estimatedMean]}
          slot={state.estimationMethod === 'unweighted' ? 1 : 0}
          size={8}
        />
      </Plot>
    </Figure>
  )
}
