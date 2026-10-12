import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function InteractionLogsVisualizer() {
  const state = useFigureState({
    viewMode: choice<'examination_curve' | 'observed_ctr'>(
      [
        { value: 'examination_curve', label: 'Examination Propensity P(E=1 | rank k)' },
        { value: 'observed_ctr', label: 'Observed CTR P(C=1) vs True Relevance' },
      ],
      'observed_ctr',
      { label: 'feedback view' },
    ),
    positionDecay: slider(0.0, 1.5, 0.8, {
      step: 0.1,
      label: 'position bias exponent (eta: P(E) = 1 / k^eta)',
    }),
  })

  const results = useMemo(() => {
    // 5 items in fixed display rank 1..5 with true intrinsic relevance
    const ranks = [1, 2, 3, 4, 5]
    const trueRelevance = [0.25, 0.4, 0.85, 0.3, 0.7] // Item 3 is most relevant (0.85), followed by Item 5 (0.70)

    const eta = state.positionDecay

    const propensities = ranks.map((k) => 1.0 / Math.pow(k, eta))
    const observedCTR = ranks.map((_, idx) => propensities[idx] * trueRelevance[idx])
    const ipsRelevance = observedCTR.map((ctr, idx) => ctr / propensities[idx])

    // For plot:
    // If examination_curve, plot p(k) vs k
    // If observed_ctr, plot observed CTR vs rank k
    const xPositions: number[] = []
    const yValues: number[] = []

    if (state.viewMode === 'examination_curve') {
      for (let k = 1; k <= 5; k += 0.2) {
        xPositions.push(k)
        yValues.push(1.0 / Math.pow(k, eta))
      }
    } else {
      ranks.forEach((k, idx) => {
        xPositions.push(k)
        yValues.push(observedCTR[idx])
      })
    }

    // Identify which rank got the most clicks
    let maxCtr = -1
    let mostClickedRank = 1
    observedCTR.forEach((ctr, idx) => {
      if (ctr > maxCtr) {
        maxCtr = ctr
        mostClickedRank = idx + 1
      }
    })

    const rank1CTR = observedCTR[0]
    const bestItemRank = 3 // index 2, true relevance 0.85
    const bestItemCTR = observedCTR[2]

    const rankingInversion = mostClickedRank !== bestItemRank

    return {
      xPositions,
      yValues,
      ranks,
      observedCTR,
      trueRelevance,
      propensities,
      ipsRelevance,
      rank1CTR,
      bestItemCTR,
      mostClickedRank,
      rankingInversion,
    }
  }, [state.positionDecay, state.viewMode])

  const axisX = useAxis({
    label: 'Display Position / Rank (k)',
    range: [0.8, 5.2],
  })

  const axisY = useAxis({
    label: state.viewMode === 'examination_curve' ? 'Examination Propensity' : 'Observed Click Rate',
    range: [0, 1.05],
  })

  return (
    <Figure
      title="Position Bias and Inverse Propensity Weighting in Interaction Telemetry"
      state={state}
      hoverReadout={false}
      caption="Item 3 is the most relevant (r=0.85), but sits at rank 3. As position bias increases, rank 1 receives disproportionate clicks despite mediocre true quality (r=0.25). Inverse propensity scoring (IPS) weights each click by 1/P(Examined | rank) to recover true item quality."
      readouts={
        <>
          <Readout
            label="Most Clicked Item"
            value={`Rank ${results.mostClickedRank} (${results.rankingInversion ? 'Biased Inversion!' : 'True Best Item'})`}
          />
          <Readout label="Rank 1 CTR (True r=0.25)" value={`${(results.rank1CTR * 100).toFixed(1)}%`} />
          <Readout label="Rank 3 CTR (True r=0.85)" value={`${(results.bestItemCTR * 100).toFixed(1)}%`} />
          <Readout
            label="IPS Debiasing"
            value={results.rankingInversion ? 'IPS restores correct rank order' : 'True order preserved'}
          />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Examination Decay & Click Attenuation">
        {state.viewMode === 'examination_curve' ? (
          <Curve name="Examination P(E=1 | k)" x={results.xPositions} y={results.yValues} slot={0} width={2.5} />
        ) : (
          <>
            <Points name="Observed CTR P(C=1)" x={results.ranks} y={results.observedCTR} slot={1} size={8} />
            <Points name="True Intrinsic Relevance" x={results.ranks} y={results.trueRelevance} slot={0} size={6} />
          </>
        )}
      </Plot>
    </Figure>
  )
}
