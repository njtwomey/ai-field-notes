import { useMemo } from 'react'
import { Figure, int, Plot, Points, Readout, Segments, slider, useAxis, useFigureState } from 'aifn-render'
import { normal, stream } from 'aifn-compute/foundation/random'

export function MultipleImputationExplorer() {
  const state = useFigureState({
    M: int(5, { min: 2, max: 15, label: 'imputations M' }),
    missingRate: slider(0.1, 0.6, 0.35, { step: 0.05, label: 'missingness fraction' }),
    seed: int(3, { min: 1, max: 20, label: 'random seed' }),
  })

  const results = useMemo(() => {
    const rng = stream(state.seed)
    const M = state.M
    const p = state.missingRate

    const TRUE_Q = 2.0
    // Within-imputation sampling variance scales with sample size and missing rate
    // With N = 200, base variance sigma^2 / N ~ 0.04
    const baseU = 0.04 * (1 + 0.5 * p)
    // Between-imputation variance B scales with missing fraction p
    const betweenStd = Math.sqrt(0.06 * p)

    const qEstimates: number[] = []
    const uEstimates: number[] = []

    for (let m = 0; m < M; m++) {
      // Each imputation samples from posterior distribution of missing data
      const q_m = TRUE_Q + normal(rng, 0, betweenStd) + normal(rng, 0, Math.sqrt(baseU) * 0.4)
      const u_m = Math.max(0.01, baseU + normal(rng, 0, 0.005))
      qEstimates.push(q_m)
      uEstimates.push(u_m)
    }

    // Rubin's rules
    const qBar = qEstimates.reduce((a, b) => a + b, 0) / M
    const uBar = uEstimates.reduce((a, b) => a + b, 0) / M

    let ssq = 0
    for (let m = 0; m < M; m++) {
      const diff = qEstimates[m] - qBar
      ssq += diff * diff
    }
    const B = M > 1 ? ssq / (M - 1) : 0
    const totalT = uBar + (1 + 1 / M) * B
    const seTotal = Math.sqrt(totalT)
    const seWithin = Math.sqrt(uBar)
    const seBetween = Math.sqrt((1 + 1 / M) * B)

    // Fraction of missing information (FMI)
    const fmi = totalT > 1e-6 ? ((1 + 1 / M) * B) / totalT : 0
    const relativeIncreaseVar = uBar > 1e-6 ? ((1 + 1 / M) * B) / uBar : 0

    // Coordinates for individual imputations m = 1..M
    const impX: number[] = []
    const impY: number[] = []
    const impSegments: { from: readonly [number, number]; to: readonly [number, number] }[] = []

    for (let m = 0; m < M; m++) {
      const x = m + 1
      const y = qEstimates[m]
      const halfCi = 1.96 * Math.sqrt(uEstimates[m])
      impX.push(x)
      impY.push(y)
      impSegments.push({
        from: [x, y - halfCi],
        to: [x, y + halfCi],
      })
    }

    // Pooled estimate plotted at x = M + 1.5
    const pooledX = [M + 1.5]
    const pooledY = [qBar]
    const pooledSegments: { from: readonly [number, number]; to: readonly [number, number] }[] = [
      {
        from: [M + 1.5, qBar - 1.96 * seTotal],
        to: [M + 1.5, qBar + 1.96 * seTotal],
      },
    ]

    return {
      qBar,
      uBar,
      B,
      totalT,
      seTotal,
      seWithin,
      seBetween,
      fmi,
      relativeIncreaseVar,
      impX,
      impY,
      impSegments,
      pooledX,
      pooledY,
      pooledSegments,
      maxX: M + 2.5,
    }
  }, [state.M, state.missingRate, state.seed])

  const xAxis = useAxis({
    label: 'imputation dataset m (1..M) and pooled estimate',
    range: [0.5, results.maxX],
  })
  const yAxis = useAxis({ label: 'parameter estimate Q̂ (with 95% CI)', range: [1.0, 3.0] })

  return (
    <Figure
      title="Rubin's pooling rules and total variance decomposition"
      state={state}
      hoverReadout={false}
      caption="Across M completed datasets, substantive estimates Q̂^(m) fluctuate around the pooled mean. Each individual imputation reports an overly optimistic confidence interval based strictly on within-imputation variance Ū. Rubin's rules expand the pooled confidence interval by adding between-imputation variance (1 + 1/M)B, faithfully propagating missing-data uncertainty into the final inference."
      readouts={
        <>
          <Readout label="pooled estimate Q̄" value={results.qBar.toFixed(3)} />
          <Readout label="total SE √T" value={results.seTotal.toFixed(3)} />
          <Readout label="within SE √Ū" value={results.seWithin.toFixed(3)} />
          <Readout label="between SE √((1+1/M)B)" value={results.seBetween.toFixed(3)} />
          <Readout label="missing info (FMI)" value={`${(results.fmi * 100).toFixed(1)}%`} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} title="Individual imputation estimates vs Rubin pooled inference">
        <Segments name="individual within-imputation 95% CIs" segments={results.impSegments} slot={1} width={2} />
        <Points name="imputation point estimates Q̂^(m)" x={results.impX} y={results.impY} slot={1} size={5} />
        <Segments
          name="Rubin pooled 95% CI (incorporating between-imputation variance B)"
          segments={results.pooledSegments}
          slot={0}
          width={3.5}
        />
        <Points name="pooled point estimate Q̄" x={results.pooledX} y={results.pooledY} slot={0} size={7} />
      </Plot>
    </Figure>
  )
}
