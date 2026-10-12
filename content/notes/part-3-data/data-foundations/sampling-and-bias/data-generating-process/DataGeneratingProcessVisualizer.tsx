import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function DataGeneratingProcessVisualizer() {
  const state = useFigureState({
    viewMode: choice<'collider_sample' | 'full_population'>(
      [
        { value: 'collider_sample', label: 'Selected Cohort (Collider conditioning: X1 + X2 >= tau)' },
        { value: 'full_population', label: 'Full Population Data-Generating Process' },
      ],
      'collider_sample',
      { label: 'sampling lens' },
    ),
    threshold: slider(-0.5, 1.8, 0.8, {
      step: 0.1,
      label: 'selection hurdle threshold tau (admission / recording cutoff)',
    }),
  })

  const results = useMemo(() => {
    // Generate 60 deterministic synthetic candidate points (X1, X2)
    // with zero true correlation in the unselected population
    const allX1: number[] = []
    const allX2: number[] = []

    const seed = 42
    for (let i = 0; i < 60; i += 1) {
      // Deterministic quasi-normal spread
      const u1 = (Math.sin(i * 12.9898 + seed) * 43758.5453) % 1
      const u2 = (Math.cos(i * 78.233 + seed) * 43758.5453) % 1
      const x1 = Math.sin(i * 0.45) * 1.5 + (u1 - 0.5) * 0.8
      const x2 = Math.cos(i * 0.45) * 1.5 + (u2 - 0.5) * 0.8
      allX1.push(x1)
      allX2.push(x2)
    }

    const tau = state.threshold
    const selectedX1: number[] = []
    const selectedX2: number[] = []
    const rejectedX1: number[] = []
    const rejectedX2: number[] = []

    allX1.forEach((x1, idx) => {
      const x2 = allX2[idx]
      if (x1 + x2 >= tau) {
        selectedX1.push(x1)
        selectedX2.push(x2)
      } else {
        rejectedX1.push(x1)
        rejectedX2.push(x2)
      }
    })

    const nSelected = selectedX1.length
    const acceptanceRate = (nSelected / allX1.length) * 100.0

    // Compute sample correlation on selected subset
    let meanX1 = 0
    let meanX2 = 0
    selectedX1.forEach((x, idx) => {
      meanX1 += x
      meanX2 += selectedX2[idx]
    })
    meanX1 = nSelected > 0 ? meanX1 / nSelected : 0
    meanX2 = nSelected > 0 ? meanX2 / nSelected : 0

    let cov = 0
    let varX1 = 0
    let varX2 = 0
    selectedX1.forEach((x, idx) => {
      const dx1 = x - meanX1
      const dx2 = selectedX2[idx] - meanX2
      cov += dx1 * dx2
      varX1 += dx1 * dx1
      varX2 += dx2 * dx2
    })

    const corrSelected = varX1 > 0 && varX2 > 0 ? cov / Math.sqrt(varX1 * varX2) : 0.0

    // Linear fit line for selected points
    const slope = varX1 > 0 ? cov / varX1 : 0
    const intercept = meanX2 - slope * meanX1

    const xGrid = [-2.0, -1.0, 0.0, 1.0, 2.0]
    const fittedY = xGrid.map((x) => intercept + slope * x)
    const popZeroY = xGrid.map(() => 0.0)

    return {
      allX1,
      allX2,
      selectedX1,
      selectedX2,
      rejectedX1,
      rejectedX2,
      acceptanceRate,
      corrSelected,
      xGrid,
      fittedY,
      popZeroY,
    }
  }, [state.threshold])

  const axisX = useAxis({
    label: 'Attribute 1 (X1) [e.g., Coding Competence]',
    range: [-2.5, 2.5],
  })

  const axisY = useAxis({
    label: 'Attribute 2 (X2) [e.g., Social Charm]',
    range: [-2.5, 2.5],
  })

  return (
    <Figure
      title="Data-Generating Process and Collider Selection Bias (Berkson's Fallacy)"
      state={state}
      hoverReadout={false}
      caption="Attributes X1 and X2 are completely independent in the true data-generating process (r ~ 0.0). However, when conditioning on admission or dataset inclusion (X1 + X2 >= tau), an artificial negative correlation emerges in the observed data: high X1 compensates for low X2, creating a spurious trade-off."
      readouts={
        <>
          <Readout label="Population DGP Correlation" value="rho = 0.00 (Independent)" />
          <Readout
            label="Observed Sample Correlation"
            value={`r = ${results.corrSelected.toFixed(2)} (Spurious Negative)`}
          />
          <Readout label="Inclusion / Selection Rate" value={`${results.acceptanceRate.toFixed(1)}%`} />
          <Readout label="Bias Mechanism" value="Berkson's Collider Bias: S = 1 opens spurious path" />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="True Independence vs Conditioning on a Collider">
        {state.viewMode === 'full_population' ? (
          <>
            <Points
              name="Unselected Population Units"
              x={results.rejectedX1}
              y={results.rejectedX2}
              slot={1}
              size={6}
            />
            <Points name="Selected Units (S=1)" x={results.selectedX1} y={results.selectedX2} slot={0} size={7} />
            <Curve
              name="Population Flat Trend (r=0)"
              x={results.xGrid}
              y={results.popZeroY}
              slot={0}
              dashed
              width={1.8}
            />
          </>
        ) : (
          <>
            <Points
              name="Selected Sample Observations"
              x={results.selectedX1}
              y={results.selectedX2}
              slot={0}
              size={8}
            />
            <Curve name="Spurious Sample Fit Line" x={results.xGrid} y={results.fittedY} slot={2} width={2.5} />
          </>
        )}
      </Plot>
    </Figure>
  )
}
