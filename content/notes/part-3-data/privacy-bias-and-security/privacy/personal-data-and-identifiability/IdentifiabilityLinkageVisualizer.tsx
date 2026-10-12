import { useMemo } from 'react'
import { choice, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function IdentifiabilityLinkageVisualizer() {
  const state = useFigureState({
    quasiIdentifiers: choice<'gender_only' | 'gender_zip' | 'gender_zip_dob'>(
      [
        { value: 'gender_zip_dob', label: '3 Quasi-identifiers: {Gender, 5-digit ZIP, Date of Birth}' },
        { value: 'gender_zip', label: '2 Quasi-identifiers: {Gender, 5-digit ZIP}' },
        { value: 'gender_only', label: '1 Quasi-identifier: {Gender only}' },
      ],
      'gender_zip_dob',
      { label: 'quasi-identifier set' },
    ),
    cohortSize: slider(200, 1000, 500, { step: 100, label: 'population cohort size' }),
  })

  const results = useMemo(() => {
    const N = state.cohortSize
    let uniqueCount = 0
    let minGroupSize = 1
    let avgGroupSize = 1

    // Simulation of population uniqueness based on Sweeney's demographic distribution
    if (state.quasiIdentifiers === 'gender_only') {
      uniqueCount = 0
      minGroupSize = Math.floor(N * 0.45)
      avgGroupSize = Math.floor(N / 2)
    } else if (state.quasiIdentifiers === 'gender_zip') {
      // 2 attributes: ~15% unique records in municipal area
      uniqueCount = Math.round(N * 0.16)
      minGroupSize = 1
      avgGroupSize = 8
    } else {
      // 3 attributes: {Gender, ZIP, DOB} -> Sweeney's 87% uniqueness
      uniqueCount = Math.round(N * 0.87)
      minGroupSize = 1
      avgGroupSize = 1.15
    }

    const uniquePct = (uniqueCount / N) * 100
    const nonUniqueCount = N - uniqueCount

    // Points for 2D distribution view (grid of cohort records: red if unique/re-identifiable, blue if safe)
    const pointsX: number[] = []
    const pointsY: number[] = []
    const isUnique: boolean[] = []

    const COLS = 25
    for (let i = 0; i < N; i++) {
      const x = i % COLS
      const y = Math.floor(i / COLS)
      pointsX.push(x)
      pointsY.push(y)
      isUnique.push(i < uniqueCount)
    }

    const uniqueX: number[] = []
    const uniqueY: number[] = []
    const safeX: number[] = []
    const safeY: number[] = []

    for (let i = 0; i < N; i++) {
      if (isUnique[i]) {
        uniqueX.push(pointsX[i])
        uniqueY.push(pointsY[i])
      } else {
        safeX.push(pointsX[i])
        safeY.push(pointsY[i])
      }
    }

    return {
      uniquePct,
      uniqueCount,
      nonUniqueCount,
      minGroupSize,
      avgGroupSize,
      uniqueX,
      uniqueY,
      safeX,
      safeY,
    }
  }, [state.quasiIdentifiers, state.cohortSize])

  const axisX = useAxis({
    label: 'cohort population sample grid (horizontal index)',
    range: [-1, 25],
  })
  const axisY = useAxis({
    label: 'cohort population sample grid (vertical index)',
    range: [-1, 42],
  })

  return (
    <Figure
      title="Personal data and identifiability: quasi-identifiers and re-identification via linkage attacks"
      state={state}
      hoverReadout={false}
      caption="Evaluating re-identification risk in pseudo-anonymized datasets. While direct identifiers (names, SSNs) are removed, combinations of innocuous quasi-identifiers create unique combinations. In a standard population, combining Gender, 5-digit ZIP, and Date of Birth uniquely identifies 87% of individuals (red points), enabling re-identification by joining hospital records with public voter registries (Sweeney 1997)."
      readouts={
        <>
          <Readout
            label="uniquely re-identifiable"
            value={`${results.uniquePct.toFixed(1)}% (${results.uniqueCount} records)`}
          />
          <Readout label="minimum equivalence group size" value={`${results.minGroupSize} individuals`} />
          <Readout label="average group size" value={`${results.avgGroupSize.toFixed(1)} individuals`} />
          <Readout
            label="linkage attack vulnerability"
            value={
              state.quasiIdentifiers === 'gender_zip_dob'
                ? 'Critical Risk (87% Re-identification)'
                : state.quasiIdentifiers === 'gender_zip'
                  ? 'Moderate Risk (16% Re-identification)'
                  : 'Low Risk (0% Re-identification)'
            }
          />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Population cohort: uniquely identifiable (red) vs protected (blue)">
        {results.safeX.length > 0 && (
          <Points
            name="shared equivalence group (protected individuals)"
            x={results.safeX}
            y={results.safeY}
            slot={0}
            size={4}
          />
        )}
        {results.uniqueX.length > 0 && (
          <Points
            name="uniquely identifiable records (vulnerable to linkage)"
            x={results.uniqueX}
            y={results.uniqueY}
            slot={1}
            size={4}
          />
        )}
      </Plot>
    </Figure>
  )
}
