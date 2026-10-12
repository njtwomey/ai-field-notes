import { useMemo } from 'react'
import { Curve, Figure, Handle, int, Plot, Readout, slider, useAxis, useFigureState } from 'aifn-render'

function normalPdf(x: number, mean: number, sd: number): number {
  const z = (x - mean) / sd
  return (1 / (sd * Math.sqrt(2 * Math.PI))) * Math.exp(-0.5 * z * z)
}

export function BlockingAndMatching() {
  const state = useFigureState({
    windowSize: int(5, { min: 1, max: 20, step: 1, label: 'blocking window w' }),
    fieldCount: int(4, { min: 2, max: 8, step: 1, label: 'comparison fields' }),
    tLower: slider(-10, 5, -2, { step: 0.5, label: 'lower threshold T_λ' }),
    tUpper: slider(0, 15, 6, { step: 0.5, label: 'upper threshold T_μ' }),
  })

  const { windowSize, fieldCount, tLower, tUpper } = state

  // Total dataset size N = 1000 records -> all pairs = N(N-1)/2 = 499,500
  const totalRecords = 1000
  const allPairs = (totalRecords * (totalRecords - 1)) / 2

  // Sorted neighborhood blocking: candidate pairs approx N * w
  const candidatePairs = Math.min(allPairs, totalRecords * windowSize)
  const reductionRatio = 1 - candidatePairs / allPairs
  // Pair completeness rises with window size
  const pairCompleteness = 1 - Math.exp(-0.4 * windowSize)

  // Fellegi-Sunter log-likelihood ratio distributions for Matches (M) and Non-matches (U)
  // Non-matches centre around -2 * fields, Matches centre around +3 * fields
  const meanU = -1.8 * fieldCount
  const sdU = Math.sqrt(fieldCount) * 1.2
  const meanM = 2.4 * fieldCount
  const sdM = Math.sqrt(fieldCount) * 1.5

  const densityCurves = useMemo(() => {
    const xs: number[] = []
    const densityU: number[] = []
    const densityM: number[] = []
    for (let x = -20; x <= 25; x += 0.25) {
      xs.push(x)
      densityU.push(normalPdf(x, meanU, sdU))
      densityM.push(normalPdf(x, meanM, sdM))
    }
    return { xs, densityU, densityM }
  }, [meanU, sdU, meanM, sdM])

  const lowerBound = Math.min(tLower, tUpper - 0.5)
  const upperBound = Math.max(tUpper, lowerBound + 0.5)

  const xAxis = useAxis({ label: 'Fellegi-Sunter match weight R', range: [-18, 22] })
  const yAxis = useAxis({ label: 'density', range: [0, 0.25] })

  return (
    <Figure
      title="Candidate blocking and the Fellegi-Sunter decision rule"
      state={state}
      hoverReadout={false}
      caption="Topological blocking reduces the N(N−1)/2 search space to candidate pairs within a window w of sorted keys. Candidate pairs are scored by their Fellegi-Sunter log-weight R = ∑ w_k. Two decision thresholds split pairs into non-matches (R ≤ T_λ), clerical review (T_λ < R < T_μ), and definite matches (R ≥ T_μ). Drag the threshold handles to balance automated classification against clerical review burden."
      readouts={
        <>
          <Readout label="pair reduction" value={`${(reductionRatio * 100).toFixed(1)}%`} />
          <Readout label="pair completeness" value={`${(pairCompleteness * 100).toFixed(1)}%`} />
          <Readout label="lower threshold T_λ" value={lowerBound.toFixed(1)} />
          <Readout label="upper threshold T_μ" value={upperBound.toFixed(1)} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} title="distributions of match weight for non-matches and matches">
        <Curve name="non-matches (U)" x={densityCurves.xs} y={densityCurves.densityU} slot={0} />
        <Curve name="matches (M)" x={densityCurves.xs} y={densityCurves.densityM} slot={1} />
        <Curve name="T_λ boundary" x={[lowerBound, lowerBound]} y={[0, 0.25]} slot={0} dashed />
        <Curve name="T_μ boundary" x={[upperBound, upperBound]} y={[0, 0.25]} slot={1} dashed />
        <Handle
          kind="x"
          at={lowerBound}
          label="T_λ"
          onDrag={(x) => state.set('tLower', Math.min(x, upperBound - 0.5))}
        />
        <Handle
          kind="x"
          at={upperBound}
          label="T_μ"
          onDrag={(x) => state.set('tUpper', Math.max(x, lowerBound + 0.5))}
        />
      </Plot>
    </Figure>
  )
}
