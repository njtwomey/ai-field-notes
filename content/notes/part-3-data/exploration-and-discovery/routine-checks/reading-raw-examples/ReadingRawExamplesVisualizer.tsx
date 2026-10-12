import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, useAxis, useFigureState } from 'aifn-render'

export function ReadingRawExamplesVisualizer() {
  const state = useFigureState({
    distributionPattern: choice<'bimodal_empty_center' | 'sentinel_pollution'>(
      [
        { value: 'bimodal_empty_center', label: 'Bimodal Mixture (Mean = 50, but zero mass at center)' },
        { value: 'sentinel_pollution', label: 'Sentinel Code Pollution (-999 values hidden by aggregations)' },
      ],
      'bimodal_empty_center',
      { label: 'data distribution pathology' },
    ),
    inspectionLens: choice<'naive_head' | 'random_sample' | 'extrema_inspection'>(
      [
        { value: 'naive_head', label: 'df.head(5) (First 5 rows sorted by ID: Looks perfectly normal)' },
        { value: 'random_sample', label: 'Random Sampling (Draws from high-density modes, misses bugs)' },
        { value: 'extrema_inspection', label: 'Extrema Inspection (Min/Max slices: Instantly exposes flaws)' },
      ],
      'extrema_inspection',
      { label: 'inspection strategy' },
    ),
  })

  const results = useMemo(() => {
    const xGrid: number[] = []
    const densityCurve: number[] = []

    // 100 points grid in [0 .. 100]
    for (let x = 0; x <= 100; x += 1) {
      xGrid.push(x)
      let d = 0
      if (state.distributionPattern === 'bimodal_empty_center') {
        // Two peaks at 20 and 80, zero mass at 50
        const d1 = Math.exp(-0.5 * Math.pow((x - 25) / 8, 2))
        const d2 = Math.exp(-0.5 * Math.pow((x - 75) / 8, 2))
        d = (d1 + d2) * 20.0
      } else {
        // Normal peak at 52, but sentinel cliff at 0
        const d1 = Math.exp(-0.5 * Math.pow((x - 52) / 12, 2))
        const dSentinel = x <= 3 ? 1.5 : 0
        d = (d1 + dSentinel) * 25.0
      }
      densityCurve.push(d)
    }

    // Sampled points depending on inspection lens
    let sampledX: number[] = []
    let sampledY: number[] = []
    let flawDetected = ''

    if (state.distributionPattern === 'bimodal_empty_center') {
      if (state.inspectionLens === 'naive_head') {
        sampledX = [23, 25, 27, 24, 26]
        sampledY = sampledX.map(() => 15.0)
        flawDetected = 'Missed: Looks like a single cluster around 25'
      } else if (state.inspectionLens === 'random_sample') {
        sampledX = [22, 26, 74, 78, 24, 76]
        sampledY = sampledX.map(() => 15.0)
        flawDetected = 'Partial: Sees two clusters, but missed empty gap rationale'
      } else {
        // Extrema inspection pulls min, median, max
        sampledX = [5, 25, 50, 75, 95]
        sampledY = [2.0, 19.5, 0.1, 19.5, 2.0]
        flawDetected = 'Exposed: Mean=50 sits in a complete vacuum of zero data!'
      }
    } else {
      // Sentinel pollution
      if (state.inspectionLens === 'naive_head') {
        sampledX = [48, 52, 55, 50, 53]
        sampledY = sampledX.map(() => 20.0)
        flawDetected = 'Missed: Clean values near 50'
      } else if (state.inspectionLens === 'random_sample') {
        sampledX = [45, 50, 54, 58, 62]
        sampledY = sampledX.map(() => 20.0)
        flawDetected = 'Missed: 98% of data is clean, random sample missed sentinels'
      } else {
        sampledX = [0, 40, 52, 70, 99]
        sampledY = [35.0, 15.0, 25.0, 10.0, 1.0]
        flawDetected = 'Exposed: Sentinel value -999 (coded as 0) spotted at minimum boundary!'
      }
    }

    return {
      xGrid,
      densityCurve,
      sampledX,
      sampledY,
      flawDetected,
    }
  }, [state.distributionPattern, state.inspectionLens])

  const axisX = useAxis({
    label: 'Feature Measurement Value X (Aggregate Mean = 50.0)',
    range: [0, 100],
  })

  const axisY = useAxis({
    label: 'Density / Sample Frequency',
    range: [0, 42],
  })

  return (
    <Figure
      title="What Aggregates Conceal: Diagnosing Pathologies by Reading Raw Examples"
      state={state}
      hoverReadout={false}
      caption="Summary statistics (mean, variance) conceal critical pathologies. An aggregate mean of 50.0 can represent two polarized clusters with an empty center, or a clean distribution corrupted by boundary sentinels. Inspecting raw examples at the extrema instantly unmasks the truth."
      readouts={
        <>
          <Readout label="Aggregate Mean" value="50.0 (Unchanged across pathologies)" />
          <Readout label="Aggregate Std" value="15.0 (Unchanged across pathologies)" />
          <Readout label="Diagnostic Result" value={results.flawDetected} />
          <Readout
            label="Methodology"
            value={state.inspectionLens === 'extrema_inspection' ? 'Extrema sampling active' : 'Blind to boundary bugs'}
          />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Underlying Data Distribution vs Inspected Instances">
        <Curve name="True Feature Density" x={results.xGrid} y={results.densityCurve} slot={0} width={2.5} />
        <Points name="Inspected Raw Examples" x={results.sampledX} y={results.sampledY} slot={1} size={8} />
      </Plot>
    </Figure>
  )
}
