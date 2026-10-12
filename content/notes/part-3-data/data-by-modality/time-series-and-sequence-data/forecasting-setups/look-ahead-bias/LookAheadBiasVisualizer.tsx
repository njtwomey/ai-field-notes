import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function LookAheadBiasVisualizer() {
  const state = useFigureState({
    visualizerMode: choice<'centred_window' | 'vintage_revisions' | 'survivorship_bias'>(
      [
        { value: 'centred_window', label: 'Filter Leakage: Causal Trailing vs Centred Bilateral Window' },
        {
          value: 'vintage_revisions',
          label: 'Data Vintages: Preliminary Real-Time vs Revised Restatements (Croushore 2001)',
        },
        {
          value: 'survivorship_bias',
          label: 'Survivorship Bias: Point-in-Time Universe vs Current Survivors (López de Prado 2018)',
        },
      ],
      'centred_window',
      { label: 'leakage mechanism' },
    ),
    windowSize: slider(3, 21, 9, {
      step: 2,
      label: 'smoothing filter window W',
    }),
    horizonSteps: slider(1, 12, 4, {
      step: 1,
      label: 'forecasting step horizon H',
    }),
  })

  const { visualizerMode, windowSize, horizonSteps } = state

  const results = useMemo(() => {
    if (visualizerMode === 'centred_window') {
      // Timeline around decision boundary t = 10
      const steps = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
      // Ground truth signal with a sharp structural shift at t = 10 (drops from 70 to 30)
      // Causal trailing filter: uses strictly past points [t - W, t]. Only starts dropping AFTER t = 10!
      const causalFilter = steps.map((t) => {
        if (t <= 10) return 70
        const elapsed = t - 10
        return Math.max(30, 70 - elapsed * (40 / Math.max(2, windowSize)))
      })
      // Centred bilateral filter: uses future points [t - W/2, t + W/2]. Starts dropping BEFORE t = 10 (peeks into future drop!)
      const halfW = Math.floor(windowSize / 2)
      const centredFilter = steps.map((t) => {
        if (t <= 10 - halfW) return 70
        if (t >= 10 + halfW) return 30
        const progress = (t - (10 - halfW)) / (2 * halfW)
        return 70 - progress * 40
      })

      const leakBefore10 = halfW * 2
      const isLeaking = halfW >= 1

      return {
        x: steps,
        y: centredFilter,
        baseline: causalFilter,
        stat1: `Filter window W = ${windowSize} steps`,
        stat2: `Future lookahead: +${halfW} steps into future`,
        stat3: isLeaking ? `Leaking ${leakBefore10} steps ahead` : 'Minimal leak',

        warning:
          'Centred rolling windows incorporate future post-decision observations, predicting regime crashes before they occur',
        axisXLabel: 'Decision Timeline (t=10 represents forecast cutoff)',
        axisYLabel: 'Filtered Covariate Feature Value',
        xRange: [-0.5, 20.5] as [number, number],
        yRange: [20, 80] as [number, number],
      }
    } else if (visualizerMode === 'vintage_revisions') {
      // Macroeconomic quarters / reporting vintages: t = 1 to 12
      const quarters = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
      // Real-time preliminary release (noisy, first published vintage)
      const preliminaryVintage = quarters.map((q) => 2.0 + Math.sin(q * 0.8) * 1.5 + (q % 2 === 0 ? 0.6 : -0.6))
      // Final revised series (re-benchmarked years later with complete census data)
      const revisedVintage = quarters.map((q) => 2.0 + Math.sin(q * 0.8) * 1.5)

      const currRevisionGap = Math.abs(preliminaryVintage[horizonSteps - 1] - revisedVintage[horizonSteps - 1])

      return {
        x: quarters,
        y: revisedVintage,
        baseline: preliminaryVintage,
        stat1: `Evaluation horizon: Q${horizonSteps}`,
        stat2: `Revision gap: ${currRevisionGap.toFixed(2)}%`,
        stat3: 'Vintage mismatch',
        warning:
          'Backtesting on revised data creates phantom predictive power: the revised numbers were not known at forecast time',
        axisXLabel: 'Reporting Period Vintage (Quarters)',
        axisYLabel: 'Indicator Value (% Growth)',
        xRange: [0.5, 12.5] as [number, number],
        yRange: [0, 5] as [number, number],
      }
    } else {
      // Survivorship bias: Universe active entity count over 10 years
      const years = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
      // Point-in-time true universe: 500 initial firms, natural default/attrition rate (~5% per year)
      const trueUniverse = years.map((yr) => Math.round(500 * Math.pow(0.95, yr - 1)))
      // Survivor-only universe: filtered retrospectively to only include firms active in Year 10 (always constant surviving count)
      const survivingFinal = Math.round(500 * Math.pow(0.95, 9))
      const survivorUniverse = years.map(() => survivingFinal)

      const currentTrue = trueUniverse[horizonSteps - 1]
      const deltaFailed = currentTrue - survivingFinal

      return {
        x: years,
        y: survivorUniverse,
        baseline: trueUniverse,
        stat1: `Year ${horizonSteps} point-in-time entities: ${currentTrue}`,
        stat2: `Final survivors: ${survivingFinal}`,
        stat3: `Ignored failed units: ${deltaFailed}`,
        warning:
          'Conditioning universe membership on present-day survival discards all bankrupt entities, inflating historical returns',
        axisXLabel: 'Historical Observation Horizon (Years)',
        axisYLabel: 'Sampled Universe Entity Count',
        xRange: [0.5, 10.5] as [number, number],
        yRange: [250, 520] as [number, number],
      }
    }
  }, [visualizerMode, windowSize, horizonSteps])

  const axisX = useAxis({
    label: results.axisXLabel,
    range: results.xRange,
  })

  const axisY = useAxis({
    label: results.axisYLabel,
    range: results.yRange,
  })

  return (
    <Figure
      title="Look-Ahead Bias: Centred Window Leakage, Data Vintages, and Survivorship Distortion"
      state={state}
      hoverReadout={false}
      caption="Anatomy of look-ahead leakage in time series: Future peeking in bilateral centred filters, macro revision discrepancy between preliminary vintages and restatements, and retrospective survivorship bias."
      readouts={
        <>
          <Readout label="Temporal Setup" value={results.stat1} />
          <Readout label="Discrepancy Scale" value={results.stat2} />
          <Readout
            label={
              visualizerMode === 'centred_window'
                ? 'Integrity Flag'
                : visualizerMode === 'vintage_revisions'
                  ? 'Status'
                  : 'Defunct Units'
            }
            value={results.stat3}
            color="var(--red-600)"
          />
          <Readout label="Methodological Directive" value={results.warning} color="var(--amber-600)" />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Temporal Causality Evaluation Profile">
        <Curve
          name={
            visualizerMode === 'centred_window'
              ? 'Strict Causal Trailing Filter (Point-in-Time Legal)'
              : visualizerMode === 'vintage_revisions'
                ? 'Real-Time Preliminary Release (What Was Actually Known)'
                : 'True Point-in-Time Universe (Includes Failed Units)'
          }
          x={results.x}
          y={results.baseline}
          slot={2}
          dashed
          width={1.8}
        />
        <Curve
          name={
            visualizerMode === 'centred_window'
              ? 'Centred Bilateral Filter (Peeks into Future Drop)'
              : visualizerMode === 'vintage_revisions'
                ? 'Revised Restated Data (Subsequent Post-Hoc Revisions)'
                : 'Retrospective Survivor-Only Universe (Severely Biased)'
          }
          x={results.x}
          y={results.y}
          slot={0}
          width={2.5}
        />
        <Points name="Measurement Coordinates" x={results.x} y={results.y} slot={0} size={7} />
      </Plot>
    </Figure>
  )
}
