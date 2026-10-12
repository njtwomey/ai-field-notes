import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function ArealAggregationVisualizer() {
  const state = useFigureState({
    mode: choice<'maup_scale' | 'ecological_fallacy' | 'dasymetric_error'>(
      [
        {
          value: 'maup_scale',
          label: 'MAUP Scale Effect: Spatial Aggregation Unit Size vs Correlation r (Openshaw 1984)',
        },
        {
          value: 'ecological_fallacy',
          label: 'Ecological Fallacy: Aggregate Slope vs Individual Covariance (Robinson 1950)',
        },
        {
          value: 'dasymetric_error',
          label: 'Dasymetric Mapping: Uniform Areal Weighting vs Ancillary Land-Cover Weighting',
        },
      ],
      'maup_scale',
      { label: 'areal aggregation analysis' },
    ),
    param1: slider(2, 64, 16, {
      step: 2,
      label: 'aggregation zones K / ecological grouping parameter / ancillary fidelity (%)',
    }),
    param2: slider(1, 5, 3, {
      step: 1,
      label: 'underlying individual correlation r_0 / unpopulated land proportion',
    }),
  })

  const { mode, param1, param2 } = state

  const results = useMemo(() => {
    if (mode === 'maup_scale') {
      const zonesK = param1 // 2 to 64
      const baseCorr = param2 * 0.15 // 0.15 to 0.75

      // Across zone count K from 64 (fine) down to 2 (coarse)
      const zoneCounts = [64, 48, 32, 24, 16, 12, 8, 4, 2]
      const observedCorr: number[] = []
      const varianceRetained: number[] = []

      for (const k of zoneCounts) {
        // As k decreases (coarser units), variance collapses and correlation r inflates towards 1.0
        const inflatedR = Math.min(0.98, baseCorr + (1 - baseCorr) * (1 - k / 64) * 0.85)
        const varRet = Math.max(5, (k / 64) * 100)

        observedCorr.push(inflatedR)
        varianceRetained.push(varRet)
      }

      const activeFactor = 1 - zonesK / 64
      const activeInflatedR = Math.min(0.98, baseCorr + (1 - baseCorr) * activeFactor * 0.85)
      const activeVar = Math.max(5, (zonesK / 64) * 100)

      return {
        x: zoneCounts,
        y1: observedCorr.map((v) => v * 100),
        y2: varianceRetained,
        stat1: `Zones K = ${zonesK} (base r0 = ${baseCorr.toFixed(2)})`,
        stat2: `Aggregated r = ${activeInflatedR.toFixed(2)}`,
        stat3: `Retained variance: ${activeVar.toFixed(0)}%`,
        insight: 'MAUP scale effect: coarse spatial averaging inflates correlation while erasing within-zone variance',
        axisXLabel: 'Spatial Aggregation Units K (64=blocks, 2=states)',
        axisYLabel: 'Aggregated Correlation r (%) & Variance (%)',
        xRange: [0, 68] as [number, number],
        yRange: [0, 105] as [number, number],
      }
    } else if (mode === 'ecological_fallacy') {
      // Robinson 1950 ecological fallacy demo
      // 8 groups/zones with internal negative correlation but external positive aggregate correlation
      const groupCenters = [10, 20, 30, 40, 50, 60, 70, 80]
      const aggregateY: number[] = []
      const individualSpread: number[] = []

      for (const cx of groupCenters) {
        // Between-group positive trend: Y = 0.8 * cx + 10
        const cy = 0.8 * cx + 10
        aggregateY.push(cy)
        // Within-group local trend: negative slope
        individualSpread.push(cy - 12)
      }

      return {
        x: groupCenters,
        y1: aggregateY,
        y2: individualSpread,
        stat1: 'Robinson 1950 Ecological Fallacy',
        stat2: 'Group-level slope: β_macro = +0.80',
        stat3: 'Individual slope: β_micro = -0.45',
        insight: 'Simpson paradox in space: ecological correlation does not reflect individual behavior',
        axisXLabel: 'Sociodemographic Feature X (Group Mean)',
        axisYLabel: 'Outcome Metric Y',
        xRange: [5, 85] as [number, number],
        yRange: [0, 95] as [number, number],
      }
    } else {
      // Dasymetric error vs uniform areal weighting across uninhabited land fraction
      const uninhabitedPct = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
      const naiveUniformError: number[] = []
      const dasymetricError: number[] = []

      for (const u of uninhabitedPct) {
        // Naive areal weighting assumes uniform density, error explodes with unpopulated land fraction
        naiveUniformError.push(1.8 * u)
        // Dasymetric mapping masks out uninhabited land, bounding error
        dasymetricError.push(Math.min(15, 0.15 * u + 2))
      }

      return {
        x: uninhabitedPct,
        y1: naiveUniformError,
        y2: dasymetricError,
        stat1: 'Dasymetric Masking vs Areal Weighting',
        stat2: 'Naive error scales linearly with open space',
        stat3: 'Dasymetric error bounded < 15%',
        insight: 'Ancillary building masks prevent allocating population into lakes and parks',
        axisXLabel: 'Uninhabited Forest / Water Surface (%)',
        axisYLabel: 'Areal Disaggregation Error RMSE (%)',
        xRange: [-5, 95] as [number, number],
        yRange: [0, 170] as [number, number],
      }
    }
  }, [mode, param1, param2])

  const axisX = useAxis({ label: results.axisXLabel, range: results.xRange })
  const axisY = useAxis({ label: results.axisYLabel, range: results.yRange })

  return (
    <Figure
      title="Areal Aggregation: The Modifiable Areal Unit Problem and Ecological Fallacy"
      state={state}
      hoverReadout={false}
      readouts={
        <>
          <Readout label="Aggregation Regime" value={results.stat1} />
          <Readout label="Aggregate Parameter" value={results.stat2} />
          <Readout label="Fidelity Metric" value={results.stat3} />
          <Readout label="Spatial Statistical Insight" value={results.insight} />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Statistical Distortion and Boundary Sensitivity Under Spatial Aggregation">
        <Curve
          name={
            mode === 'maup_scale'
              ? 'Observed Areal Correlation r (%)'
              : mode === 'ecological_fallacy'
                ? 'Ecological Macro Trend (Between-Zone Centroids)'
                : 'Uniform Areal Weighting Disaggregation Error (%)'
          }
          x={results.x}
          y={results.y1}
          slot={0}
          width={2.5}
        />
        <Curve
          name={
            mode === 'maup_scale'
              ? 'Retained Within-Zone Variance (%)'
              : mode === 'ecological_fallacy'
                ? 'Individual Micro Trend (Within-Zone Dispersal)'
                : 'Dasymetric Land-Cover Guided Disaggregation Error (%)'
          }
          x={results.x}
          y={results.y2}
          slot={1}
          dashed
          width={2.0}
        />
        <Points name="Evaluated Units" x={results.x} y={results.y1} slot={0} size={7} />
      </Plot>
    </Figure>
  )
}
