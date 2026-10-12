import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function SpatialFeaturesVisualizer() {
  const state = useFigureState({
    mode: choice<'spatial_autocorrelation' | 'buffered_cv' | 'distance_decay'>(
      [
        {
          value: 'spatial_autocorrelation',
          label: 'Spatial Autocorrelation: Variogram Semivariance gamma(h) & Moran I (Moran 1950)',
        },
        {
          value: 'buffered_cv',
          label: 'Spatial Cross-Validation: Buffer Exclusion Radius vs Generalization R² (Tobler 1970)',
        },
        {
          value: 'distance_decay',
          label: 'Neighbourhood Weighting: Distance Decay Kernels W_ij vs Separation d',
        },
      ],
      'spatial_autocorrelation',
      { label: 'spatial feature analysis' },
    ),
    param1: slider(10, 80, 40, {
      step: 5,
      label: 'variogram spatial range a (km) / buffer radius r (km) / decay parameter',
    }),
    param2: slider(1, 5, 3, {
      step: 1,
      label: 'Moran I autocorrelation scale / nugget effect / kernel power alpha',
    }),
  })

  const { mode, param1, param2 } = state

  const results = useMemo(() => {
    if (mode === 'spatial_autocorrelation') {
      const rangeA = param1 // 10 to 80 km
      const nugget = param2 * 5 // 5 to 25% nugget
      const sill = 100 // total sill

      // Distance lags h from 0 to 100 km
      const lags = [0, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]
      const semivariances: number[] = []
      const autocorrelations: number[] = []

      for (const h of lags) {
        let gamma = nugget
        if (h <= rangeA) {
          // Spherical variogram model: gamma(h) = c0 + (c - c0) * (1.5 * (h/a) - 0.5 * (h/a)^3)
          gamma += (sill - nugget) * (1.5 * (h / rangeA) - 0.5 * Math.pow(h / rangeA, 3))
        } else {
          gamma = sill
        }
        semivariances.push(gamma)
        // Autocorrelation: rho(h) = 1 - gamma(h) / sill
        autocorrelations.push(Math.max(0, (1 - gamma / sill) * 100))
      }

      const activeMoranI = Math.max(0.05, 0.95 - (nugget / 100) * 0.8)

      return {
        x: lags,
        y1: semivariances,
        y2: autocorrelations,
        stat1: `Spatial range a = ${rangeA} km`,
        stat2: `Nugget = ${nugget}%, Sill = ${sill}`,
        stat3: `Moran's I = +${activeMoranI.toFixed(2)}`,
        insight: 'Observations closer than spatial range a violate independent identical distribution (i.i.d.)',
        axisXLabel: 'Spatial Separation Lag h (km)',
        axisYLabel: 'Semivariance γ(h) & Spatial Correlation ρ(h) (%)',
        xRange: [-2, 105] as [number, number],
        yRange: [0, 105] as [number, number],
      }
    } else if (mode === 'buffered_cv') {
      const bufferR = param1 // 10 to 80 km
      const variogramRange = 35 // km

      // Buffer exclusion distance from 0 to 60 km
      const bufferDistances = [0, 5, 10, 15, 20, 25, 30, 35, 40, 50, 60]
      const randomCV: number[] = []
      const spatialBufferedCV: number[] = []

      for (const b of bufferDistances) {
        // Random CV remains deceptively optimistic ~ 88% R2
        randomCV.push(88.0)
        // Buffered CV drops as spatial leakage is eliminated, stabilizing once b >= variogramRange
        const leakageFactor = Math.max(0, 1 - b / variogramRange)
        const honestR2 = 58.0 + 30.0 * Math.pow(leakageFactor, 1.8)
        spatialBufferedCV.push(honestR2)
      }

      const activeLeakage = Math.max(0, 1 - bufferR / variogramRange)
      const activeHonest = 58.0 + 30.0 * Math.pow(activeLeakage, 1.8)

      return {
        x: bufferDistances,
        y1: randomCV,
        y2: spatialBufferedCV,
        stat1: `Buffer r_buffer = ${bufferR} km`,
        stat2: 'Random k-fold: R² = 88.0% (leaked)',
        stat3: `Buffered CV: R² = ${activeHonest.toFixed(1)}%`,
        insight: `Spatial leakage inflates apparent score by +${(88.0 - activeHonest).toFixed(1)}% R²`,
        axisXLabel: 'Spatial Exclusion Buffer Radius (km)',
        axisYLabel: 'Cross-Validation Performance R² (%)',
        xRange: [-2, 65] as [number, number],
        yRange: [45, 100] as [number, number],
      }
    } else {
      // Distance decay comparison: inverse distance vs negative exponential
      const dists = [1, 2, 5, 10, 15, 20, 25, 30, 40, 50]
      const inversePower: number[] = []
      const negExponential: number[] = []

      const alpha = 1.0 + param2 * 0.3 // 1.3 to 2.5
      const gammaScale = 0.08

      for (const d of dists) {
        inversePower.push(100 / Math.pow(d, alpha * 0.8))
        negExponential.push(100 * Math.exp(-gammaScale * d))
      }

      return {
        x: dists,
        y1: inversePower,
        y2: negExponential,
        stat1: `Power law α = ${alpha.toFixed(1)}`,
        stat2: `Exponential γ = ${gammaScale}`,
        stat3: 'Tobler First Law: near things related',
        insight: 'Inverse power decay emphasizes hyper-local neighbors; exponential decays smoothly',
        axisXLabel: 'Euclidean / Network Distance d (km)',
        axisYLabel: 'Spatial Weight W_ij (% of peak)',
        xRange: [0, 55] as [number, number],
        yRange: [0, 105] as [number, number],
      }
    }
  }, [mode, param1, param2])

  const axisX = useAxis({ label: results.axisXLabel, range: results.xRange })
  const axisY = useAxis({ label: results.axisYLabel, range: results.yRange })

  return (
    <Figure
      title="Spatial Features: Autocorrelation, Semivariance, and Buffered Cross-Validation"
      state={state}
      hoverReadout={false}
      readouts={
        <>
          <Readout label="Spatial Modeling Param" value={results.stat1} />
          <Readout label="Baseline / Variogram" value={results.stat2} />
          <Readout label="Validation Fidelity" value={results.stat3} />
          <Readout label="Geographic Insight" value={results.insight} />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Spatial Dependence Structure and Cross-Validation Diagnostics">
        <Curve
          name={
            mode === 'spatial_autocorrelation'
              ? 'Spherical Semivariance γ(h) (%)'
              : mode === 'buffered_cv'
                ? 'Naive Random k-fold CV (Autocorrelation Leaked R² %)'
                : 'Inverse Distance Power Decay W_ij ~ d^(-α)'
          }
          x={results.x}
          y={results.y1}
          slot={0}
          width={2.5}
        />
        <Curve
          name={
            mode === 'spatial_autocorrelation'
              ? 'Spatial Autocorrelation Function ρ(h) (%)'
              : mode === 'buffered_cv'
                ? 'Spatial Buffered Block CV (Honest Generalization R² %)'
                : 'Negative Exponential Distance Decay W_ij ~ exp(-γd)'
          }
          x={results.x}
          y={results.y2}
          slot={1}
          dashed
          width={2.0}
        />
        <Points name="Sampling Lag Distances" x={results.x} y={results.y1} slot={0} size={7} />
      </Plot>
    </Figure>
  )
}
