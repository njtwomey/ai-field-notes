import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function CrsVisualizer() {
  const state = useFigureState({
    mode: choice<'mercator_distortion' | 'longitude_compression' | 'utm_scale'>(
      [
        {
          value: 'mercator_distortion',
          label: 'Web Mercator (EPSG:3857): Scale Factor k = sec(phi) & Area Inflation (Snyder 1987)',
        },
        {
          value: 'longitude_compression',
          label: 'Geodesic vs Planar: Longitude Metric Length vs Latitude phi (Vincenty 1975)',
        },
        {
          value: 'utm_scale',
          label: 'UTM Projection: Scale Factor Across 6° Zone (Central Meridian k0 = 0.9996)',
        },
      ],
      'mercator_distortion',
      { label: 'crs projection analysis' },
    ),
    param1: slider(0, 80, 50, {
      step: 5,
      label: 'latitude phi (degrees) / target location / zone longitude offset',
    }),
    param2: slider(1, 10, 5, {
      step: 1,
      label: 'longitude separation delta lambda (degrees) / UTM zone width / scale factor',
    }),
  })

  const { mode, param1, param2 } = state

  const results = useMemo(() => {
    if (mode === 'mercator_distortion') {
      const lat = param1 // 0 to 80 deg
      const lats = [0, 10, 20, 30, 40, 50, 60, 70, 75, 80]
      const linearScale: number[] = []
      const areaInflation: number[] = []

      for (const phi of lats) {
        const rad = (phi * Math.PI) / 180
        const sec = 1 / Math.cos(rad)
        linearScale.push(sec)
        areaInflation.push(Math.min(35, sec * sec))
      }

      const activeRad = (lat * Math.PI) / 180
      const activeLinear = 1 / Math.cos(activeRad)
      const activeArea = activeLinear * activeLinear

      return {
        x: lats,
        y1: linearScale,
        y2: areaInflation,
        stat1: `Latitude φ = ${lat}°`,
        stat2: `Linear scale k = ${activeLinear.toFixed(2)}x`,
        stat3: `Area inflation k² = ${activeArea.toFixed(2)}x`,
        insight: `At ${lat}°, areas appear ${(activeArea * 100 - 100).toFixed(0)}% larger than reality`,
        axisXLabel: 'Latitude φ (degrees)',
        axisYLabel: 'Scale Factor (linear k & area k²)',
        xRange: [-2, 85] as [number, number],
        yRange: [0, 36] as [number, number],
      }
    } else if (mode === 'longitude_compression') {
      const dLon = param2 // 1 to 10 deg
      const lats = [0, 10, 20, 30, 40, 50, 60, 70, 80]
      const trueDistances: number[] = []
      const naiveDistances: number[] = []

      // Earth radius R ~ 6371 km
      // 1 deg latitude ~ 111.13 km
      // 1 deg longitude at latitude phi = 111.32 * cos(phi) km
      for (const phi of lats) {
        const rad = (phi * Math.PI) / 180
        const kmPerDegLon = 111.32 * Math.cos(rad)
        trueDistances.push(kmPerDegLon * dLon)
        // Naive Euclidean assumes 1 deg = 111.32 km regardless of latitude
        naiveDistances.push(111.32 * dLon)
      }

      const activeLat = param1
      const activeRad = (activeLat * Math.PI) / 180
      const activeTrue = 111.32 * Math.cos(activeRad) * dLon
      const activeNaive = 111.32 * dLon
      const errorPct = ((activeNaive - activeTrue) / activeTrue) * 100

      return {
        x: lats,
        y1: naiveDistances,
        y2: trueDistances,
        stat1: `Δλ = ${dLon}° at φ = ${activeLat}°`,
        stat2: `True distance: ${activeTrue.toFixed(0)} km`,
        stat3: `Naive Euclidean: ${activeNaive.toFixed(0)} km`,
        insight: `Euclidean metric distance error: +${errorPct.toFixed(1)}%`,
        axisXLabel: 'Latitude φ (degrees)',
        axisYLabel: `Distance for Δλ = ${dLon}° (km)`,
        xRange: [-2, 85] as [number, number],
        yRange: [0, 1200] as [number, number],
      }
    } else {
      // UTM 6-degree zone scale factor from central meridian (dLon = 0) to zone boundary (dLon = ±3 deg)
      const offsets = [-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
      const scaleFactors: number[] = []
      const nominalScales: number[] = []

      for (const offset of offsets) {
        // UTM scale factor: k = k0 * (1 + (x / R)^2 / 2), where k0 = 0.9996
        const xDist = offset * 111.32 * Math.cos((40 * Math.PI) / 180) // at ~40 deg lat
        const k = 0.9996 * (1 + Math.pow(xDist / 6378, 2) / 2)
        scaleFactors.push(k)
        nominalScales.push(1.0)
      }

      return {
        x: offsets,
        y1: scaleFactors,
        y2: nominalScales,
        stat1: 'Central Meridian k0 = 0.9996',
        stat2: 'Zone Edge k ≈ 1.0004',
        stat3: 'Max Zone Distortion: < 0.04%',
        insight: 'UTM conformal projection bounds local scale error within 1 part in 2,500',
        axisXLabel: 'Longitude Offset from Central Meridian (°)',
        axisYLabel: 'UTM Grid Scale Factor k',
        xRange: [-3.5, 3.5] as [number, number],
        yRange: [0.999, 1.001] as [number, number],
      }
    }
  }, [mode, param1, param2])

  const axisX = useAxis({ label: results.axisXLabel, range: results.xRange })
  const axisY = useAxis({ label: results.axisYLabel, range: results.yRange })

  return (
    <Figure
      title="Coordinate Reference Systems: Projections, Geodesic Curvature, and Distortions"
      state={state}
      hoverReadout={false}
      readouts={
        <>
          <Readout label="Geographic Location" value={results.stat1} />
          <Readout label="Primary Metric" value={results.stat2} />
          <Readout label="Distortion / Scale" value={results.stat3} />
          <Readout label="Geodetic Insight" value={results.insight} />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Metric Distortion Across Latitude and Projection Systems">
        <Curve
          name={
            mode === 'mercator_distortion'
              ? 'Mercator Linear Scale Factor k = sec(φ)'
              : mode === 'longitude_compression'
                ? 'Naive Flat Euclidean Distance (km)'
                : 'UTM Local Grid Scale Factor k'
          }
          x={results.x}
          y={results.y1}
          slot={0}
          width={2.5}
        />
        <Curve
          name={
            mode === 'mercator_distortion'
              ? 'Mercator Area Inflation Factor k² = sec²(φ)'
              : mode === 'longitude_compression'
                ? 'True Geodesic Distance on Ellipsoid (km)'
                : 'Ideal Unit Scale k = 1.0000'
          }
          x={results.x}
          y={results.y2}
          slot={1}
          dashed
          width={2.0}
        />
        <Points name="Evaluated Locations" x={results.x} y={results.y1} slot={0} size={7} />
      </Plot>
    </Figure>
  )
}
