import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function VectorAndRasterVisualizer() {
  const state = useFigureState({
    mode: choice<'resolution_quantization' | 'rtree_efficiency' | 'zonal_overlap'>(
      [
        {
          value: 'resolution_quantization',
          label: 'Raster GSD vs Vector Boundary: Quantization Error & Mixed Pixels',
        },
        {
          value: 'rtree_efficiency',
          label: 'Spatial Join Scaling: Naive Pairwise O(N·M) vs R-Tree Indexed O(N·log M)',
        },
        {
          value: 'zonal_overlap',
          label: 'Zonal Statistics: Cell Overlap Coverage vs Polygon Area Estimation',
        },
      ],
      'resolution_quantization',
      { label: 'geospatial data structure' },
    ),
    param1: slider(1, 50, 10, {
      step: 1,
      label: 'ground sampling distance GSD (m) / query points N (x100) / cell size',
    }),
    param2: slider(10, 100, 50, {
      step: 10,
      label: 'parcel boundary radius (m) / target polygon count M / threshold (%)',
    }),
  })

  const { mode, param1, param2 } = state

  const results = useMemo(() => {
    if (mode === 'resolution_quantization') {
      const gsd = param1 // 1 to 50 m
      const parcelRadius = param2 * 2 // 20 to 200 m
      const trueArea = Math.PI * parcelRadius * parcelRadius

      const gsdSteps = [1, 2, 5, 10, 15, 20, 25, 30, 40, 50]
      const mixedPixelPct: number[] = []
      const relativeAreaError: number[] = []

      for (const curGsd of gsdSteps) {
        // Boundary perimeter ~ 2 * pi * R
        // Boundary band of width curGsd contains approx (2 * pi * R * curGsd) / curGsd^2 pixels = 2 * pi * R / curGsd
        const totalCells = trueArea / (curGsd * curGsd)
        const boundaryCells = (2 * Math.PI * parcelRadius) / curGsd
        const mixedFraction = Math.min(100, (boundaryCells / Math.max(1, totalCells)) * 100)
        // Quantization area error bounded by fraction of boundary cells
        const areaErr = Math.min(50, 0.5 * mixedFraction)

        mixedPixelPct.push(mixedFraction)
        relativeAreaError.push(areaErr)
      }

      const activeTotalCells = Math.round(trueArea / (gsd * gsd))
      const activeBoundary = (2 * Math.PI * parcelRadius) / gsd
      const activeMixed = Math.min(100, (activeBoundary / Math.max(1, activeTotalCells)) * 100)

      return {
        x: gsdSteps,
        y1: mixedPixelPct,
        y2: relativeAreaError,
        stat1: `GSD = ${gsd} m, Radius = ${parcelRadius} m`,
        stat2: `Grid: ${activeTotalCells.toLocaleString()} cells`,
        stat3: `Mixed boundary: ${activeMixed.toFixed(1)}%`,
        insight: 'Coarse GSD creates high boundary mixed-pixel ambiguity in land cover classification',
        axisXLabel: 'Ground Sampling Distance GSD (m)',
        axisYLabel: 'Mixed Pixels & Boundary Area Error (%)',
        xRange: [0, 55] as [number, number],
        yRange: [0, 100] as [number, number],
      }
    } else if (mode === 'rtree_efficiency') {
      const nPoints = param1 * 100 // 100 to 5,000 points
      const mPolys = param2 * 10 // 100 to 1,000 polygons

      // Evaluation counts for query sets
      const pointCounts = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
      const naiveOperations: number[] = []
      const rtreeOperations: number[] = []

      for (const curN of pointCounts) {
        // Naive: N * M operations (in millions)
        const naiveOps = (curN * mPolys) / 1000000
        // R-Tree: N * (log2(M) + candidates_in_box)
        const rtreeOps = (curN * (Math.log2(mPolys) + 4)) / 1000000
        naiveOperations.push(naiveOps)
        rtreeOperations.push(rtreeOps)
      }

      const activeNaive = (nPoints * mPolys) / 1000000
      const activeRtree = (nPoints * (Math.log2(mPolys) + 4)) / 1000000
      const speedup = activeNaive / Math.max(0.001, activeRtree)

      return {
        x: pointCounts,
        y1: naiveOperations,
        y2: rtreeOperations,
        stat1: `N = ${nPoints.toLocaleString()}, M = ${mPolys} polys`,
        stat2: `Naive ops: ${activeNaive.toFixed(2)}M checks`,
        stat3: `R-Tree: ${activeRtree.toFixed(3)}M checks`,
        insight: `R-tree bounding box indexing delivers ~${speedup.toFixed(0)}x speedup`,
        axisXLabel: 'Query Points N',
        axisYLabel: 'Geometric Intersect Operations (Millions)',
        xRange: [300, 5200] as [number, number],
        yRange: [0, 5.5] as [number, number],
      }
    } else {
      // Zonal overlap: cell coverage threshold vs area estimation precision
      const thresholds = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
      const inclusionFractions: number[] = []
      const areaFidelities: number[] = []

      for (const t of thresholds) {
        // Center-point rule (t=50) vs all-touched rule (t=0)
        inclusionFractions.push(100 - 0.7 * t)
        areaFidelities.push(100 - Math.abs(t - 50) * 0.8)
      }

      return {
        x: thresholds,
        y1: inclusionFractions,
        y2: areaFidelities,
        stat1: 'Zonal Statistics Overlap Rule',
        stat2: 'All-touched vs Center-point',
        stat3: 'Optimal at 50% cell threshold',
        insight: 'All-touched rule over-estimates polygon areas by including perimeter pixels',
        axisXLabel: 'Cell Overlap Area Threshold (%)',
        axisYLabel: 'Selected Cells (%) & Area Fidelity (%)',
        xRange: [5, 105] as [number, number],
        yRange: [20, 105] as [number, number],
      }
    }
  }, [mode, param1, param2])

  const axisX = useAxis({ label: results.axisXLabel, range: results.xRange })
  const axisY = useAxis({ label: results.axisYLabel, range: results.yRange })

  return (
    <Figure
      title="Vector and Raster Geospatial Data: Resolution, Spatial Joins, and Zonal Aggregation"
      state={state}
      hoverReadout={false}
      readouts={
        <>
          <Readout label="Geospatial Setting" value={results.stat1} />
          <Readout label="Raw Scale / Baseline" value={results.stat2} />
          <Readout label="Indexed / Fidelity Metric" value={results.stat3} />
          <Readout label="Spatial Data Insight" value={results.insight} />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Spatial Data Model Computational and Discretization Behavior">
        <Curve
          name={
            mode === 'resolution_quantization'
              ? 'Mixed Perimeter Pixels (%)'
              : mode === 'rtree_efficiency'
                ? 'Naive Pairwise Operations O(N·M) (Millions)'
                : 'Selected Raster Cell Count (%)'
          }
          x={results.x}
          y={results.y1}
          slot={0}
          width={2.5}
        />
        <Curve
          name={
            mode === 'resolution_quantization'
              ? 'Boundary Discretization Error (%)'
              : mode === 'rtree_efficiency'
                ? 'R-Tree Indexed Operations O(N·log M) (Millions)'
                : 'Zonal Metric Area Estimation Fidelity (%)'
          }
          x={results.x}
          y={results.y2}
          slot={1}
          dashed
          width={2.0}
        />
        <Points name="Evaluated Configurations" x={results.x} y={results.y1} slot={0} size={7} />
      </Plot>
    </Figure>
  )
}
