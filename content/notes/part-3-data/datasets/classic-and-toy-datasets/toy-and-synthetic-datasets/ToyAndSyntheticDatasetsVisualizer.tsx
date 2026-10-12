import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function ToyAndSyntheticDatasetsVisualizer() {
  const state = useFigureState({
    mode: choice<'two_moons' | 'concentric_circles' | 'swiss_roll_unfolding'>(
      [
        {
          value: 'two_moons',
          label: 'Two Moons: Interlocking Non-Convex Manifolds vs Noise Level σ',
        },
        {
          value: 'concentric_circles',
          label: 'Concentric Circles: Planar Non-Separability vs Radial Basis Transform',
        },
        {
          value: 'swiss_roll_unfolding',
          label: 'Swiss Roll: Geodesic Manifold Distance vs Euclidean Shortcut (Tenenbaum 2000)',
        },
      ],
      'two_moons',
      { label: 'synthetic dataset generator' },
    ),
    param1: slider(5, 40, 15, {
      step: 5,
      label: 'noise level sigma (x0.01) / inner radius r1 / roll turns (tau)',
    }),
    param2: slider(1, 5, 3, {
      step: 1,
      label: 'sample density N (x100) / separation margin / geodesic neighborhood k',
    }),
  })

  const { mode, param1, param2 } = state

  const results = useMemo(() => {
    if (mode === 'two_moons') {
      const noise = param1 * 0.01 // 0.05 to 0.40
      // Generate two crescent moons:
      // Moon 1: x = cos(t), y = sin(t), t in [0, pi]
      // Moon 2: x = 1 - cos(t), y = 0.5 - sin(t), t in [0, pi]
      const tSteps = [0, 0.35, 0.7, 1.05, 1.4, 1.75, 2.1, 2.45, 2.8, 3.14]
      const moon1X: number[] = []
      const moon1Y: number[] = []
      const moon2X: number[] = []
      const moon2Y: number[] = []

      for (const t of tSteps) {
        moon1X.push(Math.cos(t))
        moon1Y.push(Math.sin(t))
        moon2X.push(1 - Math.cos(t))
        moon2Y.push(0.5 - Math.sin(t))
      }

      // Expected Bayes accuracy drops with noise
      const bayesAcc = Math.max(70, 100 - 90 * noise)

      return {
        x1: moon1X,
        y1: moon1Y,
        x2: moon2X,
        y2: moon2Y,
        stat1: `Gaussian noise σ = ${noise.toFixed(2)}`,
        stat2: `Bayes optimal bound: ${bayesAcc.toFixed(1)}%`,
        stat3: 'Linear classifier max: ~78%',
        insight: 'Interlocking non-convex manifolds require non-linear kernels or depth ≥ 2 MLPs',
        axisXLabel: 'Coordinate x₁',
        axisYLabel: 'Coordinate x₂',
        xRange: [-1.5, 2.5] as [number, number],
        yRange: [-1.0, 1.5] as [number, number],
      }
    } else if (mode === 'concentric_circles') {
      const rInner = 0.8 + (param1 / 40) * 0.6 // 0.8 to 1.4
      const rOuter = 2.4

      const angles = [0, 0.63, 1.26, 1.88, 2.51, 3.14, 3.77, 4.4, 5.03, 5.65, 6.28]
      const innerX: number[] = []
      const innerY: number[] = []
      const outerX: number[] = []
      const outerY: number[] = []

      for (const a of angles) {
        innerX.push(rInner * Math.cos(a))
        innerY.push(rInner * Math.sin(a))
        outerX.push(rOuter * Math.cos(a))
        outerY.push(rOuter * Math.sin(a))
      }

      const margin = rOuter - rInner

      return {
        x1: innerX,
        y1: innerY,
        x2: outerX,
        y2: outerY,
        stat1: `Radii: r₁ = ${rInner.toFixed(2)}, r₂ = ${rOuter.toFixed(2)}`,
        stat2: `Margin Δr = ${margin.toFixed(2)}`,
        stat3: 'Polar separability: 100%',
        insight: 'Radially symmetric data is completely linearly non-separable in Euclidean (x₁, x₂)',
        axisXLabel: 'Planar Coordinate x₁',
        axisYLabel: 'Planar Coordinate x₂',
        xRange: [-3.0, 3.0] as [number, number],
        yRange: [-3.0, 3.0] as [number, number],
      }
    } else {
      // Swiss roll: 2D manifold spiral in 3D: (t * cos(t), y, t * sin(t))
      // Plotting manifold parameter t vs Euclidean shortcut distance vs true geodesic distance
      const tVals = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
      const t0 = 1.5
      const euclideanDists: number[] = []
      const geodesicDists: number[] = []

      for (const t of tVals) {
        // True arc length along Archimedean spiral: s ~ 0.5 * (t * sqrt(1 + t^2) + asinh(t))
        const s = 0.5 * (t * Math.sqrt(1 + t * t) - t0 * Math.sqrt(1 + t0 * t0))
        // Euclidean straight-line distance cuts across spiral roll layers
        const xDiff = t * Math.cos(t) - t0 * Math.cos(t0)
        const zDiff = t * Math.sin(t) - t0 * Math.sin(t0)
        const eucl = Math.sqrt(xDiff * xDiff + zDiff * zDiff)

        geodesicDists.push(s)
        euclideanDists.push(eucl)
      }

      return {
        x1: tVals,
        y1: euclideanDists,
        y2: geodesicDists,
        x2: tVals,
        stat1: 'Tenenbaum 2000 Swiss Roll',
        stat2: 'Manifold intrinsic dimension d = 2',
        stat3: 'Embedded in ambient D = 3',
        insight: 'Euclidean distance cuts through empty space across roll layers; Isomap follows the manifold',
        axisXLabel: 'Spiral Parameter t (radians)',
        axisYLabel: 'Distance from Origin Point (units)',
        xRange: [1.2, 4.8] as [number, number],
        yRange: [0, 12] as [number, number],
      }
    }
  }, [mode, param1, param2])

  const axisX = useAxis({ label: results.axisXLabel, range: results.xRange })
  const axisY = useAxis({ label: results.axisYLabel, range: results.yRange })

  return (
    <Figure
      title="Toy and Synthetic Datasets: Manifold Geometry, Noise Margins, and Dimensionality"
      state={state}
      hoverReadout={false}
      readouts={
        <>
          <Readout label="Synthetic Setup" value={results.stat1} />
          <Readout label="Boundary Characteristic" value={results.stat2} />
          <Readout label="Linear Separability" value={results.stat3} />
          <Readout label="Manifold Insight" value={results.insight} />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Geometric Manifold Configuration and Class Separation">
        <Points name="Class 0 / Layer 1" x={results.x1} y={results.y1} slot={0} size={8} />
        {mode === 'swiss_roll_unfolding' ? (
          <Curve name="Geodesic Manifold Distance" x={results.x2} y={results.y2} slot={1} dashed width={2.0} />
        ) : (
          <Points name="Class 1 / Outer Layer" x={results.x2} y={results.y2} slot={1} size={8} />
        )}
      </Plot>
    </Figure>
  )
}
