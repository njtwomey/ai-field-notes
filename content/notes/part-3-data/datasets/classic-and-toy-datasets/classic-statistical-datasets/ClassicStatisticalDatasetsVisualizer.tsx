import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function ClassicStatisticalDatasetsVisualizer() {
  const state = useFigureState({
    mode: choice<'anscombe_quartet' | 'old_faithful' | 'iris_separation'>(
      [
        {
          value: 'anscombe_quartet',
          label: "Anscombe's Quartet: Identical Summary Stats vs Distinct Distributions (Anscombe 1973)",
        },
        {
          value: 'old_faithful',
          label: 'Old Faithful Geyser: Bimodal Duration vs Waiting Time Clustering',
        },
        {
          value: 'iris_separation',
          label: "Fisher's Iris: Linear Separability of Setosa vs Versicolor/Virginica (Fisher 1936)",
        },
      ],
      'anscombe_quartet',
      { label: 'benchmark dataset selection' },
    ),
    param1: slider(1, 4, 1, {
      step: 1,
      label: 'Anscombe quartet dataset (I, II, III, IV) / cluster variance / feature pair',
    }),
    param2: slider(1, 5, 3, {
      step: 1,
      label: 'regression line overlay / waiting time threshold / species boundary',
    }),
  })

  const { mode, param1, param2 } = state

  const results = useMemo(() => {
    if (mode === 'anscombe_quartet') {
      const quartetId = param1 // 1, 2, 3, 4

      // Exact coordinates of Anscombe's 4 datasets (11 points each)
      const dataI: [number, number][] = [
        [10.0, 8.04],
        [8.0, 6.95],
        [13.0, 7.58],
        [9.0, 8.81],
        [11.0, 8.33],
        [14.0, 9.96],
        [6.0, 7.24],
        [4.0, 4.26],
        [12.0, 10.84],
        [7.0, 4.82],
        [5.0, 5.68],
      ]
      const dataII: [number, number][] = [
        [10.0, 9.14],
        [8.0, 8.14],
        [13.0, 8.74],
        [9.0, 8.77],
        [11.0, 9.26],
        [14.0, 8.1],
        [6.0, 6.13],
        [4.0, 3.1],
        [12.0, 9.13],
        [7.0, 7.26],
        [5.0, 4.74],
      ]
      const dataIII: [number, number][] = [
        [10.0, 7.46],
        [8.0, 6.77],
        [13.0, 12.74],
        [9.0, 7.11],
        [11.0, 7.81],
        [14.0, 8.84],
        [6.0, 6.08],
        [4.0, 5.39],
        [12.0, 8.15],
        [7.0, 6.42],
        [5.0, 5.73],
      ]
      const dataIV: [number, number][] = [
        [8.0, 6.58],
        [8.0, 5.76],
        [8.0, 7.71],
        [8.0, 8.84],
        [8.0, 8.47],
        [8.0, 7.04],
        [8.0, 5.25],
        [19.0, 12.5],
        [8.0, 5.56],
        [8.0, 7.91],
        [8.0, 6.89],
      ]

      const points = quartetId === 1 ? dataI : quartetId === 2 ? dataII : quartetId === 3 ? dataIII : dataIV
      const xCoords = points.map((p) => p[0])
      const yCoords = points.map((p) => p[1])

      // All 4 have exact regression line y = 3.00 + 0.500 * x
      const regLineX = [3, 5, 8, 10, 14, 18, 20]
      const regLineY = regLineX.map((x) => 3.0 + 0.5 * x)

      const quartetDesc =
        quartetId === 1
          ? 'Quartet I: Standard linear relationship with Gaussian residual variance'
          : quartetId === 2
            ? 'Quartet II: Nonlinear quadratic parabolic curve (linear model misspecified)'
            : quartetId === 3
              ? 'Quartet III: Strict collinear linear relationship with single high-residual vertical outlier'
              : 'Quartet IV: Clustered x values with single high-leverage influential point'

      return {
        x: xCoords,
        y: yCoords,
        regX: regLineX,
        regY: regLineY,
        stat1: `Anscombe Dataset ${quartetId === 1 ? 'I' : quartetId === 2 ? 'II' : quartetId === 3 ? 'III' : 'IV'}`,
        stat2: 'Mean x = 9.00, y = 7.50',
        stat3: 'Regression: y = 3.00 + 0.50x, r = 0.816',
        insight: quartetDesc,
        axisXLabel: 'Explanatory Feature x',
        axisYLabel: 'Response Target y',
        xRange: [2, 21] as [number, number],
        yRange: [2, 14] as [number, number],
      }
    } else if (mode === 'old_faithful') {
      // Old Faithful: Eruption duration (1.5 to 5.2 min) vs Waiting time (40 to 95 min)
      // Two bimodal clusters: short eruptions (~2.0 min) with short wait (~54 min),
      // and long eruptions (~4.4 min) with long wait (~80 min).
      const durations = [1.8, 2.0, 2.2, 1.9, 2.1, 2.3, 1.7, 3.8, 4.0, 4.2, 4.4, 4.6, 4.8, 3.9, 4.5]
      const waitTimes = [51, 54, 58, 52, 55, 60, 48, 76, 80, 84, 82, 88, 91, 79, 85]

      const waitThreshold = 40 + param2 * 10 // 50 to 90 min

      return {
        x: durations,
        y: waitTimes,
        regX: [1.5, 5.0],
        regY: [33 + 10.7 * 1.5, 33 + 10.7 * 5.0],
        stat1: 'Old Faithful (272 eruptions)',
        stat2: 'Bimodal Geyser Eruptions',
        stat3: `Wait Threshold = ${waitThreshold} min`,
        insight: 'Strong positive correlation and clear 2-component mixture of Gaussians',
        axisXLabel: 'Eruption Duration (minutes)',
        axisYLabel: 'Waiting Time to Next Eruption (minutes)',
        xRange: [1.2, 5.2] as [number, number],
        yRange: [40, 96] as [number, number],
      }
    } else {
      // Fisher Iris: Petal length (1.0 to 7.0 cm) vs Petal width (0.1 to 2.5 cm)
      // Setosa (L: 1-2, W: 0.1-0.6), Versicolor (L: 3-5, W: 1.0-1.8), Virginica (L: 4.5-6.9, W: 1.4-2.5)
      const petalL = [1.4, 1.3, 1.5, 1.4, 1.7, 4.7, 4.5, 4.9, 4.0, 4.6, 6.0, 5.1, 5.9, 5.6, 5.8]
      const petalW = [0.2, 0.2, 0.2, 0.2, 0.4, 1.4, 1.5, 1.5, 1.3, 1.3, 2.5, 1.9, 2.1, 1.8, 2.2]

      return {
        x: petalL,
        y: petalW,
        regX: [1.0, 6.5],
        regY: [-0.36 + 0.41 * 1.0, -0.36 + 0.41 * 6.5],
        stat1: 'Fisher Iris (150 specimens, 3 species)',
        stat2: 'Setosa linearly separable',
        stat3: 'Versicolor/Virginica boundary overlap',
        insight: 'Classic benchmark for discriminant analysis and multiclass clustering',
        axisXLabel: 'Petal Length (cm)',
        axisYLabel: 'Petal Width (cm)',
        xRange: [0.8, 6.8] as [number, number],
        yRange: [0, 2.8] as [number, number],
      }
    }
  }, [mode, param1, param2])

  const axisX = useAxis({ label: results.axisXLabel, range: results.xRange })
  const axisY = useAxis({ label: results.axisYLabel, range: results.yRange })

  return (
    <Figure
      title="Classic Statistical Datasets: Distributions, Modality, and Model Assumptions"
      state={state}
      hoverReadout={false}
      readouts={
        <>
          <Readout label="Active Benchmark" value={results.stat1} />
          <Readout label="Statistical Summary" value={results.stat2} />
          <Readout label="Model Property" value={results.stat3} />
          <Readout label="Diagnostic Insight" value={results.insight} />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Observation Scatter and Statistical Structure">
        <Points name="Benchmark Observations" x={results.x} y={results.y} slot={0} size={8} />
        <Curve name="Fitted Linear Model / Trajectory" x={results.regX} y={results.regY} slot={1} dashed width={2.0} />
      </Plot>
    </Figure>
  )
}
