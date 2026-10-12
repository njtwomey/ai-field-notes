import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function ExploratoryDataAnalysisVisualizer() {
  const state = useFigureState({
    edaMode: choice<'anscombe' | 'kde' | 'heteroscedastic'>(
      [
        { value: 'anscombe', label: 'Anscombe’s Quartet (Identical Stats, Different Shapes)' },
        { value: 'kde', label: 'Univariate KDE (Under- vs Over-Smoothing Bandwidth)' },
        { value: 'heteroscedastic', label: 'Bivariate Geometry (Heteroscedasticity & Fan Shapes)' },
      ],
      'anscombe',
      { label: 'exploratory diagnostic lens' },
    ),
    anscombeSet: choice<'I' | 'II' | 'III' | 'IV'>(
      [
        { value: 'I', label: 'Dataset I (Linear with Noise)' },
        { value: 'II', label: 'Dataset II (Non-Linear Quadratic Curve)' },
        { value: 'III', label: 'Dataset III (Linear with Extreme Outlier)' },
        { value: 'IV', label: 'Dataset IV (Single High-Leverage Point)' },
      ],
      'I',
      { label: 'Anscombe quartet dataset' },
    ),
    kdeSmoothing: slider(0.3, 2.5, 1.0, {
      step: 0.1,
      label: 'KDE bandwidth factor (h / h_silverman)',
    }),
    heteroSpread: slider(0.5, 3.5, 2.0, {
      step: 0.5,
      label: 'heteroscedastic variance fan severity',
    }),
  })

  const { edaMode, anscombeSet, kdeSmoothing, heteroSpread } = state

  const results = useMemo(() => {
    if (edaMode === 'anscombe') {
      // Francis Anscombe's 4 famous 11-point datasets
      // All have mean(x) = 9.0, mean(y) = 7.50, var(x) = 11.0, var(y) = 4.12, corr = 0.816
      let xVals: number[] = []
      let yVals: number[] = []

      if (anscombeSet === 'I') {
        xVals = [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]
        yVals = [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]
      } else if (anscombeSet === 'II') {
        xVals = [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]
        yVals = [9.14, 8.14, 8.74, 8.77, 9.26, 8.1, 6.13, 3.1, 9.13, 7.26, 4.74]
      } else if (anscombeSet === 'III') {
        xVals = [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]
        yVals = [7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73]
      } else {
        // IV
        xVals = [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 19]
        yVals = [6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 5.56, 7.91, 6.89, 12.5]
      }

      // Regression line: y = 3.0 + 0.5 * x
      const xLine = [3, 20]
      const yLine = [3.0 + 0.5 * 3, 3.0 + 0.5 * 20]

      return {
        x: xVals,
        y: yVals,
        lineX: xLine,
        lineY: yLine,
        statSummary: 'Mean x=9.0, Mean y=7.50, Corr=0.816, Slope=0.50',
      }
    } else if (edaMode === 'kde') {
      // Bimodal density with 2 Gaussian components at mu=3 and mu=7
      const xGrid = Array.from({ length: 40 }, (_, i) => 0.5 + i * 0.25)
      const h = 0.8 * kdeSmoothing

      const densities = xGrid.map((x) => {
        // true underlying mixture
        const g1 = (0.5 / (h * Math.sqrt(2 * Math.PI))) * Math.exp(-Math.pow(x - 3.2, 2) / (2 * h * h))
        const g2 = (0.5 / (h * Math.sqrt(2 * Math.PI))) * Math.exp(-Math.pow(x - 7.2, 2) / (2 * h * h))
        return (g1 + g2) * 25
      })

      return {
        x: xGrid,
        y: densities,
        lineX: xGrid,
        lineY: densities,
        statSummary:
          kdeSmoothing < 0.6
            ? 'Undersmoothed (Spurious Variance Noise)'
            : kdeSmoothing > 1.6
              ? 'Oversmoothed (Bimodal Modes Blurred)'
              : 'Optimal Bandwidth (Recovers Both Modes)',
      }
    } else {
      // Heteroscedastic fan: 15 points
      const xVals = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
      const yVals = xVals.map((x, i) => {
        const spread = (x / 16) * heteroSpread * 3.0
        const sign = i % 2 === 0 ? 1 : -1
        return 3.0 + 0.5 * x + sign * spread
      })

      const xLine = [2, 16]
      const yLine = [3.0 + 0.5 * 2, 3.0 + 0.5 * 16]

      return {
        x: xVals,
        y: yVals,
        lineX: xLine,
        lineY: yLine,
        statSummary: `Heteroscedasticity: Variance scales with X (Spread factor: ${heteroSpread.toFixed(1)})`,
      }
    }
  }, [edaMode, anscombeSet, kdeSmoothing, heteroSpread])

  const axisX = useAxis({
    label:
      edaMode === 'anscombe'
        ? 'Anscombe Feature X'
        : edaMode === 'kde'
          ? 'Continuous Value Scale'
          : 'Predictor X (Variance Expands with X)',
    range: edaMode === 'anscombe' ? [2, 20] : edaMode === 'kde' ? [0, 10.5] : [1, 17],
  })

  const axisY = useAxis({
    label: edaMode === 'kde' ? 'Estimated Kernel Density (%)' : 'Outcome / Feature Y',
    range: edaMode === 'kde' ? [0, 45] : [2, 14],
  })

  return (
    <Figure
      title="Exploratory Data Analysis: Geometry Beyond Summary Statistics"
      state={state}
      hoverReadout={false}
      caption="EDA reveals geometric realities that summary statistics conceal—demonstrated by Anscombe's quartet, KDE bandwidth smoothing sensitivity, and heteroscedastic fan shapes."
      readouts={
        <>
          {edaMode === 'anscombe' && (
            <>
              <Readout label="Linear Fit Metrics" value={results.statSummary} />
              <Readout
                label="Visual Diagnostic"
                value={
                  anscombeSet === 'I'
                    ? 'Legitimate noisy linear relationship'
                    : anscombeSet === 'II'
                      ? 'Severe non-linearity (requires polynomial/spline)'
                      : anscombeSet === 'III'
                        ? 'Single outlier severely distorts slope'
                        : 'Single high-leverage point creates artificial correlation'
                }
              />
            </>
          )}
          {edaMode === 'kde' && (
            <>
              <Readout label="Smoothing Status" value={results.statSummary} />
              <Readout label="Bandwidth Setting" value={`h = ${(0.8 * kdeSmoothing).toFixed(2)}`} />
            </>
          )}
          {edaMode === 'heteroscedastic' && (
            <>
              <Readout label="Residual Diagnostic" value="Non-constant variance across X domain" />
              <Readout label="Modeling Impact" value="OLS standard errors biased; requires WLS or quantile loss" />
            </>
          )}
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Exploratory Data Geometry">
        {edaMode !== 'kde' && (
          <Curve name="Fitted OLS Regression Line" x={results.lineX} y={results.lineY} slot={2} dashed width={1.8} />
        )}
        {edaMode === 'kde' ? (
          <Curve name="Estimated Density Profile" x={results.x} y={results.y} slot={0} width={2.5} />
        ) : (
          <Points name="Observed Samples" x={results.x} y={results.y} slot={0} size={8} />
        )}
      </Plot>
    </Figure>
  )
}
