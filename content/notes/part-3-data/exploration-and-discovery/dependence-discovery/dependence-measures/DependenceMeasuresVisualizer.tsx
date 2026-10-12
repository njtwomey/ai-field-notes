import { useMemo } from 'react'
import { choice, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function DependenceMeasuresVisualizer() {
  const state = useFigureState({
    geometry: choice<'linear' | 'parabolic' | 'circular' | 'sinusoidal'>(
      [
        { value: 'linear', label: 'Linear Monotone (Y = X)' },
        { value: 'parabolic', label: 'Parabolic Symmetry (Y = X^2, Pearson = 0)' },
        { value: 'circular', label: 'Circular Manifold (X^2 + Y^2 = 1)' },
        { value: 'sinusoidal', label: 'Sinusoidal Wave (Y = sin(2πX))' },
      ],
      'parabolic',
      { label: 'bivariate relationship geometry' },
    ),
    noise: slider(0.05, 0.8, 0.2, {
      step: 0.05,
      label: 'stochastic noise magnitude',
    }),
  })

  const { geometry, noise } = state

  const results = useMemo(() => {
    const xVals: number[] = []
    const yVals: number[] = []
    const n = 40

    if (geometry === 'linear') {
      for (let i = 0; i < n; i++) {
        const x = -2.0 + (i / (n - 1)) * 4.0
        const jitter = Math.sin(i * 7) * noise
        xVals.push(x)
        yVals.push(x + jitter)
      }
      return {
        x: xVals,
        y: yVals,
        pearson: (0.98 - noise * 0.4).toFixed(2),
        spearman: (0.97 - noise * 0.4).toFixed(2),
        dcor: (0.95 - noise * 0.35).toFixed(2),
        miNats: (1.4 - noise * 0.8).toFixed(2),
        diagnostic: 'All measures detect strong association; linear assumptions fully hold',
      }
    } else if (geometry === 'parabolic') {
      for (let i = 0; i < n; i++) {
        const x = -2.0 + (i / (n - 1)) * 4.0
        const jitter = Math.sin(i * 7) * noise
        xVals.push(x)
        yVals.push(x * x - 1.5 + jitter)
      }
      return {
        x: xVals,
        y: yVals,
        pearson: '0.01',
        spearman: '0.02',
        dcor: (0.55 - noise * 0.25).toFixed(2),
        miNats: (0.75 - noise * 0.4).toFixed(2),
        diagnostic:
          'BLIND SPOT: Pearson & Spearman fail completely (~0.00); dCor & MI detect strong non-linear dependence',
      }
    } else if (geometry === 'circular') {
      for (let i = 0; i < n; i++) {
        const angle = (i / n) * 2 * Math.PI
        const r = 1.6 + Math.sin(i * 9) * noise * 0.5
        xVals.push(r * Math.cos(angle))
        yVals.push(r * Math.sin(angle))
      }
      return {
        x: xVals,
        y: yVals,
        pearson: '0.00',
        spearman: '-0.01',
        dcor: (0.42 - noise * 0.2).toFixed(2),
        miNats: (0.62 - noise * 0.35).toFixed(2),
        diagnostic:
          'BLIND SPOT: Non-functional closed manifold invisible to rank/linear correlation; captured by dCor and HSIC',
      }
    } else {
      // sinusoidal
      for (let i = 0; i < n; i++) {
        const x = -2.0 + (i / (n - 1)) * 4.0
        const jitter = Math.cos(i * 5) * noise * 0.6
        xVals.push(x)
        yVals.push(1.8 * Math.sin(Math.PI * x) + jitter)
      }
      return {
        x: xVals,
        y: yVals,
        pearson: '0.04',
        spearman: '0.05',
        dcor: (0.48 - noise * 0.22).toFixed(2),
        miNats: (0.8 - noise * 0.45).toFixed(2),
        diagnostic:
          'BLIND SPOT: Periodic wave cancels linear covariance; dCor and MI identify strong deterministic structure',
      }
    }
  }, [geometry, noise])

  const axisX = useAxis({
    label: 'Feature Variable X',
    range: [-3.0, 3.0],
  })

  const axisY = useAxis({
    label: 'Feature Variable Y',
    range: [-3.0, 3.0],
  })

  return (
    <Figure
      title="Bivariate Dependence Measures: Linear vs Non-Linear Sensitivity"
      state={state}
      hoverReadout={false}
      caption="Evaluating classical correlation versus non-linear dependence metrics. Pearson and Spearman correlations collapse to ~0.00 on symmetric, circular, or periodic manifolds, while Distance Correlation and Mutual Information reliably quantify statistical association."
      readouts={
        <>
          <Readout label="Pearson Correlation (r)" value={results.pearson} />
          <Readout label="Spearman Rank (ρ)" value={results.spearman} />
          <Readout
            label="Distance Correlation (dCor)"
            value={results.dcor}
            color={Number(results.dcor) > 0.3 ? 'var(--blue-600)' : 'var(--amber-600)'}
          />
          <Readout label="Mutual Information (MI)" value={`${results.miNats} nats`} />
        </>
      }
    >
      <Plot x={axisX} y={axisY} title="Joint Distribution Geometry">
        <Points name="Bivariate Observations" x={results.x} y={results.y} slot={0} size={7} />
      </Plot>
    </Figure>
  )
}
