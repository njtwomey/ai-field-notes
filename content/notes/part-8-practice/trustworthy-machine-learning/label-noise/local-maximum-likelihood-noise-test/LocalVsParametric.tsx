import { useMemo } from 'react'
import { Curve, Figure, float, formatNumber, int, Plot, Points, Readout, useAxis, useFigureState } from 'aifn-render'
import {
  asymmetricXor,
  boundaryLine,
  corrupt,
  fitLogistic,
  localTest,
  parametricTest,
  xorAnchors,
  xorBoundary,
} from '../_shared/noiseTests'

const BOUNDARY = xorBoundary()
const XS = [-5, 7]
const SHOWN = 500
const pValue = (p: number) => (p < 1e-4 ? '< 0.0001' : formatNumber(p))

/**
 * The asymmetric XOR data of Yang et al. (2024). The same anchors (true posterior 1/2) feed the parametric test
 * (global logistic regression) and the local-likelihood test (local linear logistic regression at each anchor).
 */
export function LocalVsParametric() {
  const state = useFigureState({
    alpha: float(0, { min: 0, max: 0.4, step: 0.01, label: 'α = P(flip | y = 1)' }),
    beta: float(0.1, { min: 0, max: 0.4, step: 0.01, label: 'β = P(flip | y = 0)' }),
    n: int(1000, { min: 200, max: 2000, step: 100, label: 'training points N' }),
    k: int(8, { min: 1, max: 16, step: 1, label: 'anchors k' }),
    h: float(1, { min: 0.3, max: 2.5, step: 0.05, label: 'bandwidth h' }),
    seed: int(3, { ge: 0, label: 'seed' }),
  })

  const sample = useMemo(() => asymmetricXor(state.n, state.seed), [state.n, state.seed])
  const noisy = useMemo(() => corrupt(sample, state.alpha, state.beta), [sample, state.alpha, state.beta])
  const anchors = useMemo(() => xorAnchors(state.k, state.seed + 11), [state.k, state.seed])
  const fit = useMemo(() => fitLogistic(sample.X, noisy), [sample, noisy])
  const par = useMemo(() => parametricTest(fit, anchors), [fit, anchors])
  const loc = useMemo(() => localTest(sample.X, noisy, anchors, state.h), [sample, noisy, anchors, state.h])

  const series = [
    { name: 'true boundary', x: BOUNDARY.map((p) => p[0]), y: BOUNDARY.map((p) => p[1]), muted: true },
    {
      name: 'points',
      x: sample.X.slice(0, SHOWN).map((p) => p[0]),
      y: sample.X.slice(0, SHOWN).map((p) => p[1]),
      group: noisy.slice(0, SHOWN),
      groupNames: ['noisy label 0', 'noisy label 1'],
    },
    { name: 'logistic regression boundary', x: XS, y: boundaryLine(fit.theta, XS), slot: 2 },
    { name: 'anchors', x: anchors.map((a) => a[0]), y: anchors.map((a) => a[1]), emphasis: true },
  ] as const

  const xAxis = useAxis({ label: 'x₁', range: [-5, 7] })
  const yAxis = useAxis({ label: 'x₂', range: [-5, 7], equal: xAxis })
  return (
    <Figure
      title="Parametric and local tests on asymmetric XOR"
      purpose="Compare a parametric and a local test for class-conditional label noise on data that a linear model cannot fit."
      state={state}
      caption="Class 1 is a mixture of Gaussians at (4, 4) and (−2, −2), class 0 at (−1, 1) and (1, −1). The anchors lie on the true boundary (grey), where the posterior is 1/2. A straight logistic-regression boundary cannot follow it, so the parametric test rejects even with clean labels (set α = β = 0). The local fit follows the boundary; its p-values stay larger under the null and fall as β − α grows. Very small bandwidths leave anchors with few neighbours; large ones bias the local fit towards a global one."

      readouts={
        <>
          <Readout label="parametric: mean fitted posterior" value={formatNumber(par.etaBar)} />
          <Readout label="parametric p-value" value={pValue(par.p)} />
          <Readout label="local: mean fitted posterior" value={formatNumber(loc.etaBar)} />
          <Readout label="local p-value" value={pValue(loc.p)} />
          <Readout
            label="noisy posterior at anchors, (1 − α + β)/2"
            value={formatNumber((1 - state.alpha + state.beta) / 2)}
          />
        </>
      }
    >
      <Plot
        x={xAxis}
        y={yAxis}
        ariaLabel={'Asymmetric XOR data with the true boundary, a logistic-regression boundary and anchor points'}
      >
        <Points {...series[0]} />
        <Points {...series[1]} />
        <Curve {...series[2]} />
        <Points {...series[3]} />
      </Plot>
    </Figure>
  )
}
