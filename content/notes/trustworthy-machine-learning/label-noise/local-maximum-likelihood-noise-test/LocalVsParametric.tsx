import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamButton,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type XYSeries,
} from '@/components/viz'
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
  const alpha = useParam(0, { min: 0, max: 0.4, step: 0.01 })
  const beta = useParam(0.1, { min: 0, max: 0.4, step: 0.01 })
  const n = useParam(1000, { min: 200, max: 2000, step: 100 })
  const k = useParam(8, { min: 1, max: 16, step: 1 })
  const h = useParam(1, { min: 0.3, max: 2.5, step: 0.05 })
  const [seed, setSeed] = useState(3)

  const sample = useMemo(() => asymmetricXor(n.value, seed), [n.value, seed])
  const noisy = useMemo(() => corrupt(sample, alpha.value, beta.value), [sample, alpha.value, beta.value])
  const anchors = useMemo(() => xorAnchors(k.value, seed + 11), [k.value, seed])
  const fit = useMemo(() => fitLogistic(sample.X, noisy), [sample, noisy])
  const par = useMemo(() => parametricTest(fit, anchors), [fit, anchors])
  const loc = useMemo(() => localTest(sample.X, noisy, anchors, h.value), [sample, noisy, anchors, h.value])

  const series: XYSeries[] = [
    { name: 'true boundary', type: 'scatter', x: BOUNDARY.map((p) => p[0]), y: BOUNDARY.map((p) => p[1]), muted: true },
    {
      name: 'points',
      type: 'scatter',
      x: sample.X.slice(0, SHOWN).map((p) => p[0]),
      y: sample.X.slice(0, SHOWN).map((p) => p[1]),
      group: noisy.slice(0, SHOWN),
      groupNames: ['noisy label 0', 'noisy label 1'],
    },
    { name: 'logistic regression boundary', type: 'line', x: XS, y: boundaryLine(fit.theta, XS), slot: 2 },
    { name: 'anchors', type: 'scatter', x: anchors.map((a) => a[0]), y: anchors.map((a) => a[1]), emphasis: true },
  ]

  return (
    <Interactive
      title="Parametric and local tests on asymmetric XOR"
      caption="Class 1 is a mixture of Gaussians at (4, 4) and (−2, −2), class 0 at (−1, 1) and (1, −1). The anchors lie on the true boundary (grey), where the posterior is 1/2. A straight logistic-regression boundary cannot follow it, so the parametric test rejects even with clean labels (set α = β = 0). The local fit follows the boundary; its p-values stay larger under the null and fall as β − α grows. Very small bandwidths leave anchors with few neighbours; large ones bias the local fit towards a global one."
      controls={
        <>
          <ParamSlider label="α = P(flip | y = 1)" param={alpha} />
          <ParamSlider label="β = P(flip | y = 0)" param={beta} />
          <ParamSlider label="training points N" param={n} format={(v) => String(v)} />
          <ParamSlider label="anchors k" param={k} format={(v) => String(v)} />
          <ParamSlider label="bandwidth h" param={h} />
          <ParamButton onClick={() => setSeed((s) => s + 1)}>New sample</ParamButton>
        </>
      }
      readout={
        <>
          <Readout label="parametric: mean fitted posterior" value={formatNumber(par.etaBar)} />
          <Readout label="parametric p-value" value={pValue(par.p)} />
          <Readout label="local: mean fitted posterior" value={formatNumber(loc.etaBar)} />
          <Readout label="local p-value" value={pValue(loc.p)} />
          <Readout
            label="noisy posterior at anchors, (1 − α + β)/2"
            value={formatNumber((1 - alpha.value + beta.value) / 2)}
          />
        </>
      }
    >
      <XYChart
        series={series}
        xLabel="x₁"
        yLabel="x₂"
        xRange={[-5, 7]}
        yRange={[-5, 7]}
        equalAspect
        ariaLabel="Asymmetric XOR data with the true boundary, a logistic-regression boundary and anchor points"
      />
    </Interactive>
  )
}
