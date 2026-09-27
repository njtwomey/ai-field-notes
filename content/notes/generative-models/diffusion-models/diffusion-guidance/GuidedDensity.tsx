import { useMemo } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, useParam, type XYSeries } from '@/components/viz'
import { linspace } from '@/lib/math'

/**
 * Two equally likely classes, p(x | A) = N(−1, 1) and p(x | B) = N(1, 1). Classifier-free guidance with weight w
 * targets p̃(x | A) ∝ p(x | A)^{1+w} p(x)^{−w} = p(x | A) p(A | x)^w (up to a constant), computed here on a grid.
 */
const X = linspace(-5, 5, 401)
const DX = X[1] - X[0]
const normal = (x: number, m: number) => Math.exp(-((x - m) ** 2) / 2) / Math.sqrt(2 * Math.PI)
const P_A = X.map((x) => normal(x, -1))
const P_B = X.map((x) => normal(x, 1))
const P = X.map((_, i) => 0.5 * P_A[i] + 0.5 * P_B[i])
const POST_A = X.map((_, i) => (0.5 * P_A[i]) / P[i])

export function GuidedDensity() {
  const w = useParam(2, { min: 0, max: 8, step: 0.1 })

  const view = useMemo(() => {
    const unnorm = X.map((_, i) => Math.exp((1 + w.value) * Math.log(P_A[i]) - w.value * Math.log(P[i])))
    const z = unnorm.reduce((s, v) => s + v, 0) * DX
    const guided = unnorm.map((v) => v / z)
    const mean = guided.reduce((s, p, i) => s + p * X[i], 0) * DX
    const sd = Math.sqrt(guided.reduce((s, p, i) => s + p * (X[i] - mean) ** 2, 0) * DX)
    const confidence = guided.reduce((s, p, i) => s + p * POST_A[i], 0) * DX
    const series: XYSeries[] = [
      { name: 'p(x), both classes', type: 'line', x: X, y: P, muted: true },
      { name: 'p(x | A), unguided', type: 'line', x: X, y: P_A, slot: 0, dashed: true },
      { name: 'p(x | B)', type: 'line', x: X, y: P_B, slot: 1, dashed: true },
      { name: 'guided target for A', type: 'line', x: X, y: guided, slot: 0, area: true },
    ]
    return { series, mean, sd, confidence }
  }, [w.value])

  return (
    <Interactive
      title="What guidance does to a class-conditional distribution"
      caption="Class A is N(−1, 1) and class B is N(1, 1). Guidance with weight w targets p(x | A) p(A | x)^w, renormalised. At w = 0 this is the plain conditional. Larger w moves mass away from the region class B also occupies and narrows the distribution: samples become more typical of A and less diverse."
      controls={<ParamSlider label="guidance weight w" param={w} />}
      readout={
        <>
          <Readout label="mean" value={formatNumber(view.mean)} />
          <Readout label="standard deviation" value={formatNumber(view.sd)} />
          <Readout label="mean classifier confidence p(A | x)" value={formatNumber(view.confidence)} />
        </>
      }
    >
      <XYChart series={view.series} xLabel="x" yLabel="density" xRange={[-5, 5]} yRange={[0, undefined]} />
    </Interactive>
  )
}
