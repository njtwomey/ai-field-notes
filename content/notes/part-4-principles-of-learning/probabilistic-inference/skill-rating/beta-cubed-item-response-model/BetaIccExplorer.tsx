import { useMemo, useState } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, type Handle, type XYSeries } from 'aifn-render'
import { linspace } from '@/lib/math'
import { incompleteBeta, logGamma } from '@/lib/math/special'

const THETA = linspace(0.005, 0.995, 100)
const P_GRID = linspace(0.0025, 0.9975, 200)
const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v))

/** α and β of the response distribution Beta(α, β) at ability θ. */
const shape = (t: number, d: number, a: number) => ({
  al: Math.pow(t / d, a),
  be: Math.pow((1 - t) / (1 - d), a),
})

/** The q-quantile of Beta(α, β) by bisection on the regularised incomplete beta function. */
function betaQuantile(q: number, al: number, be: number): number {
  let lo = 0
  let hi = 1
  for (let k = 0; k < 40; k++) {
    const mid = (lo + hi) / 2
    if (incompleteBeta(mid, al, be) < q) lo = mid
    else hi = mid
  }
  return (lo + hi) / 2
}

function shapeName(a: number): string {
  if (Math.abs(a) < 0.025) return 'flat: responses ignore ability'
  const dir = a > 0 ? 'increasing' : 'decreasing'
  const m = Math.abs(a)
  if (Math.abs(m - 1) < 0.025) return `${dir}, no inflection`
  return m > 1 ? `${dir} sigmoid` : `${dir} anti-sigmoid`
}

export function BetaIccExplorer() {
  const [delta, setDelta] = useState(0.5)
  const [a, setA] = useState(2)
  const [theta0, setTheta0] = useState(0.7)

  const curves = useMemo<XYSeries[]>(() => {
    const mean = THETA.map((t) => {
      const { al, be } = shape(t, delta, a)
      return al / (al + be)
    })
    const q10 = THETA.map((t) => {
      const { al, be } = shape(t, delta, a)
      return betaQuantile(0.1, al, be)
    })
    const q90 = THETA.map((t) => {
      const { al, be } = shape(t, delta, a)
      return betaQuantile(0.9, al, be)
    })
    return [
      { name: 'mean response (ICC)', type: 'line', x: THETA, y: mean, emphasis: true },
      { name: '10% quantile', type: 'line', x: THETA, y: q10, dashed: true, slot: 0 },
      { name: '90% quantile', type: 'line', x: THETA, y: q90, dashed: true, slot: 0 },
    ]
  }, [delta, a])

  const { al, be } = shape(theta0, delta, a)
  const density = useMemo<XYSeries[]>(() => {
    const logB = logGamma(al) + logGamma(be) - logGamma(al + be)
    const y = P_GRID.map((p) => Math.exp((al - 1) * Math.log(p) + (be - 1) * Math.log(1 - p) - logB))
    return [
      { name: `response density at θ = ${formatNumber(theta0)}`, type: 'line', x: P_GRID, y, area: true, slot: 1 },
    ]
  }, [al, be, theta0])
  const yTop = Math.min(8, Math.max(...density[0].y) * 1.05)

  const handles: Handle[] = [
    { kind: 'x', at: delta, label: 'δ', onDrag: (x) => setDelta(Math.round(clamp(x, 0.05, 0.95) * 100) / 100) },
    { kind: 'x', at: theta0, label: 'θ', onDrag: (x) => setTheta0(Math.round(clamp(x, 0.02, 0.98) * 100) / 100) },
  ]

  return (
    <Interactive
      title="β³ item characteristic curves and response distributions"
      caption="The top chart shows the mean response of one item against ability (solid) and the 10% and 90% quantiles of the Beta response distribution (dashed). Drag the line marked δ to move the difficulty and the line marked θ to pick an ability; the bottom chart shows the response density at that ability. Discrimination above 1 gives a sigmoid, between 0 and 1 an anti-sigmoid that is steep near θ = 0 and θ = 1, and a negative value reverses the curve. At θ = δ the response is uniform on (0, 1) whatever the discrimination."
      controls={
        <>
          <ParamSlider label="difficulty δ" value={delta} onChange={setDelta} min={0.05} max={0.95} step={0.01} />
          <ParamSlider label="discrimination a" value={a} onChange={setA} min={-3} max={4} step={0.05} />
          <ParamSlider label="ability θ" value={theta0} onChange={setTheta0} min={0.02} max={0.98} step={0.01} />
        </>
      }
      readout={
        <>
          <Readout label="shape:" value={shapeName(a)} />
          <Readout label="α" value={formatNumber(al)} />
          <Readout label="β" value={formatNumber(be)} />
          <Readout label="mean response" value={formatNumber(al / (al + be))} />
          <Readout label="slope at θ = δ" value={formatNumber(a / (4 * delta * (1 - delta)))} />
        </>
      }
    >
      <div className="space-y-2">
        <XYChart
          series={curves}
          xLabel="ability θ"
          yLabel="response p"
          xRange={[0, 1]}
          yRange={[0, 1]}
          height={280}
          handles={handles}
        />
        <XYChart
          series={density}
          xLabel="response p"
          yLabel="density"
          xRange={[0, 1]}
          yRange={[0, yTop]}
          height={200}
        />
      </div>
    </Interactive>
  )
}
