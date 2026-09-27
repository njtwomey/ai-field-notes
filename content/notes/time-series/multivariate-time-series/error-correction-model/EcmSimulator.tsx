import { useMemo } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, useParam, type XYSeries } from '@/components/viz'
import { rng } from '@/lib/math'

const T = 300

export function EcmSimulator() {
  const alphaY = useParam(-0.2, { min: -1, max: 0, step: 0.05 })
  const alphaX = useParam(0, { min: 0, max: 1, step: 0.05 })
  const seed = useParam(4, { min: 1, max: 30, step: 1 })

  const sim = useMemo(() => {
    const g = rng(seed.value)
    const x = [0]
    const y = [0]
    for (let t = 1; t < T; t++) {
      // Equilibrium y = x. A positive gap pulls y down (alpha_y < 0) and pushes x up (alpha_x > 0).
      const gap = y[t - 1] - x[t - 1]
      x.push(x[t - 1] + alphaX.value * gap + g.normal())
      y.push(y[t - 1] + alphaY.value * gap + g.normal())
    }
    return { x, y, gap: y.map((v, i) => v - x[i]) }
  }, [alphaY.value, alphaX.value, seed.value])

  const charts = useMemo(() => {
    const t = Array.from({ length: T }, (_, i) => i)
    const levels: XYSeries[] = [
      { name: 'x_t', type: 'line', x: t, y: sim.x, slot: 0 },
      { name: 'y_t', type: 'line', x: t, y: sim.y, slot: 1 },
    ]
    const gap: XYSeries[] = [
      { name: 'gap y_t − x_t', type: 'line', x: t, y: sim.gap, slot: 2 },
      { name: 'equilibrium', type: 'line', x: [0, T - 1], y: [0, 0], muted: true, dashed: true },
    ]
    return { levels, gap }
  }, [sim])

  // The gap follows an AR(1): z_t = (1 + alpha_y - alpha_x) z_{t-1} + noise.
  const root = 1 + alphaY.value - alphaX.value
  const stable = Math.abs(root) < 1
  const halfLife = !stable
    ? 'never'
    : root <= 0
      ? 'under 1 step'
      : `${formatNumber(Math.log(0.5) / Math.log(root))} steps`

  return (
    <Interactive
      title="Error correction between two random walks"
      caption="Both series take a Gaussian step each period. In addition, y corrects by α_y times the previous gap y − x, and x by α_x times the same gap. With α_y = α_x = 0 the two are independent random walks and drift apart. With any correction the gap is stationary, so the series are cointegrated with β = 1: they wander together. The gap follows an AR(1) with coefficient 1 + α_y − α_x, so either series, or both, can do the correcting."
      controls={
        <>
          <ParamSlider label="adjustment of y, α_y" param={alphaY} format={(v) => v.toFixed(2)} />
          <ParamSlider label="adjustment of x, α_x" param={alphaX} format={(v) => v.toFixed(2)} />
          <ParamSlider label="seed" param={seed} format={(v) => String(v)} withArrows />
        </>
      }
      readout={
        <>
          <Readout label="AR coefficient of the gap" value={formatNumber(root)} />
          <Readout label="half-life of a deviation" value={halfLife} />
          <Readout label="cointegrated" value={stable ? 'yes' : 'no'} />
        </>
      }
    >
      <div className="space-y-4">
        <XYChart series={charts.levels} xLabel="t" yLabel="level" xRange={[0, T - 1]} height={240} />
        <XYChart series={charts.gap} xLabel="t" yLabel="gap" xRange={[0, T - 1]} height={160} />
      </div>
    </Interactive>
  )
}
