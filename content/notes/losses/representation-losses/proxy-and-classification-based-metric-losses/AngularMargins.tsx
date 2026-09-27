import { useMemo } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, useParam, type XYSeries } from '@/components/viz'
import { linspace } from '@/lib/math'

const toRad = (deg: number) => (deg * Math.PI) / 180
const toDeg = (rad: number) => (rad * 180) / Math.PI

/** Bisection for the angle θ in [0, φ] where the target logit g(θ) equals the rival logit cos(φ − θ). */
function boundary(g: (t: number) => number, phi: number): number | null {
  const h = (t: number) => g(t) - Math.cos(phi - t)
  let lo = 0
  let hi = phi
  if (h(lo) < 0) return null
  if (h(hi) > 0) return phi
  for (let i = 0; i < 50; i++) {
    const mid = (lo + hi) / 2
    if (h(mid) > 0) lo = mid
    else hi = mid
  }
  return (lo + hi) / 2
}

/**
 * Two classes whose weight vectors are φ apart. A feature in their plane at angle θ from class 1's weight is at angle
 * φ − θ from class 2's. Each curve is the class-1 logit (divided by the scale s) under one loss; where it crosses the
 * rival's logit is that loss's training decision boundary.
 */
export function AngularMargins() {
  const phiDeg = useParam(90, { min: 30, max: 120, step: 5 })
  const mCos = useParam(0.35, { min: 0, max: 0.8, step: 0.05 })
  const mArc = useParam(0.5, { min: 0, max: 1, step: 0.05 })

  const r = useMemo(() => {
    const phi = toRad(phiDeg.value)
    const theta = linspace(0, phi, 181)
    const deg = theta.map(toDeg)
    // cos(θ + m) stops decreasing once θ + m > π; the widget's ranges keep θ + m below π.
    const soft = (t: number) => Math.cos(t)
    const cosface = (t: number) => Math.cos(t) - mCos.value
    const arcface = (t: number) => Math.cos(t + mArc.value)
    const series: XYSeries[] = [
      { name: 'normalised softmax: cos θ', type: 'line', x: deg, y: theta.map(soft), slot: 0 },
      { name: 'CosFace: cos θ − m', type: 'line', x: deg, y: theta.map(cosface), slot: 1 },
      { name: 'ArcFace: cos(θ + m)', type: 'line', x: deg, y: theta.map(arcface), slot: 2 },
      {
        name: 'rival class: cos(φ − θ)',
        type: 'line',
        x: deg,
        y: theta.map((t) => Math.cos(phi - t)),
        emphasis: true,
        dashed: true,
      },
    ]
    const b = [boundary(soft, phi), boundary(cosface, phi), boundary(arcface, phi)].map((v) =>
      v === null ? 'none' : `${formatNumber(toDeg(v))}°`,
    )
    return { series, b }
  }, [phiDeg.value, mCos.value, mArc.value])

  return (
    <Interactive
      title="Where each margin puts the training decision boundary"
      caption="Two class weight vectors lie φ apart, and a feature lies between them at angle θ from class 1. Each coloured curve is class 1's logit divided by the scale s; the dashed curve is class 2's. During training, class 1 wins where its curve is above the dashed one. Normalised softmax splits the angle in half. ArcFace moves the boundary toward class 1 by exactly m/2 radians, whatever φ is; CosFace moves it by an amount that depends on φ. At test time both classes use plain cosine, so the training margin leaves a gap between the classes."
      controls={
        <>
          <ParamSlider label="angle between class weights φ (degrees)" param={phiDeg} format={(v) => v.toFixed(0)} />
          <ParamSlider label="CosFace margin m" param={mCos} />
          <ParamSlider label="ArcFace margin m (radians)" param={mArc} />
        </>
      }
      readout={
        <>
          <Readout label="softmax boundary θ" value={r.b[0]} />
          <Readout label="CosFace boundary θ" value={r.b[1]} />
          <Readout label="ArcFace boundary θ" value={r.b[2]} />
        </>
      }
    >
      <XYChart series={r.series} xLabel="angle θ to class 1's weight (degrees)" yLabel="logit / s" yRange={[-1.2, 1]} />
    </Interactive>
  )
}
