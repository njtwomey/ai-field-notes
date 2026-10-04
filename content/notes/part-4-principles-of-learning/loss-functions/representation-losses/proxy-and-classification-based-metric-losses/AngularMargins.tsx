import { useMemo } from 'react'
import { Curve, Figure, float, formatNumber, Plot, Readout, slider, useAxis, useFigureState } from 'aifn-render'
import { linspace, toFlat } from 'aifn/foundation/tensor'

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
  const state = useFigureState({
    phiDeg: slider(30, 120, 90, {
      step: 5,
      label: 'angle between class weights φ (degrees)',
      format: (v) => v.toFixed(0),
    }),
    mCos: float(0.35, { min: 0, max: 0.8, step: 0.05, label: 'CosFace margin m' }),
    mArc: slider(0, 1, 0.5, { step: 0.05, label: 'ArcFace margin m (radians)' }),
  })

  const r = useMemo(() => {
    const phi = toRad(state.phiDeg)
    const theta = toFlat(linspace(0, phi, 181))
    const deg = theta.map(toDeg)
    // cos(θ + m) stops decreasing once θ + m > π; the widget's ranges keep θ + m below π.
    const soft = (t: number) => Math.cos(t)
    const cosface = (t: number) => Math.cos(t) - state.mCos
    const arcface = (t: number) => Math.cos(t + state.mArc)
    const series = [
      { name: 'normalised softmax: cos θ', x: deg, y: theta.map(soft), slot: 0 },
      { name: 'CosFace: cos θ − m', x: deg, y: theta.map(cosface), slot: 1 },
      { name: 'ArcFace: cos(θ + m)', x: deg, y: theta.map(arcface), slot: 2 },
      {
        name: 'rival class: cos(φ − θ)',
        x: deg,
        y: theta.map((t) => Math.cos(phi - t)),
        emphasis: true,
        dashed: true,
      },
    ] as const
    const b = [boundary(soft, phi), boundary(cosface, phi), boundary(arcface, phi)].map((v) =>
      v === null ? 'none' : `${formatNumber(toDeg(v))}°`,
    )
    return { series, b }
  }, [state.phiDeg, state.mCos, state.mArc])

  const xAxis = useAxis({ label: "angle θ to class 1's weight (degrees)", hold: 'union' })
  const yAxis = useAxis({ label: 'logit / s', range: [-1.2, 1] })
  return (
    <Figure
      title="Where each margin puts the training decision boundary"
      state={state}
      caption="Two class weight vectors lie φ apart, and a feature lies between them at angle θ from class 1. Each coloured curve is class 1's logit divided by the scale s; the dashed curve is class 2's. During training, class 1 wins where its curve is above the dashed one. Normalised softmax splits the angle in half. ArcFace moves the boundary toward class 1 by exactly m/2 radians, whatever φ is; CosFace moves it by an amount that depends on φ. At test time both classes use plain cosine, so the training margin leaves a gap between the classes."

      readouts={
        <>
          <Readout label="softmax boundary θ" value={r.b[0]} />
          <Readout label="CosFace boundary θ" value={r.b[1]} />
          <Readout label="ArcFace boundary θ" value={r.b[2]} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis}>
        <Curve {...r.series[0]} />
        <Curve {...r.series[1]} />
        <Curve {...r.series[2]} />
        <Curve {...r.series[3]} />
      </Plot>
    </Figure>
  )
}
