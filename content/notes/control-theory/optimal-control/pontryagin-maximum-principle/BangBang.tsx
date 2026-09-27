import { useMemo } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, useParam, type XYSeries } from '@/components/viz'

/**
 * Time-optimal control of the double integrator ẋ₁ = x₂, ẋ₂ = u with |u| ≤ 1, solved in closed form: one arc with
 * u = ∓1 until the state meets the switching curve x₁ = −½x₂|x₂|, then one arc with u = ±1 along it to the origin.
 */
function solve(x1: number, x2: number) {
  const s = x1 + 0.5 * x2 * Math.abs(x2)
  // σ is the sign of the first arc's input: −1 above the switching curve, +1 below.
  const sigma = s > 0 ? -1 : 1
  const c = -sigma * x1 + 0.5 * x2 * x2
  const t1 = -sigma * x2 + Math.sqrt(Math.max(c, 0))
  const total = t1 + Math.sqrt(Math.max(c, 0))
  const n = 200
  const t: number[] = []
  const p: number[] = []
  const v: number[] = []
  const u: number[] = []
  for (let i = 0; i <= n; i++) {
    const tau = (total * i) / n
    let pos: number
    let vel: number
    let ui: number
    if (tau <= t1) {
      ui = sigma
      vel = x2 + sigma * tau
      pos = x1 + x2 * tau + 0.5 * sigma * tau * tau
    } else {
      const v1 = x2 + sigma * t1
      const p1 = x1 + x2 * t1 + 0.5 * sigma * t1 * t1
      const d = tau - t1
      ui = -sigma
      vel = v1 - sigma * d
      pos = p1 + v1 * d - 0.5 * sigma * d * d
    }
    t.push(tau)
    p.push(pos)
    v.push(vel)
    u.push(ui)
  }
  return { t, p, v, u, t1, total, sigma }
}

const CURVE_V = Array.from({ length: 121 }, (_, i) => -3 + (6 * i) / 120)

export function BangBang() {
  const x1 = useParam(2, { min: -4, max: 4, step: 0.05 })
  const x2 = useParam(1, { min: -3, max: 3, step: 0.05 })
  const sol = useMemo(() => solve(x1.value, x2.value), [x1.value, x2.value])

  const phase: XYSeries[] = [
    {
      name: 'switching curve x₁ = −½x₂|x₂|',
      type: 'line',
      x: CURVE_V.map((v) => -0.5 * v * Math.abs(v)),
      y: CURVE_V,
      muted: true,
    },
    { name: 'optimal trajectory', type: 'line', x: sol.p, y: sol.v, slot: 0 },
    { name: 'origin', type: 'scatter', x: [0], y: [0], emphasis: true },
  ]
  const input: XYSeries[] = [{ name: 'u(t)', type: 'line', x: sol.t, y: sol.u, slot: 1 }]

  return (
    <Interactive
      title="Time-optimal control is bang-bang"
      caption="The fastest way to bring a unit mass to rest at the origin with a force bounded by |u| ≤ 1. Drag the initial state (position x₁, velocity x₂). The minimum principle allows only u = +1 or u = −1 with at most one switch. Each arc is a parabola; the switch happens where the first parabola meets the switching curve (grey), the only curve along which full thrust in one direction ends exactly at the origin."
      controls={
        <>
          <ParamSlider label="initial position x₁" param={x1} />
          <ParamSlider label="initial velocity x₂" param={x2} />
        </>
      }
      readout={
        <>
          <Readout label="first arc" value={`u = ${sol.sigma > 0 ? '+1' : '−1'}`} />
          <Readout label="switch time" value={formatNumber(sol.t1)} />
          <Readout label="minimum time T" value={formatNumber(sol.total)} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <XYChart
          series={phase}
          xLabel="position x₁"
          yLabel="velocity x₂"
          xRange={[-6, 6]}
          yRange={[-4, 4]}
          equalAspect
          handles={[
            {
              kind: 'point',
              at: [x1.value, x2.value],
              onDrag: ([p, v]) => {
                x1.set(p)
                x2.set(v)
              },
              label: 'initial state',
            },
          ]}
        />
        <XYChart series={input} xLabel="time t" yLabel="input u" yRange={[-1.5, 1.5]} height={260} />
      </div>
    </Interactive>
  )
}
