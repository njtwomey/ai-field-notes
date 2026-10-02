import { useMemo } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, useParam, type XYSeries } from 'aifn-render'
import { dlqr, eig2, formatComplex, matvec, type Mat } from '../../_shared/control'

const H = 0.1
const STEPS = 100
/** Double integrator (position, velocity) under a zero-order hold with sample time H. */
const A: Mat = [
  [1, H],
  [0, 1],
]
const B: Mat = [[(H * H) / 2], [H]]

export function LqrDoubleIntegrator() {
  const logQ1 = useParam(0, { min: -2, max: 2, step: 0.1 })
  const logQ2 = useParam(-1, { min: -2, max: 2, step: 0.1 })
  const logR = useParam(0, { min: -2, max: 2, step: 0.1 })
  const p0 = useParam(3, { min: -4, max: 4, step: 0.05 })
  const v0 = useParam(0, { min: -4, max: 4, step: 0.05 })

  const sol = useMemo(() => {
    const Q: Mat = [
      [10 ** logQ1.value, 0],
      [0, 10 ** logQ2.value],
    ]
    const R: Mat = [[10 ** logR.value]]
    const { P, K } = dlqr(A, B, Q, R)
    const Acl: Mat = A.map((row, i) => row.map((v, j) => v - B[i][0] * K[0][j]))
    return { Q, R, P, K: K[0], poles: eig2(Acl) }
  }, [logQ1.value, logQ2.value, logR.value])

  const traj = useMemo(() => {
    let x = [p0.value, v0.value]
    const t: number[] = []
    const pos: number[] = []
    const vel: number[] = []
    const u: number[] = []
    let cost = 0
    for (let k = 0; k <= STEPS; k++) {
      const uk = -(sol.K[0] * x[0] + sol.K[1] * x[1])
      t.push(k * H)
      pos.push(x[0])
      vel.push(x[1])
      u.push(uk)
      cost += sol.Q[0][0] * x[0] ** 2 + sol.Q[1][1] * x[1] ** 2 + sol.R[0][0] * uk ** 2
      const ax = matvec(A, x)
      x = [ax[0] + B[0][0] * uk, ax[1] + B[1][0] * uk]
    }
    // The optimal infinite-horizon cost from x₀ is x₀ᵀPx₀; the sum above truncates it at 10 s.
    const predicted = sol.P[0][0] * p0.value ** 2 + 2 * sol.P[0][1] * p0.value * v0.value + sol.P[1][1] * v0.value ** 2
    return { t, pos, vel, u, cost, predicted }
  }, [sol, p0.value, v0.value])

  const phase: XYSeries[] = [
    { name: 'state trajectory', type: 'line', x: traj.pos, y: traj.vel, slot: 0 },
    { name: 'target', type: 'scatter', x: [0], y: [0], muted: true },
  ]
  const time: XYSeries[] = [
    { name: 'position', type: 'line', x: traj.t, y: traj.pos, slot: 0 },
    { name: 'velocity', type: 'line', x: traj.t, y: traj.vel, slot: 1 },
    { name: 'control u', type: 'line', x: traj.t, y: traj.u, slot: 2 },
  ]
  const peakU = Math.max(...traj.u.map(Math.abs))
  const upper = sol.poles.find((z) => z.im >= 0) ?? sol.poles[0]

  return (
    <Interactive
      title="LQR on a double integrator"
      caption="A unit mass on a frictionless track, pushed by a force u, sampled every 0.1 s. The regulator minimises Σ q₁·position² + q₂·velocity² + r·u². Drag the starting state in the phase plane (left). Raising q₁ relative to r buys a faster return to the origin with larger forces; raising r makes the controller frugal and slow; raising q₂ damps the motion. Whatever the weights, the closed loop is stable. The optimal cost x₀ᵀPx₀ equals the simulated sum once the state has settled within the 10 s shown."
      controls={
        <>
          <ParamSlider label="position weight q₁" param={logQ1} format={(v) => formatNumber(10 ** v)} />
          <ParamSlider label="velocity weight q₂" param={logQ2} format={(v) => formatNumber(10 ** v)} />
          <ParamSlider label="control weight r" param={logR} format={(v) => formatNumber(10 ** v)} />
        </>
      }
      readout={
        <>
          <Readout label="gain K" value={`[${formatNumber(sol.K[0])}, ${formatNumber(sol.K[1])}]`} />
          <Readout
            label="closed-loop eigenvalues"
            value={
              formatComplex(upper, 3) +
              (Math.abs(upper.im) > 1e-9 ? ' and conjugate' : `, ${formatComplex(sol.poles[1], 3)}`)
            }
          />
          <Readout label="cost x₀ᵀPx₀" value={formatNumber(traj.predicted)} />
          <Readout label="simulated cost (10 s)" value={formatNumber(traj.cost)} />
          <Readout label="peak |u|" value={formatNumber(peakU)} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <XYChart
          series={phase}
          xLabel="position"
          yLabel="velocity"
          xRange={[-4, 4]}
          yRange={[-4, 4]}
          equalAspect
          handles={[
            {
              kind: 'point',
              at: [p0.value, v0.value],
              onDrag: ([p, v]) => {
                p0.set(p)
                v0.set(v)
              },
              label: 'initial state',
            },
          ]}
        />
        <XYChart series={time} xLabel="time t (s)" yLabel="value" xRange={[0, STEPS * H]} height={320} />
      </div>
    </Interactive>
  )
}
