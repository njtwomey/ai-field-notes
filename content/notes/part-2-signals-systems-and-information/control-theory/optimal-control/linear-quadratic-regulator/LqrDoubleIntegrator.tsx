import { useMemo } from 'react'
import {
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  Plot,
  Points,
  Readout,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
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
  const state = useFigureState({
    logQ1: float(0, {
      min: -2,
      max: 2,
      step: 0.1,
      label: 'position weight q₁',
      points_per_decade: 2,
      logTransform: 'value-is-log',
      format: (v) => formatNumber(10 ** v),
    }),
    logQ2: float(-1, {
      min: -2,
      max: 2,
      step: 0.1,
      label: 'velocity weight q₂',
      points_per_decade: 2,
      logTransform: 'value-is-log',
      format: (v) => formatNumber(10 ** v),
    }),
    logR: float(0, {
      min: -2,
      max: 2,
      step: 0.1,
      label: 'control weight r',
      points_per_decade: 2,
      logTransform: 'value-is-log',
      format: (v) => formatNumber(10 ** v),
    }),
    p0: slider(-4, 4, 3, { step: 0.05, onChart: true }),
    v0: slider(-4, 4, 0, { step: 0.05, onChart: true }),
  })

  const sol = useMemo(() => {
    const Q: Mat = [
      [10 ** state.logQ1, 0],
      [0, 10 ** state.logQ2],
    ]
    const R: Mat = [[10 ** state.logR]]
    const { P, K } = dlqr(A, B, Q, R)
    const Acl: Mat = A.map((row, i) => row.map((v, j) => v - B[i][0] * K[0][j]))
    return { Q, R, P, K: K[0], poles: eig2(Acl) }
  }, [state.logQ1, state.logQ2, state.logR])

  const traj = useMemo(() => {
    let x = [state.p0, state.v0]
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
    const predicted = sol.P[0][0] * state.p0 ** 2 + 2 * sol.P[0][1] * state.p0 * state.v0 + sol.P[1][1] * state.v0 ** 2
    return { t, pos, vel, u, cost, predicted }
  }, [sol, state.p0, state.v0])

  const phase = [
    { name: 'state trajectory', x: traj.pos, y: traj.vel, slot: 0 },
    { name: 'target', x: [0], y: [0], muted: true },
  ] as const
  const time = [
    { name: 'position', x: traj.t, y: traj.pos, slot: 0 },
    { name: 'velocity', x: traj.t, y: traj.vel, slot: 1 },
    { name: 'control u', x: traj.t, y: traj.u, slot: 2 },
  ] as const
  const peakU = Math.max(...traj.u.map(Math.abs))
  const upper = sol.poles.find((z) => z.im >= 0) ?? sol.poles[0]

  const xAxis = useAxis({ label: 'position', range: [-4, 4] })
  const yAxis = useAxis({ label: 'velocity', range: [-4, 4], equal: xAxis })
  const xAxis2 = useAxis({ label: 'time t (s)', range: [0, STEPS * H] })
  const yAxis2 = useAxis({ label: 'value', hold: 'union' })
  return (
    <Figure
      title="LQR on a double integrator"
      state={state}
      caption="A unit mass on a frictionless track, pushed by a force u, sampled every 0.1 s. The regulator minimises Σ q₁·position² + q₂·velocity² + r·u². Drag the starting state in the phase plane (left). Raising q₁ relative to r buys a faster return to the origin with larger forces; raising r makes the controller frugal and slow; raising q₂ damps the motion. Whatever the weights, the closed loop is stable. The optimal cost x₀ᵀPx₀ equals the simulated sum once the state has settled within the 10 s shown."

      readouts={
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
        <Plot x={xAxis} y={yAxis}>
          <Curve {...phase[0]} />
          <Points {...phase[1]} />
          <Handle
            kind="point"
            at={[state.p0, state.v0]}
            onDrag={([p, v]) => {
              state.set('p0', p)
              state.set('v0', v)
            }}
            label="initial state"
          />
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={320}>
          <Curve {...time[0]} />
          <Curve {...time[1]} />
          <Curve {...time[2]} />
        </Plot>
      </div>
    </Figure>
  )
}
