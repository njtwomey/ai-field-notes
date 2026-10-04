import { useMemo } from 'react'
import {
  Curve,
  Figure,
  float,
  formatNumber,
  int,
  Plot,
  Readout,
  Segments,
  setting,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { dlqr, type Mat } from '../../_shared/control'

const H = 0.2
const STEPS = 60
const X0 = [-5, 0]
/** Double integrator (position, velocity) under a zero-order hold. */
const A: Mat = [
  [1, H],
  [0, 1],
]
const B = [(H * H) / 2, H]
const Q: Mat = [
  [1, 0],
  [0, 0],
]
const R = 0.01
const LQR = dlqr(A, [[B[0]], [B[1]]], Q, [[R]])

const step = (x: number[], u: number) => [x[0] + H * x[1] + B[0] * u, x[1] + B[1] * u]

/**
 * Condensed QP for horizon N: predicted states are X = Φx + ΓU, and the cost is ½UᵀHU + (Fx)ᵀU + const, where
 * H = 2(ΓᵀQ̄Γ + R·I) and F = 2ΓᵀQ̄Φ. Q̄ is block diagonal with Q, and P (or Q) on the last block.
 */
function condense(n: number, terminal: boolean) {
  // Aʲ⁻¹⁻ⁱB for the double integrator is (B₀ + (j−1−i)·H·B₁, B₁).
  const gamma = (j: number, i: number) => (i < j ? [B[0] + (j - 1 - i) * H * B[1], B[1]] : [0, 0])
  const weight = (j: number): Mat => (j === n && terminal ? LQR.P : Q)
  const Hm: number[][] = Array.from({ length: n }, () => new Array(n).fill(0))
  const F: number[][] = Array.from({ length: n }, () => [0, 0])
  for (let j = 1; j <= n; j++) {
    const W = weight(j)
    // Aʲ = [[1, jH], [0, 1]]
    const Aj = [
      [1, j * H],
      [0, 1],
    ]
    for (let a = 0; a < n; a++) {
      const ga = gamma(j, a)
      if (!ga[0] && !ga[1]) continue
      const wg = [W[0][0] * ga[0] + W[0][1] * ga[1], W[1][0] * ga[0] + W[1][1] * ga[1]]
      for (let b = 0; b < n; b++) {
        const gb = gamma(j, b)
        Hm[a][b] += 2 * (wg[0] * gb[0] + wg[1] * gb[1])
      }
      for (let c = 0; c < 2; c++) F[a][c] += 2 * (wg[0] * Aj[0][c] + wg[1] * Aj[1][c])
    }
  }
  for (let a = 0; a < n; a++) Hm[a][a] += 2 * R
  return { Hm, F }
}

/** Box-constrained QP by cyclic coordinate descent, which is exact per coordinate and converges for H ≻ 0. */
function solveBoxQp(Hm: number[][], f: number[], umax: number, warm: number[]): number[] {
  const u = [...warm]
  for (let sweep = 0; sweep < 300; sweep++) {
    let change = 0
    for (let i = 0; i < u.length; i++) {
      let g = f[i]
      for (let j = 0; j < u.length; j++) if (j !== i) g += Hm[i][j] * u[j]
      const next = Math.max(-umax, Math.min(umax, -g / Hm[i][i]))
      change = Math.max(change, Math.abs(next - u[i]))
      u[i] = next
    }
    if (change < 1e-9) break
  }
  return u
}

function runMpc(n: number, umax: number, terminal: boolean) {
  const { Hm, F } = condense(n, terminal)
  let x = [...X0]
  let warm = new Array(n).fill(0)
  const pos: number[] = []
  const u: number[] = []
  const plans: number[][] = []
  let cost = 0
  for (let k = 0; k < STEPS; k++) {
    const f = F.map((row) => row[0] * x[0] + row[1] * x[1])
    const plan = solveBoxQp(Hm, f, umax, warm)
    // The predicted positions of this plan, for the "plan at step k" overlay.
    let xp = [...x]
    plans.push([x[0], ...plan.map((v) => (xp = step(xp, v))[0])])
    pos.push(x[0])
    u.push(plan[0])
    cost += x[0] ** 2 + R * plan[0] ** 2
    x = step(x, plan[0])
    warm = [...plan.slice(1), plan[plan.length - 1]]
  }
  pos.push(x[0])
  return { pos, u, plans, cost }
}

function runSaturatedLqr(umax: number) {
  let x = [...X0]
  const pos: number[] = []
  const u: number[] = []
  let cost = 0
  for (let k = 0; k < STEPS; k++) {
    const uk = Math.max(-umax, Math.min(umax, -(LQR.K[0][0] * x[0] + LQR.K[0][1] * x[1])))
    pos.push(x[0])
    u.push(uk)
    cost += x[0] ** 2 + R * uk ** 2
    x = step(x, uk)
  }
  pos.push(x[0])
  return { pos, u, cost }
}

const T = Array.from({ length: STEPS + 1 }, (_, k) => k * H)

export function MpcHorizon() {
  const state = useFigureState({
    horizon: int(3, { min: 1, max: 30, step: 1, label: 'horizon N (steps)', format: (v) => String(v) }),
    umax: float(1, { min: 0.25, max: 3, step: 0.05, label: 'input limit u_max' }),
    terminal: setting(true, 'terminal cost xᵀPx'),
    at: float(0, { min: 0, max: STEPS - 1, step: 1, label: 'show plan at step k', format: (v) => String(v) }),
  })

  const mpc = useMemo(
    () => runMpc(state.horizon, state.umax, state.terminal),
    [state.horizon, state.umax, state.terminal],
  )
  const sat = useMemo(() => runSaturatedLqr(state.umax), [state.umax])

  const k = state.at
  const plan = mpc.plans[k]
  const position = [
    { name: 'target', x: [0, STEPS * H], y: [0, 0], muted: true },
    { name: 'MPC', x: T, y: mpc.pos, slot: 0 },
    { name: 'saturated LQR', x: T, y: sat.pos, slot: 1, dashed: true },
    { name: `plan made at step k`, x: plan.map((_, j) => (k + j) * H), y: plan, slot: 2 },
  ] as const
  const input = [
    { name: 'MPC input', x: T.slice(0, STEPS), y: mpc.u, slot: 0 },
    { name: 'saturated LQR input', x: T.slice(0, STEPS), y: sat.u, slot: 1, dashed: true },
  ] as const
  const overshoot = Math.max(0, ...mpc.pos)

  const xAxis = useAxis({ label: 'time t (s)', range: [0, STEPS * H] })
  const yAxis = useAxis({ label: 'position', range: [-6, 5] })
  const xAxis2 = useAxis({ label: 'time t (s)', range: [0, STEPS * H] })
  const yAxis2 = useAxis({ label: 'input u', hold: 'union' })
  return (
    <Figure
      title="Receding horizon with an input limit"
      state={state}
      caption="A unit mass starts at position −5, at rest, and must be brought to 0 with a force limited to |u| ≤ u_max; sample time 0.2 s, stage cost position² + 0.01·u². At every step MPC solves a quadratic program over the next N inputs, applies the first, and re-plans. The coloured segment is the plan made at step k: the controller only ever executes its first move. The dashed curve clips the unconstrained LQR input at the limit instead, and overshoots because it does not know the limit is coming. Short horizons without a terminal cost can fail to converge at all; the terminal cost xᵀPx from the LQR makes even N = 1 behave like saturated LQR, and a longer horizon lets the controller brake in time."

      readouts={
        <>
          <Readout label="cost, MPC" value={formatNumber(mpc.cost)} />
          <Readout label="cost, saturated LQR" value={formatNumber(sat.cost)} />
          <Readout label="MPC overshoot past 0" value={formatNumber(overshoot)} />
          <Readout label="final position (12 s)" value={formatNumber(mpc.pos[STEPS])} />
        </>
      }
    >
      <div className="space-y-2">
        <Plot x={xAxis} y={yAxis} height={260}>
          <Curve {...position[0]} />
          <Curve {...position[1]} />
          <Curve {...position[2]} />
          <Curve {...position[3]} />
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={180}>
          <Curve {...input[0]} />
          <Curve {...input[1]} />
          <Segments
            segments={[
              { from: [0, state.umax], to: [STEPS * H, state.umax] },
              { from: [0, -state.umax], to: [STEPS * H, -state.umax] },
            ]}
          />
        </Plot>
      </div>
    </Figure>
  )
}
