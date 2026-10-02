/**
 * "Control: from LQR to MPC": the sampled double integrator under bounded force, regulated by receding-horizon MPC
 * (core `mpcController` + `recedingHorizon`) and by the LQR gain with its input clipped. The player walks the closed
 * loop; at each step the MPC plan (predicted states and inputs over the horizon) is drawn against what was applied.
 */
import { dlqr, mpcController, recedingHorizon, type RecedingHorizonState } from 'aifn/dynamics/control'
import { toFlat, toRows } from 'aifn/foundation/tensor'
import { trace } from 'aifn/foundation/trace'
import { useMemo } from 'react'
import { Player, usePlayhead } from '@lab/controls'
import { Dashboard, DashboardCell, DashboardRow, Figure } from '@lab/layout'
import { row, slider, toggle, useFigureState } from '@lab/state'
import { Annotation, Curve, Handle, Plot, Plots, Points, Readout, formatNumber, useAxis } from '@lab/viz'

const fmt = (v: number) => formatNumber(v)
const DT = 0.1
const STEPS = 80
const A = [
  [1, DT],
  [0, 1],
]
const B = [[0.5 * DT * DT], [DT]]

type Run = {
  t: number[]
  p: number[]
  v: number[]
  u: number[]
  cost: number
  plans: { p: number[]; v: number[]; u: number[] }[]
  status: string[]
  /** The slack the applied plan's first predicted step needed (soft bound), per step. */
  slack: number[]
}

/** The receding-horizon loop, recorded at every step with its plan. */
function mpcRun(
  horizon: number,
  uMax: number,
  vMax: number | null,
  R: number,
  x0: [number, number],
  rho: number | null,
): Run {
  const Q = [
    [1, 0],
    [0, 0.1],
  ]
  const c = mpcController({
    A,
    B,
    Q,
    R: [[R]],
    horizon,
    uMin: -uMax,
    uMax,
    ...(vMax !== null ? { xMin: [-Infinity, -vMax], xMax: [Infinity, vMax] } : {}),
    // A purely quadratic slack penalty: the bound gives a little whenever leaning on it pays.
    ...(rho !== null ? { soft: { quadratic: rho, linear: 0 } } : {}),
  })
  const tr = trace(recedingHorizon(c), { x0 }, STEPS, { keep: 'all' })
  const states = tr.steps as readonly RecedingHorizonState[]
  const run: Run = { t: [], p: [], v: [], u: [], cost: 0, plans: [], status: [], slack: [] }
  for (const s of states) {
    const x = toFlat(s.x)
    const u = toFlat(s.u)[0]
    run.t.push(s.t * DT)
    run.p.push(x[0])
    run.v.push(x[1])
    run.u.push(u)
    run.cost += x[0] * x[0] + 0.1 * x[1] * x[1] + R * u * u
    const X = toRows(s.predictedX)
    run.plans.push({ p: X.map((r) => r[0]), v: X.map((r) => r[1]), u: toFlat(s.predictedU) })
    run.status.push(s.status)
    run.slack.push(Math.max(0, ...X.slice(1).map((r) => Math.abs(r[1]) - (vMax ?? Infinity))))
  }
  return run
}

/** The LQR gain with the force clipped to ±uMax (no prediction, so no knowledge of the bounds). */
function clippedLqrRun(uMax: number, R: number, x0: [number, number]) {
  const K = toFlat(
    dlqr(
      { A, B },
      [
        [1, 0],
        [0, 0.1],
      ],
      [[R]],
    ).K,
  )
  let [p, v] = x0
  const out = { t: [] as number[], p: [] as number[], v: [] as number[], u: [] as number[], cost: 0, K }
  for (let k = 0; k <= STEPS; k++) {
    const u = Math.max(-uMax, Math.min(uMax, -(K[0] * p + K[1] * v)))
    out.t.push(k * DT)
    out.p.push(p)
    out.v.push(v)
    out.u.push(u)
    out.cost += p * p + 0.1 * v * v + R * u * u
    ;[p, v] = [p + DT * v + 0.5 * DT * DT * u, v + DT * u]
  }
  return out
}

export function MpcSpecimen() {
  const state = useFigureState({
    design: row('1 · MPC design', {
      horizon: slider(1, 40, 12, { label: 'horizon N (steps)', step: 1 }),
      logR: slider(-2, 1, -1, { label: 'log₁₀ force weight R', step: 0.1 }),
    }),
    bounds: row('2 · constraints', {
      uMax: slider(0.2, 3, 1, { label: 'force bound |u| ≤ u_max', step: 0.05 }),
      vBound: toggle(true, 'velocity bound |v| ≤ v_max'),
      vMax: slider(0.3, 3, 1.2, { label: 'v_max', step: 0.05 }),
      soft: toggle(false, 'soft velocity bound (slack)'),
      logRho: slider(-1, 4, 1, { label: 'log₁₀ slack weight ρ', step: 0.1 }),
    }),
    p0: slider(-5, 5, 4, { onChart: true, label: 'start position p₀' }),
    v0: slider(-1.2, 1.2, 0, { onChart: true, label: 'start velocity v₀' }),
  })
  const { horizon, logR } = state.design
  const { uMax, vBound, vMax, soft, logRho } = state.bounds
  const R = 10 ** logR
  const rho = vBound && soft ? 10 ** logRho : null
  const mpc = useMemo(
    () => mpcRun(horizon, uMax, vBound ? vMax : null, R, [state.p0, state.v0], rho),
    [horizon, uMax, vBound, vMax, R, state.p0, state.v0, rho],
  )
  const lqr = useMemo(() => clippedLqrRun(uMax, R, [state.p0, state.v0]), [uMax, R, state.p0, state.v0])
  const [i, setI] = usePlayhead(mpc.t.length)
  const plan = mpc.plans[i]
  const planT = plan.p.map((_, k) => mpc.t[i] + k * DT)
  const planUT = plan.u.map((_, k) => mpc.t[i] + k * DT)
  const vMaxSeen = (vs: number[]) => Math.max(...vs.map(Math.abs))
  const infeasible = mpc.status.filter((s) => s !== 'optimal').length
  const time = useAxis({ label: 't (s)', range: [0, STEPS * DT + 0.1 * horizon * DT] })
  const pos = useAxis({ label: 'position p', hold: 'union', key: `${state.p0}` })
  const vel = useAxis({ label: 'velocity v', hold: 'union', key: `${state.p0}` })
  const force = useAxis({ label: 'force u', range: [-3.2, 3.2] })
  const pp = useAxis({ label: 'position p', range: [-5.5, 5.5] })
  const pv = useAxis({ label: 'velocity v', range: [-3.2, 3.2] })
  const slice = (xs: number[]) => xs.slice(0, i + 1)
  return (
    <Figure
      title="Control: from LQR to MPC"
      purpose="LQR is optimal with no constraints; clipping its force respects the actuator but not the state bound. MPC solves a constrained quadratic program over a horizon at every step and applies only the first input: the receding horizon."
      defaultSize="XL"
      state={state}
      controls={
        <Player
          value={i}
          onChange={setI}
          count={mpc.t.length}
          format={(k) => `t = ${(k * DT).toFixed(1)} s`}
          label="3 · time t"
        />
      }
      readouts={{
        'at t': (
          <>
            <Readout label="t" value={`${fmt(mpc.t[i])} s`} />
            <Readout label="MPC force u(t)" value={fmt(mpc.u[i])} />
            <Readout label="plan status" value={mpc.status[i]} />
            {rho !== null && <Readout label="planned excess over v_max" value={fmt(mpc.slack[i])} />}
          </>
        ),
        'whole run': (
          <>
            <Readout label="cost Σ p² + 0.1v² + Ru² (MPC, clipped LQR)" value={`${fmt(mpc.cost)}, ${fmt(lqr.cost)}`} />
            <Readout label="max |v| (MPC, clipped LQR)" value={`${fmt(vMaxSeen(mpc.v))}, ${fmt(vMaxSeen(lqr.v))}`} />
            <Readout label="LQR gain K" value={lqr.K.map(fmt).join(', ')} />
            <Readout label="steps with no feasible plan" value={infeasible} />
          </>
        ),
      }}
      caption={`The sampled double integrator (Δt = ${DT} s) starts at (p₀, v₀) and is driven to rest at the origin with Q = diag(1, 0.1) and force weight R. Drag the start point on the phase plane. Play to walk the closed loop: solid lines are what happened up to t (MPC in the first colour, LQR with its force clipped to ±u_max in the third), the dashed line is the MPC plan made at t over its horizon of N steps, and grey shows each whole run. Dashed horizontal lines are the bounds. Clipped LQR saturates and overshoots the velocity bound; MPC brakes early because its plan sees the bound coming. With a short horizon MPC is myopic and can still run into the bound (a step with no feasible plan drops the state bound and is counted); with a long one its first move approaches the constrained optimum. With no active constraint MPC's first input equals the LQR input, since its terminal weight is the LQR cost-to-go. Switch on the soft velocity bound to give the bound a slack s ≥ 0 per step, |v| ≤ v_max + s, paid for by ρs² in the objective: every plan is then feasible (no step drops the bound), and the plan leans past v_max by as much as the trade with the tracking cost is worth, less as ρ grows. Adding a linear penalty λs with λ above the bound's multiplier would make the penalty exact (core option soft.linear).`}
    >
      <Dashboard>
        <DashboardRow minHeight={420}>
          <DashboardCell>
            <Plot x={pp} y={pv} title="phase plane">
              {vBound && <Annotation y={vMax} dashed />}
              {vBound && <Annotation y={-vMax} dashed />}
              <Curve name="whole runs" x={[...mpc.p, NaN, ...lqr.p]} y={[...mpc.v, NaN, ...lqr.v]} muted />
              <Curve name="MPC" x={slice(mpc.p)} y={slice(mpc.v)} slot={0} />
              <Curve name="clipped LQR" x={slice(lqr.p)} y={slice(lqr.v)} slot={2} />
              <Curve name="MPC plan at t" x={plan.p} y={plan.v} slot={0} dashed live />
              <Points name="MPC" x={[mpc.p[i]]} y={[mpc.v[i]]} slot={0} live />
              <Points name="clipped LQR" x={[lqr.p[i]]} y={[lqr.v[i]]} slot={2} live />
              <Handle {...state.handle(['p0', 'v0'], { label: 'start' })} />
            </Plot>
          </DashboardCell>
          <DashboardCell ratio={1.3}>
            <Plots rows={3} hoverGroup>
              <Plot x={time} y={pos} legend={false}>
                <Curve name="whole runs" x={[...mpc.t, NaN, ...lqr.t]} y={[...mpc.p, NaN, ...lqr.p]} muted />
                <Curve name="MPC" x={slice(mpc.t)} y={slice(mpc.p)} slot={0} />
                <Curve name="clipped LQR" x={slice(lqr.t)} y={slice(lqr.p)} slot={2} />
                <Curve name="MPC plan" x={planT} y={plan.p} slot={0} dashed live />
              </Plot>
              <Plot x={time} y={vel} legend={false}>
                {vBound && <Annotation y={vMax} dashed />}
                {vBound && <Annotation y={-vMax} dashed />}
                <Curve name="whole runs" x={[...mpc.t, NaN, ...lqr.t]} y={[...mpc.v, NaN, ...lqr.v]} muted />
                <Curve name="MPC" x={slice(mpc.t)} y={slice(mpc.v)} slot={0} />
                <Curve name="clipped LQR" x={slice(lqr.t)} y={slice(lqr.v)} slot={2} />
                <Curve name="MPC plan" x={planT} y={plan.v} slot={0} dashed live />
              </Plot>
              <Plot x={time} y={force} legend={false}>
                <Annotation y={uMax} dashed />
                <Annotation y={-uMax} dashed />
                <Curve name="MPC" x={slice(mpc.t)} y={slice(mpc.u)} slot={0} />
                <Curve name="clipped LQR" x={slice(lqr.t)} y={slice(lqr.u)} slot={2} />
                <Curve name="MPC plan" x={planUT} y={plan.u} slot={0} dashed live />
              </Plot>
            </Plots>
          </DashboardCell>
        </DashboardRow>
      </Dashboard>
    </Figure>
  )
}
