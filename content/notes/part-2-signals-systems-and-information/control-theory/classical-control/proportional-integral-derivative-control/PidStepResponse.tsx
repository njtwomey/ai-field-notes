import { useMemo } from 'react'
import {
  Button,
  Curve,
  Figure,
  formatNumber,
  Plot,
  Readout,
  seriesLayers,
  type SeriesSpec,
  setting,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { rk4 } from '../../_shared/control'

const DT = 0.02
const T_END = 40
/** A load disturbance steps onto the plant input at this time, to show how each action rejects it. */
const T_DIST = 20
const DIST = 0.5
const U_MAX = 2

type Result = { t: number[]; y: number[]; u: number[] }

/**
 * Plant 1/(s+1)³ in companion form: y = x₁, ẏ = x₂. The controller is u = Kp·e + Ki·∫e − Kd·ẏ, with the derivative
 * acting on the measurement so that a reference step gives no derivative kick.
 */
function simulate(kp: number, ki: number, kd: number, limit: boolean, antiWindup: boolean): Result {
  const control = (s: number[]) => {
    const e = 1 - s[0]
    const raw = kp * e + ki * s[3] - kd * s[1]
    const u = limit ? Math.max(-U_MAX, Math.min(U_MAX, raw)) : raw
    // Conditional integration: stop integrating while the actuator is saturated and the error would push it further.
    const freeze = limit && antiWindup && Math.abs(raw) > U_MAX && Math.sign(e) === Math.sign(raw)
    return { e, u, freeze }
  }
  const t: number[] = []
  const y: number[] = []
  const u: number[] = []
  let s = [0, 0, 0, 0]
  const steps = Math.round(T_END / DT)
  for (let k = 0; k <= steps; k++) {
    const d = k * DT >= T_DIST ? DIST : 0
    const f = (z: number[]) => {
      const c = control(z)
      return [z[1], z[2], -z[0] - 3 * z[1] - 3 * z[2] + c.u + d, c.freeze ? 0 : c.e]
    }
    t.push(k * DT)
    y.push(s[0])
    u.push(control(s).u)
    s = rk4(f, s, DT)
    if (!Number.isFinite(s[0]) || Math.abs(s[0]) > 50) break
  }
  return { t, y, u }
}

function metrics({ t, y, u }: Result) {
  const before = t.findIndex((v) => v >= T_DIST)
  if (before < 0) return null
  const final = y[before - 1]
  const peak = Math.max(...y.slice(0, before))
  let settle = 0
  for (let i = 0; i < before; i++) if (Math.abs(y[i] - final) > 0.02 * Math.abs(final)) settle = t[i]
  const tail = y.slice(before - 100, before)
  // Bounded and nearly still over the two seconds before the disturbance; otherwise the loop is unstable or too slow.
  const settled = Math.max(...tail) - Math.min(...tail) < 0.1 * Math.max(Math.abs(final), 0.1)
  return {
    overshoot: Math.max(0, ((peak - final) / Math.abs(final)) * 100),
    settle,
    error: 1 - final,
    settled,
    uPeak: Math.max(...u.map(Math.abs)),
  }
}

// Ziegler–Nichols ultimate-gain rules for this plant: Ku = 8, Tu = 2π/√3.
const TU = (2 * Math.PI) / Math.sqrt(3)
const ZN = { kp: 4.8, ki: 4.8 / (TU / 2), kd: 4.8 * (TU / 8) }

export function PidStepResponse() {
  const state = useFigureState({
    kp: slider(0, 10, 1, { step: 0.05, label: 'proportional gain Kp' }),
    ki: slider(0, 5, 0, { step: 0.05, label: 'integral gain Ki' }),
    kd: slider(0, 4, 0, { step: 0.05, label: 'derivative gain Kd' }),
    limit: setting(false, `actuator limit |u| ≤ ${U_MAX}`),
    antiWindup: setting(false, 'anti-windup'),
  })
  const { kp, ki, kd, limit, antiWindup } = state

  const result = useMemo(() => simulate(kp, ki, kd, limit, antiWindup), [kp, ki, kd, limit, antiWindup])
  const m = metrics(result)
  const series = [
    { name: 'reference r', x: [0, T_END], y: [1, 1], muted: true },
    { name: 'output y', x: result.t, y: result.y, slot: 0 },
  ] as const
  const control: SeriesSpec[] = [{ name: 'control u', type: 'line', x: result.t, y: result.u, slot: 1 }]
  const setAll = (p: number, i: number, d: number) => {
    state.set('kp', p)
    state.set('ki', i)
    state.set('kd', d)
  }

  const xAxis = useAxis({ label: 'time t', range: [0, T_END] })
  const yAxis = useAxis({ label: 'output y', range: [-0.5, 2.5] })
  const xAxis2 = useAxis({ label: 'time t', range: [0, T_END] })
  const yAxis2 = useAxis({ label: 'control u', hold: 'union' })
  return (
    <Figure
      title="PID control of a third-order lag"
      state={state}
      caption={`Unit step response of the plant 1/(s+1)³ under PID control, with a load disturbance of ${DIST} added to the plant input at t = ${T_DIST}. With P only the output settles short of the reference, with error 1/(1+Kp). Integral gain drives that error to zero, faster for larger Ki, and does the same for the disturbance, at the cost of more overshoot. Derivative gain adds damping. The Ziegler–Nichols settings (Kp = 4.8, Ki = 2.65, Kd = 2.18) respond fast but overshoot by about half. With the actuator limited to |u| ≤ ${U_MAX}, the integrator keeps accumulating while the input is saturated and the overshoot grows (windup); anti-windup stops the integration while saturated.`}
      controls={
        <>
          <Button variant="outline" size="sm" onClick={() => setAll(4, 0, 0)}>
            P only
          </Button>
          <Button variant="outline" size="sm" onClick={() => setAll(ZN.kp, ZN.ki, ZN.kd)}>
            Ziegler–Nichols
          </Button>
        </>
      }
      readouts={
        m && m.settled ? (
          <>
            <Readout label="overshoot of final value" value={`${formatNumber(m.overshoot)} %`} />
            <Readout
              label="2 % settling time"
              value={m.settle > T_DIST - 1 ? `> ${T_DIST - 1} s` : `${formatNumber(m.settle)} s`}
            />
            <Readout label="steady-state error" value={formatNumber(Math.abs(m.error) < 5e-4 ? 0 : m.error)} />
            <Readout label="peak |u|" value={formatNumber(m.uPeak)} />
          </>
        ) : (
          <Readout label="closed loop" value="unstable or not settled by t = 20" />
        )
      }
    >
      <div className="space-y-2">
        <Plot x={xAxis} y={yAxis} height={260}>
          <Curve {...series[0]} />
          <Curve {...series[1]} />
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={170}>
          {seriesLayers(control)}
        </Plot>
      </div>
    </Figure>
  )
}
