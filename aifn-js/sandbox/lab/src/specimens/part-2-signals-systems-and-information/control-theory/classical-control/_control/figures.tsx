import { transferFunction } from 'aifn/systems'
import { pidLoop, type AntiWindup } from 'aifn-methods/dynamics/control'
import { toFlat, type Tensor } from 'aifn/foundation/tensor'
import { trace } from 'aifn/foundation/trace'
import { useMemo } from 'react'
import { Player, usePlayhead } from '@lab/controls'
import { ControlRow, Figure } from '@lab/layout'
import { choice, row, slider, toggle, useFigureState } from '@lab/state'
import { Curve, formatNumber, Handle, Plot, Plots, Points, Readout, useAxis } from '@lab/viz'

const flat = (t: Tensor) => toFlat(t)
const fmt = (v: number) => formatNumber(v)

/** Several traces as one NaN-separated line: the whole run drawn faintly behind the part played so far. */
function ghost(t: readonly number[], ys: readonly (readonly number[])[]) {
  const x: number[] = []
  const y: number[] = []
  for (const v of ys) {
    x.push(...t, NaN)
    y.push(...v, NaN)
  }
  return { x, y }
}

/** A trace up to position i (inclusive), and its point at i. */
const upTo = (t: readonly number[], y: readonly number[], i: number) => ({ x: t.slice(0, i + 1), y: y.slice(0, i + 1) })
const timeLabel = (t: readonly number[]) => (i: number) => `t = ${(t[i] ?? 0).toFixed(2)} s`

// ── PID step response ────────────────────────────────────────────────────────────────────────────────────────────────

const PLANT = transferFunction([1], [1, 3, 3, 1]) // 1/(s + 1)³

export function PidSpecimen() {
  const state = useFigureState({
    gains: row('1 · gains', {
      kp: slider(0, 6, 2, { label: 'k_p', step: 0.05 }),
      ki: slider(0, 3, 1, { label: 'k_i', step: 0.05 }),
      kd: slider(0, 4, 0.5, { label: 'k_d', step: 0.05 }),
    }),
    actuator: row('2 · actuator', {
      limit: toggle(true, 'limit |u| ≤ 1.2'),
      antiWindup: choice(
        [
          { value: 'none', label: 'none' },
          { value: 'clamp', label: 'conditional integration' },
          { value: 'back-calculation', label: 'back-calculation' },
        ],
        'none',
        { label: 'anti-windup' },
      ),
    }),
    setpoint: slider(0.2, 1.1, 1, { onChart: true, label: 'set point' }),
  })
  const { kp, ki, kd } = state.gains
  const { limit } = state.actuator
  const antiWindup = state.actuator.antiWindup as AntiWindup
  const { setpoint } = state
  const run = useMemo(
    () =>
      trace(
        pidLoop(
          PLANT,
          { kp, ki, kd, filter: 0.05 },
          {
            dt: 0.02,
            tEnd: 25,
            setpoint: (t) => (t >= 1 ? setpoint : 0),
            uMin: limit ? -1.2 : undefined,
            uMax: limit ? 1.2 : undefined,
            antiWindup,
            derivativeOn: 'measurement',
          },
        ),
        {},
        2000,
        { record: { y: (s) => s.y, r: (s) => s.r, u: (s) => s.u, uRaw: (s) => s.uRaw, t: (s) => s.time } },
      ),
    [kp, ki, kd, limit, antiWindup, setpoint],
  )
  const t = flat(run.series.t)
  const y = flat(run.series.y)
  const u = flat(run.series.u)
  const uRaw = flat(run.series.uRaw)
  const r = flat(run.series.r)
  const peak = Math.max(...y)
  const saturatedFor = run.steps.filter((s) => s.saturated).length * 0.02
  const [i, setI] = usePlayhead(t.length)
  const ghostY = useMemo(() => ghost(t, [y]), [t, y])
  const ghostU = useMemo(() => ghost(t, [uRaw, u]), [t, uRaw, u])
  const yNow = upTo(t, y, i)
  const uNow = upTo(t, u, i)
  const rawNow = upTo(t, uRaw, i)
  const time = useAxis({ label: 't (s)' })
  const out = useAxis({ label: 'output', hold: 'union' })
  const inp = useAxis({ label: 'input', hold: 'union' })
  return (
    <Figure
      title="PID step response with actuator limits"
      purpose="Proportional action speeds the response, integral action removes the steady-state error, derivative action damps it; when the actuator saturates the integrator winds up and overshoots unless anti-windup stops it."
      state={state}
      defaultSize="L"
      controls={
        <>
          <ControlRow label="3 · time">
            <Player value={i} onChange={setI} count={t.length} format={timeLabel(t)} label="t" />
          </ControlRow>
        </>
      }
      readouts={
        <>
          <Readout label="y(t), u(t)" value={`${fmt(y[i])}, ${fmt(u[i])}`} />
          <Readout label="integral state" value={fmt(run.steps[i]?.integral ?? NaN)} />
          <Readout label="overshoot" value={`${fmt(Math.max(0, (100 * (peak - setpoint)) / setpoint))} %`} />
          <Readout label="final error" value={fmt(run.steps.at(-1)!.e)} />
          <Readout label="time saturated" value={`${fmt(saturatedFor)} s`} />
        </>
      }
      caption="Plant 1/(s + 1)³ under a PID controller (derivative on the measurement, filtered with T_f = 0.05 s) sampled every 0.02 s; the set point steps at t = 1 s. Play to watch the loop respond, with the whole run in grey. Drag the dashed set-point line on the top panel (up to 1.1, below the actuator's reach). With the limit on and no anti-windup, the integral keeps growing while u is pinned at 1.2, which is the overshoot; turn anti-windup on to compare."
    >
      <Plots rows={2} heights={[3, 2]} hoverGroup>
        <Plot x={time} y={out}>
          <Curve name="set point r" x={t} y={r} dashed muted />
          <Curve name="whole run" x={ghostY.x} y={ghostY.y} muted thin />
          <Curve name="output y" x={yNow.x} y={yNow.y} slot={0} live />
          <Points name="y now" x={[t[i]]} y={[y[i]]} slot={0} live />
          <Handle {...state.handle('setpoint', { label: 'r', axis: 'y' })} />
        </Plot>
        <Plot x={time} y={inp}>
          <Curve name="whole run" x={ghostU.x} y={ghostU.y} muted thin />
          <Curve name="controller output (unlimited)" x={rawNow.x} y={rawNow.y} slot={3} dashed live />
          <Curve name="applied u" x={uNow.x} y={uNow.y} slot={2} live />
          <Points name="u now" x={[t[i]]} y={[u[i]]} slot={2} live />
        </Plot>
      </Plots>
    </Figure>
  )
}
