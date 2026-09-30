import { transferFunction } from 'aifn/systems'
import { pidLoop, type AntiWindup } from 'aifn-applied/dynamics/control'
import { toFlat, type Tensor } from 'aifn/foundation/tensor'
import { trace } from 'aifn/foundation/trace'
import { useMemo, useState } from 'react'
import { Player, Select, Slider, Switch, usePlayhead } from '@lab/controls'
import { ControlRow, Figure } from '@lab/layout'
import { Panel, Readout, Subplots, XYChart, formatNumber, type Handle, type XYSeries } from '@lab/viz'

const flat = (t: Tensor) => toFlat(t)
const fmt = (v: number) => formatNumber(v)

/** Several traces as one NaN-separated line: the whole run drawn faintly behind the part played so far. */
function ghost(name: string, t: readonly number[], ys: readonly (readonly number[])[]): XYSeries {
  const x: number[] = []
  const y: number[] = []
  for (const v of ys) {
    x.push(...t, NaN)
    y.push(...v, NaN)
  }
  return { name, type: 'line', muted: true, x, y }
}

/** A trace up to position i (inclusive), and its point at i. */
const upTo = (t: readonly number[], y: readonly number[], i: number) => ({ x: t.slice(0, i + 1), y: y.slice(0, i + 1) })
const at = (t: readonly number[], y: readonly number[], i: number) => ({ x: [t[i]], y: [y[i]] })
const timeLabel = (t: readonly number[]) => (i: number) => `t = ${(t[i] ?? 0).toFixed(2)} s`

// ── PID step response ────────────────────────────────────────────────────────────────────────────────────────────────

const PLANT = transferFunction([1], [1, 3, 3, 1]) // 1/(s + 1)³

export function PidSpecimen() {
  const [kp, setKp] = useState(2)
  const [ki, setKi] = useState(1)
  const [kd, setKd] = useState(0.5)
  const [limit, setLimit] = useState(true)
  const [antiWindup, setAntiWindup] = useState<AntiWindup>('none')
  const [setpoint, setSetpoint] = useState(1)
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
  const output = useMemo(
    (): XYSeries[] => [
      { name: 'set point r', type: 'line', x: t, y: r, dashed: true, muted: true },
      ghost('whole run', t, [y]),
    ],
    [t, r, y],
  )
  const input = useMemo((): XYSeries[] => [ghost('whole run', t, [uRaw, u])], [t, uRaw, u])
  const liveOutput: XYSeries[] = [
    { name: 'output y', type: 'line', slot: 0, ...upTo(t, y, i) },
    { name: 'output y', type: 'scatter', slot: 0, ...at(t, y, i) },
  ]
  const liveInput: XYSeries[] = [
    { name: 'controller output (unlimited)', type: 'line', slot: 3, dashed: true, ...upTo(t, uRaw, i) },
    { name: 'applied u', type: 'line', slot: 2, ...upTo(t, u, i) },
    { name: 'applied u', type: 'scatter', slot: 2, ...at(t, u, i) },
  ]
  const handles: Handle[] = [
    { kind: 'y', at: setpoint, label: 'set point', onDrag: (v) => setSetpoint(Math.max(0.2, Math.min(1.1, v))) },
  ]
  return (
    <Figure
      title="PID step response with actuator limits"
      description="Proportional action speeds the response, integral action removes the steady-state error, derivative action damps it; when the actuator saturates the integrator winds up and overshoots unless anti-windup stops it."
      defaultSize="L"
      controls={
        <>
          <ControlRow label="1 · gains">
            <Slider label="k_p" value={kp} min={0} max={6} step={0.05} onChange={setKp} />
            <Slider label="k_i" value={ki} min={0} max={3} step={0.05} onChange={setKi} />
            <Slider label="k_d" value={kd} min={0} max={4} step={0.05} onChange={setKd} />
          </ControlRow>
          <ControlRow label="2 · actuator">
            <Switch label="limit |u| ≤ 1.2" checked={limit} onChange={setLimit} />
            <Select
              label="anti-windup"
              value={antiWindup}
              onChange={setAntiWindup}
              options={[
                { value: 'none', label: 'none' },
                { value: 'clamp', label: 'conditional integration' },
                { value: 'back-calculation', label: 'back-calculation' },
              ]}
            />
          </ControlRow>
          <ControlRow label="3 · time">
            <Player value={i} onChange={setI} count={t.length} duration={5} format={timeLabel(t)} label="t" />
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
      <Subplots rows={2} sharex heightRatios={[3, 2]} hoverGroup rescaleOnChange={false}>
        <Panel>
          <XYChart xLabel="t (s)" yLabel="output" handles={handles} series={output} live={liveOutput} />
        </Panel>
        <Panel>
          <XYChart xLabel="t (s)" yLabel="input" series={input} live={liveInput} />
        </Panel>
      </Subplots>
    </Figure>
  )
}
