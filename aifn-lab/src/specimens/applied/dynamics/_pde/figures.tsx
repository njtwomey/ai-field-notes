import {
  gridPoints,
  heatEquation,
  transportEquation,
  waveEquation,
  type PdeState,
  type TimeScheme,
} from 'aifn-applied/dynamics/pde'
import { toFlat } from 'aifn/foundation/tensor'
import { trace } from 'aifn/foundation/trace'
import { useMemo, useState } from 'react'
import { Player, Select, Slider, usePlayhead } from '@lab/controls'
import { ControlRow, Figure } from '@lab/layout'
import { Panel, Readout, Subplots, XYChart, formatNumber, type XYSeries } from '@lab/viz'

const fmt = (v: number) => formatNumber(v)
const NONE: XYSeries[] = []

// ---------------------------------------------------------------------------------------------------------------------
// The heat equation and its stability limit.

const HEAT_GRID = { a: 0, b: 1, n: 41 }
const HEAT_X = toFlat(gridPoints(HEAT_GRID))
const dx = 1 / (HEAT_GRID.n - 1)
// A step profile plus a little grid-scale roughness, which is what an unstable explicit scheme amplifies first.
const STEP = (x: number) => (x > 0.3 && x < 0.6 ? 1 : 0)
const SCHEMES: { value: TimeScheme; label: string }[] = [
  { value: 'explicit', label: 'explicit (FTCS)' },
  { value: 'implicit', label: 'implicit Euler' },
  { value: 'crank-nicolson', label: 'Crank–Nicolson' },
]

const HEAT_T = 0.1
const HEAT_INITIAL: XYSeries[] = [
  { name: 'u(x, 0)', type: 'line', x: HEAT_X, y: HEAT_X.map(STEP), dashed: true, muted: true },
]

export function HeatSpecimen() {
  const [scheme, setScheme] = useState<TimeScheme>('explicit')
  const [r, setR] = useState(0.45)
  const D = 1
  const dt = (r * dx * dx) / D
  const steps = Math.max(1, Math.round(HEAT_T / dt))
  const tr = useMemo(
    () =>
      trace(
        heatEquation({
          diffusivity: D,
          grid: HEAT_GRID,
          boundary: { kind: 'dirichlet', left: 0, right: 0 },
          dt,
          scheme,
        }),
        { u0: STEP },
        steps,
        { every: Math.max(1, Math.floor(steps / 100)), stopOnNonFinite: false },
      ),
    [dt, scheme, steps],
  )
  const [at, setAt] = usePlayhead(tr.steps.length)
  const now = tr.steps[at] as PdeState
  const u = toFlat(now.u)
  const live: XYSeries[] = [{ name: 'u(x, t)', type: 'line', slot: 0, x: HEAT_X, y: u }]
  return (
    <Figure
      title="The heat equation and its stability limit"
      description="Explicit time stepping of u_t = u_xx is stable only while r = DΔt/Δx² ≤ ½; past it the finest grid oscillation grows by |1 − 4r| per step. The implicit schemes are stable at any r."
      controls={
        <>
          <ControlRow label="1 · scheme">
            <Select label="time stepping" value={scheme} onChange={setScheme} options={SCHEMES} />
            <Slider label="r = DΔt/Δx²" value={r} min={0.05} max={1.5} step={0.01} onChange={setR} />
          </ControlRow>
          <ControlRow label="2 · time">
            <Player
              className="col-span-full"
              value={at}
              onChange={setAt}
              count={tr.steps.length}
              duration={4}
              label="t"
              format={(i) => fmt((tr.steps[i] as PdeState).time)}
            />
          </ControlRow>
        </>
      }
      readouts={
        <>
          <Readout label={now.stability.number} value={fmt(now.stability.value)} />
          <Readout label="limit" value={Number.isFinite(now.stability.limit) ? fmt(now.stability.limit) : 'none'} />
          <Readout label="stable" value={now.stability.stable ? 'yes' : 'no'} />
          <Readout label="steps so far" value={now.t} />
          <Readout label="max |u|" value={fmt(Math.max(...u.map(Math.abs)))} />
        </>
      }
      caption="A unit step on [0.3, 0.6] with u = 0 at both ends, on 41 points. Play to watch it diffuse up to t = 0.1. With the explicit scheme, move r past 0.5: the solution breaks into a sawtooth that grows without bound. Crank–Nicolson stays bounded but, at large r, lets the sharp edges ring for a while; implicit Euler smooths them at once."
    >
      <XYChart series={HEAT_INITIAL} live={live} xLabel="x" yLabel="u" xRange={[0, 1]} yRange={[-0.5, 1.5]} />
    </Figure>
  )
}

// ---------------------------------------------------------------------------------------------------------------------
// Transport schemes after one period.

const T_GRID = { a: 0, b: 1, n: 101 }
const T_X = toFlat(gridPoints(T_GRID))
const PROFILE = (x: number) => (x > 0.1 && x < 0.3 ? 1 : 0) + Math.exp(-200 * (x - 0.65) ** 2)
const TRANSPORT: { value: 'upwind' | 'lax-friedrichs' | 'lax-wendroff'; label: string; slot: number }[] = [
  { value: 'upwind', label: 'upwind', slot: 0 },
  { value: 'lax-friedrichs', label: 'Lax–Friedrichs', slot: 1 },
  { value: 'lax-wendroff', label: 'Lax–Wendroff', slot: 2 },
]

/** The exact solution of u_t + u_x = 0 on the periodic unit interval: the profile shifted by t. */
const shifted = (t: number) => T_X.map((x) => PROFILE((((x - t) % 1) + 1) % 1))

export function TransportSchemesSpecimen() {
  const [nu, setNu] = useState(0.8)
  const [periods, setPeriods] = useState(1)
  const runs = useMemo(() => {
    const dt = (nu * 0.01) / 1
    const steps = Math.round(periods / dt)
    // About 60 frames per period, at most 300.
    const every = Math.max(1, Math.round(steps / Math.min(300, 60 * periods)))
    return TRANSPORT.map((m) => {
      const tr = trace(transportEquation({ velocity: 1, grid: T_GRID, dt, scheme: m.value }), { u0: PROFILE }, steps, {
        every,
        stopOnNonFinite: false,
      })
      return { ...m, frames: tr.steps as PdeState[] }
    })
  }, [nu, periods])
  const count = runs[0].frames.length
  const [at, setAt] = usePlayhead(count, 1)
  const now = runs[0].frames[at]
  const live: XYSeries[] = [
    { name: 'exact (the profile shifted by t)', type: 'line', x: T_X, y: shifted(now.time), dashed: true, muted: true },
    ...runs.map(({ label, slot, frames }) => ({
      name: label,
      slot,
      type: 'line' as const,
      x: T_X,
      y: toFlat(frames[Math.min(at, frames.length - 1)].u),
    })),
  ]
  const s0 = now.stability
  return (
    <Figure
      title="Transport schemes after whole periods"
      description="On u_t + u_x = 0 the exact solution returns to its start after each period; upwind and Lax–Friedrichs smear it (numerical diffusion), Lax–Wendroff keeps the bump but rings behind the square edges (dispersion)."
      controls={
        <>
          <ControlRow label="1 · discretisation">
            <Slider label="Courant number ν = cΔt/Δx" value={nu} min={0.1} max={1.2} step={0.05} onChange={setNu} />
            <Slider label="periods" value={periods} min={1} max={5} step={1} onChange={setPeriods} />
          </ControlRow>
          <ControlRow label="2 · time">
            <Player
              className="col-span-full"
              value={at}
              onChange={setAt}
              count={count}
              duration={3 + periods}
              startReason="the figure compares the schemes after whole periods, where the exact solution is the initial profile again"
              label="t"
              format={(i) => fmt(runs[0].frames[i].time)}
            />
          </ControlRow>
        </>
      }
      readouts={
        <>
          <Readout label={s0.number} value={fmt(s0.value)} />
          <Readout label="CFL condition ν ≤ 1" value={s0.stable ? 'holds' : 'violated'} />
          <Readout label="t" value={fmt(now.time)} />
        </>
      }
      caption="A square pulse and a Gaussian bump on a periodic grid of 100 cells, moving right at speed 1. Play to watch the schemes carry it round; the player starts at the end, after whole periods, where the exact solution is the initial profile again. At ν = 1 upwind and Lax–Wendroff shift exactly one cell per step and are exact; past 1 every scheme blows up."
    >
      <XYChart series={NONE} live={live} xLabel="x" yLabel="u" xRange={[0, 1]} yRange={[-0.5, 1.5]} />
    </Figure>
  )
}

// ---------------------------------------------------------------------------------------------------------------------
// The wave equation.

const W_GRID = { a: 0, b: 1, n: 101 }
const W_X = toFlat(gridPoints(W_GRID))
const PLUCK = (x: number) => Math.exp(-300 * (x - 0.3) ** 2)

const W_T = 2
const W_INITIAL: XYSeries[] = [{ name: 'u(x, 0)', type: 'line', x: W_X, y: W_X.map(PLUCK), dashed: true, muted: true }]

export function WaveSpecimen() {
  const [nu, setNu] = useState(0.9)
  const dt = nu * 0.01
  const steps = Math.round(W_T / dt)
  const run = useMemo(() => {
    const tr = trace(waveEquation({ speed: 1, grid: W_GRID, dt }), { u0: PLUCK }, steps, {
      every: Math.max(1, Math.round(steps / 200)),
      record: { energy: (s) => s.energy },
      stopOnNonFinite: false,
    })
    const e = toFlat(tr.series.energy)
    const drift = e.map((v) => (v - e[0]) / e[0])
    const finite = drift.filter(Number.isFinite).map(Math.abs)
    // The drift axis: its largest value over the run, capped so a blow-up does not flatten the stable part.
    const top = Math.min(1, Math.max(1e-6, ...finite)) * 1.1
    return { steps: tr.steps, t: tr.steps.map((s) => s.time), drift, top }
  }, [dt, steps])
  const [at, setAt] = usePlayhead(run.steps.length)
  const now = run.steps[at]
  const liveU: XYSeries[] = [{ name: 'u(x, t)', type: 'line', slot: 0, x: W_X, y: toFlat(now.u) }]
  const liveE: XYSeries[] = [
    { name: 'relative energy drift', type: 'line', slot: 1, x: run.t.slice(0, at + 1), y: run.drift.slice(0, at + 1) },
    { name: 'now', type: 'scatter', emphasis: true, x: [now.time], y: [run.drift[at]] },
  ]
  return (
    <Figure
      title="The wave equation by leapfrog"
      description="A plucked string splits into two pulses moving at speed c and reflecting, inverted, from the fixed ends; leapfrog conserves the discrete energy while ν ≤ 1."
      defaultSize="L"
      controls={
        <>
          <ControlRow label="1 · discretisation">
            <Slider label="Courant number ν" value={nu} min={0.2} max={1.05} step={0.01} onChange={setNu} />
          </ControlRow>
          <ControlRow label="2 · time">
            <Player
              className="col-span-full"
              value={at}
              onChange={setAt}
              count={run.steps.length}
              duration={5}
              label="t"
              format={(i) => fmt(run.steps[i].time)}
            />
          </ControlRow>
        </>
      }
      readouts={
        <>
          <Readout label="ν" value={fmt(now.stability.value)} />
          <Readout label="CFL condition ν ≤ 1" value={now.stability.stable ? 'holds' : 'violated'} />
          <Readout label="energy drift at t" value={fmt(run.drift[at])} />
        </>
      }
      caption="u(x, 0) a narrow bump at x = 0.3, at rest, fixed ends. Play to watch the two pulses travel, reflect and cross; below, the relative change of the discrete energy up to t. The pulses keep their shape best at ν = 1 (exact on the grid); smaller ν adds a dispersive wake; ν just above 1 blows up after a few crossings."
    >
      <Subplots rows={2} heightRatios={[1.3, 1]}>
        <Panel>
          <XYChart series={W_INITIAL} live={liveU} xLabel="x" yLabel="u" xRange={[0, 1]} yRange={[-1.1, 1.1]} />
        </Panel>
        <Panel>
          <XYChart
            series={NONE}
            live={liveE}
            xLabel="t"
            yLabel="(E − E₀)/E₀"
            xRange={[0, W_T]}
            yRange={[-run.top, run.top]}
            legend={false}
          />
        </Panel>
      </Subplots>
    </Figure>
  )
}
