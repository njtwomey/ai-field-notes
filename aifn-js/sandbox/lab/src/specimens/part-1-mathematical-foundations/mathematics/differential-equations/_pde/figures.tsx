import {
  gridPoints,
  heatEquation,
  transportEquation,
  waveEquation,
  type PdeState,
  type TimeScheme,
} from 'aifn-methods/dynamics/pde'
import { toFlat } from 'aifn/foundation/tensor'
import { trace } from 'aifn/foundation/trace'
import { useMemo } from 'react'
import { Player, usePlayhead } from '@lab/controls'
import { ControlRow, Figure } from '@lab/layout'
import { choice, row, slider, useFigureState } from '@lab/state'
import { Curve, formatNumber, Plot, Plots, Points, Readout, useAxis } from '@lab/viz'

const fmt = (v: number) => formatNumber(v)

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
const HEAT_U0 = HEAT_X.map(STEP)

export function HeatSpecimen() {
  const state = useFigureState({
    setup: row('1 · scheme', {
      scheme: choice(SCHEMES, 'explicit', { label: 'time stepping' }),
      r: slider(0.05, 1.5, 0.45, { label: 'r = DΔt/Δx²', step: 0.01 }),
    }),
  })
  const scheme = state.setup.scheme as TimeScheme
  const { r } = state.setup
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
  const xa = useAxis({ label: 'x', range: [0, 1] })
  const ua = useAxis({ label: 'u', range: [-0.5, 1.5] })
  return (
    <Figure
      title="The heat equation and its stability limit"
      purpose="Explicit time stepping of u_t = u_xx is stable only while r = DΔt/Δx² ≤ ½; past it the finest grid oscillation grows by |1 − 4r| per step. The implicit schemes are stable at any r."
      state={state}
      controls={
        <>
          <ControlRow label="2 · time">
            <Player
              className="col-span-full"
              value={at}
              onChange={setAt}
              count={tr.steps.length}
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
      <Plot x={xa} y={ua}>
        <Curve name="u(x, 0)" x={HEAT_X} y={HEAT_U0} dashed muted />
        <Curve name="u(x, t)" x={HEAT_X} y={u} slot={0} live />
      </Plot>
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
  const state = useFigureState({
    setup: row('1 · discretisation', {
      nu: slider(0.1, 1.2, 0.8, { label: 'Courant number ν = cΔt/Δx', step: 0.05 }),
      periods: slider(1, 5, 1, { label: 'periods', step: 1 }),
    }),
  })
  const { nu, periods } = state.setup
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
  const exact = shifted(now.time)
  const s0 = now.stability
  const xa = useAxis({ label: 'x', range: [0, 1] })
  const ua = useAxis({ label: 'u', range: [-0.5, 1.5] })
  return (
    <Figure
      title="Transport schemes after whole periods"
      purpose="On u_t + u_x = 0 the exact solution returns to its start after each period; upwind and Lax–Friedrichs smear it (numerical diffusion), Lax–Wendroff keeps the bump but rings behind the square edges (dispersion)."
      state={state}
      controls={
        <>
          <ControlRow label="2 · time">
            <Player
              className="col-span-full"
              value={at}
              onChange={setAt}
              count={count}
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
      <Plot x={xa} y={ua}>
        <Curve name="exact (the profile shifted by t)" x={T_X} y={exact} dashed muted />
        {runs.map(({ label, slot, frames }) => (
          <Curve key={label} name={label} x={T_X} y={toFlat(frames[Math.min(at, frames.length - 1)].u)} slot={slot} />
        ))}
      </Plot>
    </Figure>
  )
}

// ---------------------------------------------------------------------------------------------------------------------
// The wave equation.

const W_GRID = { a: 0, b: 1, n: 101 }
const W_X = toFlat(gridPoints(W_GRID))
const PLUCK = (x: number) => Math.exp(-300 * (x - 0.3) ** 2)

const W_T = 2
const W_U0 = W_X.map(PLUCK)

export function WaveSpecimen() {
  const state = useFigureState({
    setup: row('1 · discretisation', { nu: slider(0.2, 1.05, 0.9, { label: 'Courant number ν', step: 0.01 }) }),
  })
  const { nu } = state.setup
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
  const xa = useAxis({ label: 'x', range: [0, 1] })
  const ua = useAxis({ label: 'u', range: [-1.1, 1.1] })
  const ta = useAxis({ label: 't', range: [0, W_T] })
  const ea = useAxis({ label: '(E − E₀)/E₀', range: [-run.top, run.top] })
  return (
    <Figure
      title="The wave equation by leapfrog"
      purpose="A plucked string splits into two pulses moving at speed c and reflecting, inverted, from the fixed ends; leapfrog conserves the discrete energy while ν ≤ 1."
      state={state}
      defaultSize="L"
      controls={
        <>
          <ControlRow label="2 · time">
            <Player
              className="col-span-full"
              value={at}
              onChange={setAt}
              count={run.steps.length}
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
      <Plots rows={2} heights={[1.3, 1]}>
        <Plot x={xa} y={ua}>
          <Curve name="u(x, 0)" x={W_X} y={W_U0} dashed muted />
          <Curve name="u(x, t)" x={W_X} y={toFlat(now.u)} slot={0} live />
        </Plot>
        <Plot x={ta} y={ea} legend={false}>
          <Curve name="relative energy drift" x={run.t.slice(0, at + 1)} y={run.drift.slice(0, at + 1)} slot={1} live />
          <Points name="now" x={[now.time]} y={[run.drift[at]]} emphasis live />
        </Plot>
      </Plots>
    </Figure>
  )
}
