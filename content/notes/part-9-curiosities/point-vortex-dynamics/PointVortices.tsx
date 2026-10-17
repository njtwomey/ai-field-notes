import { useMemo } from 'react'
import { rungeKutta } from 'aifn-compute/dynamics/ode'
import { trace } from 'aifn-compute/foundation/trace'
import { toFlat } from 'aifn-compute/foundation/tensor'
import {
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  Player,
  Plot,
  Points,
  Readout,
  setting,
  slider,
  useAxis,
  useComputed,
  useFigureState,
  usePlayhead,
  variants,
  type ParamDef,
} from 'aifn-render'
import { invariants, kirchhoff, tracerGrid, type Vortex } from './vortices'

type Box = { x: [number, number]; y: [number, number] }

/** A starting arrangement: its vortices, the view, how long to run, and whether to follow the vortices. */
type Setup = { label: string; vortices: Vortex[]; box: Box; duration: number; follow: boolean }

const BOX: Box = { x: [-4, 4], y: [-3, 3] }

const SETUPS: Record<string, Setup> = {
  pair: {
    label: 'opposite pair',
    vortices: [
      { x: -2.5, y: 0.5, kappa: 1 },
      { x: -2.5, y: -0.5, kappa: -1 },
    ],
    box: BOX,
    duration: 6,
    follow: false,
  },
  like: {
    label: 'like pair',
    vortices: [
      { x: -0.5, y: 0, kappa: 1 },
      { x: 0.5, y: 0, kappa: 1 },
    ],
    box: BOX,
    duration: 2 * Math.PI,
    follow: false,
  },
  leapfrog: {
    label: 'two pairs leapfrog',
    vortices: [
      { x: 0, y: 1, kappa: 1 },
      { x: 0, y: -1, kappa: -1 },
      { x: 0, y: 0.5, kappa: 1 },
      { x: 0, y: -0.5, kappa: -1 },
    ],
    box: BOX,
    duration: 18,
    follow: true,
  },
  three: {
    label: 'three vortices',
    vortices: [
      { x: -1, y: -0.4, kappa: 1 },
      { x: 1, y: -0.6, kappa: 1 },
      { x: 0.2, y: 1, kappa: 1 },
    ],
    box: BOX,
    duration: 30,
    follow: false,
  },
  four: {
    label: 'four vortices',
    vortices: [
      { x: -1, y: -0.4, kappa: 1 },
      { x: 1, y: -0.6, kappa: 1 },
      { x: 0.2, y: 1, kappa: 1 },
      { x: 0.6, y: 0.2, kappa: 1 },
    ],
    box: BOX,
    duration: 30,
    follow: false,
  },
}

const SUBSCRIPT = ['₁', '₂', '₃', '₄']

/** One case of the setup choice: a position (on the chart) and a strength slider per vortex, a duration and a frame. */
const caseOf = (s: Setup) => {
  const params: Record<string, ParamDef> = {}
  s.vortices.forEach((v, i) => {
    params[`x${i + 1}`] = slider(s.box.x[0], s.box.x[1], v.x, { onChart: true, step: 0.05 })
    params[`y${i + 1}`] = slider(s.box.y[0], s.box.y[1], v.y, { onChart: true, step: 0.05 })
    params[`k${i + 1}`] = slider(-2, 2, v.kappa, { label: `Γ${SUBSCRIPT[i]}/2π`, step: 0.05 })
  })
  params.duration = float(Math.round(s.duration * 100) / 100, {
    label: 'duration t',
    gt: 0,
    le: 200,
    suggestions: [2 * Math.PI, 10, 20, 36, 60],
  })
  params.follow = setting(s.follow, 'follow the vortices')
  return { label: s.label, params }
}

/** Frames kept for the player. */
const FRAMES = 300
/** Tracers per unit length on the grid seeded over the view (none right beside a vortex), and their largest number. */
const TRACER_DENSITY = 3.6
const MAX_TRACERS = 2500
/** The tracers' smoothing length: within it of a vortex a tracer feels a smoothed velocity. */
const TRACER_CORE = 0.05
/** The twin run starts with vortex 1 moved this far in x. */
const TWIN_NUDGE = 1e-6

type Run = { frames: number[][]; times: number[] }

/** Integrates the state from z0 with RK4 and keeps about FRAMES frames. */
function integrate(kappa: number[], z0: number[], duration: number, dt: number): Run {
  const steps = Math.max(1, Math.round(duration / dt))
  const every = Math.max(1, Math.round(steps / FRAMES))
  const tr = trace(rungeKutta(kirchhoff(kappa, TRACER_CORE), 'rk4', { stepSize: dt }), { x0: z0 }, steps, {
    every,
    keep: 'all',
    timing: false,
    stopOnNonFinite: false,
  })
  return { frames: tr.steps.map((s) => toFlat(s.x)), times: tr.steps.map((s) => s.time) }
}

const fmt = (v: number) => formatNumber(v)

/** Point vortices moving under Kirchhoff's equations, with passive tracers and a twin run for sensitivity. */
export function PointVortices() {
  const state = useFigureState({
    setup: variants(
      {
        pair: caseOf(SETUPS.pair),
        like: caseOf(SETUPS.like),
        leapfrog: caseOf(SETUPS.leapfrog),
        three: caseOf(SETUPS.three),
        four: caseOf(SETUPS.four),
      },
      { label: '1 · vortices', choiceLabel: 'start', initial: 'pair' },
    ),
    dt: float(0.01, { label: '2 · RK4 step Δt', gt: 0, le: 0.1, suggestions: [0.002, 0.005, 0.01, 0.02, 0.05] }),
    tracers: setting(true, 'tracers'),
  })
  const key = state.setup.key
  const setup = SETUPS[key]
  const values = state.setup.values as unknown as Record<string, number | boolean>
  const n = setup.vortices.length
  // The values object is rebuilt on every render, so the vortices are memoised on their numbers.
  const signature = setup.vortices
    .map((_, i) => `${values[`x${i + 1}`]},${values[`y${i + 1}`]},${values[`k${i + 1}`]}`)
    .join(';')
  const vortices = useMemo<Vortex[]>(
    () =>
      signature.split(';').map((s) => {
        const [x, y, kappa] = s.split(',').map(Number)
        return { x, y, kappa }
      }),
    [signature],
  )
  const duration = values.duration as number
  const follow = values.follow as boolean
  const { dt } = state
  const kappa = useMemo(() => vortices.map((v) => v.kappa), [vortices])
  const z0 = useMemo(() => vortices.flatMap((v) => [v.x, v.y]), [vortices])

  // The vortices alone (cheap: recomputed on every drag) and a twin nudged by 10⁻⁶.
  const main = useMemo(() => integrate(kappa, z0, duration, dt), [kappa, z0, duration, dt])
  const twin = useMemo(() => {
    const nudged = [...z0]
    nudged[0] += TWIN_NUDGE
    return integrate(kappa, nudged, duration, dt)
  }, [kappa, z0, duration, dt])

  // The tracers, seeded over the view (stretched ahead of the vortices when following them), run once the drag ends.
  const tracerRun = useComputed(
    () => {
      if (!state.tracers) return null
      const box: Box = { x: [...setup.box.x], y: [...setup.box.y] }
      if (follow) {
        const last = main.frames[main.frames.length - 1]
        let dx = 0
        let dy = 0
        for (let i = 0; i < n; i++) {
          dx += (last[2 * i] - z0[2 * i]) / n
          dy += (last[2 * i + 1] - z0[2 * i + 1]) / n
        }
        box.x = [box.x[0] + Math.min(0, dx), box.x[1] + Math.max(0, dx)]
        box.y = [box.y[0] + Math.min(0, dy), box.y[1] + Math.max(0, dy)]
      }
      const width = box.x[1] - box.x[0]
      const height = box.y[1] - box.y[0]
      const density = Math.min(TRACER_DENSITY, Math.sqrt(MAX_TRACERS / (width * height)))
      const grid = tracerGrid(box, Math.ceil(width * density), Math.ceil(height * density), vortices, 0.15)
      return integrate(kappa, [...z0, ...grid], duration, dt)
    },
    [state.tracers, setup, follow, main, n, z0, vortices, kappa, duration, dt],
    { mode: 'release' },
  )

  const count = main.frames.length
  const [frame, setFrame] = usePlayhead(count)
  const at = Math.min(frame, count - 1)

  // The displayed frame: the lab frame, or one moving with the mean vortex position (equal to the lab frame at t = 0).
  const offsets = useMemo(
    () =>
      main.frames.map((z) => {
        if (!follow) return [0, 0]
        let ox = 0
        let oy = 0
        for (let i = 0; i < n; i++) {
          ox += (z[2 * i] - z0[2 * i]) / n
          oy += (z[2 * i + 1] - z0[2 * i + 1]) / n
        }
        return [ox, oy]
      }),
    [main, follow, n, z0],
  )

  const paths = useMemo(
    () =>
      vortices.map((_, i) => ({
        x: main.frames.map((z, f) => z[2 * i] - offsets[f][0]),
        y: main.frames.map((z, f) => z[2 * i + 1] - offsets[f][1]),
      })),
    [vortices, main, offsets],
  )

  const z = main.frames[at]
  const [ox, oy] = offsets[at]
  const now = {
    x: vortices.map((_, i) => z[2 * i] - ox),
    y: vortices.map((_, i) => z[2 * i + 1] - oy),
    group: vortices.map((v) => (v.kappa >= 0 ? 0 : 1)),
  }
  const tracers = useMemo(() => {
    const run = tracerRun.value
    if (!run) return null
    const f = Math.min(at, run.frames.length - 1)
    const w = run.frames[f]
    const xs: number[] = []
    const ys: number[] = []
    for (let k = 2 * n; k < w.length; k += 2) {
      xs.push(w[k] - ox)
      ys.push(w[k + 1] - oy)
    }
    return xs.length > 0 ? { x: xs, y: ys } : null
  }, [tracerRun.value, at, n, ox, oy])

  const start = invariants(main.frames[0], kappa)
  const current = invariants(z, kappa)
  const zt = twin.frames[Math.min(at, twin.frames.length - 1)]
  let separation = 0
  for (let k = 0; k < 2 * n; k++) separation += (z[k] - zt[k]) ** 2
  separation = Math.sqrt(separation)

  const xAxis = useAxis({ label: follow ? 'x − x̄(t) + x̄(0)' : 'x', range: setup.box.x })
  const yAxis = useAxis({ label: 'y', range: setup.box.y, equal: xAxis })
  return (
    <Figure
      title="Point vortices"
      state={state}
      aspect={4 / 3}
      controls={
        <Player
          value={at}
          onChange={setFrame}
          count={count}
          label="time"
          format={(v) => `t = ${formatNumber(main.times[Math.min(v, count - 1)])}`}
        />
      }
      readouts={{
        now: (
          <>
            <Readout label="t" value={fmt(main.times[at])} />
            <Readout label="twin separation" value={separation.toExponential(1)} />
          </>
        ),
        conserved: (
          <>
            <Readout label="H/2π" value={`${fmt(current.energy)} (start ${fmt(start.energy)})`} />
            <Readout label="impulse (Q, P)/2π" value={`(${fmt(current.qx)}, ${fmt(current.py)})`} />
            <Readout label="angular impulse I/2π" value={`${fmt(current.angular)} (start ${fmt(start.angular)})`} />
          </>
        ),
      }}
      caption="Drag the numbered handles to move each vortex's starting point, and set each strength Γ/2π with its slider (positive turns counter-clockwise). Play the time. An opposite pair translates; a like pair rotates about its centre of vorticity; two coaxial pairs leapfrog; three vortices move regularly while the tracers between them mix; four vortices move chaotically. The twin run starts with vortex 1 moved by 10⁻⁶: its separation grows about linearly in time for two or three vortices and exponentially for four. Grey dots are passive tracers carried by the flow, recomputed when a drag ends. H, (Q, P) and I stay at their starting values up to the integration error."
    >
      <Plot x={xAxis} y={yAxis}>
        {state.tracers && tracers && (
          <Points name="tracers" x={tracers.x} y={tracers.y} muted dense stale={tracerRun.stale} />
        )}
        {paths.map((p, i) => (
          <Curve
            key={`path-${i}`}
            name={vortices[i].kappa >= 0 ? 'path, Γ > 0' : 'path, Γ < 0'}
            x={p.x.slice(0, at + 1)}
            y={p.y.slice(0, at + 1)}
            slot={vortices[i].kappa >= 0 ? 0 : 1}
            width={1.5}
            silent
          />
        ))}
        <Points
          name="vortices"
          x={now.x}
          y={now.y}
          group={now.group}
          groupNames={['Γ > 0 (counter-clockwise)', 'Γ < 0 (clockwise)']}
          size={10}
        />
        {vortices.map((_, i) => (
          <Handle
            key={`start-${key}-${i}`}
            {...state.handle([`setup.x${i + 1}`, `setup.y${i + 1}`], { label: `${i + 1}` })}
          />
        ))}
      </Plot>
    </Figure>
  )
}
