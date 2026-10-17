import { useMemo } from 'react'
import { heatEquation } from 'aifn-methods/dynamics/pde'
import { trace } from 'aifn-compute/foundation/trace'
import { toFlat } from 'aifn-compute/foundation/tensor'
import {
  choice,
  Curve,
  Figure,
  float,
  formatNumber,
  Player,
  Plot,
  Readout,
  slider,
  toggle,
  useAxis,
  useFigureState,
  usePlayhead,
} from 'aifn-render'
import {
  characteristics,
  initialPhi,
  initialVelocity,
  measureShock,
  velocityFromPhi,
  type ProfileName,
} from './burgers'

const L = 2 * Math.PI
/** Frames kept for the player. */
const FRAMES = 120
/** Grid points across the predicted shock width 8ν/Δu with Δu = 2A, within the grid limits below. */
const POINTS_PER_WIDTH = 12
const MIN_POINTS = 201
const MAX_POINTS = 1601
/** The explicit scheme's diffusion number r = νΔt/Δx²: below ½ it is stable and keeps φ positive. */
const DIFFUSION_NUMBER = 0.4
/** The most grid-point updates (points × time steps) per run, about half a second; a coarser grid keeps within it. */
const WORK = 8e6

const fmt = (v: number) => formatNumber(v)

/** Burgers' equation solved through the heat equation: φ by the engine's finite differences, mapped back to u. */
export function BurgersShock() {
  const state = useFigureState({
    profile: choice<ProfileName>(
      [
        { value: 'sine', label: 'sine, u₀ = A sin x' },
        { value: 'step', label: 'step, u₀ = ±A' },
      ],
      'sine',
      { label: 'initial profile' },
    ),
    amp: slider(0.5, 2, 1, { label: 'amplitude A', step: 0.05 }),
    nu: float(0.05, { label: 'viscosity ν', ge: 0.005, le: 0.5, suggestions: [0.01, 0.02, 0.05, 0.1, 0.2] }),
    duration: float(3, { label: 'duration t', gt: 0, le: 6, suggestions: [1, 2, 3, 5] }),
    inviscid: toggle(true, 'inviscid characteristics'),
  })
  const { profile, amp, nu, duration } = state

  const run = useMemo(() => {
    const target = Math.ceil(L / ((4 * nu) / amp / POINTS_PER_WIDTH)) + 1
    // Points × steps ≈ n · tν(n − 1)²/(rL²), so the work budget caps n.
    const affordable = Math.floor(Math.cbrt((WORK * DIFFUSION_NUMBER * L * L) / (duration * nu)))
    const n = Math.max(MIN_POINTS, Math.min(MAX_POINTS, target, affordable))
    const dx = L / (n - 1)
    const dt = (DIFFUSION_NUMBER * dx * dx) / nu
    const steps = Math.max(1, Math.ceil(duration / dt))
    const every = Math.max(1, Math.round(steps / FRAMES))
    const tr = trace(
      heatEquation({
        diffusivity: nu,
        grid: { a: 0, b: L, n },
        boundary: { kind: 'periodic' },
        dt,
        scheme: 'explicit',
      }),
      { u0: initialPhi(profile, amp, nu) },
      steps,
      { every, keep: 'all', timing: false, stopOnNonFinite: false },
    )
    const xs = Array.from({ length: n }, (_, i) => i * dx)
    const phis = tr.steps.map((s) => toFlat(s.u))
    const us = phis.map((p) => velocityFromPhi(p, dx, nu))
    return { n, dx, xs, phis, us, times: tr.steps.map((s) => s.time) }
  }, [profile, amp, nu, duration])

  const count = run.times.length
  const [frame, setFrame] = usePlayhead(count)
  const at = Math.min(frame, count - 1)
  const t = run.times[at]
  const u = run.us[at]
  const phi = run.phis[at]
  const phiMax = Math.max(...phi)
  const phiShown = useMemo(() => phi.map((p) => p / phiMax), [phi, phiMax])
  const shock = measureShock(u, run.dx)
  const predicted = (8 * nu) / shock.jump

  const u0 = useMemo(() => initialVelocity(profile, amp), [profile, amp])
  const start = useMemo(() => run.xs.map(u0), [run, u0])
  const chars = useMemo(() => characteristics(u0, t, 600), [u0, t])
  const breaking = profile === 'sine' ? 1 / amp : 0

  const xAxis = useAxis({ label: 'x', range: [0, L] })
  const uAxis = useAxis({ label: 'u', range: [-1.4 * amp, 1.4 * amp] })
  const xAxis2 = useAxis({ label: 'x', range: [0, L] })
  const pAxis = useAxis({ label: 'φ / max φ', range: [0, 1.05] })
  return (
    <Figure
      title="A Burgers shock from the heat equation"
      state={state}
      controls={
        <Player
          value={at}
          onChange={setFrame}
          count={count}
          label="time"
          format={(v) => `t = ${formatNumber(run.times[Math.min(v, count - 1)])}`}
        />
      }
      readouts={{
        now: (
          <>
            <Readout label="t" value={fmt(t)} />
            <Readout
              label="breaking time 1/A"
              value={profile === 'sine' ? fmt(breaking) : '0 (a jump from the start)'}
            />
            <Readout label="grid" value={`${run.n} points, Δx = ${fmt(run.dx)}`} />
          </>
        ),
        shock: (
          <>
            <Readout label="jump Δu" value={fmt(shock.jump)} />
            <Readout label="measured width Δu / max(−uₓ)" value={fmt(shock.width)} />
            <Readout label="predicted width 8ν/Δu" value={fmt(predicted)} />
            <Readout label="ratio" value={fmt(shock.width / predicted)} />
            <Readout label="grid points across the width" value={fmt(shock.width / run.dx)} />
          </>
        ),
      }}
      caption="Top: the Burgers velocity u(x, t) on a periodic interval, recovered as u = −2ν φₓ/φ from φ, which the heat equation φₜ = ν φₓₓ evolves (bottom, scaled to a maximum of 1). The engine solves the heat equation by explicit finite differences. Play the time. The sine steepens until it breaks at t = 1/A and then carries a shock at x = π; the step makes a shock at x = π and a rarefaction fan at x = 0 from the start. The dashed curve is the inviscid solution by characteristics, which folds over where the viscous solution has its shock. Type a smaller ν: the shock narrows in proportion, and the measured width approaches 8ν/Δu once the shock is narrow compared with the wave."
    >
      <Plot x={xAxis} y={uAxis} height={280}>
        <Curve name="u at t = 0" x={run.xs} y={start} muted />
        {state.inviscid && <Curve name="inviscid (characteristics)" x={chars.x} y={chars.u} slot={1} dashed silent />}
        <Curve name="u(x, t)" x={run.xs} y={Array.from(u)} slot={0} />
      </Plot>
      <Plot x={xAxis2} y={pAxis} height={180}>
        <Curve name="φ(x, t) / max φ" x={run.xs} y={Array.from(phiShown)} slot={2} />
      </Plot>
    </Figure>
  )
}
