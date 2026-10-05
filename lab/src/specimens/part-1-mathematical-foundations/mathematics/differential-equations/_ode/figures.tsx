import {
  amplification,
  bdf,
  dormandPrince,
  hamiltonianSystem,
  implicitEuler,
  rungeKutta,
  stabilityRegion,
  symplectic,
  type AdaptiveState,
  type OdeState,
  type Rhs,
  type StabilityMethod,
  type SymplecticMethod,
} from 'aifn-compute/dynamics/ode'
import { toFlat, toRows, type Tensor } from 'aifn-compute/foundation/tensor'
import { trace, type Algorithm, type Trace } from 'aifn-compute/foundation/trace'
import { useMemo } from 'react'
import { Player, usePlayhead } from 'aifn-render/controls'
import { Figure } from 'aifn-render/layout'
import { choice, row, setting, slider, useComputed, useFigureState } from 'aifn-render/state'
import { TracePanel } from '@lab/views'
import { Contours, Curve, Handle, Plot, Plots, Points, Raster, Readout, formatNumber, useAxis } from 'aifn-render/viz'

const fmt = (v: number) => formatNumber(v)

/** The index of the last entry of an ascending `ts` at or before `t` (0 if none). */
function upTo(ts: ArrayLike<number>, t: number): number {
  let lo = 0
  let hi = ts.length - 1
  if (hi < 0 || ts[0] > t) return 0
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1
    if (ts[mid] <= t) lo = mid
    else hi = mid - 1
  }
  return lo
}

/** The extent of several arrays, ignoring non-finite values, padded by `pad` of its span. */
function extent(arrays: ArrayLike<number>[], pad = 0.05): [number, number] {
  let lo = Infinity
  let hi = -Infinity
  for (const a of arrays)
    for (let i = 0; i < a.length; i++)
      if (Number.isFinite(a[i])) {
        lo = Math.min(lo, a[i])
        hi = Math.max(hi, a[i])
      }
  if (!(hi >= lo)) return [0, 1]
  const d = (hi - lo || Math.abs(hi) || 1) * pad
  return [lo - d, hi + d]
}

/** A log axis range: whole decades around the positive values. */
function decades(arrays: ArrayLike<number>[]): [number, number] {
  const positive = arrays.map((a) => Array.from(a).filter((v) => v > 0))
  const [lo, hi] = extent(positive, 0)
  return [10 ** Math.floor(Math.log10(lo)), 10 ** Math.ceil(Math.log10(hi))]
}

/** Frames of a Player over [0, tEnd]: `n` + 1 equally spaced times. */
const frameTime = (i: number, n: number, tEnd: number) => (tEnd * i) / n

// ---------------------------------------------------------------------------------------------------------------------
// 1. A stiff problem: x′ = −λ(x − cos t). After a fast transient the solution follows the slow cos t, but an explicit
// method's step stays capped by stability (|1 + hλ·…| ≤ 1) however smooth the solution is.

type StiffRun = { name: string; slot: number; tr: Trace<OdeState> }

function stiffRuns(lambda: number, h: number): StiffRun[] {
  // Plain-array right-hand side (fast); the implicit methods get its Jacobian −λ directly.
  const f: Rhs = (t, x) => [-lambda * (toFlat(x)[0] - Math.cos(t))]
  const jacobian = () => [[-lambda]]
  const tEnd = 6
  const opts = { x0: [0] }
  const record = { x: (s: OdeState) => toFlat(s.x)[0], h: (s: OdeState) => s.stepSize }
  const run = <S extends OdeState>(alg: Algorithm<{ x0: number[] }, S>) =>
    trace(alg, opts, 200_000, { record, stopOnNonFinite: false }) as unknown as Trace<OdeState>
  return [
    { name: 'Dormand–Prince (adaptive)', slot: 0, tr: run(dormandPrince(f, { tEnd, rtol: 1e-4, atol: 1e-7 })) },
    { name: 'implicit Euler', slot: 1, tr: run(implicitEuler(f, { stepSize: h, tEnd, jacobian })) },
    { name: 'BDF2', slot: 2, tr: run(bdf(f, 2, { stepSize: h, tEnd, jacobian })) },
  ]
}

const STIFF_END = 6
const STIFF_FRAMES = 240

export function StiffStepsSpecimen() {
  const state = useFigureState({
    problem: row('1 · problem', { logLambda: slider(0, 3.5, 2.5, { label: 'log₁₀ stiffness λ', step: 0.1 }) }),
    implicit: row('2 · implicit step', { h: slider(0.02, 1, 0.1, { label: 'h (implicit Euler, BDF2)', step: 0.02 }) }),
  })
  const { logLambda } = state.problem
  const { h } = state.implicit
  const [frame, setFrame] = usePlayhead(STIFF_FRAMES + 1)
  const lambda = 10 ** logLambda
  const runs = useMemo(() => {
    return stiffRuns(lambda, h).map((r) => {
      const t = r.tr.steps.map((s) => s.time)
      const x = toFlat(r.tr.series.x)
      // The size of the step that arrived at each time (none at t = 0).
      const hs = r.tr.steps.map((s, k) => (k === 0 ? NaN : Math.abs(s.stepSize)))
      return { ...r, t, x, hs }
    })
  }, [lambda, h])
  const exact = useMemo(() => {
    const ts = Array.from({ length: 400 }, (_, i) => (STIFF_END * i) / 399)
    const k = lambda * lambda + 1
    return {
      x: ts,
      y: ts.map(
        (t) => (lambda * (lambda * Math.cos(t) + Math.sin(t))) / k - ((lambda * lambda) / k) * Math.exp(-lambda * t),
      ),
    }
  }, [lambda])
  const ranges = useMemo(
    () => ({ x: extent([exact.y, ...runs.map((r) => r.x)]), h: decades(runs.map((r) => r.hs)) }),
    [exact, runs],
  )
  const tNow = frameTime(frame, STIFF_FRAMES, STIFF_END)
  const at = runs.map((r) => upTo(r.t, tNow + 1e-12))
  const dpAll = runs[0].tr.steps as AdaptiveState[]
  const dp = dpAll[at[0]]
  const rejected = dpAll.slice(1, at[0] + 1).reduce((n, s) => n + s.attempts.length - 1, 0)
  const time = useAxis({ label: 't', range: [0, STIFF_END] })
  const xa = useAxis({ label: 'x', range: ranges.x })
  const ha = useAxis({ label: 'step size h', log: true, range: ranges.h })
  return (
    <Figure
      title="Step sizes on a stiff problem"
      purpose="Stiffness caps an explicit method's step by stability, not accuracy: once the transient has died, Dormand–Prince still takes steps of about 3/λ, while implicit Euler and BDF2 follow the slow solution with any step."
      defaultSize="L"
      state={state}
      controls={
        <Player
          value={frame}
          onChange={setFrame}
          count={STIFF_FRAMES + 1}
          label="3 · time t"
          format={(i) => `t = ${frameTime(i, STIFF_FRAMES, STIFF_END).toFixed(3)}`}
        />
      }
      readouts={{
        'so far': (
          <>
            <Readout label="t" value={tNow.toFixed(3)} />
            <Readout label="Dormand–Prince steps" value={`${dp.t} (${rejected} rejected)`} />
            <Readout label="Dormand–Prince evaluations" value={dp.evaluations} />
            <Readout label="BDF2 evaluations" value={runs[2].tr.steps[at[2]].evaluations} />
          </>
        ),
        'step size': (
          <>
            <Readout label="Dormand–Prince step now" value={at[0] > 0 ? fmt(Math.abs(dp.stepSize)) : '–'} />
            <Readout label="stability cap 3.3/λ" value={fmt(3.3 / lambda)} />
          </>
        ),
      }}
      caption="x′ = −λ(x − cos t), x(0) = 0 on [0, 6]. Play to watch the three solvers advance, each with its current point. Top: the solutions so far and the exact one (grey). Bottom: the size of every step taken so far (log scale). Dormand–Prince's steps grow through the transient and then plateau at its stability limit; raise λ and the plateau falls in proportion, and the work grows with it."
    >
      <Plots rows={2} heights={[3, 2]} hoverGroup>
        <Plot x={time} y={xa}>
          <Curve name="exact" x={exact.x} y={exact.y} muted />
          {runs.map(({ name, slot, t, x, hs }, i) => (
            <Curve
              key={name}
              name={name}
              x={t.slice(0, at[i] + 1)}
              y={x.slice(0, at[i] + 1)}
              slot={slot}
              showPoints={hs.length < 80}
            />
          ))}
          {runs.map(({ name, slot, t, x }, i) => (
            <Points key={name} name={name} x={[t[at[i]]]} y={[x[at[i]]]} slot={slot} live />
          ))}
        </Plot>
        <Plot x={time} y={ha}>
          {runs.map(({ name, slot, t, hs }, i) => (
            <Curve
              key={name}
              name={name}
              x={t.slice(1, at[i] + 1)}
              y={hs.slice(1, at[i] + 1)}
              slot={slot}
              showPoints={hs.length < 80}
            />
          ))}
        </Plot>
      </Plots>
    </Figure>
  )
}

// ---------------------------------------------------------------------------------------------------------------------
// 2. Energy drift on the pendulum: H(q, p) = ½p² − cos q.

const PENDULUM_H = {
  potential: (q: Tensor) => -Math.cos(toFlat(q)[0]),
  potentialGradient: (q: Tensor) => [Math.sin(toFlat(q)[0])],
}

type EnergyMethod = SymplecticMethod | 'rk4' | 'euler'
const ENERGY_METHODS: { value: EnergyMethod; label: string; slot: number }[] = [
  { value: 'rk4', label: 'RK4', slot: 0 },
  { value: 'euler', label: 'explicit Euler', slot: 1 },
  { value: 'symplectic-euler', label: 'symplectic Euler', slot: 2 },
  { value: 'leapfrog', label: 'leapfrog', slot: 3 },
  { value: 'velocity-verlet', label: 'velocity Verlet', slot: 4 },
]

function energyRun(method: EnergyMethod, h: number, q0: number, tEnd: number) {
  const steps = Math.round(tEnd / h)
  const every = Math.max(1, Math.floor(steps / 1500))
  if (method === 'rk4' || method === 'euler') {
    const { rhs, energy } = hamiltonianSystem(PENDULUM_H)
    const tr = trace(rungeKutta(rhs, method, { stepSize: h }), { x0: [q0, 0] }, steps, {
      every,
      stopOnNonFinite: false,
      record: { q: (s) => toFlat(s.x)[0], p: (s) => toFlat(s.x)[1], e: (s) => energy(s.x) },
    })
    const e = toFlat(tr.series.e)
    return {
      t: tr.steps.map((s) => s.time),
      q: toFlat(tr.series.q),
      p: toFlat(tr.series.p),
      error: e.map((v) => v - e[0]),
    }
  }
  const tr = trace(symplectic(PENDULUM_H, method, { stepSize: h }), { q0: [q0], p0: [0] }, steps, {
    every,
    stopOnNonFinite: false,
    record: { q: (s) => toFlat(s.q)[0], p: (s) => toFlat(s.p)[0], e: (s) => s.energyError },
  })
  return { t: tr.steps.map((s) => s.time), q: toFlat(tr.series.q), p: toFlat(tr.series.p), error: toFlat(tr.series.e) }
}

const ENERGY_END = 2000
const ENERGY_FRAMES = 400

export function EnergyDriftSpecimen() {
  const state = useFigureState({
    start: row('1 · start', { q0: slider(-3, 3, 2, { label: 'q₀ (rad)', step: 0.05 }) }),
    integrators: row('2 · integrators', {
      h: slider(0.05, 0.8, 0.5, { label: 'step h', step: 0.05 }),
      rk4: setting(true, 'RK4'),
      euler: setting(false, 'explicit Euler'),
      'symplectic-euler': setting(false, 'symplectic Euler'),
      leapfrog: setting(true, 'leapfrog'),
      'velocity-verlet': setting(false, 'velocity Verlet'),
    }),
  })
  const { q0 } = state.start
  const shown = state.integrators
  const { h } = shown
  const [frame, setFrame] = usePlayhead(ENERGY_FRAMES + 1)
  const key = ENERGY_METHODS.map((m) => (shown[m.value] ? 1 : 0)).join('')
  // Up to 40 000 steps per integrator: dragging q₀ reruns as often as a frame allows, dimming the old runs meanwhile.
  const computed = useComputed(
    () =>
      ENERGY_METHODS.filter((_, k) => key[k] === '1').map((m) => ({ ...m, r: energyRun(m.value, h, q0, ENERGY_END) })),
    [h, q0, key],
  )
  const runs = computed.value
  const driftRange = useMemo(() => extent(runs.map(({ r }) => r.error)), [runs])
  const tNow = frameTime(frame, ENERGY_FRAMES, ENERGY_END)
  const at = runs.map(({ r }) => upTo(r.t, tNow + 1e-9))
  const q = useAxis({ label: 'q', range: [-3.5, 3.5] })
  const p = useAxis({ label: 'p', range: [-2.5, 2.5] })
  const time = useAxis({ label: 't', range: [0, ENERGY_END] })
  const e = useAxis({ label: 'H − H₀', range: driftRange })
  return (
    <Figure
      title="Energy drift on the pendulum"
      purpose="A symplectic integrator keeps the pendulum's energy error bounded for ever; RK4, though more accurate per step, lets it drift steadily, so its orbit spirals inwards."
      defaultSize="L"
      state={state}
      controls={
        <Player
          value={frame}
          onChange={setFrame}
          count={ENERGY_FRAMES + 1}
          label="3 · time t"
          format={(i) => `t = ${Math.round(frameTime(i, ENERGY_FRAMES, ENERGY_END))}`}
        />
      }
      readouts={{
        [`H − H₀ at t = ${Math.round(tNow)}`]: runs.map(({ label, r }, i) => (
          <Readout key={label} label={label} value={fmt(r.error[at[i]])} />
        )),
      }}
      caption="Pendulum H = ½p² − cos q from (q₀, 0) over t ∈ [0, 2000]. Play to trace the orbits and the energy error in time; the whole runs are drawn faintly behind. Left: the orbits so far in the phase plane, with each integrator's current point (drag q₀ along the axis). Right: H − H₀ so far. Leapfrog and velocity Verlet oscillate within O(h²); symplectic Euler within O(h); RK4 loses energy steadily; explicit Euler gains it and spirals out."
    >
      <Plots cols={2} widths={[1, 1.4]}>
        <Plot x={q} y={p}>
          {/* The whole run, faint, behind the part played so far: the point shows before playing. */}
          {runs.map(({ label, slot, r }) => (
            <Curve key={`${label} all`} name={label} x={r.q} y={r.p} slot={slot} width={0.5} muted />
          ))}
          {runs.map(({ label, slot, r }, i) => (
            <Curve
              key={label}
              name={label}
              x={r.q.slice(0, at[i] + 1)}
              y={r.p.slice(0, at[i] + 1)}
              slot={slot}
              thin
              stale={computed.stale}
            />
          ))}
          {runs.map(({ label, slot, r }, i) => (
            <Points key={label} name={label} x={[r.q[at[i]]]} y={[r.p[at[i]]]} slot={slot} live />
          ))}
          <Handle kind="point" at={[q0, 0]} label="q₀" onDrag={([v]) => state.set('start.q0', v)} />
        </Plot>
        <Plot x={time} y={e}>
          {runs.map(({ label, slot, r }) => (
            <Curve key={`${label} all`} name={label} x={r.t} y={r.error} slot={slot} thin />
          ))}
          {runs.map(({ label, slot, r }, i) => (
            <Curve
              key={label}
              name={label}
              x={r.t.slice(0, at[i] + 1)}
              y={r.error.slice(0, at[i] + 1)}
              slot={slot}
              stale={computed.stale}
            />
          ))}
        </Plot>
      </Plots>
    </Figure>
  )
}

// ---------------------------------------------------------------------------------------------------------------------
// 3. Stability regions.

const STABILITY_LEVELS = [0]
const STABILITY_METHODS = [
  { value: 'euler', label: 'explicit Euler' },
  { value: 'heun', label: 'Heun' },
  { value: 'rk4', label: 'RK4' },
  { value: 'dormand-prince', label: 'Dormand–Prince' },
  { value: 'implicit-euler', label: 'implicit Euler' },
  { value: 'implicit-trapezoid', label: 'implicit trapezoid' },
  { value: 'bdf2', label: 'BDF2' },
  { value: 'bdf3', label: 'BDF3' },
] as const

export function StabilityRegionSpecimen() {
  const state = useFigureState({
    method: row('1 · method', { method: choice(STABILITY_METHODS, 'rk4', { label: 'method' }) }),
    zr: slider(-6, 3, -2, { onChart: true, label: 'Re z' }),
    zi: slider(-4.5, 4.5, 1.5, { onChart: true, label: 'Im z' }),
  })
  const method = state.method.method
  const z: [number, number] = [state.zr, state.zi]
  const region = useMemo(
    () => stabilityRegion(method as StabilityMethod, { real: [-6, 3], imag: [-4.5, 4.5], nx: 91, ny: 91 }),
    [method],
  )
  const grid = useMemo(() => {
    // log₁₀ of the amplification, clipped, so 0 (the boundary) sits in the middle of a diverging scale.
    const rows = toRows(region.amplification).map((r) => r.map((v) => Math.max(-1.5, Math.min(1.5, Math.log10(v)))))
    return { x: toFlat(region.real), y: toFlat(region.imag), z: rows }
  }, [region])
  const a = amplification(method as StabilityMethod, z[0], z[1])
  const re = useAxis({ label: 'Re z' })
  const im = useAxis({ label: 'Im z', equal: re })
  return (
    <Figure
      title="Stability regions"
      purpose="A method is stable on x′ = λx when z = hλ lies where its amplification is at most 1 (blue); an explicit method's region is bounded, so a large negative λ forces a small h."
      state={state}
      readouts={{
        'at z': (
          <>
            <Readout label="z = hλ" value={`${fmt(z[0])} ${z[1] < 0 ? '−' : '+'} ${fmt(Math.abs(z[1]))}i`} />
            <Readout label="amplification" value={fmt(a)} />
            <Readout label="stable" value={a <= 1 + 1e-12 ? 'yes' : 'no'} />
          </>
        ),
      }}
      caption="Colour: log₁₀ of the amplification |R(z)| (the largest characteristic root for BDF), clipped to ±1.5; the contour is the boundary |R| = 1. Drag z. The implicit methods contain the whole left half-plane (A-stability); BDF3 misses a sliver near the imaginary axis."
    >
      <Plot x={re} y={im}>
        <Raster
          x={grid.x}
          y={grid.y}
          z={grid.z}
          scale="diverging"
          range={[-1.5, 1.5]}
          valueLabel="log₁₀ amplification"
        />
        <Contours x={grid.x} y={grid.y} z={grid.z} levels={STABILITY_LEVELS} />
        <Handle {...state.handle(['zr', 'zi'], { label: 'z' })} />
      </Plot>
    </Figure>
  )
}

// ---------------------------------------------------------------------------------------------------------------------
// 4. Dormand–Prince as a trace.

const lotkaVolterra: Rhs = (_t, x) => {
  const [a, b] = toFlat(x)
  return [1.5 * a - a * b, a * b - 3 * b]
}

export function AdaptiveTraceSpecimen() {
  const state = useFigureState({
    tolerance: row('1 · tolerance', { logTol: slider(-8, -1, -4, { label: 'log₁₀ rtol', step: 0.5 }) }),
  })
  const { logTol } = state.tolerance
  const tr = useMemo(
    () =>
      trace(
        dormandPrince(lotkaVolterra, { tEnd: 15, rtol: 10 ** logTol, atol: 10 ** (logTol - 3) }),
        { x0: [10, 5] },
        5000,
        {
          record: {
            'step size': (s: AdaptiveState) => s.stepSize,
            'error estimate': (s: AdaptiveState) => (Number.isFinite(s.error) ? s.error : 0),
            'attempts this step': (s: AdaptiveState) => s.attempts.length,
          },
        },
      ),
    [logTol],
  )
  const path = useMemo(() => {
    const X = tr.steps.map((s) => toFlat(s.x))
    return { x: X.map((v) => v[0]), y: X.map((v) => v[1]) }
  }, [tr])
  return (
    <Figure
      purpose="An adaptive Runge–Kutta method picks each step so its error estimate meets the tolerance: steps shrink where the orbit turns fast, grow on the slow stretches, and a step that misses is retried smaller."
      title="Dormand–Prince step by step"
      defaultSize="L"
      state={state}
      caption="Lotka–Volterra from (10, 5). Play or step: each position is one accepted step, drawn as a point on the orbit so far; its state lists the attempts (rejected ones had error norm above 1). The step size shrinks where the orbit turns fast and grows along the slow stretches."
    >
      <TracePanel
        trace={tr}
        show={['step size', 'error estimate']}
        renderState={(s, { position }) => <AdaptiveOrbit path={path} position={position} state={s} />}
      />
    </Figure>
  )
}

/** The Lotka–Volterra orbit up to one accepted step: the whole orbit muted, the steps so far and the current point. */
function AdaptiveOrbit({
  path,
  position,
  state,
}: {
  path: { x: number[]; y: number[] }
  position: number
  state: AdaptiveState
}) {
  const x = toFlat(state.x)
  const prey = useAxis({ label: 'prey' })
  const predators = useAxis({ label: 'predators' })
  return (
    <Plot x={prey} y={predators}>
      <Curve name="orbit" x={path.x} y={path.y} muted />
      <Curve
        name="accepted steps"
        x={path.x.slice(0, position + 1)}
        y={path.y.slice(0, position + 1)}
        slot={0}
        showPoints
      />
      <Points name="current" x={[x[0]]} y={[x[1]]} emphasis live />
    </Plot>
  )
}
