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
} from 'aifn/dynamics/ode'
import { toFlat, toRows, type Tensor } from 'aifn/foundation/tensor'
import { trace, type Algorithm, type Trace } from 'aifn/foundation/trace'
import { useMemo, useState } from 'react'
import { Player, Select, Slider, Switch, usePlayhead } from '@lab/controls'
import { ControlRow, Figure } from '@lab/layout'
import { TraceView } from '@lab/views'
import { Heatmap, Panel, Readout, Subplots, XYChart, formatNumber, type Handle, type XYSeries } from '@lab/viz'

const fmt = (v: number) => formatNumber(v)
const NO_SERIES: XYSeries[] = []

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
  const [logLambda, setLogLambda] = useState(2.5)
  const [h, setH] = useState(0.1)
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
  const solution = useMemo((): XYSeries[] => [{ name: 'exact', type: 'line', ...exact, muted: true }], [exact])
  const tNow = frameTime(frame, STIFF_FRAMES, STIFF_END)
  const at = runs.map((r) => upTo(r.t, tNow + 1e-12))
  const liveSolution: XYSeries[] = [
    ...runs.map(({ name, slot, t, x, hs }, i) => ({
      name,
      slot,
      type: 'line' as const,
      showPoints: hs.length < 80,
      x: t.slice(0, at[i] + 1),
      y: x.slice(0, at[i] + 1),
    })),
    ...runs.map(({ name, slot, t, x }, i) => ({
      name,
      slot,
      type: 'scatter' as const,
      x: [t[at[i]]],
      y: [x[at[i]]],
    })),
  ]
  const liveSteps: XYSeries[] = runs.map(({ name, slot, t, hs }, i) => ({
    name,
    slot,
    type: 'line',
    showPoints: hs.length < 80,
    x: t.slice(1, at[i] + 1),
    y: hs.slice(1, at[i] + 1),
  }))
  const dpAll = runs[0].tr.steps as AdaptiveState[]
  const dp = dpAll[at[0]]
  const rejected = dpAll.slice(1, at[0] + 1).reduce((n, s) => n + s.attempts.length - 1, 0)
  return (
    <Figure
      title="Step sizes on a stiff problem"
      description="Stiffness caps an explicit method's step by stability, not accuracy: once the transient has died, Dormand–Prince still takes steps of about 3/λ, while implicit Euler and BDF2 follow the slow solution with any step."
      defaultSize="L"
      controls={
        <>
          <ControlRow label="1 · problem">
            <Slider label="log₁₀ stiffness λ" value={logLambda} min={0} max={3.5} step={0.1} onChange={setLogLambda} />
          </ControlRow>
          <ControlRow label="2 · implicit step">
            <Slider label="h (implicit Euler, BDF2)" value={h} min={0.02} max={1} step={0.02} onChange={setH} />
          </ControlRow>
          <ControlRow label="3 · time">
            <Player
              value={frame}
              onChange={setFrame}
              count={STIFF_FRAMES + 1}
              duration={4}
              label="t"
              format={(i) => `t = ${frameTime(i, STIFF_FRAMES, STIFF_END).toFixed(3)}`}
            />
          </ControlRow>
        </>
      }
      readouts={
        <>
          <Readout label="t" value={tNow.toFixed(3)} />
          <Readout label="Dormand–Prince steps so far" value={`${dp.t} (${rejected} rejected)`} />
          <Readout label="Dormand–Prince evaluations so far" value={dp.evaluations} />
          <Readout label="BDF2 evaluations so far" value={runs[2].tr.steps[at[2]].evaluations} />
          <Readout label="Dormand–Prince step now" value={at[0] > 0 ? fmt(Math.abs(dp.stepSize)) : '–'} />
          <Readout label="stability cap 3.3/λ" value={fmt(3.3 / lambda)} />
        </>
      }
      caption="x′ = −λ(x − cos t), x(0) = 0 on [0, 6]. Play to watch the three solvers advance, each with its current point. Top: the solutions so far and the exact one (grey). Bottom: the size of every step taken so far (log scale). Dormand–Prince's steps grow through the transient and then plateau at its stability limit; raise λ and the plateau falls in proportion, and the work grows with it."
    >
      <Subplots rows={2} sharex heightRatios={[3, 2]} hoverGroup>
        <Panel>
          <XYChart
            series={solution}
            live={liveSolution}
            xRange={[0, STIFF_END]}
            yRange={ranges.x}
            xLabel="t"
            yLabel="x"
          />
        </Panel>
        <Panel>
          <XYChart
            series={NO_SERIES}
            live={liveSteps}
            xRange={[0, STIFF_END]}
            yRange={ranges.h}
            yLog
            xLabel="t"
            yLabel="step size h"
          />
        </Panel>
      </Subplots>
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
  const [h, setH] = useState(0.5)
  const [q0, setQ0] = useState(2)
  const [shown, setShown] = useState<Record<EnergyMethod, boolean>>({
    rk4: true,
    euler: false,
    'symplectic-euler': false,
    leapfrog: true,
    'velocity-verlet': false,
  })
  const [frame, setFrame] = usePlayhead(ENERGY_FRAMES + 1)
  const runs = useMemo(
    () => ENERGY_METHODS.filter((m) => shown[m.value]).map((m) => ({ ...m, r: energyRun(m.value, h, q0, ENERGY_END) })),
    [h, q0, shown],
  )
  const driftRange = useMemo(() => extent(runs.map(({ r }) => r.error)), [runs])
  const tNow = frameTime(frame, ENERGY_FRAMES, ENERGY_END)
  const at = runs.map(({ r }) => upTo(r.t, tNow + 1e-9))
  const phase: XYSeries[] = [
    ...runs.map(({ label, slot, r }, i) => ({
      name: label,
      slot,
      type: 'line' as const,
      thin: true,
      x: r.q.slice(0, at[i] + 1),
      y: r.p.slice(0, at[i] + 1),
    })),
    ...runs.map(({ label, slot, r }, i) => ({
      name: label,
      slot,
      type: 'scatter' as const,
      x: [r.q[at[i]]],
      y: [r.p[at[i]]],
    })),
  ]
  const drift: XYSeries[] = runs.map(({ label, slot, r }, i) => ({
    name: label,
    slot,
    type: 'line',
    x: r.t.slice(0, at[i] + 1),
    y: r.error.slice(0, at[i] + 1),
  }))
  const handles: Handle[] = [
    { kind: 'point', at: [q0, 0], onDrag: ([q]) => setQ0(Math.max(-3, Math.min(3, q))), label: 'q₀' },
  ]
  return (
    <Figure
      title="Energy drift on the pendulum"
      description="A symplectic integrator keeps the pendulum's energy error bounded for ever; RK4, though more accurate per step, lets it drift steadily, so its orbit spirals inwards."
      defaultSize="L"
      controls={
        <>
          <ControlRow label="1 · start">
            <Slider label="q₀ (rad)" value={q0} min={-3} max={3} step={0.05} onChange={setQ0} />
          </ControlRow>
          <ControlRow label="2 · integrators">
            <Slider label="step h" value={h} min={0.05} max={0.8} step={0.05} onChange={setH} />
            {ENERGY_METHODS.map((m) => (
              <Switch
                key={m.value}
                label={m.label}
                checked={shown[m.value]}
                onChange={(v: boolean) => setShown((s) => ({ ...s, [m.value]: v }))}
              />
            ))}
          </ControlRow>
          <ControlRow label="3 · time">
            <Player
              value={frame}
              onChange={setFrame}
              count={ENERGY_FRAMES + 1}
              duration={5}
              label="t"
              format={(i) => `t = ${Math.round(frameTime(i, ENERGY_FRAMES, ENERGY_END))}`}
            />
          </ControlRow>
        </>
      }
      readouts={runs.map(({ label, r }, i) => (
        <Readout key={label} label={`${label}: H − H₀ at t = ${Math.round(tNow)}`} value={fmt(r.error[at[i]])} />
      ))}
      caption="Pendulum H = ½p² − cos q from (q₀, 0) over t ∈ [0, 2000]. Play to trace the orbits and the energy error in time. Left: the orbits so far in the phase plane, with each integrator's current point (drag q₀ along the axis). Right: H − H₀ so far. Leapfrog and velocity Verlet oscillate within O(h²); symplectic Euler within O(h); RK4 loses energy steadily; explicit Euler gains it and spirals out."
    >
      <Subplots cols={2} widthRatios={[1, 1.4]}>
        <Panel>
          <XYChart
            series={NO_SERIES}
            live={phase}
            handles={handles}
            xLabel="q"
            yLabel="p"
            xRange={[-3.5, 3.5]}
            yRange={[-2.5, 2.5]}
          />
        </Panel>
        <Panel>
          <XYChart
            series={NO_SERIES}
            live={drift}
            xRange={[0, ENERGY_END]}
            yRange={driftRange}
            xLabel="t"
            yLabel="H − H₀"
          />
        </Panel>
      </Subplots>
    </Figure>
  )
}

// ---------------------------------------------------------------------------------------------------------------------
// 3. Stability regions.

const STABILITY_METHODS: { value: string; label: string }[] = [
  { value: 'euler', label: 'explicit Euler' },
  { value: 'heun', label: 'Heun' },
  { value: 'rk4', label: 'RK4' },
  { value: 'dormand-prince', label: 'Dormand–Prince' },
  { value: 'implicit-euler', label: 'implicit Euler' },
  { value: 'implicit-trapezoid', label: 'implicit trapezoid' },
  { value: 'bdf2', label: 'BDF2' },
  { value: 'bdf3', label: 'BDF3' },
]

export function StabilityRegionSpecimen() {
  const [method, setMethod] = useState('rk4')
  const [z, setZ] = useState<[number, number]>([-2, 1.5])
  const region = useMemo(
    () => stabilityRegion(method as StabilityMethod, { real: [-6, 3], imag: [-4.5, 4.5], nx: 91, ny: 91 }),
    [method],
  )
  const grid = useMemo(() => {
    // log₁₀ of the amplification, clipped, so 0 (the boundary) sits in the middle of a diverging scale.
    const rows = toRows(region.amplification).map((r) => r.map((v) => Math.max(-1.5, Math.min(1.5, Math.log10(v)))))
    return { x: toFlat(region.real), y: toFlat(region.imag), z: rows }
  }, [region])
  const contours = useMemo(() => ({ levels: [0] }), [])
  const a = amplification(method as StabilityMethod, z[0], z[1])
  return (
    <Figure
      title="Stability regions"
      description="A method is stable on x′ = λx when z = hλ lies where its amplification is at most 1 (blue); an explicit method's region is bounded, so a large negative λ forces a small h."
      controls={<Select label="method" value={method} onChange={setMethod} options={STABILITY_METHODS} />}
      readouts={
        <>
          <Readout label="z = hλ" value={`${fmt(z[0])} ${z[1] < 0 ? '−' : '+'} ${fmt(Math.abs(z[1]))}i`} />
          <Readout label="amplification" value={fmt(a)} />
          <Readout label="stable" value={a <= 1 + 1e-12 ? 'yes' : 'no'} />
        </>
      }
      caption="Colour: log₁₀ of the amplification |R(z)| (the largest characteristic root for BDF), clipped to ±1.5; the contour is the boundary |R| = 1. Drag z. The implicit methods contain the whole left half-plane (A-stability); BDF3 misses a sliver near the imaginary axis."
    >
      <Heatmap
        x={grid.x}
        y={grid.y}
        z={grid.z}
        scale="diverging"
        range={[-1.5, 1.5]}
        contours={contours}
        xLabel="Re z"
        yLabel="Im z"
        valueLabel="log₁₀ amplification"
        equalAspect
        handles={[{ kind: 'point', at: z, onDrag: setZ, label: 'z' }]}
      />
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
  const [logTol, setLogTol] = useState(-4)
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
    <TraceView
      title="Dormand–Prince step by step"
      trace={tr}
      show={['step size', 'error estimate']}
      startAtFirst
      defaultSize="L"
      controls={<Slider label="log₁₀ rtol" value={logTol} min={-8} max={-1} step={0.5} onChange={setLogTol} />}
      caption="Lotka–Volterra from (10, 5). Play or step: each position is one accepted step, drawn as a point on the orbit so far; its state lists the attempts (rejected ones had error norm above 1). The step size shrinks where the orbit turns fast and grows along the slow stretches."
      renderState={(s, { position }) => <AdaptiveOrbit path={path} position={position} state={s} />}
    />
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
  const orbit = useMemo((): XYSeries[] => [{ name: 'orbit', type: 'line', ...path, muted: true }], [path])
  const x = toFlat(state.x)
  const live: XYSeries[] = [
    {
      name: 'accepted steps',
      type: 'line',
      showPoints: true,
      slot: 0,
      x: path.x.slice(0, position + 1),
      y: path.y.slice(0, position + 1),
    },
    { name: 'current', type: 'scatter', x: [x[0]], y: [x[1]], emphasis: true },
  ]
  return <XYChart xLabel="prey" yLabel="predators" series={orbit} live={live} />
}
