import {
  fixedPoints,
  invariantManifolds,
  pushForwardDensityFrames,
  streamline,
  trajectory,
  transportDensityFrames,
  type Box,
  type FixedPoint,
  type VectorField,
} from 'aifn-compute/dynamics/fields'
import { directionField, nullclines, type Segments } from '@lab/drawing/fields'
import { limitCycle } from 'aifn-methods/dynamics/nonlinear'
import { normals, stream } from 'aifn-compute/foundation/random'
import {
  get,
  imagPart,
  mul,
  neg,
  realPart,
  sin,
  stack,
  sub,
  toFlat,
  toRows,
  type Tensor,
} from 'aifn-compute/foundation/tensor'
import { useMemo } from 'react'
import { Player, usePlayhead } from 'aifn-render/controls'
import { Figure } from 'aifn-render/layout'
import { choice, row, slider, toggle, useComputed, useFigureState, variants } from 'aifn-render/state'
import {
  Curve,
  Handle,
  Plot,
  Points,
  Raster,
  Readout,
  Segments as SegmentsLayer,
  formatNumber,
  useAxis,
} from 'aifn-render/viz'

type Segment = { from: [number, number]; to: [number, number] }

const fmt = (v: number) => formatNumber(Math.abs(v) < 1e-12 ? 0 : v)

/** Segments as one line series, NaN-separated, so a whole level set is one legend entry. */
function segmentLine(s: Segments): { x: number[]; y: number[] } {
  const a = toRows(s.start)
  const b = toRows(s.end)
  const x: number[] = []
  const y: number[] = []
  a.forEach((p, k) => x.push(p[0], b[k][0], NaN) && y.push(p[1], b[k][1], NaN))
  return { x, y }
}

const curve = (m: Tensor) => {
  const r = toRows(m)
  return { x: r.map((p) => p[0]), y: r.map((p) => p[1]) }
}

// ---------------------------------------------------------------------------------------------------------------------
// Planar systems, written with tensor primitives so that fixed points and manifolds can use their Jacobians.

type System = {
  value: string
  label: string
  param: { label: string; min: number; max: number; initial: number }
  field: (p: number) => VectorField
  box: [[number, number], [number, number]]
  start: [number, number]
  /** A section crossed by the limit cycle, if the system has one. */
  section?: { point: number[]; normal: number[] }
}

const SYSTEM_LIST: System[] = [
  {
    value: 'pendulum',
    label: 'damped pendulum',
    param: { label: 'damping c', min: 0, max: 1, initial: 0.15 },
    field: (c) => (x) => stack([get(x, 1), sub(neg(sin(get(x, 0))), mul(c, get(x, 1)))]),
    box: [
      [-7, 7],
      [-3.5, 3.5],
    ],
    start: [-6, 2.6],
  },
  {
    value: 'van-der-pol',
    label: 'Van der Pol oscillator',
    param: { label: 'μ', min: 0.1, max: 4, initial: 1 },
    field: (mu) => (x) => {
      const [a, b] = [get(x, 0), get(x, 1)]
      return stack([b, sub(mul(mul(mu, sub(1, mul(a, a))), b), a)])
    },
    box: [
      [-3, 3],
      [-5, 5],
    ],
    start: [0.3, 0.2],
    section: { point: [0, 0], normal: [1, 0] },
  },
  {
    value: 'duffing',
    label: 'damped Duffing oscillator',
    param: { label: 'damping δ', min: 0, max: 1, initial: 0.25 },
    field: (d) => (x) => {
      const [a, b] = [get(x, 0), get(x, 1)]
      return stack([b, sub(sub(a, mul(a, mul(a, a))), mul(d, b))])
    },
    box: [
      [-2, 2],
      [-1.5, 1.5],
    ],
    start: [-1.2, 1.3],
  },
  {
    value: 'lotka-volterra',
    label: 'Lotka–Volterra with crowding',
    param: { label: 'crowding ε', min: 0, max: 0.5, initial: 0 },
    field: (e) => (x) => {
      const [a, b] = [get(x, 0), get(x, 1)]
      return stack([mul(a, sub(sub(1, b), mul(e, a))), mul(b, sub(a, 1))])
    },
    box: [
      [0, 3.5],
      [0, 3],
    ],
    start: [0.5, 0.6],
  },
]

const SYSTEMS = Object.fromEntries(SYSTEM_LIST.map((s) => [s.value, s])) as Record<string, System>
/** Each system with its own parameter and its own start x₀ (placed on the chart, moved by its handle). */
const caseOf = (s: System) => ({
  label: s.label,
  params: {
    p: slider(s.param.min, s.param.max, s.param.initial, { label: s.param.label, step: 0.01 }),
    sx: slider(s.box[0][0], s.box[0][1], s.start[0], { onChart: true, label: 'x₀' }),
    sy: slider(s.box[1][0], s.box[1][1], s.start[1], { onChart: true, label: 'y₀' }),
  },
})
const SYSTEM = variants(
  {
    pendulum: caseOf(SYSTEMS.pendulum),
    'van-der-pol': caseOf(SYSTEMS['van-der-pol']),
    duffing: caseOf(SYSTEMS.duffing),
    'lotka-volterra': caseOf(SYSTEMS['lotka-volterra']),
  },
  { label: '1 · system', choiceLabel: 'system', initial: 'duffing' },
)

/** The time a phase portrait plays over, and its frames. */
const PORTRAIT_T = 15
const PORTRAIT_FRAMES = 150
/** RK4 steps per frame of the main trajectory. */
const PATH_SUBSTEPS = 6

export function PhasePortraitSpecimen() {
  const state = useFigureState({
    system: SYSTEM,
    reveal: row('2 · reveal', {
      showNullclines: toggle(true, 'nullclines'),
      showManifolds: toggle(true, 'saddle manifolds'),
      showCycle: toggle(false, 'limit cycle (Van der Pol)'),
      showParticles: toggle(true, 'particles'),
    }),
  })
  const which = state.system.key
  const system = SYSTEMS[which]
  const { p, sx, sy } = state.system.values
  const x0 = useMemo((): [number, number] => [sx, sy], [sx, sy])
  const { showNullclines, showManifolds, showParticles } = state.reveal
  const showCycle = state.reveal.showCycle && !!system.section
  const [frame, setFrame] = usePlayhead(PORTRAIT_FRAMES + 1)

  const f = useMemo(() => system.field(p), [system, p])
  const box = system.box as Box
  const grid = useMemo(() => ({ x: system.box[0], y: system.box[1] }), [system])
  const xAxis = useAxis({ label: 'x', range: system.box[0] })
  const yAxis = useAxis({ label: 'y', range: system.box[1] })
  const arrows = useMemo<Segment[]>(() => {
    const d = directionField(f, { x: grid.x, y: grid.y, nx: 24, ny: 18 }, { scale: 0.6 })
    const a = toRows(d.start)
    const b = toRows(d.end)
    return a.map((s, k) => ({ from: [s[0], s[1]], to: [b[k][0], b[k][1]] }))
  }, [f, grid])
  const clines = useMemo(() => nullclines(f, { x: grid.x, y: grid.y, nx: 81, ny: 81 }).map(segmentLine), [f, grid])
  const fps = useMemo<FixedPoint[]>(() => fixedPoints(f, box, { seeds: 9 }), [f, box])
  const manifolds = useMemo(() => {
    if (!showManifolds) return []
    return fps
      .filter((q) => q.kind === 'saddle')
      .flatMap((q) => {
        const m = invariantManifolds(f, q, { t: 30, steps: 1500, bounds: box })
        return [
          ...m.stable.map((c) => ({ kind: 'stable', ...curve(c) })),
          ...m.unstable.map((c) => ({ kind: 'unstable', ...curve(c) })),
        ]
      })
  }, [f, fps, box, showManifolds])
  const cycle = useMemo(
    () => (showCycle && system.section ? limitCycle(f, x0, system.section, { orbitPoints: 400 }) : null),
    [f, x0, system, showCycle],
  )
  // The trajectory from x₀, PATH_SUBSTEPS RK4 steps per frame; it may stop early at a fixed point or the box's edge.
  // 900 RK4 steps through tensor primitives: dragging x₀ recomputes it as often as a frame allows.
  const pathRun = useComputed(() => {
    const s = streamline(f, x0, {
      t: PORTRAIT_T,
      steps: PORTRAIT_FRAMES * PATH_SUBSTEPS,
      direction: 'forward',
      bounds: box,
    })
    return curve(s)
  }, [f, x0, box])
  const path = pathRun.value
  // A cloud of particles on a grid over the box, each carried by the flow: its position at every frame.
  const particles = useMemo(() => {
    if (!showParticles) return null
    const [[x0l, x1l], [y0l, y1l]] = system.box
    const nx = 12
    const ny = 10
    const frames: { x: number[]; y: number[] }[] = Array.from({ length: PORTRAIT_FRAMES + 1 }, () => ({ x: [], y: [] }))
    for (let i = 0; i < nx; i++)
      for (let j = 0; j < ny; j++) {
        const start = [x0l + ((i + 0.5) * (x1l - x0l)) / nx, y0l + ((j + 0.5) * (y1l - y0l)) / ny]
        const rows = toRows(trajectory(f, start, PORTRAIT_T, { steps: PORTRAIT_FRAMES }).x)
        rows.forEach((r, k) => {
          // A particle that has left the box (or blown up) is dropped rather than drawn at the edge.
          const inside = r[0] >= x0l && r[0] <= x1l && r[1] >= y0l && r[1] <= y1l
          frames[k].x.push(inside ? r[0] : NaN)
          frames[k].y.push(inside ? r[1] : NaN)
        })
      }
    return frames
  }, [f, system, showParticles])

  const join = (cs: { x: number[]; y: number[] }[]) => ({
    x: cs.flatMap((c) => [...c.x, NaN]),
    y: cs.flatMap((c) => [...c.y, NaN]),
  })
  const stable = useMemo(() => join(manifolds.filter((m) => m.kind === 'stable')), [manifolds])
  const unstable = useMemo(() => join(manifolds.filter((m) => m.kind === 'unstable')), [manifolds])
  const cycleCurve = useMemo(() => (cycle ? curve(cycle.orbit) : null), [cycle])
  const fixed = useMemo(() => ({ x: fps.map((q) => toFlat(q.point)[0]), y: fps.map((q) => toFlat(q.point)[1]) }), [fps])
  // The moving parts: the particles, the trajectory so far and its head.
  const upTo = Math.min(path.x.length - 1, frame * PATH_SUBSTEPS)
  const cloud = particles?.[frame]
  const time = (k: number) => `t = ${formatNumber((PORTRAIT_T * k) / PORTRAIT_FRAMES)}`
  return (
    <Figure
      title="Phase portrait"
      purpose="Fixed points, their linearisation and the nullclines organise every trajectory of a planar flow: drag the start and play the flow to watch the path and a cloud of particles follow the arrows towards an attractor."
      defaultSize="L"
      state={state}
      controls={
        <Player
          className="col-span-full"
          value={frame}
          onChange={setFrame}
          count={PORTRAIT_FRAMES + 1}
          format={time}
          label="3 · time"
        />
      }
      readouts={{
        'fixed points': (
          <>
            {fps.map((q, k) => (
              <Readout
                key={k}
                label={`(${fmt(toFlat(q.point)[0])}, ${fmt(toFlat(q.point)[1])})`}
                value={`${q.kind}; λ = ${toFlat(realPart(q.eigen.values))
                  .map((re, i) => {
                    const im = toFlat(imagPart(q.eigen.values))[i]
                    return Math.abs(im) < 1e-9 ? fmt(re) : `${fmt(re)} ${im < 0 ? '−' : '+'} ${fmt(Math.abs(im))}i`
                  })
                  .join(', ')}`}
              />
            ))}
          </>
        ),
        flow: (
          <>
            <Readout label="time" value={time(frame)} />
            {cycle && (
              <Readout
                label="limit cycle"
                value={`period ${fmt(cycle.period)}, Floquet multiplier ${fmt(cycle.multiplier)}${cycle.converged ? '' : ' (not converged)'}`}
              />
            )}
          </>
        ),
      }}
      caption={`Grey ticks: the direction field. Dashed: the nullclines, where the flow is vertical (ẋ = 0) or horizontal (ẏ = 0); fixed points (ink) sit where they cross, found by Newton's method and classified by the eigenvalues of the Jacobian. Play the time: the trajectory from x₀ is drawn up to t, and 120 particles that start on a grid over the box move with the flow (by RK4) and gather on the attractors. Drag x₀ to start the trajectory elsewhere. The stable manifolds of a saddle separate the basins of attraction.`}
    >
      <Plot x={xAxis} y={yAxis}>
        <SegmentsLayer segments={arrows} />
        {showNullclines && <Curve name="ẋ = 0 nullcline" x={clines[0].x} y={clines[0].y} slot={1} dashed />}
        {showNullclines && <Curve name="ẏ = 0 nullcline" x={clines[1].x} y={clines[1].y} slot={2} dashed />}
        {manifolds.length > 0 && <Curve name="stable manifolds" x={stable.x} y={stable.y} slot={3} />}
        {manifolds.length > 0 && <Curve name="unstable manifolds" x={unstable.x} y={unstable.y} slot={4} />}
        {cycleCurve && <Curve name="limit cycle" x={cycleCurve.x} y={cycleCurve.y} slot={5} />}
        <Points name="fixed points" x={fixed.x} y={fixed.y} emphasis />
        {cloud && <Points name="particles" x={cloud.x} y={cloud.y} muted thin live />}
        <Curve
          name="trajectory"
          x={path.x.slice(0, upTo + 1)}
          y={path.y.slice(0, upTo + 1)}
          slot={0}
          stale={pathRun.stale}
        />
        <Points name="x(t)" x={[path.x[upTo]]} y={[path.y[upTo]]} emphasis live />
        <Handle {...state.handle(['system.sx', 'system.sy'], { label: 'x₀' })} />
      </Plot>
    </Figure>
  )
}

// ---------------------------------------------------------------------------------------------------------------------
// Transport of a density along a flow (Liouville).

/** A planar flow as plain numbers (fast to follow for every grid point), with its divergence. */
type Flow = {
  value: string
  label: string
  v: (a: number, b: number) => [number, number]
  divergence: (a: number, b: number) => number
}

const FLOWS: Flow[] = [
  {
    value: 'spiral',
    label: 'spiral sink (contracting)',
    v: (a, b) => [-0.3 * a - b, a - 0.3 * b],
    divergence: () => -0.6,
  },
  {
    value: 'pendulum',
    label: 'pendulum (area-preserving, shearing)',
    v: (a, b) => [b, -Math.sin(a)],
    divergence: () => 0,
  },
  {
    value: 'saddle',
    label: 'saddle (area-preserving, stretching)',
    v: (a, b) => [a, -b],
    divergence: () => 0,
  },
  {
    value: 'source',
    label: 'nonlinear source (expanding)',
    v: (a, b) => [0.5 * a * (1 - (a * a) / 4), 0.5 * b],
    divergence: (a) => 0.5 * (1 - (3 * a * a) / 4) + 0.5,
  },
]

const N = 61
const LIM = 3
const CENTRE = [1, 0.5]
const SD = 0.4
const T_MAX = 4
const T_FRAMES = 40
const PARTICLES = 120
const rho0 = (x: Tensor) => {
  const [a, b] = toFlat(x)
  return Math.exp(-((a - CENTRE[0]) ** 2 + (b - CENTRE[1]) ** 2) / (2 * SD * SD)) / (2 * Math.PI * SD * SD)
}

export function TransportSpecimen() {
  const state = useFigureState({
    flow: row('1 · flow', {
      which: choice(
        FLOWS.map(({ value, label }) => ({ value, label })),
        'spiral',
        { label: 'flow' },
      ),
    }),
    reveal: row('2 · reveal', { particles: toggle(true, 'particles pushed by the flow') }),
  })
  const which = state.flow.which
  const particles = state.reveal.particles
  const [frame, setFrame] = usePlayhead(T_FRAMES + 1)
  const flow = FLOWS.find((f) => f.value === which)!
  const field = useMemo<VectorField>(
    () => (x) => {
      const [a, b] = toFlat(x)
      return flow.v(a, b)
    },
    [flow],
  )
  const axis = useMemo(() => Array.from({ length: N }, (_, i) => -LIM + (2 * LIM * i) / (N - 1)), [])
  const divergence = useMemo(
    () => (x: Tensor) => {
      const [a, b] = toFlat(x)
      return flow.divergence(a, b)
    },
    [flow],
  )
  // ρ(x, t) at every frame by aifn/fields: each grid point's characteristic followed backwards once over [0, T_MAX].
  const density = useMemo(() => {
    const points = axis.flatMap((y) => axis.map((x) => [x, y]))
    const rows = toRows(transportDensityFrames(field, rho0, points, T_MAX, { frames: T_FRAMES, divergence }))
    const frames = rows.map((rho) => axis.map((_, i) => rho.slice(i * N, (i + 1) * N)))
    const peak = Math.max(...rows.map((r) => Math.max(...r)))
    return { frames, peak }
  }, [field, divergence, axis])
  // 120 draws from ρ₀ carried forwards, with ∫₀ᵗ ∇·f ds along each.
  const cloud = useMemo(() => {
    const z = toRows(normals(stream('transport'), [PARTICLES, 2], 0, SD)).map(([a, b]) => [
      a + CENTRE[0],
      b + CENTRE[1],
    ])
    const moved = pushForwardDensityFrames(field, z, T_MAX, { frames: T_FRAMES, divergence })
    const x = toFlat(moved.x)
    const volume = toRows(moved.logVolume)
    return volume.map((l, k) => ({
      x: l.map((_, i) => x[(k * PARTICLES + i) * 2]),
      y: l.map((_, i) => x[(k * PARTICLES + i) * 2 + 1]),
      l,
    }))
  }, [field, divergence])
  // Flow context: streamlines through a grid of seeds, and the path of ρ₀'s centre.
  const context = useMemo(() => {
    const box: Box = [
      [-LIM, LIM],
      [-LIM, LIM],
    ]
    const seeds = [-2, 0, 2].flatMap((a) => [-2, 0, 2].map((b) => [a, b]))
    const lines = seeds.map((seed) => curve(streamline(field, seed, { t: 2.5, steps: 100, bounds: box })))
    const centre = curve(trajectory(field, CENTRE, T_MAX, { steps: 80 }).x)
    return {
      streamlines: { x: lines.flatMap((l) => [...l.x, NaN]), y: lines.flatMap((l) => [...l.y, NaN]) },
      centre,
    }
  }, [field])
  const rings = useMemo(() => {
    const x: number[] = []
    const y: number[] = []
    for (const k of [1, 2]) {
      for (let i = 0; i <= 60; i++) {
        x.push(CENTRE[0] + k * SD * Math.cos((2 * Math.PI * i) / 60))
        y.push(CENTRE[1] + k * SD * Math.sin((2 * Math.PI * i) / 60))
      }
      x.push(NaN)
      y.push(NaN)
    }
    return { x, y }
  }, [])
  const grid = density.frames[frame]
  const now = cloud[frame]
  const dots = useMemo(() => (particles ? { x: now.x, y: now.y } : { x: [], y: [] }), [now, particles])
  const xAxis = useAxis({ label: 'x' })
  const yAxis = useAxis({ label: 'y', equal: xAxis })
  // Colour by log₁₀ ρ over the two decades below the run's peak, held for the whole run: a contracting flow's peak grows
  // e^{0.6t}, so on a linear scale held to its final peak the starting density would be nearly white.
  const top = Math.log10(density.peak)
  const range = useMemo((): [number, number] => [top - 2, top], [top])
  const logGrid = useMemo(
    () => grid.map((r) => r.map((v) => Math.max(top - 2, Math.log10(Math.max(v, 1e-300))))),
    [grid, top],
  )
  const cell = (2 * LIM) / (N - 1)
  const mass = grid.flat().reduce((a, b) => a + b, 0) * cell * cell
  const t = (T_MAX * frame) / T_FRAMES
  const finite = now.l.filter(Number.isFinite)
  const time = (k: number) => `t = ${formatNumber((T_MAX * k) / T_FRAMES)}`
  return (
    <Figure
      title="Transport of a density along a flow"
      purpose="A density carried by x′ = f(x) changes along each trajectory at the rate −ρ ∇·f: area-preserving flows only reshape it, contracting ones concentrate it and expanding ones thin it out."
      state={state}
      controls={
        <Player
          className="col-span-full"
          value={frame}
          onChange={setFrame}
          count={T_FRAMES + 1}
          format={time}
          label="3 · time"
        />
      }
      readouts={{
        'at t': (
          <>
            <Readout label="t" value={fmt(t)} />
            <Readout label="mass on the grid" value={fmt(mass)} />
            <Readout label="peak density" value={fmt(Math.max(...grid.map((r) => Math.max(...r))))} />
            <Readout
              label="∫₀ᵗ ∇·f ds along the particles"
              value={finite.length ? `${fmt(Math.min(...finite))} to ${fmt(Math.max(...finite))}` : '–'}
            />
          </>
        ),
      }}
      caption="ρ(x, t) = ρ₀(φ₋ₜ(x)) · exp(−∫₀ᵗ ∇·f ds), computed by following each grid point backwards along the flow (the method of characteristics); ρ₀ is a Gaussian at (1, 0.5), its 1σ and 2σ circles drawn thin, with streamlines of the flow and the path of ρ₀'s centre. Play the time. The spiral sink contracts areas (∇·f = −0.6), so the density rises as e^{0.6t}; the pendulum and the saddle preserve area (∇·f = 0), so the peak stays put while the blob shears or stretches; the source expands it. Colour is log₁₀ ρ over two decades below the run's peak, held over the whole run, so a stronger colour means a denser blob at any time. The particles are 120 draws from ρ₀ moved forwards by the same flow: they cover the same region as the density. Mass is conserved until the density leaves the grid."
    >
      <Plot x={xAxis} y={yAxis}>
        <Raster x={axis} y={axis} z={logGrid} range={range} valueLabel="log₁₀ ρ" />
        <Curve name="streamlines" x={context.streamlines.x} y={context.streamlines.y} slot={3} thin />
        <Curve name="ρ₀ at 1σ and 2σ" x={rings.x} y={rings.y} slot={2} thin />
        <Curve name="path of ρ₀’s centre" x={context.centre.x} y={context.centre.y} slot={4} />
        <Points name="particles" x={dots.x} y={dots.y} slot={7} size={4} live />
      </Plot>
    </Figure>
  )
}
