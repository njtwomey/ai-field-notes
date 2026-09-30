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
} from 'aifn/dynamics/fields'
import { directionField, nullclines, type Segments } from '@lab/viz/drawing/fields'
import { limitCycle } from 'aifn-applied/dynamics/nonlinear'
import { normals, stream } from 'aifn/foundation/random'
import { get, mul, neg, sin, stack, sub, toFlat, toRows, type Tensor } from 'aifn/foundation/tensor'
import { useMemo, useState } from 'react'
import { Player, Select, Slider, Switch, usePlayhead } from '@lab/controls'
import { ControlRow, Figure } from '@lab/layout'
import {
  Heatmap,
  Readout,
  XYChart,
  formatNumber,
  type Handle,
  type HeatmapOverlay,
  type Segment,
  type XYSeries,
} from '@lab/viz'

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

const SYSTEMS: System[] = [
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

/** The time a phase portrait plays over, and its frames. */
const PORTRAIT_T = 15
const PORTRAIT_FRAMES = 150
/** RK4 steps per frame of the main trajectory. */
const PATH_SUBSTEPS = 6

export function PhasePortraitSpecimen() {
  const [which, setWhich] = useState('duffing')
  const system = SYSTEMS.find((s) => s.value === which)!
  const [params, setParams] = useState<Record<string, number>>(() =>
    Object.fromEntries(SYSTEMS.map((s) => [s.value, s.param.initial])),
  )
  const p = params[which]
  const [starts, setStarts] = useState<Record<string, [number, number]>>(() =>
    Object.fromEntries(SYSTEMS.map((s) => [s.value, s.start])),
  )
  const x0 = starts[which]
  const [showNullclines, setShowNullclines] = useState(true)
  const [showManifolds, setShowManifolds] = useState(true)
  const [showCycle, setShowCycle] = useState(false)
  const [showParticles, setShowParticles] = useState(true)
  const [frame, setFrame] = usePlayhead(PORTRAIT_FRAMES + 1)

  const f = useMemo(() => system.field(p), [system, p])
  const box = system.box as Box
  const grid = useMemo(() => ({ x: system.box[0], y: system.box[1] }), [system])
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
  const path = useMemo(() => {
    const s = streamline(f, x0, {
      t: PORTRAIT_T,
      steps: PORTRAIT_FRAMES * PATH_SUBSTEPS,
      direction: 'forward',
      bounds: box,
    })
    return curve(s)
  }, [f, x0, box])
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
  const series = useMemo(() => {
    const out: XYSeries[] = []
    if (showNullclines) {
      out.push({ name: 'ẋ = 0 nullcline', type: 'line', dashed: true, slot: 1, ...clines[0] })
      out.push({ name: 'ẏ = 0 nullcline', type: 'line', dashed: true, slot: 2, ...clines[1] })
    }
    if (manifolds.length) {
      out.push({
        name: 'stable manifolds',
        type: 'line',
        slot: 3,
        ...join(manifolds.filter((m) => m.kind === 'stable')),
      })
      out.push({
        name: 'unstable manifolds',
        type: 'line',
        slot: 4,
        ...join(manifolds.filter((m) => m.kind === 'unstable')),
      })
    }
    if (cycle) out.push({ name: 'limit cycle', type: 'line', slot: 5, ...curve(cycle.orbit) })
    out.push({
      name: 'fixed points',
      type: 'scatter',
      emphasis: true,
      x: fps.map((q) => toFlat(q.point)[0]),
      y: fps.map((q) => toFlat(q.point)[1]),
    })
    return out
  }, [showNullclines, clines, manifolds, cycle, fps])
  // The moving parts: the particles, the trajectory so far and its head.
  const upTo = Math.min(path.x.length - 1, frame * PATH_SUBSTEPS)
  const cloud = particles?.[frame]
  const live = useMemo(
    (): XYSeries[] => [
      { name: 'particles', type: 'scatter', muted: true, x: cloud?.x ?? [], y: cloud?.y ?? [] },
      { name: 'trajectory', type: 'line', slot: 0, x: path.x.slice(0, upTo + 1), y: path.y.slice(0, upTo + 1) },
      { name: 'x(t)', type: 'scatter', emphasis: true, x: [path.x[upTo]], y: [path.y[upTo]] },
    ],
    [cloud, path, upTo],
  )
  const clamp = (v: number, [a, b]: readonly [number, number]) => Math.min(b, Math.max(a, v))
  const handles: Handle[] = [
    {
      kind: 'point',
      at: x0,
      label: 'x₀',
      onDrag: ([a, b]) => setStarts((s) => ({ ...s, [which]: [clamp(a, grid.x), clamp(b, grid.y)] })),
    },
  ]
  const time = (k: number) => `t = ${formatNumber((PORTRAIT_T * k) / PORTRAIT_FRAMES)}`
  return (
    <Figure
      title="Phase portrait"
      description="Fixed points, their linearisation and the nullclines organise every trajectory of a planar flow: drag the start and play the flow to watch the path and a cloud of particles follow the arrows towards an attractor."
      defaultSize="L"
      controls={
        <>
          <ControlRow label="1 · system">
            <Select label="system" value={which} onChange={setWhich} options={SYSTEMS} />
            <Slider
              label={system.param.label}
              value={p}
              min={system.param.min}
              max={system.param.max}
              step={0.01}
              onChange={(v) => setParams((s) => ({ ...s, [which]: v }))}
            />
          </ControlRow>
          <ControlRow label="2 · reveal">
            <Switch label="nullclines" checked={showNullclines} onChange={setShowNullclines} />
            <Switch label="saddle manifolds" checked={showManifolds} onChange={setShowManifolds} />
            {system.section && <Switch label="limit cycle" checked={showCycle} onChange={setShowCycle} />}
            <Switch label="particles" checked={showParticles} onChange={setShowParticles} />
          </ControlRow>
          <ControlRow label="3 · time">
            <Player
              className="col-span-full"
              value={frame}
              onChange={setFrame}
              count={PORTRAIT_FRAMES + 1}
              format={time}
              label="time"
              duration={5}
            />
          </ControlRow>
        </>
      }
      readouts={
        <>
          <Readout label="time" value={time(frame)} />
          {fps.map((q, k) => (
            <Readout
              key={k}
              label={`(${fmt(toFlat(q.point)[0])}, ${fmt(toFlat(q.point)[1])})`}
              value={`${q.kind}; λ = ${toFlat(q.eigen.real)
                .map((re, i) => {
                  const im = toFlat(q.eigen.imag)[i]
                  return Math.abs(im) < 1e-9 ? fmt(re) : `${fmt(re)} ${im < 0 ? '−' : '+'} ${fmt(Math.abs(im))}i`
                })
                .join(', ')}`}
            />
          ))}
          {cycle && (
            <Readout
              label="limit cycle"
              value={`period ${fmt(cycle.period)}, Floquet multiplier ${fmt(cycle.multiplier)}${cycle.converged ? '' : ' (not converged)'}`}
            />
          )}
        </>
      }
      caption={`Grey ticks: the direction field. Dashed: the nullclines, where the flow is vertical (ẋ = 0) or horizontal (ẏ = 0); fixed points (ink) sit where they cross, found by Newton's method and classified by the eigenvalues of the Jacobian. Play the time: the trajectory from x₀ is drawn up to t, and 120 particles that start on a grid over the box move with the flow (by RK4) and gather on the attractors. Drag x₀ to start the trajectory elsewhere. The stable manifolds of a saddle separate the basins of attraction.`}
    >
      <XYChart
        series={series}
        live={live}
        segments={arrows}
        handles={handles}
        xRange={system.box[0]}
        yRange={system.box[1]}
        axisKey={which}
        rescaleOnChange={false}
        xLabel="x"
        yLabel="y"
      />
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
  const [which, setWhich] = useState('spiral')
  const [particles, setParticles] = useState(true)
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
  const overlay = useMemo(
    (): HeatmapOverlay[] => [
      { name: 'streamlines', type: 'line', thin: true, slot: 3, ...context.streamlines },
      { name: 'ρ₀ at 1σ and 2σ', type: 'line', thin: true, slot: 2, ...rings },
      { name: 'path of ρ₀’s centre', type: 'line', slot: 4, ...context.centre },
      // Drawn as the vertices of a line broken after every point: small dots that leave the density visible.
      {
        name: 'particles',
        type: 'line',
        showPoints: true,
        thin: true,
        slot: 7,
        x: particles ? now.x.flatMap((v) => [v, NaN]) : [],
        y: particles ? now.y.flatMap((v) => [v, NaN]) : [],
      },
    ],
    [context, rings, now, particles],
  )
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
      description="A density carried by x′ = f(x) changes along each trajectory at the rate −ρ ∇·f: area-preserving flows only reshape it, contracting ones concentrate it and expanding ones thin it out."
      controls={
        <>
          <ControlRow label="1 · flow">
            <Select label="flow" value={which} onChange={setWhich} options={FLOWS} />
            <Switch label="particles pushed by the flow" checked={particles} onChange={setParticles} />
          </ControlRow>
          <ControlRow label="2 · time">
            <Player
              className="col-span-full"
              value={frame}
              onChange={setFrame}
              count={T_FRAMES + 1}
              format={time}
              label="time"
              duration={4}
            />
          </ControlRow>
        </>
      }
      readouts={
        <>
          <Readout label="t" value={fmt(t)} />
          <Readout label="mass on the grid" value={fmt(mass)} />
          <Readout label="peak density" value={fmt(Math.max(...grid.map((r) => Math.max(...r))))} />
          <Readout
            label="∫₀ᵗ ∇·f ds along the particles"
            value={finite.length ? `${fmt(Math.min(...finite))} to ${fmt(Math.max(...finite))}` : '–'}
          />
        </>
      }
      caption="ρ(x, t) = ρ₀(φ₋ₜ(x)) · exp(−∫₀ᵗ ∇·f ds), computed by following each grid point backwards along the flow (the method of characteristics); ρ₀ is a Gaussian at (1, 0.5), its 1σ and 2σ circles drawn thin. Play the time. The spiral sink contracts areas (∇·f = −0.6), so the density rises as e^{0.6t}; the pendulum and the saddle preserve area (∇·f = 0), so the peak stays put while the blob shears or stretches; the source expands it. Colour is log₁₀ ρ over two decades below the run's peak, held over the whole run, so a stronger colour means a denser blob at any time. The particles are 120 draws from ρ₀ moved forwards by the same flow: they cover the same region as the density. Mass is conserved until the density leaves the grid."
    >
      <Heatmap
        x={axis}
        y={axis}
        z={logGrid}
        range={range}
        xLabel="x"
        yLabel="y"
        valueLabel="log₁₀ ρ"
        equalAspect
        axisKey={which}
        rescaleOnChange={false}
        overlay={overlay}
      />
    </Figure>
  )
}
