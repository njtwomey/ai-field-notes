import { useMemo } from 'react'
import {
  Curve,
  Figure,
  formatNumber,
  Handle,
  Player,
  Plot,
  Points,
  Readout,
  Segments,
  setting,
  Shapes,
  slider,
  useAxis,
  useComputed,
  useFigureState,
  usePlayhead,
  Vectors,
  type FilledShape,
} from 'aifn-render'
import { conic, elements, integrateOrbit, polygonArea, upTo, type V2 } from './kepler'

const fmt = (v: number) => (Number.isFinite(v) ? formatNumber(v) : '∞')

/** Frames of the Player over the integrated run. */
const FRAMES = 240
/** The number of equal time slices shaded for the second law. */
const SLICES = 12
/** The view is the square [−VIEW, VIEW]²; an unbound orbit is followed to EXIT. */
const VIEW = 2.5
const EXIT = 8
/** The start may not sit closer to the mass than this. */
const MIN_R = 0.2

const KIND_LABEL = { ellipse: 'ellipse (ℰ < 0)', parabola: 'parabola (ℰ ≈ 0)', hyperbola: 'hyperbola (ℰ > 0)' }

/** The sector swept between run indices i and j: the mass at the origin, then the orbit's points. */
function sector(x: readonly number[], y: readonly number[], i: number, j: number) {
  const sx = [0, ...x.slice(i, j + 1)]
  const sy = [0, ...y.slice(i, j + 1)]
  return { sx, sy, area: Math.abs(polygonArea(sx, sy)) }
}

export function KeplerOrbit() {
  const state = useFigureState({
    x0: slider(-2.4, 2.4, 1, { onChart: true, label: 'start x₀' }),
    y0: slider(-2.4, 2.4, 0, { onChart: true, label: 'start y₀' }),
    vx: slider(-2, 2, 0.3, { onChart: true, label: 'velocity v_x' }),
    vy: slider(-2, 2, 0.7, { onChart: true, label: 'velocity v_y' }),
    show: setting(true, 'shade equal times'),
  })
  const { x0, y0, vx, vy, show } = state
  const r0: V2 = [x0, y0]
  const v0: V2 = [vx, vy]
  const el = useMemo(() => elements([x0, y0], [vx, vy]), [x0, y0, vx, vy])
  const computed = useComputed(() => integrateOrbit([x0, y0], [vx, vy], { rExit: EXIT }), [x0, y0, vx, vy])
  const orbit = computed.value
  const exact = useMemo(() => conic(el, EXIT), [el])

  const tEnd = orbit.t[orbit.t.length - 1] ?? 0
  const [frame, setFrame] = usePlayhead(FRAMES + 1)
  const tNow = (tEnd * frame) / FRAMES
  const at = upTo(orbit.t, tNow + 1e-12)

  // Equal time slices of the run: alternate ones shaded, each with its swept area.
  const slices = useMemo(() => {
    const out: { shape: FilledShape; area: number }[] = []
    for (let k = 0; k < SLICES; k++) {
      const i = upTo(orbit.t, (tEnd * k) / SLICES + 1e-12)
      const j = upTo(orbit.t, (tEnd * (k + 1)) / SLICES + 1e-12)
      if (j <= i) continue
      const { sx, sy, area } = sector(orbit.x, orbit.y, i, j)
      out.push({
        shape: { contours: [sx.map((v, n) => [v, sy[n]] as const)], tone: 1, opacity: k % 2 ? 0.08 : 0.28 },
        area,
      })
    }
    return out
  }, [orbit, tEnd])
  const areas = slices.map((s) => s.area)
  const areaSpread = areas.length ? (Math.max(...areas) - Math.min(...areas)) / Math.max(...areas) : NaN
  const maxEnergyError = orbit.energyError.reduce((m, v) => Math.max(m, Math.abs(v)), 0)

  const xa = useAxis({ label: 'x', range: [-VIEW, VIEW] })
  const ya = useAxis({ label: 'y', range: [-VIEW, VIEW], equal: xa })

  const moveStart = ([px, py]: V2) => {
    const r = Math.hypot(px, py)
    if (r < 1e-9) {
      state.set('x0', MIN_R)
      state.set('y0', 0)
      return
    }
    const s = r < MIN_R ? MIN_R / r : 1
    state.set('x0', px * s)
    state.set('y0', py * s)
  }
  const moveTip = ([px, py]: V2) => {
    state.set('vx', px - x0)
    state.set('vy', py - y0)
  }
  const body: V2 = [orbit.x[at] ?? x0, orbit.y[at] ?? y0]
  const speed = Math.hypot(vx, vy)
  const r = Math.hypot(x0, y0)

  return (
    <Figure
      title="Orbits from one start"
      state={state}
      defaultSize="L"
      aspect={1}
      controls={
        <Player
          value={frame}
          onChange={setFrame}
          count={FRAMES + 1}
          label="time t"
          format={(i) => `t = ${((tEnd * i) / FRAMES).toFixed(2)}`}
        />
      }
      readouts={{
        'conserved quantities': (
          <>
            <Readout label="energy ℰ = v²/2 − 1/r" value={fmt(el.energy)} />
            <Readout label="angular momentum h" value={fmt(el.angularMomentum)} />
            <Readout label="eccentricity e" value={fmt(el.eccentricity)} />
            <Readout label="orbit" value={KIND_LABEL[el.kind]} />
          </>
        ),
        'the orbit': (
          <>
            <Readout label="semi-major axis a" value={el.energy < 0 ? fmt(el.semiMajorAxis) : '–'} />
            <Readout label="period T = 2π√(a³)" value={fmt(el.period)} />
            <Readout label="pericentre r_p" value={fmt(el.pericentre)} />
            <Readout
              label="speed |v₀| (circular 1/√r₀, escape √(2/r₀))"
              value={`${fmt(speed)} (${fmt(1 / Math.sqrt(r))}, ${fmt(Math.sqrt(2 / r))})`}
            />
          </>
        ),
        integration: (
          <>
            <Readout label="velocity Verlet steps (step size)" value={`${orbit.steps} (${fmt(orbit.stepSize)})`} />
            <Readout label="largest energy error" value={fmt(maxEnergyError)} />
            <Readout label="slice areas: spread / largest" value={fmt(areaSpread)} />
          </>
        ),
      }}
      caption="One body under the attraction of a unit mass at the origin (G M = 1). Drag the start point, and drag the arrow's tip to set the initial velocity. The solid curve is the orbit integrated by velocity Verlet for one period (an ellipse) or until it leaves the view (a parabola or hyperbola); the dashed curve is the conic p/(1 + e cos ν) from the readouts. Speeds below the escape speed √(2/r₀) close the orbit; speeds above it open it. The shaded wedges cover equal times, and their areas agree, which is Kepler's second law. Play to move the body along the orbit. Starts aimed almost at the mass pass so close to it that the fixed-step integrator fails, and the energy error readout shows it."
    >
      <Plot x={xa} y={ya}>
        {show && <Shapes name="equal times" shapes={slices.map((s) => s.shape)} />}
        <Curve name="conic (closed form)" x={exact.x} y={exact.y} muted dashed />
        <Curve name="orbit (velocity Verlet)" x={orbit.x} y={orbit.y} slot={0} stale={computed.stale} />
        <Segments name="radius" segments={[{ from: [0, 0], to: body }]} />
        <Points name="mass" x={[0]} y={[0]} emphasis size={12} />
        <Points name="body" x={[body[0]]} y={[body[1]]} slot={0} size={9} live />
        <Vectors vectors={[{ from: r0, to: [x0 + v0[0], y0 + v0[1]], slot: 2, label: 'v₀' }]} />
        <Handle kind="point" at={r0} label="start" onDrag={moveStart} />
        <Handle kind="point" at={[x0 + vx, y0 + vy]} label="velocity" onDrag={moveTip} />
      </Plot>
    </Figure>
  )
}
