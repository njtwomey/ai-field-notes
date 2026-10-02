import {
  bifurcationDiagram,
  cobweb,
  henonMap,
  logisticMap,
  lyapunovCurve,
  lyapunovExponent,
  lyapunovSpectrum,
  orbit,
  sineMap,
  standardMap,
  tentMap,
  type Map1,
  type MapN,
} from 'aifn-applied/dynamics/maps'
import { histogram } from 'aifn/probability/stats'
import { toFlat, toRows } from 'aifn/foundation/tensor'
import { useMemo } from 'react'
import { Player, usePlayhead } from '@lab/controls'
import { ControlRow, Dashboard, DashboardCell, DashboardRow, Figure } from '@lab/layout'
import { slider, useFigureState, variants } from '@lab/state'
import { Curve, formatNumber, Handle, Plot, Plots, Points, Raster, Readout, useAxis } from '@lab/viz'

const fmt = (v: number) => formatNumber(v)

/**
 * Visit counts of points on a grid, for drawing dense orbits as a heatmap: each row band of y is histogrammed in x by
 * `aifn/probability/stats`. Returns log(1 + count) so that rarely visited cells stay visible.
 */
function visits(
  xs: readonly number[],
  ys: readonly number[],
  xr: [number, number],
  yr: [number, number],
  nx: number,
  ny: number,
) {
  const dy = (yr[1] - yr[0]) / ny
  const rows: number[][] = []
  for (let i = 0; i < ny; i++) {
    const lo = yr[0] + i * dy
    const inRow = xs.filter((_, k) => ys[k] >= lo && ys[k] < lo + dy)
    const h = histogram(inRow.length ? inRow : [NaN], { bins: nx, range: xr })
    rows.push(toFlat(h.counts).map((c) => Math.log1p(c)))
  }
  const cx = Array.from({ length: nx }, (_, j) => xr[0] + ((j + 0.5) * (xr[1] - xr[0])) / nx)
  const cy = Array.from({ length: ny }, (_, i) => yr[0] + (i + 0.5) * dy)
  return { x: cx, y: cy, z: rows }
}

// ---------------------------------------------------------------------------------------------------------------------
// Bifurcation diagram, cobweb and Lyapunov exponent.

type Family = { label: string; make: (r: number) => Map1; range: [number, number]; initial: number }

const FAMILIES = {
  logistic: { label: 'logistic r·x(1 − x)', make: logisticMap, range: [2.5, 4], initial: 3.56 },
  sine: { label: 'sine r·sin(πx)', make: sineMap, range: [0.6, 1], initial: 0.87 },
  tent: { label: 'tent μ·min(x, 1 − x)', make: tentMap, range: [1, 2], initial: 1.5 },
} satisfies Record<string, Family>
type FamilyName = keyof typeof FAMILIES

/** The family and its parameter, each family with its own range and remembered value. */
const FAMILY = variants(
  Object.fromEntries(
    (Object.keys(FAMILIES) as FamilyName[]).map((k) => {
      const f: Family = FAMILIES[k]
      return [
        k,
        {
          label: f.label,
          params: { r: slider(f.range[0], f.range[1], f.initial, { label: 'parameter r', step: 0.001 }) },
        },
      ]
    }),
  ) as Record<FamilyName, { label: string; params: { r: ReturnType<typeof slider> } }>,
  { label: '1 · family', choiceLabel: 'map' },
)

const NR = 300
const NX = 200
/** Iterations the cobweb plays through. */
const COBWEB_N = 200
/** Positions of the r sweep. */
const SWEEP = 301

export function BifurcationSpecimen() {
  const state = useFigureState({
    family: FAMILY,
    x0: slider(0, 1, 0.2, { label: 'x₀', step: 0.001, onChart: true }),
  })
  const which = state.family.key as FamilyName
  const family: Family = FAMILIES[which]
  const r = state.family.values.r as number
  const setR = (v: number) => state.set('family.r', v)
  const x0 = state.x0
  const [n, setN] = usePlayhead(COBWEB_N + 1)

  const rs = useMemo(
    () =>
      Array.from({ length: NR }, (_, i) => family.range[0] + ((family.range[1] - family.range[0]) * (i + 0.5)) / NR),
    [family],
  )
  const diagram = useMemo(() => {
    const d = bifurcationDiagram(family.make, rs, { x0: 0.2345, transient: 400, keep: 300 })
    const rr = toFlat(d.r)
    const xx = toFlat(d.x)
    // One column per parameter: the histogram of its attractor over [0, 1].
    const columns = rs
      .map(
        (rv) =>
          histogram(
            xx.filter((_, k) => rr[k] === rv),
            { bins: NX, range: [0, 1] },
          ).counts,
      )
      .map((c) => toFlat(c))
    const z = Array.from({ length: NX }, (_, i) => columns.map((c) => Math.log1p(c[i])))
    const y = Array.from({ length: NX }, (_, i) => (i + 0.5) / NX)
    return { x: rs, y, z }
  }, [family, rs])
  const lyap = useMemo(() => toFlat(lyapunovCurve(family.make, rs, { x0: 0.2345, keep: 600 })), [family, rs])
  const map = useMemo(() => family.make(r), [family, r])
  // The whole cobweb once per map and start; the player reveals its first n iterations.
  const web = useMemo(() => {
    const w = cobweb(map, x0, COBWEB_N)
    return { x: toFlat(w.x), y: toFlat(w.y) }
  }, [map, x0])
  const lambda = useMemo(() => lyapunovExponent(map, 0.2345, { keep: 5000 }).exponent, [map])
  const late = useMemo(() => toFlat(orbit(map, x0, 8, { discard: 2000 })), [map, x0])
  const graph = useMemo(() => {
    const xs = Array.from({ length: 201 }, (_, i) => i / 200)
    return { x: xs, y: xs.map(map.f) }
  }, [map])
  const zero = useMemo(() => ({ x: [rs[0], rs[rs.length - 1]], y: [0, 0] }), [rs])
  const shown = Math.min(web.x.length, 2 * n + 1)
  const xn = web.x[shown - 1]
  const webNow = { x: web.x.slice(0, shown), y: web.y.slice(0, shown) }
  const rAxis = useAxis({ label: 'r', range: family.range, nice: false })
  const xAxis = useAxis({ label: 'x', range: [0, 1], nice: false })
  const lAxis = useAxis({ label: 'λ', range: [-2, 1] })
  const cx = useAxis({ label: 'x', range: [0, 1] })
  const cy = useAxis({ label: 'f(x)', range: [0, 1], equal: cx })
  const sweepAt = Math.round(((r - family.range[0]) / (family.range[1] - family.range[0])) * (SWEEP - 1))
  const sweepR = (k: number) => family.range[0] + ((family.range[1] - family.range[0]) * k) / (SWEEP - 1)
  return (
    <Figure
      title="Bifurcation diagram and cobweb"
      purpose="The long-run orbit of a unimodal map as its parameter grows: period doubling into chaos, with windows of order; the Lyapunov exponent turns positive where the diagram fills in."
      state={state}
      defaultSize="XL"
      controls={
        <>
          <ControlRow label="2 · sweep r">
            <Player
              className="col-span-full"
              value={sweepAt}
              onChange={(k) => setR(sweepR(k))}
              count={SWEEP}
              format={(k) => `r = ${sweepR(k).toFixed(3)}`}
              label="r"
              startReason="The sweep is a view of r, which opens at the family's chosen value."
            />
          </ControlRow>
          <ControlRow label="3 · cobweb">
            <Player
              className="col-span-full"
              value={n}
              onChange={setN}
              count={COBWEB_N + 1}
              format={(k) => `n = ${k}`}
              label="iteration"
            />
          </ControlRow>
        </>
      }
      readouts={
        <>
          <Readout label="Lyapunov exponent λ(r)" value={fmt(lambda)} />
          <Readout label="x_n" value={`n = ${Math.floor((shown - 1) / 2)}: ${fmt(xn)}`} />
          <Readout label="late orbit" value={late.map((v) => v.toFixed(3)).join(', ')} />
        </>
      }
      caption="Left: the attractor for each parameter (log visit counts of 300 iterates after 400 discarded), with the Lyapunov exponent below; drag the r line on either (or the slider). Play the sweep to move r across the range. Right: the graph of the map, the diagonal and the cobweb from x₀ (drag it), built one iteration at a time as the iteration plays: the vertical step to the graph is x_{n+1} = f(x_n), and the horizontal step to the diagonal makes that value the next input. For the logistic map λ = ln 2 at r = 4; the tent map has λ = ln μ for every μ > 1, so it is chaotic as soon as it is expanding."
    >
      <Dashboard>
        <DashboardRow minHeight={420}>
          <DashboardCell ratio={1.6}>
            <Plots rows={2} heights={[3, 1.2]}>
              <Plot x={rAxis} y={xAxis}>
                <Raster x={diagram.x} y={diagram.y} z={diagram.z} valueLabel="log(1 + visits)" colorBar={false} />
                <Handle kind="x" at={r} onDrag={setR} label="r" />
              </Plot>
              <Plot x={rAxis} y={lAxis} legend={false}>
                <Curve name="λ(r)" x={rs} y={lyap} slot={1} />
                <Curve name="zero" x={zero.x} y={zero.y} muted />
                <Handle kind="x" at={r} onDrag={setR} label="r" />
              </Plot>
            </Plots>
          </DashboardCell>
          <DashboardCell aspect="square">
            <Plot x={cx} y={cy}>
              <Curve name="f" x={graph.x} y={graph.y} slot={0} />
              <Curve name="y = x" x={[0, 1]} y={[0, 1]} muted />
              <Curve name="cobweb" x={webNow.x} y={webNow.y} slot={2} live />
              <Points name="x_n" x={[xn]} y={[xn]} emphasis live />
              <Handle {...state.handle('x0', { label: 'x₀', axis: 'x' })} />
            </Plot>
          </DashboardCell>
        </DashboardRow>
      </Dashboard>
    </Figure>
  )
}

// ---------------------------------------------------------------------------------------------------------------------
// Chaotic maps of the plane.

type PlaneMap = {
  label: string
  param: { label: string; min: number; max: number; initial: number }
  make: (p: number) => MapN
  xr: [number, number]
  yr: [number, number]
  /** Starting points of the orbits drawn. */
  starts: number[][]
}

const TAU = 2 * Math.PI
/** Iterates of the first orbit revealed by the player. */
const REVEAL = 1000
const PLANE_MAPS = {
  henon: {
    label: 'Hénon map',
    param: { label: 'a (b = 0.3)', min: 1, max: 1.42, initial: 1.4 },
    make: (a) => henonMap(a, 0.3),
    xr: [-1.5, 1.5],
    yr: [-0.45, 0.45],
    starts: [[0.1, 0.1]],
  },
  standard: {
    label: 'standard map',
    param: { label: 'K', min: 0, max: 2.5, initial: 0.97 },
    make: standardMap,
    xr: [0, TAU],
    yr: [0, TAU],
    starts: Array.from({ length: 40 }, (_, k) => [Math.PI + 0.01 * k, (TAU * (k + 0.5)) / 40]),
  },
} satisfies Record<string, PlaneMap>
type PlaneName = keyof typeof PLANE_MAPS

const PLANE = variants(
  Object.fromEntries(
    (Object.keys(PLANE_MAPS) as PlaneName[]).map((k) => {
      const m: PlaneMap = PLANE_MAPS[k]
      return [
        k,
        {
          label: m.label,
          params: { p: slider(m.param.min, m.param.max, m.param.initial, { label: m.param.label, step: 0.01 }) },
        },
      ]
    }),
  ) as Record<PlaneName, { label: string; params: { p: ReturnType<typeof slider> } }>,
  { label: '1 · map', choiceLabel: 'map' },
)

export function PlaneMapSpecimen() {
  const state = useFigureState({ map: PLANE })
  const which = state.map.key as PlaneName
  const pm: PlaneMap = PLANE_MAPS[which]
  const p = state.map.values.p as number
  const map = useMemo(() => pm.make(p), [pm, p])
  const picture = useMemo(() => {
    const per = Math.floor(30_000 / pm.starts.length)
    const xs: number[] = []
    const ys: number[] = []
    for (const s of pm.starts) {
      const pts = toRows(orbit(map, s, per, { discard: which === 'henon' ? 100 : 0 }))
      for (const [a, b] of pts) {
        xs.push(a)
        ys.push(b)
      }
    }
    return visits(xs, ys, pm.xr, pm.yr, 220, which === 'henon' ? 120 : 220)
  }, [map, pm, which])
  const spectrum = useMemo(() => toFlat(lyapunovSpectrum(map, pm.starts[0], { keep: 3000 }).exponents), [map, pm])
  // The first orbit from its start, transient included, revealed one iterate at a time.
  const first = useMemo(() => {
    const rows = toRows(orbit(map, pm.starts[0], REVEAL))
    return { x: rows.map((q) => q[0]), y: rows.map((q) => q[1]) }
  }, [map, pm])
  const [k, setK] = usePlayhead(REVEAL + 1)
  const so = { x: first.x.slice(0, k + 1), y: first.y.slice(0, k + 1) }
  const xa = useAxis({ label: which === 'henon' ? 'x' : 'θ', range: pm.xr, nice: false })
  const ya = useAxis({ label: which === 'henon' ? 'y' : 'p', range: pm.yr, nice: false })
  return (
    <Figure
      title="Chaotic maps of the plane"
      purpose="Orbits of the dissipative Hénon map collapse onto a fractal attractor; orbits of the area-preserving standard map keep to invariant curves until K breaks them into a chaotic sea."
      state={state}
      controls={
        <>
          <ControlRow label="2 · iterate">
            <Player
              className="col-span-full"
              value={k}
              onChange={setK}
              count={REVEAL + 1}
              format={(i) => `n = ${i}`}
              label="iteration"
            />
          </ControlRow>
        </>
      }
      readouts={
        <>
          <Readout label="Lyapunov spectrum" value={spectrum.map(fmt).join(', ')} />
          <Readout label="sum (mean ln|det J|)" value={fmt(spectrum[0] + spectrum[1])} />
        </>
      }
      caption="Log visit counts of 30 000 iterates (one orbit for Hénon; 40 orbits started along θ = π for the standard map). Play the iteration to watch the first orbit land point by point (ink: the current iterate): the Hénon orbit falls onto the attractor within a few steps and then fills it in no visible order, while a standard-map orbit keeps to its own curve or wanders the chaotic sea. The Lyapunov spectrum comes from the QR method along the first orbit: for Hénon the sum is ln 0.3, for the standard map 0; a positive largest exponent means chaos."
    >
      <Plot x={xa} y={ya}>
        <Raster
          x={picture.x}
          y={picture.y}
          z={picture.z}
          valueLabel="log(1 + visits)"
          colorBar={false}
          fillOpacity={0.6}
        />
        <Points name="first orbit so far" x={so.x} y={so.y} slot={7} thin live />
        <Points name="current iterate" x={[first.x[k]]} y={[first.y[k]]} emphasis live />
      </Plot>
    </Figure>
  )
}
