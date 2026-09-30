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
} from 'aifn/maps'
import { histogram } from 'aifn/stats'
import { toFlat, toRows } from 'aifn/tensor'
import { useMemo, useState } from 'react'
import { Player, Select, Slider, usePlayhead } from '@lab/controls'
import { ControlRow, Figure } from '@lab/layout'
import {
  Heatmap,
  Panel,
  Readout,
  Subplots,
  XYChart,
  formatNumber,
  type Handle,
  type HeatmapOverlay,
  type XYSeries,
} from '@lab/viz'

const fmt = (v: number) => formatNumber(v)

/**
 * Visit counts of points on a grid, for drawing dense orbits as a heatmap: each row band of y is histogrammed in x by
 * `aifn/stats`. Returns log(1 + count) so that rarely visited cells stay visible.
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
    rows.push(Array.from(h.counts, (c) => Math.log1p(c)))
  }
  const cx = Array.from({ length: nx }, (_, j) => xr[0] + ((j + 0.5) * (xr[1] - xr[0])) / nx)
  const cy = Array.from({ length: ny }, (_, i) => yr[0] + (i + 0.5) * dy)
  return { x: cx, y: cy, z: rows }
}

// ---------------------------------------------------------------------------------------------------------------------
// Bifurcation diagram, cobweb and Lyapunov exponent.

type Family = { value: string; label: string; make: (r: number) => Map1; range: [number, number]; initial: number }

const FAMILIES: Family[] = [
  { value: 'logistic', label: 'logistic r·x(1 − x)', make: logisticMap, range: [2.5, 4], initial: 3.56 },
  { value: 'sine', label: 'sine r·sin(πx)', make: sineMap, range: [0.6, 1], initial: 0.87 },
  { value: 'tent', label: 'tent μ·min(x, 1 − x)', make: tentMap, range: [1, 2], initial: 1.5 },
]

const NR = 300
const NX = 200
/** Iterations the cobweb plays through. */
const COBWEB_N = 200
/** Positions of the r sweep. */
const SWEEP = 301

export function BifurcationSpecimen() {
  const [which, setWhich] = useState('logistic')
  const family = FAMILIES.find((f) => f.value === which)!
  const [params, setParams] = useState<Record<string, number>>(() =>
    Object.fromEntries(FAMILIES.map((f) => [f.value, f.initial])),
  )
  const r = params[which]
  const setR = (v: number) =>
    setParams((p) => ({ ...p, [which]: Math.min(family.range[1], Math.max(family.range[0], v)) }))
  const [x0, setX0] = useState(0.2)
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
    const columns = rs.map(
      (rv) =>
        histogram(
          xx.filter((_, k) => rr[k] === rv),
          { bins: NX, range: [0, 1] },
        ).counts,
    )
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
  const graph = useMemo((): XYSeries[] => {
    const xs = Array.from({ length: 201 }, (_, i) => i / 200)
    return [
      { name: 'f', type: 'line', x: xs, y: xs.map(map.f), slot: 0 },
      { name: 'y = x', type: 'line', x: [0, 1], y: [0, 1], muted: true },
    ]
  }, [map])
  const lyapSeries = useMemo(
    (): XYSeries[] => [
      { name: 'λ(r)', type: 'line', x: rs, y: lyap, slot: 1 },
      { name: 'zero', type: 'line', x: [rs[0], rs[rs.length - 1]], y: [0, 0], muted: true },
    ],
    [rs, lyap],
  )
  const shown = Math.min(web.x.length, 2 * n + 1)
  const xn = web.x[shown - 1]
  const live = useMemo(
    (): XYSeries[] => [
      { name: 'cobweb', type: 'line', slot: 2, x: web.x.slice(0, shown), y: web.y.slice(0, shown) },
      { name: 'x_n', type: 'scatter', emphasis: true, x: [xn], y: [xn] },
    ],
    [web, shown, xn],
  )
  const rHandle: Handle[] = [{ kind: 'x', at: r, onDrag: setR, label: 'r' }]
  const sweepAt = Math.round(((r - family.range[0]) / (family.range[1] - family.range[0])) * (SWEEP - 1))
  const sweepR = (k: number) => family.range[0] + ((family.range[1] - family.range[0]) * k) / (SWEEP - 1)
  return (
    <Figure
      title="Bifurcation diagram and cobweb"
      description="Where the long-run orbit of a unimodal map lives as its parameter grows: period doubling into chaos, with windows of order. The cobweb shows the orbit at the chosen r, and the Lyapunov exponent turns positive exactly where the diagram fills in."
      defaultSize="XL"
      controls={
        <>
          <ControlRow label="1 · family">
            <Select label="map" value={which} onChange={setWhich} options={FAMILIES} />
            <Slider
              label="parameter r"
              value={r}
              min={family.range[0]}
              max={family.range[1]}
              step={0.001}
              onChange={setR}
            />
          </ControlRow>
          <ControlRow label="2 · sweep r">
            <Player
              className="col-span-full"
              value={sweepAt}
              onChange={(k) => setR(sweepR(k))}
              count={SWEEP}
              format={(k) => `r = ${sweepR(k).toFixed(3)}`}
              label="r"
              duration={10}
              startReason="The sweep is a view of r, which opens at the family's chosen value."
            />
          </ControlRow>
          <ControlRow label="3 · cobweb">
            <Slider label="x₀" value={x0} min={0} max={1} step={0.001} onChange={setX0} />
            <Player
              className="col-span-full"
              value={n}
              onChange={setN}
              count={COBWEB_N + 1}
              format={(k) => `n = ${k}`}
              label="iteration"
              duration={5}
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
      caption="Left: the attractor for each parameter (log visit counts of 300 iterates after 400 discarded), with the Lyapunov exponent below; drag the r line on either. Play the sweep to move r across the range. Right: the graph of the map, the diagonal and the cobweb from x₀ (drag it), built one iteration at a time as the iteration plays: the vertical step to the graph is x_{n+1} = f(x_n), and the horizontal step to the diagonal makes that value the next input. For the logistic map λ = ln 2 at r = 4; the tent map has λ = ln μ for every μ > 1, so it is chaotic as soon as it is expanding."
    >
      <Subplots cols={2} widthRatios={[1.6, 1]}>
        <Panel>
          <Subplots rows={2} sharex heightRatios={[3, 1.2]} axisKey={which} rescaleOnChange={false}>
            <Panel>
              <Heatmap
                {...diagram}
                xLabel="r"
                yLabel="x"
                valueLabel="log(1 + visits)"
                colorBar={false}
                handles={rHandle}
              />
            </Panel>
            <Panel>
              <XYChart
                series={lyapSeries}
                handles={rHandle}
                xLabel="r"
                yLabel="λ"
                xRange={family.range}
                yRange={[-2, 1]}
                legend={false}
              />
            </Panel>
          </Subplots>
        </Panel>
        <Panel>
          <XYChart
            aspect="equal"
            xRange={[0, 1]}
            yRange={[0, 1]}
            xLabel="x"
            yLabel="f(x)"
            handles={[{ kind: 'x', at: x0, onDrag: (v) => setX0(Math.min(1, Math.max(0, v))), label: 'x₀' }]}
            series={graph}
            live={live}
          />
        </Panel>
      </Subplots>
    </Figure>
  )
}

// ---------------------------------------------------------------------------------------------------------------------
// Chaotic maps of the plane.

type PlaneMap = {
  value: string
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
const PLANE_MAPS: PlaneMap[] = [
  {
    value: 'henon',
    label: 'Hénon map',
    param: { label: 'a (b = 0.3)', min: 1, max: 1.42, initial: 1.4 },
    make: (a) => henonMap(a, 0.3),
    xr: [-1.5, 1.5],
    yr: [-0.45, 0.45],
    starts: [[0.1, 0.1]],
  },
  {
    value: 'standard',
    label: 'standard map',
    param: { label: 'K', min: 0, max: 2.5, initial: 0.97 },
    make: standardMap,
    xr: [0, TAU],
    yr: [0, TAU],
    starts: Array.from({ length: 40 }, (_, k) => [Math.PI + 0.01 * k, (TAU * (k + 0.5)) / 40]),
  },
]

export function PlaneMapSpecimen() {
  const [which, setWhich] = useState('henon')
  const pm = PLANE_MAPS.find((m) => m.value === which)!
  const [params, setParams] = useState<Record<string, number>>(() =>
    Object.fromEntries(PLANE_MAPS.map((m) => [m.value, m.param.initial])),
  )
  const p = params[which]
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
  const overlay = useMemo(
    (): HeatmapOverlay[] => [
      { name: 'first orbit so far', type: 'scatter', slot: 7, x: first.x.slice(0, k + 1), y: first.y.slice(0, k + 1) },
      { name: 'current iterate', type: 'scatter', emphasis: true, x: [first.x[k]], y: [first.y[k]] },
    ],
    [first, k],
  )
  return (
    <Figure
      title="Chaotic maps of the plane"
      description="Orbits of the dissipative Hénon map collapse onto a fractal attractor; orbits of the area-preserving standard map keep to invariant curves until K breaks them into a chaotic sea."
      controls={
        <>
          <ControlRow label="1 · map">
            <Select label="map" value={which} onChange={setWhich} options={PLANE_MAPS} />
            <Slider
              label={pm.param.label}
              value={p}
              min={pm.param.min}
              max={pm.param.max}
              step={0.01}
              onChange={(v) => setParams((s) => ({ ...s, [which]: v }))}
            />
          </ControlRow>
          <ControlRow label="2 · iterate">
            <Player
              className="col-span-full"
              value={k}
              onChange={setK}
              count={REVEAL + 1}
              format={(i) => `n = ${i}`}
              label="iteration"
              duration={5}
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
      <Heatmap
        {...picture}
        xLabel={which === 'henon' ? 'x' : 'θ'}
        yLabel={which === 'henon' ? 'y' : 'p'}
        valueLabel="log(1 + visits)"
        colorBar={false}
        axisKey={which}
        fillOpacity={0.6}
        overlay={overlay}
      />
    </Figure>
  )
}
