import { useMemo } from 'react'
import {
  Bars,
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Plot,
  Raster,
  Readout,
  seriesLayers,
  type SeriesSpec,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { normal as drawNormal, stream, uniform as drawUniform } from 'aifn/foundation/random'
import { histogramDensity } from '../_shared/ode'
import { linspace, toFlat } from 'aifn/foundation/tensor'
import { normalPdf } from 'aifn/numerics/special'

const D = 0.5
const DT = 0.01
const DX = Math.sqrt(2 * D * DT) // 0.1: the step that makes the walk's variance grow like 2Dt
const FRAME = 5 // walk steps per slider step of 0.05
const FRAMES = 61
const N = 2000
const S0 = 0.25
const LO = -5
const HI = 5
const XS = toFlat(linspace(LO, HI, 201))
const T_GRID = toFlat(linspace(0, (FRAMES - 1) * FRAME * DT, FRAMES))
const X_GRID = toFlat(linspace(LO, HI, 81))

/** Which bump each particle starts in, its offset within the bump, and its random-walk displacement at every frame. */
const WALK = (() => {
  const rs = stream(21)
  const uniform = () => drawUniform(rs)
  const normal = () => drawNormal(rs)
  const side = Uint8Array.from({ length: N }, () => (uniform() < 0.5 ? 0 : 1))
  const offset = Float64Array.from({ length: N }, () => S0 * normal())
  const disp = Array.from({ length: FRAMES }, () => new Float64Array(N))
  const pos = new Float64Array(N)
  for (let f = 1; f < FRAMES; f++) {
    for (let k = 0; k < FRAME; k++) for (let i = 0; i < N; i++) pos[i] += uniform() < 0.5 ? DX : -DX
    disp[f].set(pos)
  }
  return { side, offset, disp }
})()

/** Heat-equation density at time t from two Gaussian bumps: each widens to variance S0² + 2Dt. */
const density = (x: number, t: number, c: [number, number]) => {
  const s = Math.sqrt(S0 * S0 + 2 * D * t)
  return (0.5 * (normalPdf((x - c[0]) / s) + normalPdf((x - c[1]) / s))) / s
}

export function HeatParticles() {
  const state = useFigureState({
    frame: slider(0, FRAMES - 1, 10, { step: 1, label: 'time t', format: (f) => formatNumber(T_GRID[f]) }),
    tracked: int(6, { min: 1, max: 50, step: 1, label: 'paths', format: (v) => String(v) }),
    c1: float(-1.5, { min: -3.5, max: 3.5, step: 0.05, label: 'bump 1 centre' }),
    c2: float(1, { min: -3.5, max: 3.5, step: 0.05, label: 'bump 2 centre' }),
  })
  const { c1, c2, frame, tracked, set } = state
  const c = useMemo<[number, number]>(() => [c1, c2], [c1, c2])
  const t = T_GRID[frame]

  const start = useMemo(() => Float64Array.from({ length: N }, (_, i) => c[WALK.side[i]] + WALK.offset[i]), [c])
  const positions = useMemo(() => Float64Array.from(start, (x0, i) => x0 + WALK.disp[frame][i]), [start, frame])
  const stats = useMemo(() => {
    let m = 0
    for (const x of positions) m += x
    m /= N
    let v = 0
    for (const x of positions) v += (x - m) ** 2
    return { mean: m, variance: v / N }
  }, [positions])

  const series = useMemo(() => {
    const hist = histogramDensity(positions, LO, HI, 50)
    return [
      { name: 'particles (histogram)', x: hist.x, y: hist.y, muted: true },
      { name: 'initial density', x: XS, y: XS.map((x) => density(x, 0, c)), slot: 1, dashed: true },
      { name: 'heat equation u(x, t)', x: XS, y: XS.map((x) => density(x, t, c)), slot: 0 },
    ] as const
  }, [positions, c, t])

  const field = useMemo(() => T_GRID.map((tt) => X_GRID.map((x) => density(x, tt, c))), [c])
  const paths = useMemo<SeriesSpec[]>(() => {
    const many = tracked > 1
    return Array.from({ length: tracked }, (_, k) => {
      // 97 is coprime to N, so walkers are distinct, and adding walkers keeps the ones already drawn.
      const i = (k * 97) % N
      return {
        name: many ? 'random walkers' : 'random walker',
        type: 'line' as const,
        x: T_GRID.map((_, f) => start[i] + WALK.disp[f][i]),
        y: T_GRID,
        slot: 1,
        thin: many,
      }
    })
  }, [start, tracked])
  const handles = useMemo<Handle[]>(
    () => [
      { kind: 'x', at: c1, onDrag: (v: number) => set('c1', v) },
      { kind: 'x', at: c2, onDrag: (v: number) => set('c2', v) },
    ],
    [c1, c2, set],
  )
  // Mixture variance: within-bump variance plus the spread of the two centres.
  const predicted = S0 * S0 + 2 * D * t + ((c[0] - c[1]) / 2) ** 2

  const xAxis = useAxis({ label: 'x', range: [LO, HI] })
  const yAxis = useAxis({ label: 'density', range: [0, undefined], hold: 'union' })
  const xAxis2 = useAxis({ label: 'x' })
  const yAxis2 = useAxis({ label: 't' })
  return (
    <Figure
      title="Random walkers and the heat equation"
      state={state}
      caption="2000 particles start in two bumps and take steps of ±0.1 every 0.01 time units. Left: their histogram at time t against the heat-equation solution with D = ½, the two bumps each widened to variance 0.25² + t. Right: the solution u(x, t) over time as a heat map, with the paths of some walkers as light lines; the paths slider sets how many. Drag the two vertical lines to move the starting bumps; step the time to watch both spread."

      readouts={
        <>
          <Readout label="particle mean" value={formatNumber(stats.mean)} />
          <Readout label="particle variance" value={formatNumber(stats.variance)} />
          <Readout label="predicted variance" value={formatNumber(predicted)} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <Plot x={xAxis} y={yAxis} height={320}>
          <Bars {...series[0]} />
          <Curve {...series[1]} />
          <Curve {...series[2]} />
          {(handles ?? []).map((h, i) => (
            <Handle key={i} {...h} />
          ))}
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={320}>
          <Raster x={X_GRID} y={T_GRID} z={field} scale={'sequential'} valueLabel={'u'} />
          {seriesLayers(paths, { live: true })}
        </Plot>
      </div>
    </Figure>
  )
}
