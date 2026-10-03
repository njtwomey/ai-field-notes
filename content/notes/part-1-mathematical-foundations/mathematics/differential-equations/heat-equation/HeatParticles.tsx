import { useMemo } from 'react'
import {
  Heatmap,
  Interactive,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type HeatmapOverlay,
  type XYSeries,
} from 'aifn-render'
import { linspace, rng } from '@/lib/math'
import { normalPdf } from '@/lib/math/special'
import { histogramDensity } from '../_shared/ode'

const D = 0.5
const DT = 0.01
const DX = Math.sqrt(2 * D * DT) // 0.1: the step that makes the walk's variance grow like 2Dt
const FRAME = 5 // walk steps per slider step of 0.05
const FRAMES = 61
const N = 2000
const S0 = 0.25
const LO = -5
const HI = 5
const XS = linspace(LO, HI, 201)
const T_GRID = linspace(0, (FRAMES - 1) * FRAME * DT, FRAMES)
const X_GRID = linspace(LO, HI, 81)

/** Which bump each particle starts in, its offset within the bump, and its random-walk displacement at every frame. */
const WALK = (() => {
  const { uniform, normal } = rng(21)
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
  const frame = useParam(10, { min: 0, max: FRAMES - 1, step: 1 })
  const c1 = useParam(-1.5, { min: -3.5, max: 3.5, step: 0.05 })
  const c2 = useParam(1, { min: -3.5, max: 3.5, step: 0.05 })
  const tracked = useParam(6, { min: 1, max: 50, step: 1 })
  const c = useMemo<[number, number]>(() => [c1.value, c2.value], [c1.value, c2.value])
  const t = T_GRID[frame.value]

  const start = useMemo(() => Float64Array.from({ length: N }, (_, i) => c[WALK.side[i]] + WALK.offset[i]), [c])
  const positions = useMemo(
    () => Float64Array.from(start, (x0, i) => x0 + WALK.disp[frame.value][i]),
    [start, frame.value],
  )
  const stats = useMemo(() => {
    let m = 0
    for (const x of positions) m += x
    m /= N
    let v = 0
    for (const x of positions) v += (x - m) ** 2
    return { mean: m, variance: v / N }
  }, [positions])

  const series = useMemo<XYSeries[]>(() => {
    const hist = histogramDensity(positions, LO, HI, 50)
    return [
      { name: 'particles (histogram)', type: 'bar', x: hist.x, y: hist.y, muted: true },
      { name: 'initial density', type: 'line', x: XS, y: XS.map((x) => density(x, 0, c)), slot: 1, dashed: true },
      { name: 'heat equation u(x, t)', type: 'line', x: XS, y: XS.map((x) => density(x, t, c)), slot: 0 },
    ]
  }, [positions, c, t])

  const field = useMemo(() => T_GRID.map((tt) => X_GRID.map((x) => density(x, tt, c))), [c])
  const paths = useMemo<HeatmapOverlay[]>(() => {
    const many = tracked.value > 1
    return Array.from({ length: tracked.value }, (_, k) => {
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
  }, [start, tracked.value])
  const handles = useMemo<Handle[]>(
    () => [
      { kind: 'x', at: c1.value, onDrag: c1.set },
      { kind: 'x', at: c2.value, onDrag: c2.set },
    ],
    [c1, c2],
  )
  // Mixture variance: within-bump variance plus the spread of the two centres.
  const predicted = S0 * S0 + 2 * D * t + ((c[0] - c[1]) / 2) ** 2

  return (
    <Interactive
      title="Random walkers and the heat equation"
      caption="2000 particles start in two bumps and take steps of ±0.1 every 0.01 time units. Left: their histogram at time t against the heat-equation solution with D = ½, the two bumps each widened to variance 0.25² + t. Right: the solution u(x, t) over time as a heat map, with the paths of some walkers as light lines; the paths slider sets how many. Drag the two vertical lines to move the starting bumps; step the time to watch both spread."
      controls={
        <>
          <ParamSlider label="time t" param={frame} format={(f) => formatNumber(T_GRID[f])} withArrows />
          <ParamSlider label="paths" param={tracked} withArrows format={(v) => String(v)} />
          <ParamSlider label="bump 1 centre" param={c1} />
          <ParamSlider label="bump 2 centre" param={c2} />
        </>
      }
      readout={
        <>
          <Readout label="particle mean" value={formatNumber(stats.mean)} />
          <Readout label="particle variance" value={formatNumber(stats.variance)} />
          <Readout label="predicted variance" value={formatNumber(predicted)} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <XYChart
          height={320}
          xLabel="x"
          yLabel="density"
          series={series}
          handles={handles}
          xRange={[LO, HI]}
          yRange={[0, undefined]}
        />
        <Heatmap
          x={X_GRID}
          y={T_GRID}
          z={field}
          xLabel="x"
          yLabel="t"
          valueLabel="u"
          scale="sequential"
          overlay={paths}
          height={320}
        />
      </div>
    </Interactive>
  )
}
