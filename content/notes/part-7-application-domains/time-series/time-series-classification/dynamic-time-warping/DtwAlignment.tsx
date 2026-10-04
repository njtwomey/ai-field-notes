import { useMemo } from 'react'
import {
  Curve,
  Figure,
  float,
  formatNumber,
  Plot,
  Raster,
  Readout,
  type Segment,
  Segments,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { dtw, lbKeogh } from '../_shared/tsc'

const N = 40
const INDEX = Array.from({ length: N }, (_, i) => i)

/** Two versions of the same shape, one warped in time: b(t) = a(t + warp·sin(πt)). */
function pair(warp: number) {
  const shape = (u: number) => Math.sin(2 * Math.PI * u) + 0.6 * Math.exp(-(((u - 0.7) / 0.06) ** 2))
  const a = INDEX.map((i) => shape(i / (N - 1)))
  const b = INDEX.map((i) => {
    const u = i / (N - 1)
    return shape(Math.min(1, Math.max(0, u + warp * Math.sin(Math.PI * u))))
  })
  return { a, b }
}

export function DtwAlignment() {
  const state = useFigureState({
    warp: float(0.12, { min: 0, max: 0.25, step: 0.01, label: 'amount of warping' }),
    r: float(6, { min: 0, max: N - 1, step: 1, label: 'window half-width r', format: (v) => String(v) }),
  })

  const res = useMemo(() => {
    const { a, b } = pair(state.warp)
    const full = dtw(a, b)
    const windowed = dtw(a, b, state.r)
    const lb = lbKeogh(a, b, state.r)
    const euclid = Math.sqrt(a.reduce((s, v, i) => s + (v - b[i]) ** 2, 0))
    // Pointwise cost (a_i − b_j)² as −log₁₀, so cheap matches are dark and the valley the path follows stands out.
    const z = INDEX.map((j) => INDEX.map((i) => -Math.log10((a[i] - b[j]) ** 2 + 1e-3)))
    return { a, b, full, windowed, lb, euclid, z }
  }, [state.warp, state.r])

  const path = res.windowed.path
  const overlay = [
    {
      name: 'warping path',
      x: path.map((p) => p[0]),
      y: path.map((p) => p[1]),
      emphasis: true,
    },
    {
      name: 'window edge',
      x: INDEX.filter((i) => i + state.r < N),
      y: INDEX.filter((i) => i + state.r < N).map((i) => i + state.r),
      slot: 2,
    },
    {
      name: 'window edge',
      x: INDEX.filter((i) => i - state.r >= 0),
      y: INDEX.filter((i) => i - state.r >= 0).map((i) => i - state.r),
      slot: 2,
    },
  ] as const

  // Alignment: draw b shifted down by 3 and join the points the path matches.
  const shift = 3
  const alignment: Segment[] = path
    .filter((_, k) => k % 2 === 0)
    .map(([i, j]) => ({ from: [i, res.a[i]], to: [j, res.b[j] - shift] }))
  const pairSeries = [
    { name: 'a', x: INDEX, y: res.a, slot: 0 },
    { name: 'b (shifted down)', x: INDEX, y: res.b.map((v) => v - shift), slot: 1 },
  ] as const
  const envelope = [
    { name: 'envelope of b', x: INDEX, y: res.lb.upper, dashed: true, slot: 1 },
    { name: 'envelope of b', x: INDEX, y: res.lb.lower, dashed: true, slot: 1 },
    { name: 'a', x: INDEX, y: res.a, slot: 0 },
  ] as const

  const xAxis = useAxis({ label: 'i (series a)' })
  const yAxis = useAxis({ label: 'j (series b)' })
  const xAxis2 = useAxis({ label: 't', hold: 'union' })
  const yAxis2 = useAxis({ hold: 'union' })
  const xAxis3 = useAxis({ label: 't', hold: 'union' })
  const yAxis3 = useAxis({ hold: 'union' })
  return (
    <Figure
      title="Warping one series onto another"
      state={state}
      caption="Series b is series a with its time axis bent. Left: the cost of matching a_i with b_j (log scale, dark = cheap), the optimal warping path, and the Sakoe–Chiba window of half-width r that the path must stay within. Right, top: the matched points joined. Right, bottom: LB_Keogh wraps b in an envelope r steps wide and charges a only where it leaves the envelope, a cheap lower bound on the windowed DTW distance. A window of 0 forces the diagonal path, which is the Euclidean distance."

      readouts={
        <>
          <Readout label="Euclidean" value={formatNumber(res.euclid)} />
          <Readout label="DTW, no window" value={formatNumber(res.full.distance)} />
          <Readout label="DTW, window r" value={formatNumber(res.windowed.distance)} />
          <Readout label="LB_Keogh, window r" value={formatNumber(res.lb.bound)} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <Plot x={xAxis} y={yAxis} height={360}>
          <Raster x={INDEX} y={INDEX} z={res.z} valueLabel={'−log₁₀ cost'} />
          <Curve {...overlay[0]} live />
          <Curve {...overlay[1]} live />
          <Curve {...overlay[2]} live />
        </Plot>
        <div className="min-w-0 space-y-3">
          <Plot x={xAxis2} y={yAxis2} height={170} bare>
            <Curve {...pairSeries[0]} />
            <Curve {...pairSeries[1]} />
            <Segments segments={alignment} />
          </Plot>
          <Plot x={xAxis3} y={yAxis3} height={170} bare>
            <Curve {...envelope[0]} />
            <Curve {...envelope[1]} />
            <Curve {...envelope[2]} />
          </Plot>
        </div>
      </div>
    </Figure>
  )
}
