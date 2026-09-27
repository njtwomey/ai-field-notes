import { useMemo } from 'react'
import {
  Heatmap,
  Interactive,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type HeatmapOverlay,
  type Segment,
  type XYSeries,
} from '@/components/viz'
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
  const warp = useParam(0.12, { min: 0, max: 0.25, step: 0.01 })
  const r = useParam(6, { min: 0, max: N - 1, step: 1 })

  const res = useMemo(() => {
    const { a, b } = pair(warp.value)
    const full = dtw(a, b)
    const windowed = dtw(a, b, r.value)
    const lb = lbKeogh(a, b, r.value)
    const euclid = Math.sqrt(a.reduce((s, v, i) => s + (v - b[i]) ** 2, 0))
    // Pointwise cost (a_i − b_j)² as −log₁₀, so cheap matches are dark and the valley the path follows stands out.
    const z = INDEX.map((j) => INDEX.map((i) => -Math.log10((a[i] - b[j]) ** 2 + 1e-3)))
    return { a, b, full, windowed, lb, euclid, z }
  }, [warp.value, r.value])

  const path = res.windowed.path
  const overlay: HeatmapOverlay[] = [
    {
      name: 'warping path',
      type: 'line',
      x: path.map((p) => p[0]),
      y: path.map((p) => p[1]),
      emphasis: true,
    },
    {
      name: 'window edge',
      type: 'line',
      x: INDEX.filter((i) => i + r.value < N),
      y: INDEX.filter((i) => i + r.value < N).map((i) => i + r.value),
      slot: 2,
    },
    {
      name: 'window edge',
      type: 'line',
      x: INDEX.filter((i) => i - r.value >= 0),
      y: INDEX.filter((i) => i - r.value >= 0).map((i) => i - r.value),
      slot: 2,
    },
  ]

  // Alignment: draw b shifted down by 3 and join the points the path matches.
  const shift = 3
  const alignment: Segment[] = path
    .filter((_, k) => k % 2 === 0)
    .map(([i, j]) => ({ from: [i, res.a[i]], to: [j, res.b[j] - shift] }))
  const pairSeries: XYSeries[] = [
    { name: 'a', type: 'line', x: INDEX, y: res.a, slot: 0 },
    { name: 'b (shifted down)', type: 'line', x: INDEX, y: res.b.map((v) => v - shift), slot: 1 },
  ]
  const envelope: XYSeries[] = [
    { name: 'envelope of b', type: 'line', x: INDEX, y: res.lb.upper, dashed: true, slot: 1 },
    { name: 'envelope of b', type: 'line', x: INDEX, y: res.lb.lower, dashed: true, slot: 1 },
    { name: 'a', type: 'line', x: INDEX, y: res.a, slot: 0 },
  ]

  return (
    <Interactive
      title="Warping one series onto another"
      caption="Series b is series a with its time axis bent. Left: the cost of matching a_i with b_j (log scale, dark = cheap), the optimal warping path, and the Sakoe–Chiba window of half-width r that the path must stay within. Right, top: the matched points joined. Right, bottom: LB_Keogh wraps b in an envelope r steps wide and charges a only where it leaves the envelope, a cheap lower bound on the windowed DTW distance. A window of 0 forces the diagonal path, which is the Euclidean distance."
      controls={
        <>
          <ParamSlider label="amount of warping" param={warp} />
          <ParamSlider label="window half-width r" param={r} format={(v) => String(v)} withArrows />
        </>
      }
      readout={
        <>
          <Readout label="Euclidean" value={formatNumber(res.euclid)} />
          <Readout label="DTW, no window" value={formatNumber(res.full.distance)} />
          <Readout label="DTW, window r" value={formatNumber(res.windowed.distance)} />
          <Readout label="LB_Keogh, window r" value={formatNumber(res.lb.bound)} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <Heatmap
          x={INDEX}
          y={INDEX}
          z={res.z}
          xLabel="i (series a)"
          yLabel="j (series b)"
          valueLabel="−log₁₀ cost"
          overlay={overlay}
          height={360}
        />
        <div className="min-w-0 space-y-3">
          <XYChart series={pairSeries} segments={alignment} xLabel="t" height={170} bare />
          <XYChart series={envelope} xLabel="t" height={170} bare />
        </div>
      </div>
    </Interactive>
  )
}
