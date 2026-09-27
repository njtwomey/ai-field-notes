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
  type XYSeries,
} from '@/components/viz'
import { rng } from '@/lib/math'
import { bocpd, detections } from './bocpd'

const T = 200
const MIN_SEGMENT = 12
/** Run lengths are shown in bins of two so the heatmap stays light; each bin shows the larger probability. */
const BIN = 2
const FLOOR = -6
const TIMES = Array.from({ length: T }, (_, t) => t + 1)
const RUN_BINS = Array.from({ length: Math.ceil((T + 1) / BIN) }, (_, i) => i * BIN)

/** Change times (1-based first index of each new segment), at least MIN_SEGMENT apart. */
function changeTimes(count: number, next: () => number): number[] {
  for (let attempt = 0; attempt < 500; attempt++) {
    const picks = Array.from({ length: count }, () => MIN_SEGMENT + Math.floor(next() * (T - 2 * MIN_SEGMENT)))
    picks.sort((a, b) => a - b)
    const spaced = picks.every((p, i) => i === 0 || p - picks[i - 1] >= MIN_SEGMENT)
    if (spaced) return picks.map((p) => p + 1)
  }
  return []
}

/**
 * The run-length posterior P(r_t | x_{1:t}) of Adams and MacKay's algorithm on a piecewise-constant-mean series. Each
 * segment draws a ridge along the diagonal r = t − (start − 1); a changepoint sends the mass back to the bottom.
 */
export function RunLengthPosterior() {
  const lambda = useParam(60, { min: 10, max: 250, step: 5 })
  const jump = useParam(2, { min: 0.25, max: 4, step: 0.25 })
  const noise = useParam(1, { min: 0.3, max: 2, step: 0.1 })
  const changes = useParam(4, { min: 1, max: 7, step: 1 })
  const seed = useParam(3, { min: 1, max: 40, step: 1 })

  const r = useMemo(() => {
    const g = rng(seed.value)
    const starts = changeTimes(changes.value, g.uniform)
    // Segment means take steps of ±jump, so every change has the same size.
    const means: number[] = [0]
    for (let k = 0; k < starts.length; k++) means.push(means[k] + (g.uniform() < 0.5 ? -1 : 1) * jump.value)
    const segmentOf = (t: number) => starts.filter((s) => s <= t).length
    const truth = TIMES.map((t) => means[segmentOf(t)])
    const x = truth.map((m) => m + noise.value * g.normal())
    // Prior on each segment's mean wide enough to cover every segment mean; noise known.
    const spread = Math.max(...means.map(Math.abs))
    const result = bocpd(x, {
      hazard: 1 / lambda.value,
      mu0: 0,
      sigma0: Math.max(1, spread + jump.value),
      sigma: noise.value,
    })
    const found = detections(result.map)
    // Heatmap rows are run-length bins and columns times: log₁₀ probability, floored.
    const z = RUN_BINS.map((r0) =>
      TIMES.map((_, i) => {
        const p = result.posterior[i]
        const v = Math.max(p[r0] ?? 0, p[r0 + 1] ?? 0)
        return Math.max(FLOOR, Math.log10(Math.max(v, 1e-300)))
      }),
    )
    // Match each true change to the first detection that locates it within five steps, for the delay readout.
    const matched = starts.map((s) => found.find((d) => Math.abs(d.start - s) <= 5 && d.at >= s))
    const delays = matched.flatMap((d, i) => (d ? [d.at - starts[i]] : []))
    return { x, truth, starts, result, found, z, delays }
  }, [lambda.value, jump.value, noise.value, changes.value, seed.value])

  const series: XYSeries[] = [
    { name: 'data', type: 'scatter', x: TIMES, y: r.x, muted: true },
    { name: 'true segment mean', type: 'line', x: TIMES, y: r.truth, slot: 0, dashed: true },
    { name: 'predictive mean', type: 'line', x: TIMES, y: r.result.predMean, slot: 1 },
  ]
  const overlay: HeatmapOverlay[] = [
    { name: 'MAP run length', type: 'line', x: TIMES, y: r.result.map, slot: 1 },
    { name: 'true changepoint', type: 'scatter', x: r.starts, y: r.starts.map(() => 0), emphasis: true },
  ]
  const list = (xs: number[]) => (xs.length ? xs.join(', ') : 'none')

  return (
    <Interactive
      title="The run-length posterior"
      caption="Top: a series whose mean jumps at the dashed steps, with the one-step predictive mean. Bottom: the posterior over the current run length r_t, the time since the last changepoint, on a log₁₀ colour scale (dark = probable, floored at 10⁻⁶). Within a segment the mass climbs a diagonal ridge, one step per observation; after a change it falls back towards zero and a new ridge starts. The orange line is the most probable run length. Raise the hazard rate (lower λ) to detect sooner at the cost of false alarms; shrink the jump or raise the noise to see detection get slower and less certain."
      controls={
        <>
          <ParamSlider label="expected segment length λ (hazard 1/λ)" param={lambda} />
          <ParamSlider label="size of each mean jump" param={jump} />
          <ParamSlider label="noise standard deviation σ" param={noise} />
          <ParamSlider label="number of changepoints" param={changes} format={(v) => String(v)} />
          <ParamSlider label="seed" param={seed} format={(v) => String(v)} withArrows />
        </>
      }
      readout={
        <>
          <Readout label="true changepoints" value={list(r.starts)} />
          <Readout label="located changepoints" value={list(r.found.map((d) => d.start))} />
          <Readout
            label="mean detection delay"
            value={
              r.delays.length ? `${formatNumber(r.delays.reduce((a, b) => a + b, 0) / r.delays.length)} steps` : '—'
            }
          />
        </>
      }
    >
      <div className="space-y-4">
        <XYChart series={series} xLabel="time t" yLabel="x_t" xRange={[1, T]} height={220} />
        <Heatmap
          x={TIMES}
          y={RUN_BINS}
          z={r.z}
          range={[FLOOR, 0]}
          xLabel="time t"
          yLabel="run length r_t"
          valueLabel="log₁₀ P(r_t | x_1:t)"
          overlay={overlay}
          height={320}
        />
      </div>
    </Interactive>
  )
}
