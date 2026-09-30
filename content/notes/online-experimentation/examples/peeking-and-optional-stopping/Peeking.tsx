import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamChoice,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type XYSeries,
} from '@/components/viz'
import { rng } from '@/lib/math'
import { normalQuantile } from '@/lib/math/special'
import { BOUNDARIES } from './boundaries'

type Design = 'naive' | 'pocock' | 'obf'

const EXPERIMENTS = 2000
const Y = 5
const ALPHAS = [
  { value: '0.01', label: '0.01' },
  { value: '0.05', label: '0.05' },
  { value: '0.1', label: '0.10' },
]
const DESIGNS: { value: Design; label: string }[] = [
  { value: 'naive', label: 'stop at first p < α' },
  { value: 'pocock', label: 'Pocock' },
  { value: 'obf', label: "O'Brien–Fleming" },
]

/** The critical value for |Z| at look j of K. */
function boundary(design: Design, alpha: string, K: number, j: number): number {
  const z = normalQuantile(1 - Number(alpha) / 2)
  if (design === 'naive' || K === 1) return z
  const table = BOUNDARIES[alpha]
  return design === 'pocock' ? table.pocock[K - 1] : table.obf[K - 1] * Math.sqrt(K / j)
}

/**
 * A/A experiments, where nothing differs between the arms, analysed after each of K equally sized batches. The
 * running z-statistic wanders; stopping the first time it crosses the fixed-sample threshold inflates the
 * false-positive rate. Group-sequential boundaries widen the threshold so that the overall rate stays at α.
 */
export function Peeking() {
  const looks = useParam(10, { min: 1, max: 100, step: 1 })
  const [alpha, setAlpha] = useState('0.05')
  const [design, setDesign] = useState<Design>('naive')
  const seed = useParam(1, { min: 1, max: 20, step: 1 })
  const drawn = useParam(40, { min: 1, max: 50, step: 1 })
  const K = looks.value

  // Z_j = S_j / √j, where S_j sums j independent N(0, 1) batch statistics: equal information per look.
  const paths = useMemo(() => {
    const g = rng(seed.value * 1000 + K)
    return Array.from({ length: EXPERIMENTS }, () => {
      let s = 0
      return Array.from({ length: K }, (_, i) => {
        s += g.normal()
        return s / Math.sqrt(i + 1)
      })
    })
  }, [K, seed.value])

  const r = useMemo(() => {
    const bounds = Array.from({ length: K }, (_, i) => boundary(design, alpha, K, i + 1))
    const firstCross = paths.map((z) => z.findIndex((v, i) => Math.abs(v) >= bounds[i]))
    const cumulative = bounds.map((_, j) => firstCross.filter((c) => c >= 0 && c <= j).length / EXPERIMENTS)
    return { bounds, firstCross, cumulative }
  }, [paths, design, alpha, K])

  // The first experiments of the simulated 2000 are drawn, so the count changes only the picture, never the rate.
  // Paths stop at the look where they cross. Each group is one series, with paths separated by NaN breaks.
  const drawnPaths = useMemo(() => {
    const kept = { x: [] as number[], y: [] as number[] }
    const crossed = { x: [] as number[], y: [] as number[] }
    paths.slice(0, drawn.value).forEach((z, e) => {
      const hit = r.firstCross[e] >= 0
      const stop = hit ? r.firstCross[e] : K - 1
      const target = hit ? crossed : kept
      // Every path starts from z = 0 before any data.
      target.x.push(0, ...z.slice(0, stop + 1).map((_, i) => i + 1), NaN)
      target.y.push(0, ...z.slice(0, stop + 1), NaN)
    })
    return { kept, crossed }
  }, [paths, r.firstCross, drawn.value, K])
  const lookAxis = Array.from({ length: K }, (_, i) => i + 1)
  const edge = (sign: number) => r.bounds.map((b) => sign * Math.min(b, Y + 1))
  const many = drawn.value > 1
  const pathSeries: XYSeries[] = [
    { name: 'ran to the end', type: 'line', x: drawnPaths.kept.x, y: drawnPaths.kept.y, slot: 0, thin: many },
    {
      name: 'stopped: false positive',
      type: 'line',
      x: drawnPaths.crossed.x,
      y: drawnPaths.crossed.y,
      slot: 1,
      thin: many,
    },
    {
      name: 'boundary',
      type: 'line',
      x: [...lookAxis, NaN, ...lookAxis],
      y: [...edge(1), NaN, ...edge(-1)],
      slot: 2,
      dashed: true,
    },
  ]

  const exactNaive = BOUNDARIES[alpha].naive
  const rateSeries: XYSeries[] = [
    { name: 'simulated', type: 'line', x: lookAxis, y: r.cumulative, slot: 1 },
    ...(design === 'naive'
      ? [{ name: 'exact', type: 'line' as const, x: lookAxis, y: exactNaive.slice(0, K), slot: 0, dashed: true }]
      : []),
    { name: 'α', type: 'line', x: [1, Math.max(K, 2)], y: [Number(alpha), Number(alpha)], slot: 2, dashed: true },
  ]

  const final = r.cumulative[K - 1]
  const exact = design === 'naive' ? exactNaive[K - 1] : Number(alpha)

  return (
    <Interactive
      title="Peeking at an A/A test"
      caption="Each light line is one A/A experiment: both arms are identical, so any significant result is a false positive. The z-statistic is recomputed after each of K equal batches, and the experiment stops at the first look where it crosses the boundary; stopped experiments are drawn in the second colour. The lines are the first of 2000 simulated experiments, and the paths slider sets how many are drawn, not how many are simulated. With the fixed-sample threshold at every look, the chance of stopping somewhere grows with K, far above α. Pocock's boundary raises the threshold equally at every look; O'Brien–Fleming's starts very high and falls to about the fixed-sample value at the end. Both hold the overall false-positive rate at α."
      controls={
        <>
          <ParamSlider label="looks K" param={looks} />
          <ParamChoice label="α" value={alpha} onChange={setAlpha} options={ALPHAS} />
          <ParamChoice label="stopping rule" value={design} onChange={setDesign} options={DESIGNS} />
          <ParamSlider label="paths" param={drawn} withArrows format={(v) => String(v)} />
          <ParamSlider label="seed" param={seed} />
        </>
      }
      readout={
        <>
          <Readout label="false-positive rate (simulated)" value={formatNumber(final)} />
          <Readout label={design === 'naive' ? 'exact' : 'designed'} value={formatNumber(exact)} />
          <Readout label="threshold at first look" value={formatNumber(r.bounds[0])} />
          <Readout label="threshold at last look" value={formatNumber(r.bounds[K - 1])} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <XYChart height={320} xLabel="look" yLabel="z-statistic" series={pathSeries} xRange={[0, K]} yRange={[-Y, Y]} />
        <XYChart
          height={320}
          xLabel="look"
          yLabel="P(false positive by this look)"
          series={rateSeries}
          yRange={[0, undefined]}
        />
      </div>
    </Interactive>
  )
}
