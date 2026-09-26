import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamChoice,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Segment,
  type XYSeries,
} from '@/components/viz'
import { rng } from '@/lib/math'
import { normalQuantile } from '@/lib/math/special'
import { BOUNDARIES } from './boundaries'

type Design = 'naive' | 'pocock' | 'obf'

const EXPERIMENTS = 2000
const DRAWN = 40
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

  // The first DRAWN experiments: paths stop at the look where they cross. Crossed paths are highlighted.
  const background: Segment[] = []
  const crossed = { x: [] as number[], y: [] as number[] }
  paths.slice(0, DRAWN).forEach((z, e) => {
    const stop = r.firstCross[e] >= 0 ? r.firstCross[e] : K - 1
    // Every path starts from z = 0 before any data.
    const points: [number, number][] = [[0, 0], ...z.slice(0, stop + 1).map((v, i) => [i + 1, v] as [number, number])]
    if (r.firstCross[e] >= 0) {
      points.forEach(([x, y]) => {
        crossed.x.push(x)
        crossed.y.push(y)
      })
      crossed.x.push(NaN)
      crossed.y.push(NaN)
    } else {
      for (let i = 1; i < points.length; i++) background.push({ from: points[i - 1], to: points[i] })
    }
  })
  const lookAxis = Array.from({ length: K }, (_, i) => i + 1)
  const edge = (sign: number) => r.bounds.map((b) => sign * Math.min(b, Y + 1))
  const pathSeries: XYSeries[] = [
    { name: 'stopped: false positive', type: 'line', x: crossed.x, y: crossed.y, slot: 1 },
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
      caption="Each line is one A/A experiment: both arms are identical, so any significant result is a false positive. The z-statistic is recomputed after each of K equal batches, and the experiment stops at the first look where it crosses the boundary. With the fixed-sample threshold at every look, the chance of stopping somewhere grows with K, far above α. Pocock's boundary raises the threshold equally at every look; O'Brien–Fleming's starts very high and falls to about the fixed-sample value at the end. Both hold the overall false-positive rate at α."
      controls={
        <>
          <ParamSlider label="looks K" param={looks} />
          <ParamChoice label="α" value={alpha} onChange={setAlpha} options={ALPHAS} />
          <ParamChoice label="stopping rule" value={design} onChange={setDesign} options={DESIGNS} />
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
        <XYChart
          height={320}
          xLabel="look"
          yLabel="z-statistic"
          series={pathSeries}
          segments={background}
          xRange={[0, K]}
          yRange={[-Y, Y]}
        />
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
