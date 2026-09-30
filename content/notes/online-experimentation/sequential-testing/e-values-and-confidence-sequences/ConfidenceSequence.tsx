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

const N = 2000
const STREAMS = 200
const Y = 1.2
/** Most streams that can be drawn; their running means are recorded while the miss rates are simulated. */
const MAX_DRAWN = 50
/** Drawn points of each running mean: every observation up to 20, then every fifth. */
const DRAWN_IDX = Array.from({ length: N }, (_, i) => i).filter((i) => i < 20 || (i + 1) % 5 === 0)
const ALPHAS = [
  { value: '0.01', label: '0.01' },
  { value: '0.05', label: '0.05' },
  { value: '0.1', label: '0.10' },
]

/** Half-width of the normal-mixture confidence sequence for a mean of N(θ, 1) data after n observations. */
function csHalfWidth(n: number, rho: number, alpha: number): number {
  const r2 = rho * rho
  return Math.sqrt(((1 + n * r2) / (n * n * r2)) * (2 * Math.log(1 / alpha) + Math.log(1 + n * r2)))
}

/**
 * Streams of N(0, 1) observations. After each observation, the pointwise 1 − α interval x̄ ± z/√n and the normal-mixture
 * confidence sequence x̄ ± w(n) are formed. The right chart counts, over 200 streams, how many intervals have excluded
 * the true mean at least once so far.
 */
export function ConfidenceSequence() {
  const [rho, setRho] = useState(0.5)
  const [alpha, setAlpha] = useState('0.05')
  const [seed, setSeed] = useState(1)
  const drawn = useParam(10, { min: 1, max: MAX_DRAWN, step: 1 })

  const r = useMemo(() => {
    const a = Number(alpha)
    const z = normalQuantile(1 - a / 2)
    const ns = Array.from({ length: N }, (_, i) => i + 1)
    const ci = ns.map((n) => z / Math.sqrt(n))
    const cs = ns.map((n) => csHalfWidth(n, rho, a))
    const g = rng(seed * 104729)
    const firstMissCi = new Array<number>(N).fill(0)
    const firstMissCs = new Array<number>(N).fill(0)
    // The first MAX_DRAWN of the 200 streams are recorded for drawing, so the paths slider never reruns the simulation.
    const shown: number[][] = []
    for (let s = 0; s < STREAMS; s++) {
      let sum = 0
      let missedCi = false
      let missedCs = false
      const record = s < MAX_DRAWN
      const means: number[] = []
      for (let i = 0; i < N; i++) {
        sum += g.normal()
        const m = sum / (i + 1)
        if (record) means.push(m)
        if (!missedCi && Math.abs(m) >= ci[i]) {
          missedCi = true
          firstMissCi[i]++
        }
        if (!missedCs && Math.abs(m) >= cs[i]) {
          missedCs = true
          firstMissCs[i]++
        }
      }
      if (record) shown.push(DRAWN_IDX.map((i) => means[i]))
    }
    const cumulative = (first: number[]) => {
      let c = 0
      return first.map((f) => (c += f) / STREAMS)
    }
    const band = (w: number[], sign: number) => w.map((v) => sign * Math.min(v, Y * 2))
    const path: XYSeries[] = [
      {
        name: 'pointwise interval',
        type: 'line',
        x: [...ns, NaN, ...ns],
        y: [...band(ci, 1), NaN, ...band(ci, -1)],
        slot: 1,
        dashed: true,
      },
      {
        name: 'confidence sequence',
        type: 'line',
        x: [...ns, NaN, ...ns],
        y: [...band(cs, 1), NaN, ...band(cs, -1)],
        slot: 2,
      },
      { name: 'true mean', type: 'line', x: [1, N], y: [0, 0], emphasis: true, dashed: true },
    ]
    const missCi = cumulative(firstMissCi)
    const missCs = cumulative(firstMissCs)
    const miss: XYSeries[] = [
      { name: 'pointwise interval', type: 'line', x: ns, y: missCi, slot: 1 },
      { name: 'confidence sequence', type: 'line', x: ns, y: missCs, slot: 2 },
      { name: 'α', type: 'line', x: [1, N], y: [a, a], emphasis: true, dashed: true },
    ]
    return { shown, path, miss, finalCi: missCi[N - 1], finalCs: missCs[N - 1], ratio: cs[N - 1] / ci[N - 1] }
  }, [rho, alpha, seed])

  const pathSeries = useMemo((): XYSeries[] => {
    const many = drawn.value > 1
    const x = DRAWN_IDX.map((i) => i + 1)
    const means = r.shown.slice(0, drawn.value).map((y): XYSeries => ({
      name: many ? 'running means' : 'running mean',
      type: 'line',
      x,
      y,
      slot: 0,
      thin: many,
    }))
    return [...means, ...r.path]
  }, [r, drawn.value])

  return (
    <Interactive
      title="Pointwise intervals against a confidence sequence"
      caption="Left: running means of streams of standard normal observations with true mean 0, one light line per stream; the paths slider sets how many of the 200 simulated streams are drawn. After each observation, the pointwise interval x̄ ± z₁₋α/₂/√n has coverage 1 − α at that n; the confidence sequence is wider, and covers the true mean at every n simultaneously with probability at least 1 − α. Right: over 200 streams, the share whose interval has excluded the true mean at least once by observation n. For pointwise intervals it keeps rising, as with peeking; for the confidence sequence it stays below α. The mixing scale ρ sets the sample size at which the sequence is tightest."
      controls={
        <>
          <ParamSlider label="mixing scale ρ" value={rho} onChange={setRho} min={0.05} max={2} step={0.05} />
          <ParamChoice label="α" value={alpha} onChange={setAlpha} options={ALPHAS} />
          <ParamSlider label="paths" param={drawn} withArrows format={(v) => String(v)} />
          <ParamSlider label="seed" value={seed} onChange={setSeed} min={1} max={20} step={1} />
        </>
      }
      readout={
        <>
          <Readout label={`pointwise: ever missed by n = ${N}`} value={formatNumber(r.finalCi)} />
          <Readout label="sequence: ever missed" value={formatNumber(r.finalCs)} />
          <Readout label={`width ratio at n = ${N}`} value={formatNumber(r.ratio)} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <XYChart
          height={320}
          xLabel="observations n"
          yLabel="mean"
          series={pathSeries}
          xRange={[1, N]}
          yRange={[-Y, Y]}
        />
        <XYChart
          height={320}
          xLabel="observations n"
          yLabel="P(interval has missed by n)"
          series={r.miss}
          xRange={[1, N]}
          yRange={[0, undefined]}
        />
      </div>
    </Interactive>
  )
}
