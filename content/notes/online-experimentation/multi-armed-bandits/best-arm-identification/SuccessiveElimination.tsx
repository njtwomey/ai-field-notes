import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type XYSeries,
} from '@/components/viz'
import { rng } from '@/lib/math'

const START_MEANS = [0.7, 0.6, 0.5, 0.4, 0.3]
const MAX_ROUNDS = 20000
const POINTS = 300

/** Hoeffding radius after s samples per arm, with the union bound over K arms and all rounds. */
const radius = (s: number, k: number, delta: number) => Math.sqrt(Math.log((4 * k * s * s) / delta) / (2 * s))

type Run = {
  /** Recorded rounds and, per arm, lower and upper bounds (NaN once eliminated). */
  rounds: number[]
  lower: number[][]
  upper: number[][]
  /** Round at which each arm was eliminated (the survivor: the stopping round). */
  eliminated: number[]
  winner: number
  stop: number
  /** Round at which uniform sampling with the same intervals would separate the leader from every other arm. */
  uniformStop: number
}

/**
 * Successive elimination on Bernoulli arms. Arm i's s-th reward is the s-th draw of its own seeded stream, so uniform
 * sampling, run on the same streams, sees exactly the same data for the arms both algorithms sample.
 */
function run(means: number[], delta: number, seed: number): Run {
  const k = means.length
  const streams = means.map((_, i) => rng(seed * 1000 + i))
  const sums = new Array<number>(k).fill(0)
  const active = means.map(() => true)
  const eliminated = new Array<number>(k).fill(0)
  const lower: number[][] = means.map(() => [])
  const upper: number[][] = means.map(() => [])
  const rounds: number[] = []
  let stop = 0
  let uniformStop = 0
  for (let s = 1; s <= MAX_ROUNDS; s++) {
    // Uniform sampling keeps drawing every arm, so all sums advance; elimination simply stops looking at dead arms.
    for (let i = 0; i < k; i++) sums[i] += streams[i].uniform() < means[i] ? 1 : 0
    const r = radius(s, k, delta)
    const hat = sums.map((v) => v / s)
    if (!uniformStop) {
      const lead = hat.indexOf(Math.max(...hat))
      if (hat.every((h, i) => i === lead || h + r < hat[lead] - r)) uniformStop = s
    }
    if (!stop) {
      const bestLower = Math.max(...hat.map((h, i) => (active[i] ? h - r : -Infinity)))
      for (let i = 0; i < k; i++) {
        if (active[i] && hat[i] + r < bestLower) {
          active[i] = false
          eliminated[i] = s
        }
      }
      rounds.push(s)
      for (let i = 0; i < k; i++) {
        lower[i].push(active[i] || eliminated[i] === s ? hat[i] - r : NaN)
        upper[i].push(active[i] || eliminated[i] === s ? hat[i] + r : NaN)
      }
      if (active.filter(Boolean).length === 1) stop = s
    }
    if (stop && uniformStop) break
  }
  if (!stop) stop = MAX_ROUNDS
  // Arms still active at the stop (one, or several if the cap was hit) were sampled in every round.
  active.forEach((on, i) => {
    if (on) eliminated[i] = stop
  })
  const winner = active.indexOf(true)
  // Thin the record to about POINTS rounds for drawing.
  const step = Math.max(1, Math.ceil(rounds.length / POINTS))
  const keep = rounds.map((_, j) => j).filter((j) => j % step === 0 || j === rounds.length - 1)
  return {
    rounds: keep.map((j) => rounds[j]),
    lower: lower.map((l) => keep.map((j) => l[j])),
    upper: upper.map((u) => keep.map((j) => u[j])),
    eliminated,
    winner,
    stop,
    uniformStop: uniformStop || MAX_ROUNDS,
  }
}

/** Successive elimination round by round, with draggable arm means, against uniform sampling on the same data. */
export function SuccessiveElimination() {
  const [means, setMeans] = useState(START_MEANS)
  const delta = useParam(0.05, { min: 0.001, max: 0.5, step: 0.001 })
  const seed = useParam(1, { min: 1, max: 40, step: 1 })

  const meansKey = means.join(',')
  const result = useMemo(
    () => run(meansKey.split(',').map(Number), delta.value, seed.value),
    [meansKey, delta.value, seed.value],
  )
  const k = means.length
  const best = Math.max(...means)
  const sorted = [...means].sort((a, b) => b - a)
  const gaps = means.map((m) => (m === best ? best - sorted[1] : best - m))
  const complexity = gaps.reduce((acc, g) => acc + 1 / Math.max(g, 1e-3) ** 2, 0)
  const samples = result.eliminated.reduce((a, b) => a + b, 0)
  const correct = means[result.winner] === best

  const bands: XYSeries[] = means.flatMap((_, i): XYSeries[] => [
    { name: `arm ${i + 1}`, type: 'line', x: result.rounds, y: result.upper[i], slot: i },
    { name: `arm ${i + 1}`, type: 'line', x: result.rounds, y: result.lower[i], slot: i },
  ])
  const arms = means.map((_, i) => i + 1)
  const armSeries: XYSeries[] = [
    {
      name: 'samples (share of total)',
      type: 'bar',
      x: arms,
      y: result.eliminated.map((e) => e / samples),
      muted: true,
    },
    { name: 'true mean (drag)', type: 'scatter', x: arms, y: means, emphasis: true },
  ]
  const handles: Handle[] = means.map((m, i) => ({
    kind: 'point',
    at: [i + 1, m],
    label: `arm ${i + 1}`,
    onDrag: ([, y]) =>
      setMeans((prev) => prev.map((v, j) => (j === i ? Math.min(0.95, Math.max(0.05, Math.round(y * 100) / 100)) : v))),
  }))

  return (
    <Interactive
      title="Successive elimination"
      caption="Five Bernoulli arms with draggable true means. In each round every surviving arm is sampled once, and each gets a Hoeffding interval whose width shrinks like √(log(s²/δ)/s). An arm is dropped when its upper bound falls below the highest lower bound. Arms far from the best go early; the runner-up survives longest, and its gap sets the total cost. The bars show where the samples went. Uniform sampling on the same data needs every arm to be sampled until the closest pair separates."
      controls={
        <>
          <ParamSlider label="error probability δ" param={delta} format={(v) => formatNumber(v)} />
          <ParamSlider label="seed" param={seed} format={(v) => String(v)} withArrows />
        </>
      }
      readout={
        <>
          <Readout label="recommended arm" value={`arm ${result.winner + 1} (${correct ? 'correct' : 'wrong'})`} />
          <Readout label="samples, successive elimination" value={samples.toLocaleString()} />
          <Readout label="samples, uniform with the same intervals" value={(k * result.uniformStop).toLocaleString()} />
          <Readout label="elimination rounds" value={result.eliminated.map((e, i) => `${i + 1}: ${e}`).join(', ')} />
          <Readout label="complexity H = Σ 1/Δᵢ²" value={formatNumber(complexity)} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-[2fr_1fr]">
        <XYChart
          series={bands}
          xLabel="round s (samples per surviving arm)"
          yLabel="confidence bounds"
          yRange={[0, 1]}
        />
        <XYChart
          series={armSeries}
          xLabel="arm"
          yLabel="mean / share"
          xRange={[0.5, k + 0.5]}
          yRange={[0, 1]}
          handles={handles}
        />
      </div>
    </Interactive>
  )
}
