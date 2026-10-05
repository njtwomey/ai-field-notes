import { useMemo, useState } from 'react'
import {
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Plot,
  Readout,
  seriesLayers,
  type SeriesSpec,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { stream, uniform } from 'aifn-compute/foundation/random'

const HORIZON = 400
const START_MEANS = [0.6, 0.45, 0.3]

type Trajectory = {
  /** Pull counts after t rounds, t = 0 … HORIZON. */
  n: number[][]
  /** Reward sums after t rounds. */
  s: number[][]
}

/** One run of the index policy μ̂ + √(c ln t / n) on Bernoulli arms; every arm is pulled once first. */
function runUcb(means: number[], c: number, seed: number): Trajectory {
  const k = means.length
  const r = stream(seed)
  const n = new Array<number>(k).fill(0)
  const s = new Array<number>(k).fill(0)
  const out: Trajectory = { n: [n.slice()], s: [s.slice()] }
  for (let t = 1; t <= HORIZON; t++) {
    let arm = n.findIndex((v) => v === 0)
    if (arm < 0) {
      let best = -Infinity
      for (let i = 0; i < k; i++) {
        const index = s[i] / n[i] + Math.sqrt((c * Math.log(t)) / n[i])
        if (index > best) [best, arm] = [index, i]
      }
    }
    // Draw a reward for every arm each round and keep the pulled one, so moving one arm's mean leaves the others' draws.
    const draws = means.map((m) => (uniform(r) < m ? 1 : 0))
    n[arm] += 1
    s[arm] += draws[arm]
    out.n.push(n.slice())
    out.s.push(s.slice())
  }
  return out
}

/**
 * UCB1 round by round: each arm's empirical mean and confidence interval, whose top is the index; the pull counts over
 * time against c ln t / Δ². The true means and the round are draggable.
 */
export function UcbStepper() {
  const [means, setMeans] = useState(START_MEANS)
  const state = useFigureState({
    round: int(60, { min: 3, max: HORIZON, step: 1, label: 'round t', format: (v) => String(v) }),
    c: float(2, { min: 0.05, max: 6, step: 0.05, label: 'exploration constant c (UCB1: 2)' }),
    seed: int(3, { min: 1, max: 40, step: 1, label: 'seed', format: (v) => String(v) }),
  })

  const meansKey = means.join(',')
  const traj = useMemo(
    () => runUcb(meansKey.split(',').map(Number), state.c, state.seed),
    [meansKey, state.c, state.seed],
  )
  const t = state.round
  const n = traj.n[t]
  const s = traj.s[t]
  const next = t + 1
  const radius = n.map((v) => Math.sqrt((state.c * Math.log(next)) / v))
  const est = n.map((v, i) => s[i] / v)
  const index = est.map((m, i) => m + radius[i])
  const chosen = index.indexOf(Math.max(...index))
  const best = Math.max(...means)

  const armSeries: SeriesSpec[] = [
    ...means.map((_, i): SeriesSpec => ({
      name: `arm ${i + 1}: interval`,
      type: 'line',
      x: [i + 1, i + 1],
      y: [est[i] - radius[i], index[i]],
      slot: i,
    })),
    ...means.map((_, i): SeriesSpec => ({
      name: `arm ${i + 1}: index and mean`,
      type: 'scatter',
      x: [i + 1, i + 1],
      y: [est[i], index[i]],
      slot: i,
    })),
    { name: 'true mean (drag)', type: 'scatter', x: means.map((_, i) => i + 1), y: means, emphasis: true },
  ]
  const handles: Handle[] = means.map((m, i) => ({
    kind: 'point',
    at: [i + 1, m],
    label: `arm ${i + 1}`,
    onDrag: ([, y]) =>
      setMeans((prev) => prev.map((v, j) => (j === i ? Math.min(0.95, Math.max(0.05, Math.round(y * 100) / 100)) : v))),
  }))

  const ts = useMemo(() => Array.from({ length: HORIZON }, (_, i) => i + 1), [])
  const countSeries: SeriesSpec[] = [
    ...means.map((_, i): SeriesSpec => ({
      name: `N${i + 1}(t)`,
      type: 'line',
      x: ts,
      y: ts.map((v) => traj.n[v][i]),
      slot: i,
    })),
    ...means.flatMap((m, i): SeriesSpec[] =>
      m < best
        ? [
            {
              name: `c ln t / Δ² for arm ${i + 1}`,
              type: 'line',
              x: ts,
              y: ts.map((v) => (state.c * Math.log(v)) / (best - m) ** 2),
              slot: i,
              dashed: true,
            },
          ]
        : [],
    ),
  ]

  const xAxis = useAxis({ label: 'arm', range: [0.5, 3.5] })
  const yAxis = useAxis({ label: 'reward', range: [0, 1.6] })
  const xAxis2 = useAxis({ label: 'round t', range: [0, HORIZON] })
  const yAxis2 = useAxis({ label: 'pulls', range: [0, undefined], hold: 'union' })
  return (
    <Figure
      title="UCB1 round by round"
      state={state}
      caption="Three Bernoulli arms whose true means are the draggable diamonds. Left: after t rounds, each arm's empirical mean (lower dot) and index (upper dot), the top of a confidence interval of half-width √(c ln t / N). The arm with the highest index is pulled next. Right: pull counts; drag the vertical line to move through the rounds. A suboptimal arm's count tracks the dashed curve c ln t / Δ², the count at which its index falls below the best mean. With c near 0 the policy is greedy and can lock onto a wrong arm; with large c it explores for longer."

      readouts={
        <>
          {index.map((v, i) => (
            <Readout
              key={i}
              label={`arm ${i + 1}: N, mean, index`}
              value={`${n[i]}, ${formatNumber(est[i])}, ${formatNumber(v)}`}
            />
          ))}
          <Readout label={`pulled in round ${next}`} value={`arm ${chosen + 1}`} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-[1fr_2fr]">
        <Plot x={xAxis} y={yAxis}>
          {seriesLayers(armSeries)}
          {(handles ?? []).map((h, i) => (
            <Handle key={i} {...h} />
          ))}
        </Plot>
        <Plot x={xAxis2} y={yAxis2}>
          {seriesLayers(countSeries)}
          <Handle kind="x" at={t} label="round" onDrag={(x) => state.set('round', Math.round(x))} />
        </Plot>
      </div>
    </Figure>
  )
}
