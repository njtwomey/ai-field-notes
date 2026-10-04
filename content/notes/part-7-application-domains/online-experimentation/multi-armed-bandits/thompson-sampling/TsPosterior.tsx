import { useMemo, useState } from 'react'
import {
  Figure,
  formatNumber,
  Handle,
  int,
  Player,
  Plot,
  Readout,
  seriesLayers,
  type SeriesSpec,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { sampleBeta } from '../_shared/bandits'
import { seededRand } from '../_shared/rand'
import { linspace, toFlat } from 'aifn/foundation/tensor'
import { logGamma } from 'aifn/numerics/special'

const HORIZON = 500
const START_MEANS = [0.55, 0.45, 0.3]
/** Interior grid for densities and the probability that each arm is best. */
const GRID = toFlat(linspace(0.0025, 0.9975, 200))
const DX = GRID[1] - GRID[0]

function betaPdf(a: number, b: number): number[] {
  const logNorm = logGamma(a + b) - logGamma(a) - logGamma(b)
  return GRID.map((x) => Math.exp(logNorm + (a - 1) * Math.log(x) + (b - 1) * Math.log(1 - x)))
}

/** P(arm i has the largest mean) under independent Beta posteriors: ∫ f_i(x) Π_{j≠i} F_j(x) dx on the grid. */
function probBest(counts: number[], sums: number[]): number[] {
  const pdfs = counts.map((n, i) => betaPdf(1 + sums[i], 1 + n - sums[i]))
  const cdfs = pdfs.map((f) => {
    let acc = 0
    return f.map((v) => (acc += v * DX))
  })
  const raw = pdfs.map((f, i) =>
    f.reduce((acc, v, g) => {
      let prod = v
      for (let j = 0; j < pdfs.length; j++) if (j !== i) prod *= cdfs[j][g]
      return acc + prod * DX
    }, 0),
  )
  const total = raw.reduce((a, b) => a + b, 0)
  return raw.map((v) => v / total)
}

type Trajectory = {
  n: number[][]
  s: number[][]
  /** Posterior draws used to choose the arm in round t + 1, and the arm chosen. */
  draws: number[][]
  chosen: number[]
  best: number[][]
}

function runThompson(means: number[], seed: number): Trajectory {
  const k = means.length
  const env = seededRand(seed * 7919)
  const agent = seededRand(seed * 104729 + 1)
  const n = new Array<number>(k).fill(0)
  const s = new Array<number>(k).fill(0)
  const out: Trajectory = { n: [], s: [], draws: [], chosen: [], best: [] }
  for (let t = 0; t <= HORIZON; t++) {
    const draws = n.map((c, i) => sampleBeta(1 + s[i], 1 + c - s[i], agent))
    const arm = draws.indexOf(Math.max(...draws))
    out.n.push(n.slice())
    out.s.push(s.slice())
    out.draws.push(draws)
    out.chosen.push(arm)
    out.best.push(probBest(n, s))
    const rewards = means.map((m) => (env.uniform() < m ? 1 : 0))
    n[arm] += 1
    s[arm] += rewards[arm]
  }
  return out
}

/**
 * Thompson sampling on three Bernoulli arms, one round at a time: the Beta posteriors, the draw from each and the arm
 * whose draw is largest; and, over the rounds, the posterior probability that each arm is the best.
 */
export function TsPosterior() {
  const [means, setMeans] = useState(START_MEANS)
  const state = useFigureState({
    seed: int(2, { min: 1, max: 40, step: 1, label: 'seed' }),
  })

  const meansKey = means.join(',')
  const traj = useMemo(() => runThompson(meansKey.split(',').map(Number), state.seed), [meansKey, state.seed])
  // The round, stored with the problem it belongs to: a new seed or mean restarts the walk-through at round 0.
  const problem = `${meansKey}/${state.seed}`
  const [pos, setPos] = useState({ problem, round: 0 })
  const round = pos.problem === problem ? pos.round : 0
  const setRound = (r: number) => setPos({ problem, round: r })

  const t = round
  const n = traj.n[t]
  const s = traj.s[t]
  const draws = traj.draws[t]
  const chosen = traj.chosen[t]
  const pBest = traj.best[t]

  const density: SeriesSpec[] = [
    ...means.map((_, i): SeriesSpec => ({
      name: `arm ${i + 1}: Beta(${1 + s[i]}, ${1 + n[i] - s[i]})`,
      type: 'line',
      x: GRID,
      y: betaPdf(1 + s[i], 1 + n[i] - s[i]),
      slot: i,
      area: true,
    })),
    ...means.map((_, i): SeriesSpec => ({
      name: `arm ${i + 1}: draw θ̃`,
      type: 'scatter',
      x: [draws[i]],
      y: [0],
      slot: i,
    })),
    { name: 'largest draw: pulled', type: 'scatter', x: [draws[chosen]], y: [0], emphasis: true },
  ]
  const handles: Handle[] = means.map((m, i) => ({
    kind: 'x',
    at: m,
    label: `μ${i + 1}`,
    // A new mean is a new problem, so the walk-through restarts at round 0.
    onDrag: (x) =>
      setMeans((prev) => prev.map((v, j) => (j === i ? Math.min(0.95, Math.max(0.05, Math.round(x * 100) / 100)) : v))),
  }))

  const rounds = useMemo(() => Array.from({ length: HORIZON + 1 }, (_, i) => i), [])
  const history: SeriesSpec[] = means.map((_, i) => ({
    name: `P(arm ${i + 1} is best)`,
    type: 'line',
    x: rounds,
    y: traj.best.map((b) => b[i]),
    slot: i,
  }))

  const xAxis = useAxis({ label: 'mean θ', range: [0, 1] })
  const yAxis = useAxis({ label: 'posterior density', range: [0, undefined], hold: 'union' })
  const xAxis2 = useAxis({ label: 'round t', range: [0, HORIZON] })
  const yAxis2 = useAxis({ label: 'P(best | data)', range: [0, 1] })
  return (
    <Figure
      title="Thompson sampling, one round at a time"
      state={state}
      caption="Three Bernoulli arms with uniform Beta(1, 1) priors. Top: each arm's posterior after t rounds, the draw θ̃ from each posterior (dots on the axis) and the pulled arm, whose draw is largest. The vertical guides are the true means; drag one to change the problem. Play the rounds with the player or step through them. Early on the posteriors overlap and every arm is drawn highest sometimes; as the best arm's posterior sharpens it wins almost every draw. Bottom: the posterior probability that each arm is best, which is exactly the probability that Thompson sampling pulls it next. Drag the guide to move through the rounds."
      controls={<Player value={round} onChange={setRound} count={HORIZON + 1} label="round t" />}
      readouts={
        <>
          {means.map((_, i) => (
            <Readout
              key={i}
              label={`arm ${i + 1}: pulls, successes`}
              value={`${n[i]}, ${s[i]} (P best ${formatNumber(pBest[i])})`}
            />
          ))}
          <Readout label={`pulled in round ${t + 1}`} value={`arm ${chosen + 1}`} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={300}>
        {seriesLayers(density)}
        {(handles ?? []).map((h, i) => (
          <Handle key={i} {...h} />
        ))}
      </Plot>
      <Plot x={xAxis2} y={yAxis2} height={220}>
        {seriesLayers(history)}
        <Handle kind="x" at={t} label="round" onDrag={(x) => setRound(Math.max(0, Math.min(HORIZON, Math.round(x))))} />
      </Plot>
    </Figure>
  )
}
