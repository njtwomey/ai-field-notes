import { useEffect, useMemo, useState } from 'react'
import {
  Interactive,
  ParamButton,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type XYSeries,
} from '@/components/viz'
import { linspace, rng } from '@/lib/math'
import { logGamma } from '@/lib/math/special'
import { sampleBeta } from '../_shared/bandits'

const HORIZON = 500
const START_MEANS = [0.55, 0.45, 0.3]
/** Interior grid for densities and the probability that each arm is best. */
const GRID = linspace(0.0025, 0.9975, 200)
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
  const env = rng(seed * 7919)
  const agent = rng(seed * 104729 + 1)
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
  const [round, setRound] = useState(0)
  const [playing, setPlaying] = useState(false)
  const seed = useParam(2, { min: 1, max: 40, step: 1 })

  const meansKey = means.join(',')
  const traj = useMemo(() => runThompson(meansKey.split(',').map(Number), seed.value), [meansKey, seed.value])

  const isPlaying = playing && round < HORIZON
  useEffect(() => {
    if (!isPlaying) return
    // Early rounds play one at a time; later rounds, where little changes, play faster.
    const id = setInterval(() => {
      setRound((r) => Math.min(HORIZON, r + Math.max(1, Math.floor(r / 25))))
    }, 120)
    return () => clearInterval(id)
  }, [isPlaying])

  const t = round
  const n = traj.n[t]
  const s = traj.s[t]
  const draws = traj.draws[t]
  const chosen = traj.chosen[t]
  const pBest = traj.best[t]

  const density: XYSeries[] = [
    ...means.map((_, i): XYSeries => ({
      name: `arm ${i + 1}: Beta(${1 + s[i]}, ${1 + n[i] - s[i]})`,
      type: 'line',
      x: GRID,
      y: betaPdf(1 + s[i], 1 + n[i] - s[i]),
      slot: i,
      area: true,
    })),
    ...means.map((_, i): XYSeries => ({
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
    onDrag: (x) => {
      setMeans((prev) => prev.map((v, j) => (j === i ? Math.min(0.95, Math.max(0.05, Math.round(x * 100) / 100)) : v)))
      setRound(0)
    },
  }))

  const rounds = useMemo(() => Array.from({ length: HORIZON + 1 }, (_, i) => i), [])
  const history: XYSeries[] = means.map((_, i) => ({
    name: `P(arm ${i + 1} is best)`,
    type: 'line',
    x: rounds,
    y: traj.best.map((b) => b[i]),
    slot: i,
  }))

  return (
    <Interactive
      title="Thompson sampling, one round at a time"
      caption="Three Bernoulli arms with uniform Beta(1, 1) priors. Top: each arm's posterior after t rounds, the draw θ̃ from each posterior (dots on the axis) and the pulled arm, whose draw is largest. The vertical guides are the true means; drag one to change the problem. Press Play or step with the arrows. Early on the posteriors overlap and every arm is drawn highest sometimes; as the best arm's posterior sharpens it wins almost every draw. Bottom: the posterior probability that each arm is best, which is exactly the probability that Thompson sampling pulls it next. Drag the guide to move through the rounds."
      controls={
        <>
          <ParamSlider
            label="round t"
            value={t}
            onChange={(v) => {
              setPlaying(false)
              setRound(v)
            }}
            min={0}
            max={HORIZON}
            step={1}
            format={(v) => String(v)}
            withArrows
          />
          <ParamSlider label="seed" param={seed} format={(v) => String(v)} withArrows />
          <ParamButton
            onClick={() => {
              if (isPlaying) return setPlaying(false)
              if (round >= HORIZON) setRound(0)
              setPlaying(true)
            }}
          >
            {isPlaying ? 'Pause' : 'Play'}
          </ParamButton>
        </>
      }
      readout={
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
      <XYChart
        series={density}
        xLabel="mean θ"
        yLabel="posterior density"
        xRange={[0, 1]}
        yRange={[0, undefined]}
        handles={handles}
        height={300}
      />
      <XYChart
        series={history}
        xLabel="round t"
        yLabel="P(best | data)"
        xRange={[0, HORIZON]}
        yRange={[0, 1]}
        height={220}
        handles={[
          {
            kind: 'x',
            at: t,
            label: 'round',
            onDrag: (x) => {
              setPlaying(false)
              setRound(Math.max(0, Math.min(HORIZON, Math.round(x))))
            },
          },
        ]}
      />
    </Interactive>
  )
}
