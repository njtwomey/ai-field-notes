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
} from 'aifn-render'
import { rng } from '@/lib/math'
import { ALGORITHMS, checkpoints, sampleBeta, type Rand } from '../_shared/bandits'

const K = 3
const HORIZON = 3000
const RUNS = 10
type Scenario = 'adversary' | 'stochastic' | 'switch'
type PolicyId = 'exp3' | 'ucb1' | 'ts'
const slotOf = (id: 'ucb1' | 'ts') => ALGORITHMS.find((a) => a.id === id)!.slot
const POLICIES: { id: PolicyId; label: string; slot: number }[] = [
  { id: 'exp3', label: 'EXP3', slot: 5 },
  { id: 'ucb1', label: 'UCB1', slot: slotOf('ucb1') },
  { id: 'ts', label: 'Thompson sampling', slot: slotOf('ts') },
]
const ARM_SLOTS = [2, 3, 6]

/**
 * The reward table x[t][i] of one run, fixed before play (an oblivious adversary). "adversary" simulates UCB1, which is
 * deterministic, and gives 0 to the arm it is about to pull and 1 to every other arm.
 */
function rewardTable(scenario: Scenario, r: Rand): Uint8Array[] {
  const table: Uint8Array[] = []
  if (scenario === 'adversary') {
    const n = new Array<number>(K).fill(0)
    for (let t = 1; t <= HORIZON; t++) {
      let arm = n.findIndex((c) => c === 0)
      if (arm < 0) {
        // UCB1 only ever sees rewards of 0, so its index is the bonus alone.
        const index = n.map((c) => Math.sqrt((2 * Math.log(t)) / c))
        arm = index.indexOf(Math.max(...index))
      }
      n[arm] += 1
      table.push(Uint8Array.from({ length: K }, (_, i) => (i === arm ? 0 : 1)))
    }
    return table
  }
  for (let t = 1; t <= HORIZON; t++) {
    const late = scenario === 'switch' && t > 0.4 * HORIZON
    const mu = late ? [0.3, 0.5, 0.7] : [0.7, 0.5, 0.3]
    table.push(Uint8Array.from(mu, (m) => (r.uniform() < m ? 1 : 0)))
  }
  return table
}

type Play = { regret: number[]; probs?: number[][] }

/** Plays one policy on a fixed table; regret at each checkpoint is against the best fixed arm up to that round. */
function play(id: PolicyId, table: Uint8Array[], eta: number, r: Rand, ts: number[], record: boolean): Play {
  const n = new Array<number>(K).fill(0)
  const s = new Array<number>(K).fill(0)
  const lossHat = new Array<number>(K).fill(0)
  const totals = new Array<number>(K).fill(0)
  const regret: number[] = []
  const probs: number[][] = []
  let earned = 0
  let next = 0
  for (let t = 1; t <= HORIZON; t++) {
    const x = table[t - 1]
    let arm: number
    let p: number[] = []
    if (id === 'exp3') {
      const low = Math.min(...lossHat)
      const w = lossHat.map((l) => Math.exp(-eta * (l - low)))
      const z = w.reduce((a, b) => a + b, 0)
      p = w.map((v) => v / z)
      // Inverse-cdf draw from p.
      let u = r.uniform() - p[0]
      arm = 0
      while (u > 0 && arm < K - 1) u -= p[++arm]
      // Importance-weighted loss estimate: unbiased for 1 − x, and never negative.
      lossHat[arm] += (1 - x[arm]) / p[arm]
    } else if (id === 'ucb1') {
      arm = n.findIndex((c) => c === 0)
      if (arm < 0) {
        const index = n.map((c, i) => s[i] / c + Math.sqrt((2 * Math.log(t)) / c))
        arm = index.indexOf(Math.max(...index))
      }
    } else {
      const draws = n.map((c, i) => sampleBeta(1 + s[i], 1 + c - s[i], r))
      arm = draws.indexOf(Math.max(...draws))
    }
    n[arm] += 1
    s[arm] += x[arm]
    earned += x[arm]
    for (let i = 0; i < K; i++) totals[i] += x[i]
    if (t === ts[next]) {
      regret.push(Math.max(...totals) - earned)
      if (record && id === 'exp3') probs.push(p)
      next++
    }
  }
  return record ? { regret, probs } : { regret }
}

/** EXP3 against UCB1 and Thompson sampling on a sequence built to defeat UCB1, a stochastic one and a switching one. */
export function Exp3Figure() {
  const [scenario, setScenario] = useState<Scenario>('adversary')
  const logScale = useParam(0, { min: -2, max: 2, step: 0.05 })
  const seed = useParam(1, { min: 1, max: 30, step: 1 })

  const baseEta = Math.sqrt((2 * Math.log(K)) / (K * HORIZON))
  const eta = baseEta * 10 ** logScale.value
  const result = useMemo(() => {
    const ts = checkpoints(HORIZON, 200)
    const sums = new Map(POLICIES.map((p) => [p.id, new Array<number>(ts.length).fill(0)] as const))
    let probs: number[][] = []
    for (let run = 0; run < RUNS; run++) {
      const table = rewardTable(scenario, rng(seed.value * 7919 + run))
      for (const p of POLICIES) {
        const out = play(p.id, table, eta, rng(seed.value * 104729 + run + 1), ts, run === 0)
        const acc = sums.get(p.id)!
        out.regret.forEach((v, j) => (acc[j] += v / RUNS))
        if (out.probs) probs = out.probs
      }
    }
    return { ts, sums, probs }
  }, [scenario, eta, seed.value])

  const regretSeries: XYSeries[] = [
    ...POLICIES.map((p): XYSeries => ({
      name: p.label,
      type: 'line',
      x: result.ts,
      y: result.sums.get(p.id)!,
      slot: p.slot,
    })),
    {
      name: 'EXP3 bound √(2TK ln K)',
      type: 'line',
      x: result.ts,
      y: result.ts.map((t) => Math.sqrt(2 * t * K * Math.log(K))),
      emphasis: true,
      dashed: true,
    },
  ]
  const probSeries: XYSeries[] = ARM_SLOTS.map((slot, i) => ({
    name: `P(arm ${i + 1})`,
    type: 'line',
    x: result.ts,
    y: result.probs.map((p) => p[i]),
    slot,
  }))

  return (
    <Interactive
      title="EXP3 against a fixed sequence of rewards"
      caption="Three arms whose 0/1 rewards for every round are fixed in advance. Left: regret against the best single arm in hindsight, averaged over seeded runs. The first sequence is built by simulating UCB1: each round the arm UCB1 is about to pull pays 0 and the others pay 1. UCB1 is deterministic, so it earns nothing, and its regret grows linearly. EXP3 randomises and, on the same sequence, stays under its √(2TK ln K) guarantee. On stochastic rewards EXP3 is worse than UCB1 and Thompson sampling, which exploit the stochastic structure. Right: EXP3's arm probabilities in one run. Too large a learning rate η commits early and reacts to noise; too small a rate learns slowly."
      controls={
        <>
          <ParamChoice
            label="rewards"
            value={scenario}
            onChange={setScenario}
            options={[
              { value: 'adversary', label: 'built against UCB1' },
              { value: 'stochastic', label: 'stochastic' },
              { value: 'switch', label: 'best arm switches at 40%' },
            ]}
          />
          <ParamSlider
            label="learning rate η (× tuned value)"
            param={logScale}
            format={(v) => `${formatNumber(10 ** v)}× = ${formatNumber(baseEta * 10 ** v)}`}
          />
          <ParamSlider label="seed" param={seed} format={(v) => String(v)} withArrows />
        </>
      }
      readout={
        <>
          {POLICIES.map((p) => (
            <Readout key={p.id} label={`${p.label}: regret`} value={formatNumber(result.sums.get(p.id)!.at(-1)!)} />
          ))}
          <Readout label="bound √(2TK ln K)" value={formatNumber(Math.sqrt(2 * HORIZON * K * Math.log(K)))} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-[3fr_2fr]">
        <XYChart series={regretSeries} xLabel="round t" yLabel="regret vs best fixed arm" xRange={[0, HORIZON]} />
        <XYChart series={probSeries} xLabel="round t" yLabel="EXP3 probability" xRange={[0, HORIZON]} yRange={[0, 1]} />
      </div>
    </Interactive>
  )
}
