import { useMemo, useState } from 'react'
import {
  Interactive,
  ParamChoice,
  ParamSlider,
  ParamSwitch,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type XYSeries,
} from 'aifn-render'
import { rng } from '@/lib/math'
import { ALGORITHMS, checkpoints, sampleBeta, type Rand } from '../_shared/bandits'

const HORIZON = 5000
const RUNS = 10
/** Exploration constant ξ of the discounted and sliding-window indices, slightly above 1/2 as in the analysis. */
const XI = 0.6

type Scenario = 'abrupt' | 'drift'
type PolicyId = 'ucb1' | 'ts' | 'ducb' | 'swucb'
const slotOf = (id: 'ucb1' | 'ts') => ALGORITHMS.find((a) => a.id === id)!.slot
const POLICIES: { id: PolicyId; label: string; slot: number }[] = [
  { id: 'ucb1', label: 'UCB1', slot: slotOf('ucb1') },
  { id: 'ts', label: 'Thompson sampling', slot: slotOf('ts') },
  { id: 'ducb', label: 'discounted UCB', slot: 5 },
  { id: 'swucb', label: 'sliding-window UCB', slot: 4 },
]

/** Means of the two arms at round t: they swap at evenly spaced change points, or oscillate smoothly. */
function meansAt(t: number, scenario: Scenario, changes: number, gap: number): [number, number] {
  if (scenario === 'abrupt') {
    const segment = Math.floor(((t - 1) * (changes + 1)) / HORIZON)
    const hi = 0.5 + gap / 2
    const lo = 0.5 - gap / 2
    return segment % 2 === 0 ? [hi, lo] : [lo, hi]
  }
  const d = (gap / 2) * Math.cos((Math.PI * (changes + 1) * (t - 1)) / HORIZON)
  return [0.5 + d, 0.5 - d]
}

type Chooser = { choose: (t: number) => number; update: (arm: number, reward: number) => void }

function makeChooser(id: PolicyId, gamma: number, tau: number, r: Rand): Chooser {
  const k = 2
  const n = [0, 0]
  const s = [0, 0]
  // Sliding window: the last tau (arm, reward) pairs in a ring buffer, with running counts and sums.
  const armsBuf = new Int8Array(tau).fill(-1)
  const rewBuf = new Float64Array(tau)
  let head = 0
  const argmax = (v: number[]) => (v[1] > v[0] ? 1 : 0)
  return {
    choose: (t) => {
      if (id === 'ts') return argmax(n.map((c, i) => sampleBeta(1 + s[i], 1 + c - s[i], r)))
      const unseen = n.findIndex((c) => c < 1e-9)
      if (unseen >= 0) return unseen
      if (id === 'ucb1') return argmax(n.map((c, i) => s[i] / c + Math.sqrt((2 * Math.log(t)) / c)))
      if (id === 'ducb') {
        const total = n[0] + n[1]
        return argmax(n.map((c, i) => s[i] / c + 2 * Math.sqrt((XI * Math.log(total)) / c)))
      }
      const horizon = Math.min(t, tau)
      return argmax(n.map((c, i) => s[i] / c + Math.sqrt((XI * Math.log(horizon)) / c)))
    },
    update: (arm, reward) => {
      if (id === 'ducb') {
        for (let i = 0; i < k; i++) {
          n[i] *= gamma
          s[i] *= gamma
        }
      } else if (id === 'swucb') {
        const old = armsBuf[head]
        if (old >= 0) {
          n[old] -= 1
          s[old] -= rewBuf[head]
        }
        armsBuf[head] = arm
        rewBuf[head] = reward
        head = (head + 1) % tau
      }
      n[arm] += 1
      s[arm] += reward
    },
  }
}

/** Mean cumulative dynamic regret Σ (max_i μ_i(t) − μ_{A_t}(t)) over seeded runs; every policy sees the same draws. */
function simulate(
  id: PolicyId,
  scenario: Scenario,
  changes: number,
  gap: number,
  gamma: number,
  tau: number,
  seed: number,
) {
  const ts = checkpoints(HORIZON, 200)
  const regret = new Array<number>(ts.length).fill(0)
  for (let run = 0; run < RUNS; run++) {
    const env = rng(seed * 7919 + run)
    const policy = makeChooser(id, gamma, tau, rng(seed * 104729 + run + 1))
    let cum = 0
    let next = 0
    for (let t = 1; t <= HORIZON; t++) {
      const mu = meansAt(t, scenario, changes, gap)
      const u = env.uniform()
      const a = policy.choose(t)
      policy.update(a, u < mu[a] ? 1 : 0)
      cum += Math.max(mu[0], mu[1]) - mu[a]
      if (t === ts[next]) regret[next++] += cum
    }
  }
  return { t: ts, regret: regret.map((v) => v / RUNS) }
}

/** Two arms whose means change over time, and the dynamic regret of stationary and forgetting UCB policies. */
export function NonStationary() {
  const [scenario, setScenario] = useState<Scenario>('abrupt')
  const [enabled, setEnabled] = useState<PolicyId[]>(['ucb1', 'ts', 'ducb', 'swucb'])
  const changes = useParam(3, { min: 0, max: 10, step: 1 })
  const gap = useParam(0.3, { min: 0.05, max: 0.8, step: 0.01 })
  const memory = useParam(300, { min: 20, max: 3000, step: 10 })
  const windowSize = useParam(300, { min: 20, max: 3000, step: 10 })
  const seed = useParam(1, { min: 1, max: 30, step: 1 })

  const gamma = 1 - 1 / memory.value
  const tau = windowSize.value
  const activeKey = enabled.join(',')
  const results = useMemo(
    () =>
      new Map(
        (activeKey ? (activeKey.split(',') as PolicyId[]) : []).map(
          (id) => [id, simulate(id, scenario, changes.value, gap.value, gamma, tau, seed.value)] as const,
        ),
      ),
    [activeKey, scenario, changes.value, gap.value, gamma, tau, seed.value],
  )
  const active = POLICIES.filter((p) => results.has(p.id))

  const meanSeries = useMemo((): XYSeries[] => {
    const t = checkpoints(HORIZON, 400)
    const mu = t.map((v) => meansAt(v, scenario, changes.value, gap.value))
    return [
      { name: 'arm 1 mean', type: 'line', x: t, y: mu.map((m) => m[0]), emphasis: true },
      { name: 'arm 2 mean', type: 'line', x: t, y: mu.map((m) => m[1]), emphasis: true, dashed: true },
    ]
  }, [scenario, changes.value, gap.value])
  const regretSeries: XYSeries[] = active.map((p) => {
    const r = results.get(p.id)!
    return { name: p.label, type: 'line', x: r.t, y: r.regret, slot: p.slot }
  })

  // Garivier and Moulines' tuning for Υ breakpoints, with rewards in [0, 1] (B = 1).
  const breaks = Math.max(1, changes.value)
  const tunedTau = 2 * Math.sqrt((HORIZON * Math.log(HORIZON)) / breaks)
  const tunedMemory = 4 * Math.sqrt(HORIZON / breaks)
  const toggle = (id: PolicyId, on: boolean) =>
    setEnabled((prev) => (on ? [...prev, id] : prev.filter((x) => x !== id)))

  return (
    <Interactive
      title="Forgetting in a changing bandit"
      caption={`Two Bernoulli arms whose means swap at change points, or drift smoothly (top). Bottom: cumulative dynamic regret, measured against the best arm at each round, averaged over ${RUNS} seeded runs. UCB1 and Thompson sampling accumulate evidence for ever, so after each swap they keep pulling the formerly best arm until the new data outweigh the old; their regret jumps at every change. Discounted UCB weights a reward observed s rounds ago by γ^s, with memory 1/(1 − γ); sliding-window UCB uses only the last τ rounds. Too long a memory reacts slowly; too short a memory keeps re-exploring even when nothing changes.`}
      controls={
        <>
          <ParamChoice
            label="change"
            value={scenario}
            onChange={setScenario}
            options={[
              { value: 'abrupt', label: 'abrupt swaps' },
              { value: 'drift', label: 'smooth drift' },
            ]}
          />
          <ParamSlider label="change points (or half-cycles − 1)" param={changes} format={(v) => String(v)} />
          <ParamSlider label="gap between the arms" param={gap} />
          <ParamSlider label="discount memory 1/(1 − γ)" param={memory} format={(v) => String(v)} />
          <ParamSlider label="window τ" param={windowSize} format={(v) => String(v)} />
          <ParamSlider label="seed" param={seed} format={(v) => String(v)} withArrows />
          <div className="flex flex-wrap gap-x-4 gap-y-2 sm:col-span-2 lg:col-span-3">
            {POLICIES.map((p) => (
              <ParamSwitch
                key={p.id}
                label={p.label}
                checked={enabled.includes(p.id)}
                onChange={(on) => toggle(p.id, on)}
              />
            ))}
          </div>
        </>
      }
      readout={
        <>
          {active.map((p) => (
            <Readout key={p.id} label={`${p.label}: regret`} value={formatNumber(results.get(p.id)!.regret.at(-1)!)} />
          ))}
          <Readout label="tuned window 2√(T ln T / Υ)" value={formatNumber(tunedTau)} />
          <Readout label="tuned memory 4√(T / Υ)" value={formatNumber(tunedMemory)} />
        </>
      }
    >
      <XYChart series={meanSeries} xLabel="round t" yLabel="mean" xRange={[0, HORIZON]} yRange={[0, 1]} height={180} />
      <XYChart
        series={regretSeries}
        xLabel="round t"
        yLabel="dynamic regret"
        xRange={[0, HORIZON]}
        yRange={[0, undefined]}
      />
    </Interactive>
  )
}
