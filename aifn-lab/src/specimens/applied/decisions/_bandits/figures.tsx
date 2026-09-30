import { useMemo, useState } from 'react'
import {
  banditRun,
  epsilonGreedy,
  exp3,
  klUcb,
  laiRobbinsBound,
  regretCurves,
  thompsonBernoulli,
  ucb1,
  uniformPolicy,
  type ArmStatistics,
  type BanditPolicy,
  type BanditState,
} from 'aifn-applied/decisions/bandits'
import { bernoulliBandit } from 'aifn-applied/data/environments'
import { stream } from 'aifn/foundation/random'
import { toFlat } from 'aifn/foundation/tensor'
import { trace } from 'aifn/foundation/trace'
import { Player, Select, Slider } from '@lab/controls'
import { ControlRow, Figure } from '@lab/layout'
import { Panel, Readout, Subplots, XYChart, type XYSeries } from '@lab/viz'

const POLICIES: { make: () => BanditPolicy<unknown>; slot: number }[] = [
  { make: () => uniformPolicy() as BanditPolicy<unknown>, slot: 7 },
  { make: () => epsilonGreedy({ epsilon: 0.1 }) as BanditPolicy<unknown>, slot: 3 },
  { make: () => exp3({ gamma: 0.1 }) as BanditPolicy<unknown>, slot: 4 },
  { make: () => ucb1() as BanditPolicy<unknown>, slot: 0 },
  { make: () => klUcb() as BanditPolicy<unknown>, slot: 2 },
  { make: () => thompsonBernoulli() as BanditPolicy<unknown>, slot: 1 },
]

export function RegretSpecimen() {
  const [m1, setM1] = useState(0.45)
  const [m2, setM2] = useState(0.55)
  const [m3, setM3] = useState(0.6)
  const [runs, setRuns] = useState('10')
  const [horizon, setHorizon] = useState('1000')
  const means = useMemo(() => [m1, m2, m3], [m1, m2, m3])
  const T = Number(horizon)
  const curves = useMemo(
    () =>
      regretCurves(
        bernoulliBandit(means),
        POLICIES.map((p) => p.make()),
        { horizon: T, runs: Number(runs), stream: stream('regret'), points: 100 },
      ),
    [means, runs, T],
  )
  const t = toFlat(curves.t)
  const mean = toFlat(curves.mean)
  const k = t.length
  const series: XYSeries[] = POLICIES.map((p, i) => ({
    name: curves.names[i],
    type: 'line',
    x: t,
    y: mean.slice(i * k, (i + 1) * k),
    slot: p.slot,
  }))
  const bound = laiRobbinsBound(means, t)
  const final = POLICIES.map((_, i) => mean[i * k + k - 1])
  return (
    <Figure
      title="Regret of bandit policies"
      description="Cumulative pseudo-regret averaged over replicates with common random numbers: uniform play grows linearly, ε-greedy linearly but slower, the index and sampling policies logarithmically, near the Lai–Robbins lower bound (dashed)."
      defaultSize="L"
      controls={
        <>
          <ControlRow label="1 · Bernoulli arms">
            <Slider label="μ₁" value={m1} onChange={setM1} min={0.05} max={0.95} />
            <Slider label="μ₂" value={m2} onChange={setM2} min={0.05} max={0.95} />
            <Slider label="μ₃" value={m3} onChange={setM3} min={0.05} max={0.95} />
          </ControlRow>
          <ControlRow label="2 · experiment">
            <Select label="horizon" value={horizon} onChange={setHorizon} options={['300', '1000', '3000']} />
            <Select label="replicates" value={runs} onChange={setRuns} options={['5', '10', '20', '50']} />
          </ControlRow>
        </>
      }
      readouts={
        <>
          {POLICIES.map((_, i) => (
            <Readout key={i} label={curves.names[i]} value={final[i].toFixed(1)} />
          ))}
          <Readout label="Lai–Robbins constant" value={bound.constant.toFixed(2)} />
        </>
      }
      caption={`aifn/bandits regretCurves over ${runs} replicates (replicate k uses child(stream('regret'), k) for every policy) and laiRobbinsBound.`}
    >
      <XYChart
        series={[
          ...series,
          { name: 'Lai–Robbins ln t', type: 'line', x: t, y: toFlat(bound.bound), dashed: true, emphasis: true },
        ]}
        xLabel="round t"
        yLabel="cumulative regret"
        yRange={[0, Math.max(...final.slice(1)) * 1.3 + 1]}
      />
    </Figure>
  )
}

export function UcbStepsSpecimen() {
  const env = useMemo(() => bernoulliBandit([0.3, 0.5, 0.6]), [])
  const [policyId, setPolicyId] = useState<'ucb1' | 'thompson'>('ucb1')
  const run = useMemo(() => {
    const policy = (policyId === 'ucb1' ? ucb1() : thompsonBernoulli()) as BanditPolicy<ArmStatistics>
    return trace(banditRun(env, policy), undefined, 300, {
      stream: stream('ucb-steps'),
      record: { regret: (s) => s.cumulativeRegret },
    })
  }, [env, policyId])
  const [step, setStep] = useState(30)
  const s: BanditState<ArmStatistics> = run.steps[Math.min(step, run.steps.length - 1)]
  const arms = [1, 2, 3]
  const counts = toFlat(s.counts)
  const scores = toFlat(s.scores).map((v) => (Number.isFinite(v) ? v : 1.5))
  const empirical = Array.from(s.policy.sums, (v, i) => (s.policy.counts[i] ? v / s.policy.counts[i] : 0))
  const bars: XYSeries[] = [
    { name: policyId === 'ucb1' ? 'UCB index' : 'posterior draw', type: 'bar', x: arms, y: scores, slot: 0 },
    { name: 'empirical mean', type: 'scatter', x: arms, y: empirical, emphasis: true },
    { name: 'true mean', type: 'scatter', x: arms, y: [0.3, 0.5, 0.6], slot: 1 },
  ]
  return (
    <Figure
      title="One run, round by round"
      description="UCB1 pulls the arm with the largest optimistic index; an arm's bonus shrinks as it is pulled, so a poor arm is pulled only often enough to rule it out. Thompson sampling pulls the arm whose posterior draw is largest."
      controls={
        <>
          <ControlRow label="policy">
            <Select
              label="policy"
              value={policyId}
              onChange={setPolicyId}
              options={[
                { value: 'ucb1', label: 'UCB1' },
                { value: 'thompson', label: 'Thompson sampling' },
              ]}
            />
          </ControlRow>
          <ControlRow label="round">
            <Player
              label="round"
              value={Math.min(step, run.steps.length - 1)}
              onChange={setStep}
              count={run.steps.length}
              defaultSpeed={10}
            />
          </ControlRow>
        </>
      }
      readouts={
        <>
          <Readout label="pulls" value={counts.join(' / ')} />
          <Readout label="last arm" value={s.arm + 1} />
          <Readout label="cumulative regret" value={s.cumulativeRegret.toFixed(2)} />
        </>
      }
      caption="aifn/bandits banditRun traced for 300 rounds on Bernoulli arms with means 0.3, 0.5 and 0.6; an unpulled arm's index is infinite (drawn at 1.5)."
    >
      <Subplots cols={2}>
        <Panel>
          <XYChart series={bars} xLabel="arm" yLabel="value" yRange={[0, 1.6]} />
        </Panel>
        <Panel>
          <XYChart
            series={[
              {
                name: 'cumulative regret',
                type: 'line',
                x: Array.from(run.index),
                y: toFlat(run.series.regret),
                slot: 2,
              },
            ]}
            xLabel="round"
            yLabel="regret"
            handles={[{ kind: 'x', at: step, label: 'round', onDrag: (v) => setStep(Math.max(0, Math.round(v))) }]}
          />
        </Panel>
      </Subplots>
    </Figure>
  )
}
