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
import { Player } from '@lab/controls'
import { ControlRow, Figure } from '@lab/layout'
import { choice, row, slider, useFigureState } from '@lab/state'
import { Bars, Curve, Handle, Plot, Plots, Points, Readout, useAxis } from '@lab/viz'

const POLICIES: { make: () => BanditPolicy<unknown>; slot: number }[] = [
  { make: () => uniformPolicy() as BanditPolicy<unknown>, slot: 7 },
  { make: () => epsilonGreedy({ epsilon: 0.1 }) as BanditPolicy<unknown>, slot: 3 },
  { make: () => exp3({ gamma: 0.1 }) as BanditPolicy<unknown>, slot: 4 },
  { make: () => ucb1() as BanditPolicy<unknown>, slot: 0 },
  { make: () => klUcb() as BanditPolicy<unknown>, slot: 2 },
  { make: () => thompsonBernoulli() as BanditPolicy<unknown>, slot: 1 },
]

export function RegretSpecimen() {
  const state = useFigureState({
    arms: row('1 · Bernoulli arms', {
      m1: slider(0.05, 0.95, 0.45, { label: 'μ₁' }),
      m2: slider(0.05, 0.95, 0.55, { label: 'μ₂' }),
      m3: slider(0.05, 0.95, 0.6, { label: 'μ₃' }),
    }),
    experiment: row('2 · experiment', {
      horizon: choice([300, 1000, 3000], 1000, { label: 'horizon' }),
      runs: choice([5, 10, 20, 50], 10, { label: 'replicates' }),
    }),
  })
  const { m1, m2, m3 } = state.arms
  const { runs, horizon: T } = state.experiment
  const means = useMemo(() => [m1, m2, m3], [m1, m2, m3])
  const curves = useMemo(
    () =>
      regretCurves(
        bernoulliBandit(means),
        POLICIES.map((p) => p.make()),
        { horizon: T, runs, stream: stream('regret'), points: 100 },
      ),
    [means, runs, T],
  )
  const t = toFlat(curves.t)
  const mean = toFlat(curves.mean)
  const k = t.length
  const lines = POLICIES.map((p, i) => ({ name: curves.names[i], y: mean.slice(i * k, (i + 1) * k), slot: p.slot }))
  const bound = laiRobbinsBound(means, t)
  const final = POLICIES.map((_, i) => mean[i * k + k - 1])
  const ta = useAxis({ label: 'round t' })
  const ra = useAxis({ label: 'cumulative regret', range: [0, Math.max(...final.slice(1)) * 1.3 + 1] })
  return (
    <Figure
      title="Regret of bandit policies"
      purpose="Uniform play's regret grows linearly, ε-greedy's linearly but slower, and the index and sampling policies' logarithmically, near the Lai–Robbins lower bound (dashed)."
      state={state}
      defaultSize="L"
      readouts={
        <>
          {POLICIES.map((_, i) => (
            <Readout key={i} label={curves.names[i]} value={final[i].toFixed(1)} />
          ))}
          <Readout label="Lai–Robbins constant" value={bound.constant.toFixed(2)} />
        </>
      }
      caption={`Cumulative pseudo-regret averaged over ${runs} replicates with common random numbers (aifn regretCurves: replicate k uses child(stream('regret'), k) for every policy), and laiRobbinsBound.`}
    >
      <Plot x={ta} y={ra}>
        {lines.map((l) => (
          <Curve key={l.name} name={l.name} x={t} y={l.y} slot={l.slot} />
        ))}
        <Curve name="Lai–Robbins ln t" x={t} y={toFlat(bound.bound)} dashed emphasis />
      </Plot>
    </Figure>
  )
}

export function UcbStepsSpecimen() {
  const env = useMemo(() => bernoulliBandit([0.3, 0.5, 0.6]), [])
  const state = useFigureState({
    policy: choice(
      [
        { value: 'ucb1', label: 'UCB1' },
        { value: 'thompson', label: 'Thompson sampling' },
      ],
      'ucb1',
      { label: 'policy' },
    ),
  })
  const policyId = state.policy as 'ucb1' | 'thompson'
  const run = useMemo(() => {
    const policy = (policyId === 'ucb1' ? ucb1() : thompsonBernoulli()) as BanditPolicy<ArmStatistics>
    return trace(banditRun(env, policy), undefined, 300, {
      stream: stream('ucb-steps'),
      record: { regret: (s) => s.cumulativeRegret },
    })
  }, [env, policyId])
  const [step, setStep] = useState(0)
  const s: BanditState<ArmStatistics> = run.steps[Math.min(step, run.steps.length - 1)]
  const arms = [0, 1, 2]
  const counts = toFlat(s.counts)
  const scores = toFlat(s.scores).map((v) => (Number.isFinite(v) ? v : 1.5))
  const empirical = Array.from(s.policy.sums, (v, i) => (s.policy.counts[i] ? v / s.policy.counts[i] : 0))
  const regret = useMemo(() => ({ x: Array.from(run.index), y: toFlat(run.series.regret) }), [run])
  const armAxis = useAxis({ label: 'arm', categories: ['arm 1', 'arm 2', 'arm 3'] })
  const valueAxis = useAxis({ label: 'value', range: [0, 1.6] })
  const roundAxis = useAxis({ label: 'round' })
  const regretAxis = useAxis({ label: 'regret', hold: 'union' })
  return (
    <Figure
      title="One run, round by round"
      purpose="UCB1 pulls the arm with the largest optimistic index; an arm's bonus shrinks as it is pulled, so a poor arm is pulled only often enough to rule it out. Thompson sampling pulls the arm whose posterior draw is largest."
      state={state}
      controls={
        <>
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
      caption="Drag the round line on the regret curve or play. aifn banditRun traced for 300 rounds on Bernoulli arms with means 0.3, 0.5 and 0.6; an unpulled arm's index is infinite (drawn at 1.5)."
    >
      <Plots cols={2}>
        <Plot x={armAxis} y={valueAxis}>
          <Bars name={policyId === 'ucb1' ? 'UCB index' : 'posterior draw'} x={arms} y={scores} slot={0} width={0.6} />
          <Points name="empirical mean" x={arms} y={empirical} emphasis />
          <Points name="true mean" x={arms} y={[0.3, 0.5, 0.6]} slot={1} />
        </Plot>
        <Plot x={roundAxis} y={regretAxis} legend={false}>
          <Curve name="cumulative regret" x={regret.x} y={regret.y} slot={2} />
          <Handle kind="x" at={step} label="round" onDrag={(v) => setStep(Math.max(0, Math.round(v)))} />
        </Plot>
      </Plots>
    </Figure>
  )
}
