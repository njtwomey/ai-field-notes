/**
 * Policy-gradient agents on the gym: REINFORCE with a baseline, A2C and PPO on the cart-pole, DDPG on the pendulum's
 * continuous torque (all trained headlessly in the worker by `GymTrainer`), and offline Q-learning from logged
 * transitions with and without CQL's conservative term (`aifn-applied/gym/agents/policy`).
 */
import { useMemo, useState } from 'react'
import type { OfflineCheckpoint } from 'aifn-applied/gym/agents'
import { Player } from '@lab/controls'
import { ControlRow, Figure } from '@lab/layout'
import { call, choice, float, int, row, slider, useFigureState, type AnyValues, type Task } from '@lab/state'
import { GymTrainer, gymEnvironment, gymSetup, TrainControls, trainingRun, useTrainedRun } from '@lab/views'
import { Curve, formatNumber, Handle, Plot, Plots, Readout, useAxis } from '@lab/viz'

const fmt = (v: number, digits = 3) => (Number.isFinite(v) ? formatNumber(Number(v.toPrecision(digits))) : '—')

const AGENTS = {
  reinforceBaselineAgent: { label: 'REINFORCE with baseline', rate: 3e-3 },
  a2cAgent: { label: 'A2C', rate: 1e-3 },
  ppoAgent: { label: 'PPO (clipped)', rate: 3e-4 },
} as const
type AgentKey = keyof typeof AGENTS
const keyOf = (v: AnyValues) => String(v.agent) as AgentKey

// ── 1 · On-policy agents on the cart-pole ───────────────────────────────────────────────────────────────────────────

export function PolicyGradientCartPole() {
  const state = useFigureState({
    setup: row('1 · agent', {
      agent: choice(
        Object.entries(AGENTS).map(([value, a]) => ({ value, label: a.label })),
        'reinforceBaselineAgent',
        { label: 'agent' },
      ),
      learningRate: float(3e-3, {
        gt: 0,
        scale: 'log10',
        suggestions: [3e-4, 1e-3, 3e-3, 1e-2],
        label: 'learning rate',
      }),
      width: choice([16, 32, 64, 128], 64, { label: 'hidden width (2 layers)' }),
      nSteps: int(16, {
        ge: 1,
        le: 2048,
        suggestions: [5, 16, 64],
        label: 'A2C steps per update',
        when: (v) => keyOf(v) === 'a2cAgent',
      }),
      entropy: float(0.01, {
        ge: 0,
        suggestions: [0, 0.001, 0.01, 0.05],
        label: 'entropy bonus',
        when: (v) => keyOf(v) === 'a2cAgent',
      }),
      horizon: int(512, {
        ge: 16,
        le: 8192,
        suggestions: [128, 512, 2048],
        label: 'PPO steps per rollout',
        when: (v) => keyOf(v) === 'ppoAgent',
      }),
      epochs: int(10, {
        ge: 1,
        le: 50,
        suggestions: [1, 4, 10],
        label: 'PPO epochs',
        when: (v) => keyOf(v) === 'ppoAgent',
      }),
      clipRange: float(0.2, {
        gt: 0,
        le: 1,
        suggestions: [0.1, 0.2, 0.3],
        label: 'PPO clip ε',
        when: (v) => keyOf(v) === 'ppoAgent',
      }),
    }),
    run: trainingRun({ steps: 20_000, label: '2 · training run' }),
  })
  const key = state.setup.agent as AgentKey
  const { learningRate, width, nSteps, entropy, horizon, epochs, clipRange } = state.setup
  const setup = useMemo(() => {
    const common = { learningRate, hidden: [width, width] }
    const params =
      key === 'a2cAgent'
        ? { ...common, nSteps, entropy }
        : key === 'ppoAgent'
          ? { ...common, horizon, epochs, clipRange }
          : common
    return gymSetup('cartPoleEnvironment', {}, key, params)
  }, [key, learningRate, width, nSteps, entropy, horizon, epochs, clipRange])
  const scalars =
    key === 'ppoAgent'
      ? (['entropy', 'clip fraction', 'approx KL'] as const)
      : (['policy loss', 'value loss', 'entropy'] as const)
  return (
    <GymTrainer
      title="Policy gradients on the cart-pole"
      purpose="A softmax policy network is pushed along the gradient of expected return, log π(a|o) weighted by an advantage: REINFORCE uses whole-episode returns minus a learned baseline, A2C bootstrapped n-step returns, and PPO reuses each rollout for several epochs while clipping how far the action probabilities may move."
      state={state}
      setup={setup}
      playbackSpeed={120}
      scalars={scalars}
      actions={<span className="text-xs text-muted-foreground">suggested learning rate: {AGENTS[key].rate}</span>}
      caption={`aifn cartPoleEnvironment and ${AGENTS[key].label} (policy and value networks 4 → ${width} → ${width} → 2 and → 1, tanh, Adam at ${learningRate}). REINFORCE updates once per episode; A2C every ${nSteps} steps; PPO every ${horizon} steps with ${epochs} epochs of minibatches of 64, GAE λ = 0.95. Pick an episode on the learning curve to replay it, or evaluate the greedy policy as it stood then. The clip fraction is the share of PPO's minibatch rows whose probability ratio left [1 − ε, 1 + ε].`}
    />
  )
}

// ── 2 · DDPG on the pendulum ────────────────────────────────────────────────────────────────────────────────────────

export function DdpgPendulum() {
  const state = useFigureState({
    setup: row('1 · DDPG', {
      actorLearningRate: float(1e-3, {
        gt: 0,
        scale: 'log10',
        suggestions: [1e-4, 3e-4, 1e-3],
        label: 'actor learning rate',
      }),
      criticLearningRate: float(1e-3, {
        gt: 0,
        scale: 'log10',
        suggestions: [3e-4, 1e-3, 3e-3],
        label: 'critic learning rate',
      }),
      noise: slider(0, 0.5, 0.1, { step: 0.01, label: 'exploration noise (× u_max)' }),
      warmup: int(1000, { ge: 0, le: 20_000, suggestions: [0, 500, 1000, 2000], label: 'random steps first' }),
    }),
    evaluation: row('2 · evaluation', {
      theta0: slider(-Math.PI, Math.PI, Math.PI, { label: 'evaluation start θ₀ (rad, 0 upright)', step: 0.01 }),
    }),
    run: trainingRun({ steps: 12_000, label: '3 · training run' }),
  })
  const { actorLearningRate, criticLearningRate, noise, warmup } = state.setup
  const { theta0 } = state.evaluation
  const setup = useMemo(
    () => gymSetup('pendulumEnvironment', {}, 'ddpgAgent', { actorLearningRate, criticLearningRate, noise, warmup }),
    [actorLearningRate, criticLearningRate, noise, warmup],
  )
  const evaluationEnv = useMemo(
    () => gymEnvironment('pendulumEnvironment', { start: { theta: theta0, thetaDot: 0 } }),
    [theta0],
  )
  const rendererOptions = useMemo(
    () => ({ start: { angle: theta0, onDrag: (angle: number) => state.set('evaluation.theta0', angle) } }),
    [theta0, state],
  )
  return (
    <GymTrainer
      title="DDPG on the pendulum's continuous torque"
      purpose="With a continuous action there is no max over actions to take, so DDPG learns a deterministic actor μ(o) and moves it uphill on a learned critic Q(o, a): the deterministic policy gradient ∇_a Q(o, a)·∇_θ μ(o), from replayed transitions and slowly moving target networks."
      state={state}
      setup={{ ...setup, evaluationEnv }}
      rendererOptions={rendererOptions}
      scalars={['mean Q(o, μ(o))']}
      caption="aifn pendulumEnvironment (Gymnasium's Pendulum-v1: random start, 200 steps, reward −(θ² + 0.1 θ̇² + 0.001 u²), torque in [−2, 2]) and ddpgAgent (actor 3 → 64 → 64 → 1 with tanh, critic 4 → 64 → 64 → 1, Adam, batch 64, τ = 0.005, Gaussian noise). Training needs some 10 000 steps to swing up and hold the rod from every start (about a minute in the worker, at roughly 200 steps a second); earlier policies spin the rod round instead. Evaluate plays the greedy actor from the start θ₀: drag the start marker on the rod."
    />
  )
}

// ── 3 · Offline: CQL against offline DQN ────────────────────────────────────────────────────────────────────────────

type OfflineRun = {
  log: { n: number; returns: Float64Array }
  cql: OfflineCheckpoint[]
  dqn: OfflineCheckpoint[]
  done: boolean
}
type OfflineSettings = { episodes: number; epsilon: number; steps: number; alpha: number; seed: number }

const offlineTask = (s: OfflineSettings): Task<OfflineRun> =>
  call<OfflineRun>(
    'applied/gym/agents/policy/offlineComparison',
    call('applied/gym/environments/cartPoleEnvironment', {}),
    { episodes: s.episodes, epsilon: s.epsilon, seed: `log/${s.seed}` },
    { steps: s.steps, alpha: s.alpha, seed: `offline/${s.seed}` },
  )

export function OfflineCql() {
  const state = useFigureState({
    log: row('1 · logged data', {
      episodes: int(30, { ge: 1, le: 500, suggestions: [10, 30, 100], label: 'behaviour episodes' }),
      epsilon: slider(0, 1, 0.5, { step: 0.05, label: 'behaviour randomness ε' }),
    }),
    train: row('2 · offline training', {
      steps: int(3000, { ge: 1, suggestions: [1000, 3000, 10_000], label: 'gradient updates' }),
      alpha: float(1, { ge: 0, suggestions: [0.1, 0.5, 1, 5], label: 'CQL weight α' }),
      seed: int(1, { ge: 0, le: 9999, label: 'seed' }),
    }),
  })
  const settings: OfflineSettings = { ...state.log, ...state.train }
  const trained = useTrainedRun(settings, offlineTask)
  const r = trained.run.value
  const [picked, setPicked] = useState(0)
  const count = Math.max(r?.cql.length ?? 0, 1)
  const at = Math.min(picked, count - 1)
  const steps = trained.trained?.steps ?? settings.steps
  const stepAxis = useAxis({ label: 'gradient updates', range: [0, steps], key: steps })
  const retAxis = useAxis({ label: 'greedy return (fresh episodes)', range: [0, 520] })
  const qAxis = useAxis({ label: 'mean max_a Q(o, a) on the log', hold: 'union', key: JSON.stringify(trained.trained) })
  const behaviour = r ? Array.from(r.log.returns).reduce((a, b) => a + b, 0) / Math.max(1, r.log.returns.length) : NaN
  const series = (cs: OfflineCheckpoint[] | undefined, f: (c: OfflineCheckpoint) => number) => ({
    x: (cs ?? []).map((c) => c.step),
    y: (cs ?? []).map(f),
  })
  const step = r?.cql[at]?.step ?? 0
  const marker = (
    <Handle
      kind="x"
      at={step}
      label={`update ${step}`}
      onDrag={(v) => {
        const cs = r?.cql ?? []
        let best = 0
        cs.forEach((c, i) => {
          if (Math.abs(c.step - v) < Math.abs(cs[best].step - v)) best = i
        })
        setPicked(best)
      }}
    />
  )
  return (
    <Figure
      title="Offline RL: conservative Q-learning"
      purpose="Trained only on logged transitions, Q-learning's max picks actions the log never tried, and nothing corrects their values; CQL adds α(log Σₐ exp Q(o, a) − Q(o, a_logged)), which keeps the learnt values near the data and the greedy policy near what the data supports."
      state={state}
      defaultSize="L"
      controls={
        <>
          <TrainControls
            run={trained as never}
            progress={r ? r.cql.length / 21 / 2 + r.dqn.length / 21 / 2 : 0}
            progressText={r ? `${r.cql.length + r.dqn.length} / 42 checkpoints` : ''}
          />
          <ControlRow label="3 · checkpoint">
            <Player
              className="col-span-full"
              value={at}
              onChange={setPicked}
              count={count}
              label="checkpoint"
              format={(i) => `update ${r?.cql[i]?.step ?? 0}`}
            />
          </ControlRow>
        </>
      }
      readouts={{
        log: (
          <>
            <Readout label="transitions" value={r ? r.log.n : '—'} />
            <Readout label="behaviour return" value={fmt(behaviour)} />
          </>
        ),
        [`update ${step}`]: (
          <>
            <Readout label="CQL greedy return" value={r?.cql[at] ? fmt(r.cql[at].greedyReturn) : '—'} />
            <Readout label="offline DQN greedy return" value={r?.dqn[at] ? fmt(r.dqn[at].greedyReturn) : '—'} />
            <Readout label="CQL term" value={r?.cql[at] ? fmt(r.cql[at].cqlTerm) : '—'} />
          </>
        ),
      }}
      caption="aifn offlineComparison logs an ε-noisy threshold policy on the cart-pole (push towards the pole's lean, a random push with probability ε), then trains the same Q-network twice on the log, with the CQL term (α as set) and without it (offline DQN): Huber TD loss to a target network copied every 100 updates, Adam, batches of 64. Left: the greedy policy's mean return over five fresh episodes at each checkpoint. Right: the mean estimated value max_a Q on the logged states. Press Train; play the checkpoints or drag the marker."
    >
      <Plots cols={2}>
        <Plot x={stepAxis} y={retAxis} title={!trained.trained ? 'press Train to start' : 'greedy return'}>
          {r && <Curve name="behaviour" x={[0, steps]} y={[behaviour, behaviour]} muted dashed />}
          <Curve name="CQL" slot={0} {...series(r?.cql, (c) => c.greedyReturn)} showPoints />
          <Curve name="offline DQN" slot={1} {...series(r?.dqn, (c) => c.greedyReturn)} showPoints />
          {r && marker}
        </Plot>
        <Plot x={stepAxis} y={qAxis} title="estimated value on the log">
          <Curve name="CQL" slot={0} {...series(r?.cql, (c) => c.meanQ)} />
          <Curve name="offline DQN" slot={1} {...series(r?.dqn, (c) => c.meanQ)} />
          {r && marker}
        </Plot>
      </Plots>
    </Figure>
  )
}
