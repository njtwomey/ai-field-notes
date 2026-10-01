import { useMemo, useState } from 'react'
import { greedyActions, rollout, type RolloutState } from 'aifn-applied/gym'
import { qLearningAgent, randomAgent, valueIteration, type TdAgentState } from 'aifn-applied/gym/agents'
import { mazeEnvironment, type MAZES, type MdpEnvironment } from 'aifn-applied/gym/environments'
import type { Agent } from 'aifn/foundation/contracts'
import { stream } from 'aifn/foundation/random'
import { toFlat } from 'aifn/foundation/tensor'
import { run, seek, trace } from 'aifn/foundation/trace'
import { Player } from '@lab/controls'
import { ControlRow, Figure } from '@lab/layout'
import { choice, float, row, toggle, useFigureState } from '@lab/state'
import { GridView } from '@lab/views'
import { Curve, Handle, Plot, Plots, Readout, useAxis } from '@lab/viz'

type Layout = keyof typeof MAZES
type AgentState = TdAgentState | { action: unknown }
type State = RolloutState<number, number, number, AgentState>

const EPISODES = 150
const MAX_STEPS = 15000
const CHECKPOINT = 250
/** max_a Q(s, a) at active states, 0 at terminals. */
function maxQ(env: MdpEnvironment, Q: ArrayLike<number>): Float64Array {
  const { states: S, actions: A, terminal } = env.model
  return Float64Array.from({ length: S }, (_, s) => {
    if (terminal[s]) return 0
    let best = -Infinity
    for (let a = 0; a < A; a++) best = Math.max(best, Q[s * A + a])
    return best
  })
}

export function MazeAgentSpecimen() {
  const state = useFigureState({
    setup: row('1 · environment and agent', {
      layout: choice(
        [
          { value: 'small', label: 'small 5 × 5' },
          { value: 'classic', label: 'classic 10 × 6' },
          { value: 'traps', label: 'traps 8 × 5' },
        ],
        'small',
        { label: 'maze' },
      ),
      agent: choice(
        [
          { value: 'q', label: 'Q-learning' },
          { value: 'random', label: 'random' },
        ],
        'q',
        { label: 'agent' },
      ),
      gamma: float(0.95, { label: 'discount γ', ge: 0, lt: 1, step: 0.01, suggestions: [0.9, 0.95, 0.99] }),
      epsilon: float(0.1, {
        label: 'exploration ε',
        ge: 0,
        le: 1,
        step: 0.05,
        suggestions: [0.01, 0.05, 0.1, 0.2],
        when: (v) => v.agent === 'q',
      }),
      alpha: float(0.5, {
        label: 'step size α',
        gt: 0,
        le: 1,
        step: 0.05,
        suggestions: [0.05, 0.1, 0.5, 1],
        when: (v) => v.agent === 'q',
      }),
    }),
    view: row('2 · view', { values: toggle(false, 'values and greedy policy') }),
  })
  const layout = state.setup.layout as Layout
  const kind = state.setup.agent as 'q' | 'random'
  const { gamma, epsilon, alpha } = state.setup
  const overlay = state.view.values
  const env = useMemo(() => mazeEnvironment({ layout, gamma }), [layout, gamma])
  const agent = useMemo(
    () =>
      (kind === 'q' ? qLearningAgent({ epsilon, learningRate: alpha }) : randomAgent()) as Agent<
        AgentState,
        number,
        number
      >,
    [kind, epsilon, alpha],
  )
  const alg = useMemo(() => rollout(env, agent, { episodes: EPISODES }), [env, agent])
  const tr = useMemo(
    () =>
      trace(alg, undefined, MAX_STEPS, {
        keep: 'checkpoints',
        checkpointEvery: CHECKPOINT,
        stream: stream('maze'),
        timing: false,
        record: {
          cell: (s: State) => s.envState,
          episode: (s: State) => s.episode,
          ended: (s: State) => (s.ended ? 1 : 0),
          ret: (s: State) => s.episodeReturn,
        },
      }),
    [alg],
  )
  const series = useMemo(() => {
    const cell = toFlat(tr.series.cell)
    const episode = toFlat(tr.series.episode)
    const ended = toFlat(tr.series.ended)
    const ret = toFlat(tr.series.ret)
    const returns: number[] = []
    const endStep: number[] = []
    ended.forEach((e, i) => {
      if (!e) return
      returns.push(ret[i])
      endStep.push(i)
    })
    return { cell, episode, ended, returns, endStep }
  }, [tr])
  const steps = tr.index.length
  const [step, setStep] = useState(0)
  const at = Math.min(step, steps - 1)
  // The agent's state at this step, recomputed from the nearest checkpoint (at most CHECKPOINT − 1 steps).
  const now = useMemo(() => seek(alg, undefined, at, { checkpoints: tr }), [alg, tr, at])
  const optimal = useMemo(() => run(valueIteration(env.model, { tolerance: 1e-9 }), undefined, 2000), [env])
  const optimalPolicy = useMemo(() => toFlat(optimal.policy), [optimal])

  // The current episode's states up to this step: from its reset (step 0, or the start cell after an episode ended).
  const trail = useMemo(() => {
    let k = at
    while (k > 0 && !series.ended[k - 1]) k--
    const states = k === 0 ? [] : [env.reset(stream('maze')).state]
    for (let i = k; i <= at; i++) states.push(series.cell[i])
    return states
  }, [at, series, env])

  const learnt = 'Q' in now.agent ? (now.agent as TdAgentState) : null
  const agentPolicy = useMemo(() => (learnt ? greedyActions(env.model, learnt.Q.data) : null), [env, learnt])
  const agentValues = useMemo(() => (learnt ? maxQ(env, learnt.Q.data) : null), [env, learnt])
  const optimalQ = optimal.Q.data
  const agreement = useMemo(() => {
    if (!agentPolicy) return null
    const A = env.model.actions
    let active = 0
    let good = 0
    for (let s = 0; s < env.model.states; s++) {
      if (env.model.terminal[s]) continue
      active++
      const best = Math.max(...Array.from({ length: A }, (_, a) => optimalQ[s * A + a]))
      if (optimalQ[s * A + agentPolicy[s]] >= best - 1e-6) good++
    }
    return good / active
  }, [agentPolicy, optimalQ, env])

  const vRange = useMemo<[number, number]>(() => {
    const m = Math.max(1e-9, ...toFlat(optimal.V).map(Math.abs))
    return [-m, m]
  }, [optimal])
  const showValues = overlay && agentValues !== null
  const agentField = useMemo(
    () =>
      showValues && agentValues
        ? { values: agentValues, label: 'max_a Q(s, a)', range: vRange, colorBar: false }
        : null,
    [showValues, agentValues, vRange],
  )
  const optimalField = useMemo(() => ({ values: toFlat(optimal.V), label: 'V*(s)', range: vRange }), [optimal, vRange])
  const curve = useMemo(() => ({ x: series.returns.map((_, e) => e + 1), y: series.returns }), [series])
  const episodeNow = series.episode[at] + (series.ended[at] ? 0 : 1)

  // The two grids share one pair of axes (one toolbar entry each, one zoom) with equal units, so cells are square.
  const xa = useAxis({ label: 'x' })
  const ya = useAxis({ label: 'y', equal: xa })
  const ea = useAxis({ label: 'episode', hold: 'initial', key: tr })
  const ra = useAxis({ label: 'return (sum of rewards)', hold: 'initial', key: tr })

  return (
    <Figure
      title="Environment × agent: a maze"
      purpose="One rollout loop runs any agent in any environment: the maze declares its observation and action domains, the agent reads them, and every step is reproducible from the trace. Q-learning's greedy policy approaches the one value iteration computes from the maze's explicit model; the random agent never improves."
      state={state}
      defaultSize="XL"
      controls={
        <ControlRow label="3 · step">
          <Player label="step" value={at} onChange={setStep} count={steps} defaultSpeed={10} />
        </ControlRow>
      }
      readouts={
        <>
          <Readout label="step" value={at} />
          <Readout label="episode" value={episodeNow} />
          <Readout label="return so far" value={toFlat(tr.series.ret)[at].toFixed(0)} />
          <Readout label="episodes run" value={series.returns.length} />
          <Readout
            label="greedy actions optimal"
            value={agreement === null ? '—' : `${(100 * agreement).toFixed(0)} %`}
          />
        </>
      }
      caption={`Play, step or drag the episode line. aifn rollout of ${agent.name} on mazeEnvironment('${layout}') (goal +10, trap −20 and back to start, step −1; truncated after ${env.horizon} steps), ${EPISODES} episodes or ${MAX_STEPS} steps on stream('maze'). Top left: the agent (large mark) and its moves this episode, one arrow per direction of each cell edge walked, numbered when walked more than once (the latest move in ink); reveal "values and greedy policy" to colour cells by max_a Q and draw the agent's greedy policy. Top right: V* and π* by value iteration on env.model. Bottom: return per episode.`}
    >
      <Plots cols={2} scale={0.7}>
        <GridView
          render={env.render!}
          x={xa}
          y={ya}
          title={showValues ? `${agent.name}: max Q, greedy` : agent.name}
          value={agentField}
          policy={showValues ? agentPolicy : null}
          path={trail}
        />
        <GridView
          render={env.render!}
          x={xa}
          y={ya}
          title="value iteration on env.model: V* and π*"
          value={optimalField}
          policy={optimalPolicy}
        />
      </Plots>
      <Plot x={ea} y={ra} scale={0.3}>
        <Curve name="return per episode" x={curve.x} y={curve.y} slot={0} />
        <Handle
          kind="x"
          at={episodeNow}
          label="episode"
          onDrag={(v) => {
            const e = Math.max(1, Math.round(v))
            setStep(series.endStep[Math.min(e, series.endStep.length) - 1] ?? 0)
          }}
        />
      </Plot>
    </Figure>
  )
}
