import { useMemo, useState } from 'react'
import { mazeEnvironment, type MAZES } from 'aifn-applied/data/environments'
import { randomAgent, rollout, type RolloutState } from 'aifn-applied/decisions'
import { greedyActions, type MdpEnvironment } from 'aifn-applied/decisions/reinforcement-learning'
import { qLearningAgent, type QLearningAgentState } from 'aifn-applied/decisions/reinforcement-learning/learning'
import { valueIteration } from 'aifn-applied/decisions/reinforcement-learning/planning'
import type { Agent } from 'aifn/foundation/contracts'
import { stream } from 'aifn/foundation/random'
import { toFlat } from 'aifn/foundation/tensor'
import { run, seek, trace } from 'aifn/foundation/trace'
import { Player } from '@lab/controls'
import { ControlRow, Figure } from '@lab/layout'
import { choice, row, slider, toggle, useFigureState } from '@lab/state'
import { Curve, Handle, Plot, Plots, Points, Raster, Readout, useAxis, Vectors, type Vector } from '@lab/viz'

type Layout = keyof typeof MAZES
type AgentState = QLearningAgentState | { action: unknown }
type State = RolloutState<number, number, number, AgentState>

const EPISODES = 150
const MAX_STEPS = 15000
const CHECKPOINT = 250
const KINDS = ['wall', 'goal', 'trap', 'start'] as const

/** Cell (x, y) of state s on the environment's grid. */
const cellOf = (env: MdpEnvironment, s: number): [number, number] => {
  const w = env.render!.width
  return [s % w, Math.floor(s / w)]
}

/** The map as categorical rows (bottom row first): walls, goal, traps and start in slots 0–3; open cells blank. */
function mapRows(env: MdpEnvironment): number[][] {
  const r = env.render!
  return Array.from({ length: r.height }, (_, y) =>
    Array.from({ length: r.width }, (_, x) => KINDS.indexOf(r.cells[y * r.width + x] as (typeof KINDS)[number])),
  )
}

/** State values as heatmap rows; walls are blank. */
function valueRows(env: MdpEnvironment, V: ArrayLike<number>): number[][] {
  const r = env.render!
  return Array.from({ length: r.height }, (_, y) =>
    Array.from({ length: r.width }, (_, x) => (r.cells[y * r.width + x] === 'wall' ? NaN : V[y * r.width + x])),
  )
}

/** Greedy-policy arrows from each active cell's centre. */
function arrows(env: MdpEnvironment, policy: ArrayLike<number>): Vector[] {
  const out: Vector[] = []
  for (let s = 0; s < policy.length; s++) {
    const a = policy[s]
    if (a < 0) continue
    const [x, y] = cellOf(env, s)
    const [dx, dy] = env.render!.actionVectors[a]
    out.push({ from: [x - 0.2 * dx, y - 0.2 * dy], to: [x + 0.3 * dx, y + 0.3 * dy] })
  }
  return out
}

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

const axes = (n: number) => Array.from({ length: n }, (_, i) => i)

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
      gamma: slider(0.5, 0.99, 0.95, { label: 'discount γ' }),
      epsilon: slider(0, 0.5, 0.1, { label: 'exploration ε', when: (v) => v.agent === 'q' }),
      alpha: slider(0.05, 1, 0.5, { label: 'step size α', when: (v) => v.agent === 'q' }),
    }),
    view: row('2 · view', { values: toggle(true, 'values and greedy policy') }),
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

  // The current episode's path up to this step: from its reset (step 0, or the start cell after an episode ended).
  const trail = useMemo(() => {
    let k = at
    while (k > 0 && !series.ended[k - 1]) k--
    const cells = k === 0 ? [] : [env.reset(stream('maze')).state]
    for (let i = k; i <= at; i++) cells.push(series.cell[i])
    const xy = cells.map((c) => cellOf(env, c))
    return { x: xy.map((c) => c[0]), y: xy.map((c) => c[1]) }
  }, [at, series, env])

  const learnt = 'Q' in now.agent ? (now.agent as QLearningAgentState) : null
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

  const r = env.render!
  const gx = useMemo(() => axes(r.width), [r])
  const gy = useMemo(() => axes(r.height), [r])
  const map = useMemo(() => mapRows(env), [env])
  const vRange = useMemo<[number, number]>(() => {
    const m = Math.max(1e-9, ...toFlat(optimal.V).map(Math.abs))
    return [-m, m]
  }, [optimal])
  const showValues = overlay && agentValues !== null
  const agentRows = useMemo(() => (agentValues ? valueRows(env, agentValues) : null), [env, agentValues])
  const agentArrows = useMemo(() => (agentPolicy ? arrows(env, agentPolicy) : []), [env, agentPolicy])
  const optimalRows = useMemo(() => valueRows(env, toFlat(optimal.V)), [env, optimal])
  const optimalArrows = useMemo(() => arrows(env, optimalPolicy), [env, optimalPolicy])
  const curve = useMemo(() => ({ x: series.returns.map((_, e) => e + 1), y: series.returns }), [series])
  const episodeNow = series.episode[at] + (series.ended[at] ? 0 : 1)
  const [px, py] = cellOf(env, series.cell[at])

  const xa = useAxis({ label: 'x' })
  const ya = useAxis({ label: 'y', equal: xa })
  const ox = useAxis({ label: 'x' })
  const oy = useAxis({ label: 'y', equal: ox })
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
      caption={`Play, step or drag the episode line. aifn rollout of ${agent.name} on mazeEnvironment('${layout}') (goal +10, trap −20 and back to start, step −1; truncated after ${env.horizon} steps), ${EPISODES} episodes or ${MAX_STEPS} steps on stream('maze'). Left: the agent (large mark) and its path this episode; with the toggle, colour is max_a Q and arrows the agent's greedy policy. Middle: V* and π* by value iteration on env.model. Right: return per episode.`}
    >
      <Plots cols={3} widths={[1, 1, 1.1]}>
        <Plot x={xa} y={ya} title={showValues ? `${agent.name}: max Q, greedy` : agent.name}>
          {showValues && agentRows ? (
            <Raster
              x={gx}
              y={gy}
              z={agentRows}
              scale="diverging"
              range={vRange}
              valueLabel="max_a Q(s, a)"
              colorBar={false}
            />
          ) : (
            <Raster x={gx} y={gy} z={map} scale="categorical" categoryNames={KINDS} />
          )}
          {showValues && <Vectors vectors={agentArrows} />}
          <Curve name="path this episode" x={trail.x} y={trail.y} slot={1} showPoints />
          <Points name="agent" x={[px]} y={[py]} emphasis />
        </Plot>
        <Plot x={ox} y={oy} title="value iteration on env.model: V* and π*">
          <Raster x={gx} y={gy} z={optimalRows} scale="diverging" range={vRange} valueLabel="V*(s)" />
          <Vectors vectors={optimalArrows} />
        </Plot>
        <Plot x={ea} y={ra}>
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
      </Plots>
    </Figure>
  )
}
