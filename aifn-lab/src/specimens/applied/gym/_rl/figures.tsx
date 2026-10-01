import { useMemo, useState } from 'react'
import { episodes, type EpisodeState } from 'aifn-applied/gym'
import {
  greedyPath,
  greedyPolicy,
  policyIteration,
  qLearningAgent,
  sarsaAgent,
  valueIteration,
  valuesFromQ,
  type TdAgentState,
} from 'aifn-applied/gym/agents'
import {
  cliffWalkingEnvironment,
  frozenLakeEnvironment,
  gridworldEnvironment,
  mazeEnvironment,
} from 'aifn-applied/gym/environments'
import { stream } from 'aifn/foundation/random'
import { toFlat } from 'aifn/foundation/tensor'
import { trace } from 'aifn/foundation/trace'
import { Player } from '@lab/controls'
import { ControlRow, Figure } from '@lab/layout'
import { choice, float, row, slider, useFigureState } from '@lab/state'
import { GridView } from '@lab/views'
import { Bars, Curve, Handle, Plot, Plots, Readout, useAxis } from '@lab/viz'

// ---------------------------------------------------------------------------------------------------------------------
// 1. Value iteration and policy iteration, sweep by sweep.

type World = 'gridworld' | 'maze' | 'frozen'

export function PlanningSpecimen() {
  const state = useFigureState({
    setup: row('1 · world', {
      world: choice(
        [
          { value: 'gridworld', label: 'gridworld 4 × 3' },
          { value: 'maze', label: 'maze' },
          { value: 'frozen', label: 'FrozenLake 4 × 4 (slippery)' },
        ],
        'gridworld',
        { label: 'world' },
      ),
      gamma: slider(0.5, 0.99, 0.9, { label: 'discount γ' }),
      noise: slider(0, 0.6, 0.2, { label: 'slip probability', when: (v) => v.world !== 'frozen' }),
      stepReward: slider(-1, 0.1, -0.04, { label: 'step reward', when: (v) => v.world === 'gridworld' }),
    }),
    solve: row('2 · method', {
      method: choice(
        [
          { value: 'vi', label: 'value iteration' },
          { value: 'pi', label: 'policy iteration' },
        ],
        'vi',
        { label: 'method' },
      ),
    }),
  })
  const world = state.setup.world as World
  const { gamma, noise, stepReward } = state.setup
  const method = state.solve.method as 'vi' | 'pi'
  const env = useMemo(() => {
    if (world === 'gridworld') return gridworldEnvironment({ gamma, noise, stepReward })
    if (world === 'frozen') return frozenLakeEnvironment({ gamma })
    return mazeEnvironment({ layout: 'classic', gamma, slip: noise })
  }, [world, gamma, noise, stepReward])
  const run = useMemo(() => {
    if (method === 'vi')
      return trace(valueIteration(env.model, { tolerance: 1e-8 }), undefined, 400, {
        record: { residual: (s) => Math.max(s.residual, 1e-16) },
      })
    return trace(policyIteration(env.model), undefined, 30, { record: { changed: (s) => s.changed } })
  }, [env, method])
  const [step, setStep] = useState(0)
  const at = Math.min(step, run.steps.length - 1)
  const s = run.steps[at]
  const start = env.reset(stream(0)).state
  // A symmetric colour range about 0 from the final values, so the sign of V reads as its hue.
  const vRange = useMemo<[number, number]>(() => {
    const m = Math.max(1e-9, ...toFlat(run.steps[run.steps.length - 1].V).map(Math.abs))
    return [-m, m]
  }, [run])
  const progress = useMemo(
    () => ({
      x: Array.from(run.index),
      y: method === 'vi' ? toFlat(run.series.residual) : toFlat(run.series.changed),
    }),
    [run, method],
  )
  const field = useMemo(() => ({ values: toFlat(s.V), label: 'V(s)', range: vRange }), [s, vRange])
  const policy = useMemo(() => toFlat(s.policy), [s])
  const ka = useAxis({ label: method === 'vi' ? 'sweep k' : 'iteration', hold: 'initial', key: run })
  const ra = useAxis({
    label: method === 'vi' ? 'max |TV − V|' : 'changed',
    log: method === 'vi',
    hold: 'initial',
    key: run,
  })
  return (
    <Figure
      title="Planning on a grid: values and greedy policy per sweep"
      purpose="Value iteration spreads value outwards from the rewarding cells one step per sweep, and its greedy policy settles long before the values stop changing; policy iteration evaluates each policy exactly and needs only a handful of improvements."
      state={state}
      defaultSize="L"
      controls={
        <>
          <ControlRow label="3 · sweep">
            <Player
              label={method === 'vi' ? 'sweep' : 'iteration'}
              value={at}
              onChange={setStep}
              count={run.steps.length}
              defaultSpeed={4}
            />
          </ControlRow>
        </>
      }
      readouts={
        <>
          <Readout label="V(start)" value={toFlat(s.V)[start].toFixed(4)} />
          <Readout
            label={method === 'vi' ? 'residual' : 'changed'}
            value={'residual' in s ? s.residual.toExponential(2) : s.changed}
          />
          <Readout label="stopped" value={`${run.meta.stopped} after ${run.meta.steps}`} />
        </>
      }
      caption="Play or drag the k line. aifn valueIteration and policyIteration on env.model of gridworldEnvironment (exits +1 and −1), a text maze (goal +10, step −1) and FrozenLake; colour is V, arrows the greedy policy. Right: the residual per sweep (it falls at least as fast as γᵏ) or the actions changed per improvement."
    >
      <Plots cols={2} widths={[1.4, 1]}>
        <GridView render={env.render!} value={field} policy={policy} />
        <Plot x={ka} y={ra} legend={false}>
          {method === 'vi' ? (
            <Curve name="Bellman residual" x={progress.x} y={progress.y} slot={2} />
          ) : (
            <Bars name="states whose action changed" x={progress.x} y={progress.y} slot={2} width={0.6} />
          )}
          <Handle kind="x" at={at} label="k" onDrag={(v) => setStep(Math.max(0, Math.round(v)))} />
        </Plot>
      </Plots>
    </Figure>
  )
}

// ---------------------------------------------------------------------------------------------------------------------
// 2. Q-learning against SARSA on the cliff.

const CLIFF = cliffWalkingEnvironment()
const CLIFF_START = CLIFF.reset(stream(0)).state
type CliffState = EpisodeState<number, number, TdAgentState>
const EPISODES = 500

/** A greedy route's moves, or "loops" when it ends on a state it already visited. */
const routeLength = (path: readonly number[]) =>
  path.indexOf(path[path.length - 1]) < path.length - 1 ? 'loops' : path.length - 1

function smooth(y: number[], w = 20): number[] {
  return y.map((_, i) => {
    const lo = Math.max(0, i - w + 1)
    let s = 0
    for (let k = lo; k <= i; k++) s += y[k]
    return s / (i - lo + 1)
  })
}

export function CliffSpecimen() {
  const state = useFigureState({
    learning: row('1 · learning', {
      epsilon: float(0.1, { label: 'exploration ε', ge: 0, le: 1, step: 0.05, suggestions: [0.01, 0.05, 0.1, 0.2] }),
      alpha: float(0.5, { label: 'step size α', gt: 0, le: 1, step: 0.05, suggestions: [0.05, 0.1, 0.5, 1] }),
    }),
  })
  const { epsilon, alpha } = state.learning
  const runs = useMemo(() => {
    const opts = { epsilon, learningRate: alpha }
    const rec = { record: { reward: (s: CliffState) => s.episodeReturn }, stream: stream('cliff') }
    return {
      q: trace(episodes(CLIFF, qLearningAgent(opts)), undefined, EPISODES, rec),
      s: trace(episodes(CLIFF, sarsaAgent(opts)), undefined, EPISODES, rec),
    }
  }, [epsilon, alpha])
  const [episode, setEpisode] = useState(EPISODES)
  const e = Math.min(episode, EPISODES)
  const q = runs.q.steps[e]
  const s = runs.s.steps[e]
  const learnt = (st: CliffState) => {
    const policy = greedyPolicy(CLIFF.model, st.agent.Q)
    const path = Array.from(toFlat(greedyPath(CLIFF.model, CLIFF_START, policy)))
    const values = toFlat(valuesFromQ(CLIFF.model, st.agent.Q))
    return {
      field: {
        values,
        label: 'max_a Q(s, a)',
        range: [-20, 0] as [number, number],
        scale: 'sequential' as const,
        fillOpacity: 0.55,
      },
      path,
    }
  }
  const ql = learnt(q)
  const sl = learnt(s)
  const curves = useMemo(
    () => ({
      q: { x: Array.from(runs.q.index), y: smooth(toFlat(runs.q.series.reward)) },
      s: { x: Array.from(runs.s.index), y: smooth(toFlat(runs.s.series.reward)) },
    }),
    [runs],
  )
  // Cells, not geometry: no equal units, so the two grids share the column's height evenly. Both grids share one
  // x and one y axis (one toolbar entry each, one zoom), held at their first fit.
  const qx = useAxis({ label: 'x', hold: 'initial' })
  const qy = useAxis({ label: 'y', hold: 'initial' })
  const ea = useAxis({ label: 'episode', range: [0, EPISODES] })
  const wa = useAxis({ label: 'reward per episode', range: [-150, 0] })
  // The greedy route as move arrows in the method's colour; a policy that loops ends back on a state it visited.
  const panel = (l: typeof ql, name: string, slot: number) => (
    <GridView
      render={CLIFF.render!}
      x={qx}
      y={qy}
      title={name}
      value={l.field}
      path={l.path}
      pathSlot={slot}
      inkLatest={false}
      agent={false}
    />
  )
  return (
    <Figure
      title="Q-learning against SARSA on the cliff"
      purpose="Q-learning learns the values of the greedy policy and so the shortest path along the cliff edge, but its ε-greedy behaviour falls off now and then; SARSA learns the values of the policy it follows and takes the safer route, earning more per episode while exploring."
      state={state}
      defaultSize="XL"
      controls={
        <>
          <ControlRow label="2 · episode">
            <Player
              label="episode"
              value={e}
              onChange={setEpisode}
              count={EPISODES + 1}
              defaultSpeed={40}
              startReason="The figure compares the routes the two methods have learned, which needs all 500 episodes."
            />
          </ControlRow>
        </>
      }
      readouts={
        <>
          <Readout label="Q-learning path length" value={routeLength(ql.path)} />
          <Readout label="SARSA path length" value={routeLength(sl.path)} />
          <Readout
            label="Q-learning last 100 (mean)"
            value={(
              toFlat(runs.q.series.reward)
                .slice(-100)
                .reduce((a, b) => a + b, 0) / 100
            ).toFixed(1)}
          />
          <Readout
            label="SARSA last 100 (mean)"
            value={(
              toFlat(runs.s.series.reward)
                .slice(-100)
                .reduce((a, b) => a + b, 0) / 100
            ).toFixed(1)}
          />
        </>
      }
      caption="Drag the episode line or play. aifn episodes of qLearningAgent and sarsaAgent on cliffWalkingEnvironment (Sutton and Barto, Example 6.6), one episode per step, both on stream('cliff'); the cliff is the blank bottom row. Each grid draws the method's greedy route from the start as move arrows; a route that loops ends on a cell it already crossed. Bottom: reward per episode, a 20-episode moving average."
    >
      <Plots rows={3} heights={[1.5, 1.5, 1]}>
        {panel(ql, 'Q-learning', 0)}
        {panel(sl, 'SARSA', 1)}
        <Plot x={ea} y={wa}>
          <Curve name="Q-learning" x={curves.q.x} y={curves.q.y} slot={0} />
          <Curve name="SARSA" x={curves.s.x} y={curves.s.y} slot={1} />
          <Handle kind="x" at={e} label="episode" onDrag={(v) => setEpisode(Math.max(0, Math.round(v)))} />
        </Plot>
      </Plots>
    </Figure>
  )
}
