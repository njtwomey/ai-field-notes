import { useMemo, useState } from 'react'
import { cliffWalking, frozenLake, gridworld, maze } from 'aifn-applied/data/environments'
import { greedyPath, qLearning, sarsa } from 'aifn-applied/decisions/reinforcement-learning/learning'
import { policyIteration, valueIteration } from 'aifn-applied/decisions/reinforcement-learning/planning'
import { stateCell, type TabularMdp } from 'aifn-applied/decisions/reinforcement-learning'
import { stream } from 'aifn/foundation/random'
import { toFlat, type Tensor } from 'aifn/foundation/tensor'
import { trace } from 'aifn/foundation/trace'
import { Player } from '@lab/controls'
import { ControlRow, Figure } from '@lab/layout'
import { choice, row, slider, useFigureState } from '@lab/state'
import { Bars, Curve, Handle, Plot, Plots, Raster, Readout, useAxis, Vectors, type Vector } from '@lab/viz'

/** The value table as heatmap rows (row y = grid row y, bottom first); walls are blank. */
function gridRows(mdp: TabularMdp, V: Tensor): number[][] {
  const g = mdp.grid!
  const v = toFlat(V)
  return Array.from({ length: g.height }, (_, y) =>
    Array.from({ length: g.width }, (_, x) => (g.kinds[y * g.width + x] === 'wall' ? NaN : v[y * g.width + x])),
  )
}

/** Greedy-policy arrows from each active cell's centre. */
function arrows(mdp: TabularMdp, policy: Tensor): Vector[] {
  const g = mdp.grid!
  const p = toFlat(policy)
  const out: Vector[] = []
  p.forEach((a, s) => {
    if (a < 0) return
    const [x, y] = stateCell(g.width, s)
    const [dx, dy] = g.actionVectors[a]
    out.push({ from: [x - 0.2 * dx, y - 0.2 * dy], to: [x + 0.3 * dx, y + 0.3 * dy] })
  })
  return out
}

/** A path of states as the x and y of its cells. */
function pathCells(mdp: TabularMdp, path: Tensor) {
  const w = mdp.grid!.width
  const cells = toFlat(path).map((s) => stateCell(w, s))
  return { x: cells.map((c) => c[0]), y: cells.map((c) => c[1]) }
}

const axes = (n: number) => Array.from({ length: n }, (_, i) => i)

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
  const mdp = useMemo(() => {
    if (world === 'gridworld') return gridworld({ gamma, noise, stepReward })
    if (world === 'frozen') return frozenLake({ gamma })
    return maze(['.....#...G', '.###.#.##.', '.#...#....', '.#.####.#.', '.#......#.', 'S..####...'], {
      gamma,
      slip: noise,
    })
  }, [world, gamma, noise, stepReward])
  const run = useMemo(() => {
    if (method === 'vi')
      return trace(valueIteration(mdp, { tolerance: 1e-8 }), undefined, 400, {
        record: { residual: (s) => Math.max(s.residual, 1e-16) },
      })
    return trace(policyIteration(mdp), undefined, 30, { record: { changed: (s) => s.changed } })
  }, [mdp, method])
  const [step, setStep] = useState(0)
  const at = Math.min(step, run.steps.length - 1)
  const s = run.steps[at]
  const g = mdp.grid!
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
  const values = useMemo(() => gridRows(mdp, s.V), [mdp, s])
  const policyArrows = useMemo(() => arrows(mdp, s.policy), [mdp, s])
  const gx = useMemo(() => axes(g.width), [g])
  const gy = useMemo(() => axes(g.height), [g])
  const xa = useAxis({ label: 'x' })
  const ya = useAxis({ label: 'y', equal: xa })
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
          <Readout label="V(start)" value={toFlat(s.V)[mdp.start].toFixed(4)} />
          <Readout
            label={method === 'vi' ? 'residual' : 'changed'}
            value={'residual' in s ? s.residual.toExponential(2) : s.changed}
          />
          <Readout label="stopped" value={`${run.meta.stopped} after ${run.meta.steps}`} />
        </>
      }
      caption="Play or drag the k line. aifn valueIteration and policyIteration on gridworld (exits +1 and −1), a text maze (goal +10, step −1) and FrozenLake; colour is V, arrows the greedy policy. Right: the residual per sweep (it falls at least as fast as γᵏ) or the actions changed per improvement."
    >
      <Plots cols={2} widths={[1.4, 1]}>
        <Plot x={xa} y={ya}>
          <Raster x={gx} y={gy} z={values} scale="diverging" range={vRange} valueLabel="V(s)" />
          <Vectors vectors={policyArrows} />
        </Plot>
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

const CLIFF = cliffWalking()
const EPISODES = 500

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
      epsilon: slider(0, 0.3, 0.1, { label: 'exploration ε' }),
      alpha: slider(0.05, 1, 0.5, { label: 'step size α' }),
    }),
  })
  const { epsilon, alpha } = state.learning
  const runs = useMemo(() => {
    const opts = { epsilon, learningRate: alpha }
    const rec = { record: { reward: (s: { rewardSum: number }) => s.rewardSum }, stream: stream('cliff') }
    return {
      q: trace(qLearning(CLIFF, opts), undefined, EPISODES, rec),
      s: trace(sarsa(CLIFF, opts), undefined, EPISODES, rec),
    }
  }, [epsilon, alpha])
  const [episode, setEpisode] = useState(EPISODES)
  const e = Math.min(episode, EPISODES)
  const q = runs.q.steps[e]
  const s = runs.s.steps[e]
  const g = CLIFF.grid!
  const qPath = greedyPath(CLIFF, q.policy)
  const sPath = greedyPath(CLIFF, s.policy)
  const curves = useMemo(
    () => ({
      q: { x: Array.from(runs.q.index), y: smooth(toFlat(runs.q.series.reward)) },
      s: { x: Array.from(runs.s.index), y: smooth(toFlat(runs.s.series.reward)) },
    }),
    [runs],
  )
  // The cliff (bottom row between start and goal) is left blank.
  const valueRows = (V: Tensor) =>
    gridRows(CLIFF, V).map((row, y) => row.map((v, x) => (y === 0 && x > 0 && x < g.width - 1 ? NaN : v)))
  const gx = axes(g.width)
  const gy = axes(g.height)
  const qx = useAxis({ label: 'x' })
  // Cells, not geometry: no equal units, so the two grids share the column's height evenly.
  const qy = useAxis({ label: 'y' })
  const sx = useAxis({ label: 'x' })
  const sy = useAxis({ label: 'y' })
  const ea = useAxis({ label: 'episode', range: [0, EPISODES] })
  const wa = useAxis({ label: 'reward per episode', range: [-150, 0] })
  const panel = (V: Tensor, path: Tensor, name: string, slot: number, x: typeof qx, y: typeof qy) => {
    const p = pathCells(CLIFF, path)
    return (
      <Plot x={x} y={y} title={name}>
        <Raster x={gx} y={gy} z={valueRows(V)} range={[-20, 0]} valueLabel="max_a Q(s, a)" />
        <Curve name={`${name} greedy path`} x={p.x} y={p.y} slot={slot} showPoints />
      </Plot>
    )
  }
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
          <Readout label="Q-learning path length" value={qPath.shape[0] - 1} />
          <Readout label="SARSA path length" value={sPath.shape[0] - 1} />
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
      caption="Drag the episode line or play. aifn qLearning and sarsa on cliffWalking (Sutton and Barto, Example 6.6), one episode per step, both on stream('cliff'); the cliff is the blank bottom row. Bottom: reward per episode, a 20-episode moving average."
    >
      <Plots rows={3} heights={[1.5, 1.5, 1]}>
        {panel(q.V, qPath, 'Q-learning', 0, qx, qy)}
        {panel(s.V, sPath, 'SARSA', 1, sx, sy)}
        <Plot x={ea} y={wa}>
          <Curve name="Q-learning" x={curves.q.x} y={curves.q.y} slot={0} />
          <Curve name="SARSA" x={curves.s.x} y={curves.s.y} slot={1} />
          <Handle kind="x" at={e} label="episode" onDrag={(v) => setEpisode(Math.max(0, Math.round(v)))} />
        </Plot>
      </Plots>
    </Figure>
  )
}
