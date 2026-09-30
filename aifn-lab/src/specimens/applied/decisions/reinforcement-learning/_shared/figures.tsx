import { useMemo, useState } from 'react'
import { cliffWalking, frozenLake, gridworld, maze } from 'aifn-applied/data/environments'
import { greedyPath, qLearning, sarsa } from 'aifn-applied/decisions/reinforcement-learning/learning'
import { policyIteration, valueIteration } from 'aifn-applied/decisions/reinforcement-learning/planning'
import { stateCell, type TabularMdp } from 'aifn-applied/decisions/reinforcement-learning'
import { stream } from 'aifn/foundation/random'
import { toFlat, type Tensor } from 'aifn/foundation/tensor'
import { trace } from 'aifn/foundation/trace'
import { Player, Select, Slider } from '@lab/controls'
import { ControlRow, Figure } from '@lab/layout'
import { Heatmap, Panel, Readout, Subplots, XYChart, type HeatmapOverlay, type Vector, type XYSeries } from '@lab/viz'

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

function pathOverlay(mdp: TabularMdp, path: Tensor, name: string, slot: number): HeatmapOverlay {
  const w = mdp.grid!.width
  const cells = toFlat(path).map((s) => stateCell(w, s))
  return { name, type: 'line', x: cells.map((c) => c[0]), y: cells.map((c) => c[1]), slot, showPoints: true }
}

const axes = (n: number) => Array.from({ length: n }, (_, i) => i)

// ---------------------------------------------------------------------------------------------------------------------
// 1. Value iteration and policy iteration, sweep by sweep.

type World = 'gridworld' | 'maze' | 'frozen'

export function PlanningSpecimen() {
  const [world, setWorld] = useState<World>('gridworld')
  const [method, setMethod] = useState<'vi' | 'pi'>('vi')
  const [gamma, setGamma] = useState(0.9)
  const [noise, setNoise] = useState(0.2)
  const [stepReward, setStepReward] = useState(-0.04)
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
  const [step, setStep] = useState(3)
  const at = Math.min(step, run.steps.length - 1)
  const s = run.steps[at]
  const g = mdp.grid!
  // A symmetric colour range about 0 from the final values, so the sign of V reads as its hue.
  const vRange = useMemo<[number, number]>(() => {
    const m = Math.max(1e-9, ...toFlat(run.steps[run.steps.length - 1].V).map(Math.abs))
    return [-m, m]
  }, [run])
  const series: XYSeries[] =
    method === 'vi'
      ? [{ name: 'Bellman residual', type: 'line', x: Array.from(run.index), y: toFlat(run.series.residual), slot: 2 }]
      : [
          {
            name: 'states whose action changed',
            type: 'bar',
            x: Array.from(run.index),
            y: toFlat(run.series.changed),
            slot: 2,
          },
        ]
  return (
    <Figure
      title="Planning on a grid: values and greedy policy per sweep"
      description="Value iteration spreads value outwards from the rewarding cells one step per sweep, and its greedy policy settles long before the values stop changing; policy iteration evaluates each policy exactly and needs only a handful of improvements."
      defaultSize="L"
      controls={
        <>
          <ControlRow label="1 · world">
            <Select
              label="world"
              value={world}
              onChange={setWorld}
              options={[
                { value: 'gridworld', label: 'gridworld 4 × 3' },
                { value: 'maze', label: 'maze' },
                { value: 'frozen', label: 'FrozenLake 4 × 4 (slippery)' },
              ]}
            />
            <Slider label="discount γ" value={gamma} onChange={setGamma} min={0.5} max={0.99} />
            {world !== 'frozen' && (
              <Slider label="slip probability" value={noise} onChange={setNoise} min={0} max={0.6} />
            )}
            {world === 'gridworld' && (
              <Slider label="step reward" value={stepReward} onChange={setStepReward} min={-1} max={0.1} />
            )}
          </ControlRow>
          <ControlRow label="2 · method and sweep">
            <Select
              label="method"
              value={method}
              onChange={setMethod}
              options={[
                { value: 'vi', label: 'value iteration' },
                { value: 'pi', label: 'policy iteration' },
              ]}
            />
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
      caption="aifn/rl valueIteration and policyIteration on gridworld (exits +1 and −1), a text maze (goal +10, step −1) and FrozenLake; colour is V, arrows the greedy policy. Right: the residual per sweep (it falls at least as fast as γᵏ) or the actions changed per improvement."
    >
      <Subplots cols={2} widthRatios={[1.4, 1]}>
        <Panel>
          <Heatmap
            x={axes(g.width)}
            y={axes(g.height)}
            z={gridRows(mdp, s.V)}
            scale="diverging"
            range={vRange}
            valueLabel="V(s)"
            vectors={arrows(mdp, s.policy)}
            equalAspect
            rescaleOnChange={false}
            axisKey={`${world}:${method}`}
            holdFit="union"
            xLabel="x"
            yLabel="y"
          />
        </Panel>
        <Panel>
          <XYChart
            series={series}
            xLabel={method === 'vi' ? 'sweep k' : 'iteration'}
            yLabel={method === 'vi' ? 'max |TV − V|' : 'changed'}
            yLog={method === 'vi'}
            handles={[{ kind: 'x', at, label: 'k', onDrag: (v) => setStep(Math.max(0, Math.round(v))) }]}
          />
        </Panel>
      </Subplots>
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
  const [epsilon, setEpsilon] = useState(0.1)
  const [alpha, setAlpha] = useState(0.5)
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
  const curves: XYSeries[] = [
    { name: 'Q-learning', type: 'line', x: Array.from(runs.q.index), y: smooth(toFlat(runs.q.series.reward)), slot: 0 },
    { name: 'SARSA', type: 'line', x: Array.from(runs.s.index), y: smooth(toFlat(runs.s.series.reward)), slot: 1 },
  ]
  const panel = (V: Tensor, path: Tensor, name: string, slot: number) => (
    <Heatmap
      x={axes(g.width)}
      y={axes(g.height)}
      z={gridRows(CLIFF, V).map((row, y) => row.map((v, x) => (y === 0 && x > 0 && x < g.width - 1 ? NaN : v)))}
      valueLabel="max_a Q(s, a)"
      overlay={[pathOverlay(CLIFF, path, `${name} greedy path`, slot)]}
      xLabel="x"
      yLabel="y"
      range={[-20, 0]}
    />
  )
  return (
    <Figure
      title="Q-learning against SARSA on the cliff"
      description="Q-learning learns the values of the greedy policy and so the shortest path along the cliff edge, but its ε-greedy behaviour falls off now and then; SARSA learns the values of the policy it follows and takes the safer route, earning more per episode while exploring."
      defaultSize="XL"
      controls={
        <>
          <ControlRow label="1 · learning">
            <Slider label="exploration ε" value={epsilon} onChange={setEpsilon} min={0} max={0.3} />
            <Slider label="step size α" value={alpha} onChange={setAlpha} min={0.05} max={1} />
          </ControlRow>
          <ControlRow label="2 · episode">
            <Player label="episode" value={e} onChange={setEpisode} count={EPISODES + 1} defaultSpeed={40} />
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
      caption="aifn/rl qLearning and sarsa on cliffWalking (Sutton and Barto, Example 6.6), one episode per step, both on stream('cliff'); the cliff is the blank bottom row. Bottom: reward per episode, a 20-episode moving average."
    >
      <Subplots rows={3} heightRatios={[1.5, 1.5, 1]}>
        <Panel>{panel(q.V, qPath, 'Q-learning', 0)}</Panel>
        <Panel>{panel(s.V, sPath, 'SARSA', 1)}</Panel>
        <Panel>
          <XYChart
            series={curves}
            xLabel="episode"
            yLabel="reward per episode"
            yRange={[-150, 0]}
            handles={[{ kind: 'x', at: e, label: 'episode', onDrag: (v) => setEpisode(Math.max(0, Math.round(v))) }]}
          />
        </Panel>
      </Subplots>
    </Figure>
  )
}
