import { useMemo, useState } from 'react'
import {
  Button,
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Plot,
  Points,
  Readout,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { GridView } from 'aifn-render/gym'
import { GRID_ACTION_NAMES } from 'aifn-methods/gym'
import { policyIteration, valueIteration } from 'aifn-methods/gym/agents'
import { gridworldEnvironment } from 'aifn-methods/gym/environments'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { trace } from 'aifn-compute/foundation/trace'

type Vec2 = [number, number]

const WIDTH = 7
const HEIGHT = 5
const MAX_SWEEPS = 120
const MAX_IMPROVEMENTS = 12
const START_WALLS: Vec2[] = [
  [3, 1],
  [3, 2],
  [3, 3],
]
const START_GOAL: Vec2 = [6, 4]
const START_TRAP: Vec2 = [6, 2]
const START = 0
const RANGE: [number, number] = [-1, 1]

const clampCell = ([x, y]: Vec2): Vec2 => [
  Math.min(WIDTH - 1, Math.max(0, Math.round(x))),
  Math.min(HEIGHT - 1, Math.max(0, Math.round(y))),
]
const same = (a: Vec2, b: Vec2) => a[0] === b[0] && a[1] === b[1]
const wallKey = (walls: Vec2[]) => walls.map(([x, y]) => `${x},${y}`).join(';')

type Algorithm = 'value-iteration' | 'policy-iteration'

/** One iterate: the value table, its greedy policy (−1 where no action is taken) and the largest change from the last. */
type Iterate = { V: Float64Array; policy: Int32Array; residual: number }

/**
 * Dynamic programming on an editable gridworld (aifn `gridworldEnvironment`, solved by `valueIteration` or
 * `policyIteration` on its tabular model), drawn by the shared `GridView`. Colour is the value table after k sweeps
 * (value iteration) or k improvement steps (policy iteration); arrows are the greedy policy. Walls toggle on click; the
 * +1 goal and the −1 trap are draggable.
 */
export function GridworldPlanner({ algorithm = 'value-iteration' }: { algorithm?: Algorithm }) {
  const vi = algorithm === 'value-iteration'
  const [walls, setWalls] = useState<Vec2[]>(START_WALLS)
  const [goal, setGoal] = useState<Vec2>(START_GOAL)
  const [trap, setTrap] = useState<Vec2>(START_TRAP)
  const state = useFigureState({
    sweep: int(vi ? 5 : 1, {
      min: 0,
      max: vi ? MAX_SWEEPS : MAX_IMPROVEMENTS,
      step: 1,
      label: vi ? 'sweep k' : 'improvement step k',
    }),
    gamma: float(0.9, { min: 0.5, max: 0.99, step: 0.01, label: 'discount γ' }),
    noise: slider(0, 0.6, 0.2, { step: 0.02, label: 'noise (sideways probability)' }),
    step: float(-0.02, { min: -0.2, max: 0, step: 0.01, label: 'step reward' }),
  })

  const walled = wallKey(walls)
  const [gx, gy] = goal
  const [tx, ty] = trap
  const { gamma, noise, step: stepReward } = state
  const env = useMemo(
    () =>
      gridworldEnvironment({
        width: WIDTH,
        height: HEIGHT,
        walls: walled ? walled.split(';').map((w) => w.split(',').map(Number) as Vec2) : [],
        terminals: [
          { x: gx, y: gy, value: 1 },
          { x: tx, y: ty, value: -1 },
        ],
        start: [0, 0],
        gamma,
        noise,
        stepReward,
      }),
    [walled, gx, gy, tx, ty, gamma, noise, stepReward],
  )

  // The iterates from V₀ = 0 (value iteration) or from "always up" (policy iteration), with the sup-norm change from
  // the previous iterate. Policy iteration ends at the last policy that changed.
  const sweeps = useMemo((): Iterate[] => {
    const run = vi
      ? trace(valueIteration(env.model, { tolerance: 1e-10 }), undefined, MAX_SWEEPS)
      : trace(policyIteration(env.model), undefined, MAX_IMPROVEMENTS + 1)
    const steps = vi ? run.steps : run.steps.filter((s, i) => i === 0 || ('changed' in s && s.changed > 0))
    return steps.map((s, i) => {
      const V = Float64Array.from(toFlat(s.V))
      const prev = i > 0 ? toFlat(steps[i - 1].V) : null
      const residual = prev ? Math.max(...V.map((v, j) => Math.abs(v - prev[j]))) : NaN
      return { V, policy: Int32Array.from(toFlat(s.policy)), residual }
    })
  }, [env, vi])

  const last = sweeps.length - 1
  const k = Math.min(state.sweep, last)
  const { V, policy, residual } = sweeps[k]
  const field = useMemo(() => ({ values: V, label: 'V', range: RANGE }), [V])
  const wallXs = useMemo(() => walls.map((w) => w[0]), [walls])
  const wallYs = useMemo(() => walls.map((w) => w[1]), [walls])

  const toggleWall = (p: [number, number]) => {
    const c: Vec2 = clampCell(p)
    if (same(c, goal) || same(c, trap)) return
    setWalls((prev) => (prev.some((w) => same(w, c)) ? prev.filter((w) => !same(w, c)) : [...prev, c]))
  }
  // Dropping a terminal on a wall removes the wall; dropping it on the other terminal is refused.
  const place = (set: (c: Vec2) => void, other: Vec2) => (p: Vec2) => {
    const c = clampCell(p)
    if (same(c, other)) return
    setWalls((prev) => prev.filter((w) => !same(w, c)))
    set(c)
  }
  const reset = () => {
    setWalls(START_WALLS)
    setGoal(START_GOAL)
    setTrap(START_TRAP)
  }

  // Value iteration only: the sup-norm change per sweep against the contraction bound γᵏ · (first change).
  const residualSeries = useMemo(() => {
    const ks = sweeps.map((_, i) => i).slice(1)
    const first = sweeps[1]?.residual ?? 1
    return [
      { name: '‖V_k − V_(k−1)‖∞', x: ks, y: ks.map((i) => Math.max(sweeps[i].residual, 1e-12)) },
      {
        name: 'γ^(k−1) × first change',
        x: ks,
        y: ks.map((i) => Math.max(first * gamma ** (i - 1), 1e-12)),
        dashed: true,
        slot: 1,
      },
    ] as const
  }, [sweeps, gamma])

  const startAction = policy[START]

  const xAxis = useAxis({ label: 'sweep k', range: [1, Math.max(2, last)] })
  const yAxis = useAxis({ label: 'largest change', hold: 'union', log: true })
  return (
    <Figure
      title={vi ? 'Value iteration on a gridworld' : 'Policy iteration on a gridworld'}
      state={state}
      caption={
        vi
          ? 'Colour is the value table V_k after k sweeps of the Bellman optimality update, starting from V₀ = 0; arrows are the greedy policy with respect to V_k. Terminal cells pay +1 (goal) or −1 (trap) and end the episode; every other move pays the step reward. With noise, a move goes sideways with that probability, split between the two perpendicular directions. Click a cell to add or remove a wall (walls are the dark dots); drag the goal or the trap. Values spread outwards from the terminals one cell per sweep, and the greedy policy settles long before the values stop changing. Right: the largest change per sweep falls at least as fast as γᵏ; drag the vertical line to choose k.'
          : 'Colour is the exact value V^π_k of the k-th policy, found by solving the linear Bellman expectation equations; arrows are π_k. The first policy moves up everywhere. Each step makes the policy greedy with respect to the current values, then re-evaluates it. Terminal cells pay +1 or −1 and end the episode; every other move pays the step reward, and with noise a move goes sideways. Click a cell to add or remove a wall (walls are the dark dots); drag the goal or the trap. Policy iteration stops after a handful of steps, when no action changes.'
      }
      controls={
        <>
          <Button variant="outline" size="sm" onClick={reset}>
            Reset layout
          </Button>
        </>
      }
      readouts={
        <>
          <Readout label={vi ? 'sweeps to converge' : 'steps to converge'} value={String(last)} />
          <Readout label="shown" value={k < state.sweep ? `k = ${k} (converged)` : `k = ${k}`} />
          <Readout label="largest change at k" value={k === 0 ? '–' : formatNumber(residual)} />
          <Readout label="V at bottom-left" value={formatNumber(V[START])} />
          <Readout label="action there" value={startAction >= 0 ? GRID_ACTION_NAMES[startAction] : '–'} />
        </>
      }
    >
      <div className={vi ? 'grid gap-4 md:grid-cols-[3fr_2fr]' : ''}>
        <GridView
          render={env.render!}
          value={field}
          policy={policy}
          height={340}
          onPlotClick={toggleWall}
          ariaLabel="Gridworld value table with greedy policy arrows"
        >
          <Points name="wall" x={wallXs} y={wallYs} emphasis live />
          <Handle kind="point" at={goal} label="goal +1" onDrag={place(setGoal, trap)} />
          <Handle kind="point" at={trap} label="trap −1" onDrag={place(setTrap, goal)} />
        </GridView>
        {vi && (
          <Plot x={xAxis} y={yAxis}>
            <Curve {...residualSeries[0]} />
            <Curve {...residualSeries[1]} />
            <Handle kind="x" at={Math.max(1, k)} label="k" onDrag={(x) => state.set('sweep', Math.round(x))} />
          </Plot>
        )}
      </div>
    </Figure>
  )
}
