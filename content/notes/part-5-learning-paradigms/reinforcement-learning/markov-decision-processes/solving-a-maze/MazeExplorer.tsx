import { useMemo, useState } from 'react'
import { Button, choice, float, formatNumber, Readout, row, setting, slider, useFigureState } from 'aifn-render'
import { GridView, GymTrainer, trainingRun, type GridOverlay } from 'aifn-render/gym'
import { gymSetup, greedyActions, optimalValues, qFromValues } from 'aifn-methods/gym'
import { greedyPath, valuesFromQ, type TdAgentState } from 'aifn-methods/gym/agents'
import type { MdpEnvironment } from 'aifn-methods/gym/environments'
import { stream } from 'aifn/foundation/random'
import { toFlat } from 'aifn/foundation/tensor'
import { MAZE_OPTIONS, mazeRows, presetWalls, wallsConnect, type MazeName } from './mazes'

const MAX_STEPS = 1000

/** The optimal values of the maze (value iteration on its model), its start state and the greedy route. */
function optimum(env: MdpEnvironment) {
  const V = optimalValues(env.model)
  const start = env.reset(stream(0)).state
  const policy = greedyActions(env.model, qFromValues(env.model, V))
  const route = Array.from(toFlat(greedyPath(env.model, start, policy)))
  return { V, start, route }
}

/**
 * A maze learnt by Q-learning or SARSA (aifn `qLearningAgent` and `sarsaAgent` in `mazeEnvironment`), on the shared
 * `GymTrainer`: nothing trains until Train is pressed; then any episode can be picked on the learning curves and played,
 * over the values max_a Q(s, a) and the greedy policy the agent held after that episode.
 */
export function MazeExplorer() {
  const state = useFigureState({
    setup: row('1 · maze and agent', {
      maze: choice<MazeName>(MAZE_OPTIONS, 'routes', { label: 'maze' }),
      method: choice(
        [
          { value: 'qLearningAgent', label: 'Q-learning' },
          { value: 'sarsaAgent', label: 'SARSA' },
        ],
        'qLearningAgent',
        { label: 'method' },
      ),
      gamma: slider(0.8, 0.99, 0.95, { step: 0.01, label: 'discount γ' }),
      slip: slider(0, 0.3, 0, { step: 0.01, label: 'slip probability' }),
      alpha: float(0.5, { gt: 0, max: 1, scale: 'log10', suggestions: [0.1, 0.25, 0.5, 1], label: 'step size α' }),
      epsilon: slider(0, 0.5, 0.1, { step: 0.01, label: 'exploration ε' }),
      decay: setting(false, { label: 'decay ε as ε / (1 + e/10)' }),
    }),
    run: trainingRun({ episodes: 300, seed: 1, label: '2 · training run' }),
  })
  const { maze: name, method, gamma, slip, alpha, epsilon, decay } = state.setup

  // Clicked walls belong to their maze; another maze opens with its preset walls.
  const [layout, setLayout] = useState(() => ({ name, walls: presetWalls(name) }))
  const walls = layout.name === name ? layout.walls : presetWalls(name)
  const rows = useMemo(() => mazeRows(name, walls), [name, walls])

  const setup = useMemo(
    () =>
      gymSetup('mazeEnvironment', { layout: rows, slip, gamma, horizon: MAX_STEPS }, method, {
        learningRate: alpha,
        epsilon,
        ...(decay ? { epsilonDecay: 10 } : {}),
      }),
    [rows, slip, gamma, method, alpha, epsilon, decay],
  )
  const env = setup.env as MdpEnvironment
  const best = useMemo(() => optimum(env), [env])

  const toggleWall = ([x, y]: [number, number]) => {
    const next = wallsConnect(name, walls, Math.round(x), Math.round(y))
    if (next) setLayout({ name, walls: next })
  }

  // What the agent knew after the chosen episode: max_a Q(s, a) in colour, the greedy action as an arrow where the
  // cell's actions differ (an untouched cell has no preference, so no arrow).
  const lo = useMemo(() => Math.floor(Math.min(...best.V.filter((_, s) => !env.model.terminal[s]))), [best, env])
  const overlay = useMemo(
    () =>
      (agent: unknown): GridOverlay => {
        const Q = (agent as TdAgentState).Q
        const q = toFlat(Q)
        const A = env.model.actions
        const policy = Int32Array.from({ length: env.model.states }, (_, s) => {
          if (env.model.terminal[s]) return -1
          let a = 0
          let differ = false
          for (let b = 1; b < A; b++) {
            if (q[s * A + b] !== q[s * A]) differ = true
            if (q[s * A + b] > q[s * A + a]) a = b
          }
          return differ ? a : -1
        })
        return {
          value: {
            values: toFlat(valuesFromQ(env.model, Q)),
            label: 'max Q',
            range: [lo, 10],
            scale: 'sequential',
            fillOpacity: 0.7,
          },
          policy,
        }
      },
    [env, lo],
  )
  const rendererOptions = { overlay, onPlotClick: toggleWall }

  // With deterministic moves the shortest route has a known return: +10 at the goal and −1 for every other move.
  const shortest = slip === 0 && env.render!.cells[best.route.at(-1)!] === 'goal' ? best.route.length - 1 : null

  return (
    <GymTrainer
      title="Solving a maze"
      purpose="Q-learning and SARSA learn a maze from sampled steps; the values and the greedy policy spread back from the goal one visited cell at a time."
      state={state}
      setup={setup}
      defaultSize="L"
      scalars={decay ? ['ε'] : []}
      rendererOptions={rendererOptions}
      idleScene={
        <GridView
          render={env.render!}
          title="press Train to start; click a cell to add or remove a wall"
          onPlotClick={toggleWall}
          ariaLabel="The maze before training: walls, traps, the start and the goal"
        />
      }
      actions={
        <Button variant="outline" size="sm" onClick={() => setLayout({ name, walls: presetWalls(name) })}>
          Reset walls
        </Button>
      }
      readouts={
        <>
          <Readout label="v*(start) by value iteration" value={formatNumber(best.V[best.start])} />
          <Readout
            label="shortest route (moves, return)"
            value={shortest === null ? '–' : `${shortest}, ${formatNumber(10 - (shortest - 1))}`}
          />
        </>
      }
      caption={
        <>
          Each move pays −1, entering the goal pays +10 and ends the episode, and entering a trap pays −20 and returns
          the agent to the start. With slip, a move goes sideways with that probability. Colour is max<sub>a</sub> Q(s,
          a) after the chosen episode and arrows are the greedy policy; cells without an arrow have not been updated
          yet. The path is the chosen episode, one arrow per move, counted when a move repeats. Click a cell to add or
          remove a wall; a changed maze or setting is trained from Q = 0 when Train is pressed again.
        </>
      }
    />
  )
}
