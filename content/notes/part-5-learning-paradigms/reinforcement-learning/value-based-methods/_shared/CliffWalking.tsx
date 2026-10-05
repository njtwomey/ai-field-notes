import { useMemo, useState } from 'react'
import {
  Button,
  Curve,
  Figure,
  float,
  formatNumber,
  Plot,
  Plots,
  Readout,
  StatusText,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { GridView } from 'aifn-render/gym'
import { train } from 'aifn-methods/gym'
import { greedyPath, greedyPolicy, qLearningAgent, sarsaAgent, type TdAgentState } from 'aifn-methods/gym/agents'
import { cliffWalkingEnvironment } from 'aifn-methods/gym/environments'
import { stream } from 'aifn-compute/foundation/random'
import { toFlat } from 'aifn-compute/foundation/tensor'

const EPISODES = 500
const RUNS = 10
const SMOOTH = 10
/** Guards against an early episode wandering for ever; far above any learned episode's length. */
const MAX_STEPS = 2000
const CLIFF = cliffWalkingEnvironment({ horizon: MAX_STEPS })
const START = CLIFF.reset(stream(0)).state
const RENDER = CLIFF.render!
const GOAL = RENDER.cells.indexOf('goal')
const METHODS = [
  { label: 'Q-learning', slot: 0, agent: qLearningAgent },
  { label: 'SARSA', slot: 1, agent: sarsaAgent },
] as const
const EPISODE_AXIS = Array.from({ length: EPISODES }, (_, i) => i + 1)

const mean = (xs: ArrayLike<number>) => Array.from(xs).reduce((a, b) => a + b, 0) / xs.length

/** Moving average over the previous SMOOTH episodes, so the curves show trends rather than single falls. */
const smooth = (xs: number[]) => xs.map((_, i) => mean(xs.slice(Math.max(0, i - SMOOTH + 1), i + 1)))

/** The greedy route from the start under a learnt Q (aifn `greedyPath`: it ends on a repeated cell when it loops). */
const routeOf = (agent: TdAgentState) =>
  Array.from(toFlat(greedyPath(CLIFF.model, START, greedyPolicy(CLIFF.model, agent.Q))))

type Settings = { eps: number; alpha: number }

/** Ten training runs of each method (aifn `train`), averaged per episode, with the first greedy route that arrives. */
function compare({ eps, alpha }: Settings) {
  return METHODS.map(({ agent }) => {
    const runs = Array.from({ length: RUNS }, (_, i) =>
      train(CLIFF, agent({ epsilon: eps, learningRate: alpha }), {
        episodes: EPISODES,
        seed: 101 + i,
        maxCheckpoints: 1,
      }),
    )
    const perEpisode = EPISODE_AXIS.map((_, e) => mean(runs.map((r) => r.returns[e])))
    // With a constant step size the values keep fluctuating, and in a few runs the greedy route bumps into the top
    // wall on the way. Show the first run whose greedy route reaches the goal.
    const routes = runs.map((r) => routeOf(r.final as TdAgentState))
    const reaching = routes.filter((p) => p.at(-1) === GOAL)
    return {
      perEpisode,
      route: reaching[0] ?? routes[0],
      reaching: reaching.length,
      late: mean(perEpisode.slice(-100)),
    }
  })
}

/**
 * Q-learning against SARSA on cliff walking (aifn `cliffWalkingEnvironment`, `qLearningAgent`, `sarsaAgent`): the greedy
 * route each has learnt, drawn on the shared `GridView`, and the reward each collects per episode while it learns with
 * ε-greedy exploration. Nothing trains until Train is pressed.
 */
export function CliffWalking() {
  const state = useFigureState({
    eps: float(0.1, { min: 0, max: 0.3, step: 0.01, label: 'exploration ε' }),
    alpha: float(0.5, { min: 0.1, max: 1, step: 0.05, label: 'step size α' }),
  })
  const [trained, setTrained] = useState<Settings | null>(null)
  const stale = trained !== null && (trained.eps !== state.eps || trained.alpha !== state.alpha)
  const results = useMemo(() => (trained ? compare(trained) : null), [trained])

  const xAxis = useAxis({ label: 'x', hold: 'initial' })
  const yAxis = useAxis({ label: 'y', hold: 'initial' })
  const xAxis2 = useAxis({ label: 'episode', range: [1, EPISODES] })
  const yAxis2 = useAxis({ label: 'reward per episode', range: [-100, 0] })
  return (
    <Figure
      title="Cliff walking: Q-learning against SARSA"
      state={state}
      caption="Both agents act ε-greedily and learn with step size α; every step pays −1 and a step into the cliff pays −100 and returns the agent to the start. Top: the greedy route each has learnt after 500 episodes, as move arrows from the start (bottom left) to the goal (bottom right), from the first of 10 runs whose greedy route reaches the goal; the cliff is the row between them. Q-learning learns the values of the greedy policy, so its route runs along the cliff edge, the shortest route. SARSA learns the values of the ε-greedy policy it actually follows, for which the edge is dangerous, so its route keeps a margin. Bottom: reward per episode during learning, averaged over 10 runs and smoothed over 10 episodes. While exploration continues, SARSA collects more reward than Q-learning, whose occasional random step off the edge costs 100. At ε = 0 both follow their greedy routes and the difference disappears. Press Train to run both methods."
      controls={
        <div className="flex flex-wrap items-center gap-3">
          <Button
            size="sm"
            variant={!trained || stale ? 'default' : 'outline'}
            onClick={() => setTrained({ eps: state.eps, alpha: state.alpha })}
          >
            {trained ? 'Retrain' : 'Train'}
          </Button>
          <StatusText tone={!trained || stale ? 'attention' : 'muted'}>
            {!trained
              ? 'Not trained yet: choose ε and α, then press Train.'
              : stale
                ? 'Settings changed since this run: press Retrain to train with them.'
                : `${RUNS} runs of ${EPISODES} episodes per method`}
          </StatusText>
        </div>
      }
      readouts={
        <>
          {METHODS.map(({ label }, i) => (
            <Readout
              key={label}
              label={`${label}: greedy route length, runs reaching the goal, mean reward (last 100 episodes)`}
              value={
                results
                  ? `${results[i].route.at(-1) === GOAL ? results[i].route.length - 1 : '–'}, ${results[i].reaching} of ${RUNS}, ${formatNumber(results[i].late)}`
                  : '–'
              }
            />
          ))}
        </>
      }
    >
      <Plots rows={3} heights={[1, 1, 1.2]}>
        {METHODS.map(({ label, slot }, i) => (
          <GridView
            key={label}
            render={RENDER}
            x={xAxis}
            y={yAxis}
            title={`${label}: greedy route`}
            path={results?.[i].route ?? null}
            pathSlot={slot}
            inkLatest={false}
            agent={false}
            ariaLabel={`Cliff-walking grid with the greedy route learnt by ${label}`}
          />
        ))}
        <Plot x={xAxis2} y={yAxis2}>
          {METHODS.map(({ label, slot }, i) =>
            results ? (
              <Curve key={label} name={label} x={EPISODE_AXIS} y={smooth(results[i].perEpisode)} slot={slot} />
            ) : null,
          )}
        </Plot>
      </Plots>
    </Figure>
  )
}
