import { useMemo } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, useParam, type XYSeries } from 'aifn-render'
import { GOAL, HEIGHT, START, WIDTH, greedyPath, isCliff, train, xy, type Method } from './cliff'

const EPISODES = 500
const RUNS = 10
const SMOOTH = 10
const METHODS: { method: Method; label: string; slot: number; offset: number }[] = [
  { method: 'q-learning', label: 'Q-learning', slot: 0, offset: -0.12 },
  { method: 'sarsa', label: 'SARSA', slot: 1, offset: 0.12 },
]
const CLIFF = Array.from({ length: WIDTH * HEIGHT }, (_, s) => s).filter(isCliff)
const EPISODE_AXIS = Array.from({ length: EPISODES }, (_, i) => i + 1)

const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length

/** Moving average over the previous SMOOTH episodes, so the curves show trends rather than single falls. */
const smooth = (xs: number[]) => xs.map((_, i) => mean(xs.slice(Math.max(0, i - SMOOTH + 1), i + 1)))

/**
 * Q-learning against SARSA on cliff walking: the greedy path each has learned, and the reward each collects per
 * episode while it learns with ε-greedy exploration.
 */
export function CliffWalking() {
  const eps = useParam(0.1, { min: 0, max: 0.3, step: 0.01 })
  const alpha = useParam(0.5, { min: 0.1, max: 1, step: 0.05 })

  const results = useMemo(
    () =>
      METHODS.map(({ method }) => {
        const runs = Array.from({ length: RUNS }, (_, i) => train(method, EPISODES, eps.value, alpha.value, 101 + i))
        const perEpisode = EPISODE_AXIS.map((_, e) => mean(runs.map((r) => r.returns[e])))
        // With a constant step size the values keep fluctuating, and in a few runs the greedy path bumps into the top
        // wall on the way. Show the first run whose greedy path reaches the goal.
        const paths = runs.map((r) => greedyPath(r.Q))
        const reaching = paths.filter((p) => p.at(-1) === GOAL)
        return {
          perEpisode,
          path: reaching[0] ?? paths[0],
          reaching: reaching.length,
          late: mean(perEpisode.slice(-100)),
        }
      }),
    [eps.value, alpha.value],
  )

  const gridSeries: XYSeries[] = [
    {
      name: 'cliff (−100, back to start)',
      type: 'scatter',
      x: CLIFF.map((s) => xy(s)[0]),
      y: CLIFF.map((s) => xy(s)[1]),
      muted: true,
    },
    ...METHODS.map(({ label, slot, offset }, i): XYSeries => ({
      name: `${label}: greedy path`,
      type: 'line',
      x: results[i].path.map((s) => xy(s)[0]),
      y: results[i].path.map((s) => xy(s)[1] + offset),
      slot,
    })),
    { name: 'start and goal', type: 'scatter', x: [xy(START)[0], xy(GOAL)[0]], y: [0, 0], emphasis: true },
  ]
  const rewardSeries: XYSeries[] = METHODS.map(({ label, slot }, i) => ({
    name: label,
    type: 'line',
    x: EPISODE_AXIS,
    y: smooth(results[i].perEpisode),
    slot,
  }))

  return (
    <Interactive
      title="Cliff walking: Q-learning against SARSA"
      caption="Both agents act ε-greedily and learn with step size α; every step pays −1 and a step into the cliff pays −100 and returns the agent to the start. Top: the greedy path each has learned after 500 episodes, from the first of 10 runs whose greedy path reaches the goal. Q-learning learns the values of the greedy policy, so its path runs along the cliff edge, the shortest route. SARSA learns the values of the ε-greedy policy it actually follows, for which the edge is dangerous, so its path keeps a margin. Bottom: reward per episode during learning, averaged over 10 runs and smoothed over 10 episodes. While exploration continues, SARSA collects more reward than Q-learning, whose occasional random step off the edge costs 100. At ε = 0 both follow their greedy paths and the difference disappears."
      controls={
        <>
          <ParamSlider label="exploration ε" param={eps} />
          <ParamSlider label="step size α" param={alpha} />
        </>
      }
      readout={
        <>
          {METHODS.map(({ label }, i) => (
            <Readout
              key={label}
              label={`${label}: greedy path length, runs reaching the goal, mean reward (last 100 episodes)`}
              value={`${results[i].path.at(-1) === GOAL ? results[i].path.length - 1 : '–'}, ${results[i].reaching} of ${RUNS}, ${formatNumber(results[i].late)}`}
            />
          ))}
        </>
      }
    >
      <div className="flex flex-col gap-4">
        <XYChart
          series={gridSeries}
          xLabel="x"
          yLabel="y"
          xRange={[-0.5, WIDTH - 0.5]}
          yRange={[-0.5, HEIGHT - 0.5]}
          equalAspect
          ariaLabel="Cliff-walking grid with the greedy paths learned by Q-learning and SARSA"
        />
        <XYChart
          series={rewardSeries}
          xLabel="episode"
          yLabel="reward per episode"
          xRange={[1, EPISODES]}
          yRange={[-100, 0]}
          height={260}
        />
      </div>
    </Interactive>
  )
}
