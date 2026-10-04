import { useMemo } from 'react'
import { Curve, Figure, float, formatNumber, Plot, Readout, useAxis, useFigureState } from 'aifn-render'
import { stream, type Stream, uniform } from 'aifn/foundation/random'

const N = 5
const EPISODES = 100
const RUNS = 100
const TRUE = Array.from({ length: N }, (_, i) => (i + 1) / (N + 1))
const EPISODE_AXIS = Array.from({ length: EPISODES + 1 }, (_, i) => i)

type Method = 'td' | 'mc'

/** One episode of the random walk from the centre: the visited states and the final reward (1 on the right). */
function episode(r: Stream): { states: number[]; reward: number } {
  let s = Math.floor(N / 2)
  const states = [s]
  for (;;) {
    s += uniform(r) < 0.5 ? -1 : 1
    if (s < 0) return { states, reward: 0 }
    if (s >= N) return { states, reward: 1 }
    states.push(s)
  }
}

const rmsError = (V: number[]) => Math.sqrt(V.reduce((acc, v, i) => acc + (v - TRUE[i]) ** 2, 0) / N)

/** RMS error of the value estimates after each episode, averaged over RUNS runs. Both methods see the same walks. */
function learningCurve(method: Method, alpha: number): number[] {
  const total = new Array<number>(EPISODES + 1).fill(0)
  for (let run = 0; run < RUNS; run++) {
    const r = stream(1000 + run)
    const V = new Array<number>(N).fill(0.5)
    total[0] += rmsError(V)
    for (let e = 1; e <= EPISODES; e++) {
      const { states, reward } = episode(r)
      if (method === 'mc') {
        // Every state's return is the final reward, since intermediate rewards are 0 and γ = 1.
        for (const s of states) V[s] += alpha * (reward - V[s])
      } else {
        states.forEach((s, i) => {
          const target = i + 1 < states.length ? V[states[i + 1]] : reward
          V[s] += alpha * (target - V[s])
        })
      }
      total[e] += rmsError(V)
    }
  }
  return total.map((t) => t / RUNS)
}

/** TD(0) against constant-α Monte Carlo on the five-state random walk (Sutton and Barto, Example 6.2). */
export function RandomWalkTdMc() {
  const state = useFigureState({
    alphaTd: float(0.05, { min: 0.01, max: 0.5, step: 0.01, label: 'TD step size α' }),
    alphaMc: float(0.02, { min: 0.01, max: 0.5, step: 0.01, label: 'Monte Carlo step size α' }),
  })
  const td = useMemo(() => learningCurve('td', state.alphaTd), [state.alphaTd])
  const mc = useMemo(() => learningCurve('mc', state.alphaMc), [state.alphaMc])

  const series = [
    { name: 'TD(0)', x: EPISODE_AXIS, y: td, slot: 0 },
    { name: 'constant-α Monte Carlo', x: EPISODE_AXIS, y: mc, slot: 1 },
  ] as const

  const xAxis = useAxis({ label: 'episode', range: [0, EPISODES] })
  const yAxis = useAxis({ label: 'RMS error', range: [0, 0.25] })
  return (
    <Figure
      title="TD(0) against Monte Carlo on a random walk"
      state={state}
      caption="Five states in a row; each episode starts in the middle and steps left or right with probability 1/2 until it leaves either end. Leaving on the right pays 1, on the left 0, so the true values are 1/6, 2/6, …, 5/6. Both methods start from V = 0.5 and see the same episodes. The curves are the root-mean-squared error over the five states, averaged over 100 runs. A larger step size learns faster but settles at a higher error floor, because each update moves the estimate by a noisy amount. With equal step sizes above 0.01, TD(0) has the lower error after 100 episodes, and the best TD step size (near 0.05, error about 0.035) beats the best Monte Carlo one (near 0.02, about 0.075). The TD target depends on one random step; the Monte Carlo target depends on the whole random walk."

      readouts={
        <>
          <Readout label="TD(0) error after 100 episodes" value={formatNumber(td[EPISODES])} />
          <Readout label="Monte Carlo error after 100 episodes" value={formatNumber(mc[EPISODES])} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis}>
        <Curve {...series[0]} />
        <Curve {...series[1]} />
      </Plot>
    </Figure>
  )
}
