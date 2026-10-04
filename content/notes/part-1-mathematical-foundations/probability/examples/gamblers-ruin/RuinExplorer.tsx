import { useMemo } from 'react'
import {
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Plot,
  Points,
  Readout,
  seriesLayers,
  type SeriesSpec,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { stream, uniform as drawUniform } from 'aifn/foundation/random'

/** P(reach N before 0 | start at i), winning each unit bet with probability p. */
function winProbability(i: number, N: number, p: number): number {
  if (Math.abs(p - 0.5) < 1e-12) return i / N
  const r = (1 - p) / p
  return (1 - r ** i) / (1 - r ** N)
}

/** Expected number of bets until the gambler reaches 0 or N. */
function expectedDuration(i: number, N: number, p: number): number {
  if (Math.abs(p - 0.5) < 1e-12) return i * (N - i)
  const q = 1 - p
  return i / (q - p) - (N / (q - p)) * winProbability(i, N, p)
}

const MAX_STEPS = 200_000
/** Points drawn per path; longer paths are sampled evenly for drawing only. */
const DRAWN_POINTS = 400

/** One path of the fortune from i until it hits 0 or N, as (bet, fortune) pairs decimated for drawing. */
function fortunePath(i: number, N: number, p: number, seed: number) {
  const draws = stream(seed)
  const uniform = () => drawUniform(draws)
  let x = i
  const ys = [x]
  while (x > 0 && x < N && ys.length <= MAX_STEPS) {
    x += uniform() < p ? 1 : -1
    ys.push(x)
  }
  const stride = Math.ceil(ys.length / DRAWN_POINTS)
  const xs: number[] = []
  const vs: number[] = []
  for (let t = 0; t < ys.length; t += stride) {
    xs.push(t)
    vs.push(ys[t])
  }
  // Always keep the absorbing step, so each drawn path ends on 0 or N.
  if ((ys.length - 1) % stride !== 0) {
    xs.push(ys.length - 1)
    vs.push(ys[ys.length - 1])
  }
  return { xs, ys: vs }
}

/** Sample paths of the gambler's fortune, and the exact win probability against the starting fortune. */
export function RuinExplorer() {
  const state = useFigureState({
    p: float(0.5, { min: 0.4, max: 0.6, step: 0.002, label: 'P(win a bet) p' }),
    start: int(10, { min: 1, max: 99, step: 1, label: 'starting fortune i' }),
    target: int(20, { min: 2, max: 100, step: 1, label: 'target N' }),
    shown: int(6, { min: 1, max: 30, step: 1, label: 'paths', format: (v) => String(v) }),
    runs: int(1000, { min: 100, max: 5000, step: 100, label: 'simulated runs' }),
    seed: int(1, { min: 0, max: 30, step: 1, label: 'seed' }),
  })
  const i = Math.min(state.start, state.target - 1)

  // The win rate and mean duration come from all the simulated runs; the drawn paths are separate draws, so changing
  // how many are drawn never reruns the simulation.
  const sim = useMemo(() => {
    const draws = stream(state.seed)
    const uniform = () => drawUniform(draws)
    let wins = 0
    let totalSteps = 0
    for (let r = 0; r < state.runs; r++) {
      let x = i
      let steps = 0
      while (x > 0 && x < state.target && steps < MAX_STEPS) {
        x += uniform() < state.p ? 1 : -1
        steps++
      }
      if (x >= state.target) wins++
      totalSteps += steps
    }
    return { win: wins / state.runs, duration: totalSteps / state.runs }
  }, [state.p, state.target, state.runs, state.seed, i])

  const paths = useMemo((): SeriesSpec[] => {
    const many = state.shown > 1
    // Path k has its own stream, so adding paths leaves the existing ones unchanged.
    return Array.from({ length: state.shown }, (_, k) => {
      const { xs, ys } = fortunePath(i, state.target, state.p, state.seed * 1000 + k)
      return { name: many ? 'sample paths' : 'sample path', type: 'line', x: xs, y: ys, slot: 0, thin: many }
    })
  }, [state.p, state.target, state.seed, i, state.shown])

  const curve = useMemo(() => {
    const xs = Array.from({ length: state.target + 1 }, (_, k) => k)
    return [
      { name: 'exact P(win)', x: xs, y: xs.map((k) => winProbability(k, state.target, state.p)), slot: 0 },
      { name: 'fair game', x: [0, state.target], y: [0, 1], dashed: true, slot: 1 },
      { name: 'simulated', x: [i], y: [sim.win], emphasis: true },
    ] as const
  }, [state.target, state.p, i, sim.win])

  const xAxis = useAxis({ label: 'bet', hold: 'union' })
  const yAxis = useAxis({ label: 'fortune', range: [0, state.target] })
  const xAxis2 = useAxis({ label: 'starting fortune i', range: [0, state.target] })
  const yAxis2 = useAxis({ label: 'P(reach N before 0)', range: [0, 1] })
  return (
    <Figure
      title="Ruin or riches"
      state={state}
      caption="Left: sample paths of the gambler's fortune, one bet per step, until it reaches 0 or the target, drawn as light lines; the paths slider sets how many are drawn, not how many are simulated. Right: the exact probability of reaching the target against the starting fortune, with the share of simulated runs that did. Drag the line labelled start to change the starting fortune. A small edge against the gambler, such as p = 18/38 ≈ 0.474 in American roulette, makes a large target almost unreachable."

      readouts={
        <>
          <Readout label="P(win), exact" value={formatNumber(winProbability(i, state.target, state.p))} />
          <Readout label="P(win), simulated" value={formatNumber(sim.win)} />
          <Readout label="expected bets, exact" value={formatNumber(expectedDuration(i, state.target, state.p))} />
          <Readout label="mean bets, simulated" value={formatNumber(sim.duration)} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <Plot x={xAxis} y={yAxis} height={300}>
          {seriesLayers(paths)}
        </Plot>
        <Plot x={xAxis2} y={yAxis2} height={300}>
          <Curve {...curve[0]} />
          <Curve {...curve[1]} />
          <Points {...curve[2]} />
          <Handle kind="x" at={i} label="start" onDrag={(x) => state.set('start', Math.round(x))} />
        </Plot>
      </div>
    </Figure>
  )
}
