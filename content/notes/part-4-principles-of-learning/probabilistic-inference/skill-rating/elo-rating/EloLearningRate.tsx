import { useMemo } from 'react'
import { Curve, Figure, formatNumber, Handle, int, Plot, Readout, useAxis, useFigureState } from 'aifn-render'
import { eloExpected } from '../_shared/skill'
import { stream, uniform } from 'aifn/foundation/random'

const N = 400
const JUMP_AT = 150
const BASE = 1500
const SCALE = 400 / Math.LN10
const GAMES = Array.from({ length: N + 1 }, (_, g) => g)

/**
 * One player meets opponents rated exactly at their true strength of 1500. The player's true rating is 1500 until game
 * 150 and `after` from then on. Every game is a Bernoulli draw from the logistic Elo model with the true ratings.
 */
function simulate(K: number, after: number, seed: number): number[] {
  const r = stream(seed)
  const out = [BASE]
  let R = BASE
  for (let g = 1; g <= N; g++) {
    const truth = g <= JUMP_AT ? BASE : after
    const s = uniform(r) < eloExpected(truth, BASE) ? 1 : 0
    R += K * (s - eloExpected(R, BASE))
    out.push(R)
  }
  return out
}

/** Standard deviation of the rating over a long run at a fixed true rating equal to the opponents'. */
function stationarySd(K: number, seed: number, games = 5000): number {
  const r = stream(seed + 1000)
  let R = BASE
  let sum = 0
  let sum2 = 0
  for (let g = 0; g < games; g++) {
    R += K * ((uniform(r) < 0.5 ? 1 : 0) - eloExpected(R, BASE))
    sum += R
    sum2 += R * R
  }
  const m = sum / games
  return Math.sqrt(sum2 / games - m * m)
}

/** Games for the noise-free (expected) rating path to cover 63% of the jump. */
function meanPathTime(K: number, after: number): number {
  const p = eloExpected(after, BASE)
  const target = BASE + 0.632 * (after - BASE)
  let R = BASE
  for (let g = 1; g <= 10000; g++) {
    R += K * (p - eloExpected(R, BASE))
    if (after >= BASE ? R >= target : R <= target) return g
  }
  return Infinity
}

export function EloLearningRate() {
  const state = useFigureState({
    K: int(32, { min: 2, max: 64, step: 1, label: 'K', format: (v) => String(v) }),
    after: int(1700, { min: 1300, max: 1900, step: 10, label: 'true rating after game 150', format: (v) => String(v) }),
    seed: int(1, { min: 1, max: 30, step: 1, label: 'seed', format: (v) => String(v) }),
  })

  const res = useMemo(() => {
    const path = simulate(state.K, state.after, state.seed)
    const slow = simulate(10, state.after, state.seed)
    const p = eloExpected(state.after, BASE)
    return {
      path,
      slow,
      reach: state.after === BASE ? 0 : meanPathTime(state.K, state.after),
      noise: stationarySd(state.K, state.seed),
      theoryNoise: Math.sqrt((state.K * SCALE) / 2),
      theoryTime: SCALE / (state.K * p * (1 - p)),
    }
  }, [state.K, state.after, state.seed])

  const series = useMemo(
    () =>
      [
        {
          name: 'true rating',
          x: [0, JUMP_AT, JUMP_AT, N],
          y: [BASE, BASE, state.after, state.after],
          emphasis: true,
          dashed: true,
        },
        { name: 'K = 10', x: GAMES, y: res.slow, muted: true },
        { name: `K = ${state.K}`, x: GAMES, y: res.path, slot: 0 },
      ] as const,
    [res, state.after, state.K],
  )

  const xAxis = useAxis({ label: 'game', range: [0, N] })
  const yAxis = useAxis({ label: 'rating', range: [1200, 2000] })
  return (
    <Figure
      title="The K-factor is a learning rate"
      state={state}
      caption="A player meets opponents rated at their true 1500. The player's true rating jumps at game 150; drag the dashed line after the jump to set the new level. A large K follows the jump quickly but jitters around the truth; a small K is smooth but slow. The rating noise grows like the square root of K, and the time to adapt shrinks like 1/K."

      readouts={
        <>
          <Readout label="rating sd over 5000 games at a fixed truth" value={formatNumber(res.noise)} />
          <Readout label="theory √(K s / 2)" value={formatNumber(res.theoryNoise)} />
          <Readout label="expected path: games to cover 63% of the jump" value={String(res.reach)} />
          <Readout label="theory s / (K p(1 − p))" value={formatNumber(res.theoryTime)} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={300}>
        <Curve {...series[0]} />
        <Curve {...series[1]} />
        <Curve {...series[2]} />
        <Handle {...state.handle('after', { axis: 'y' })} />
      </Plot>
    </Figure>
  )
}
