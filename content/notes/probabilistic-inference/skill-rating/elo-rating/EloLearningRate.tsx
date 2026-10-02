import { useMemo } from 'react'
import {
  Interactive,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type XYSeries,
} from 'aifn-render'
import { rng } from '@/lib/math'
import { eloExpected } from '../_shared/skill'

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
  const r = rng(seed)
  const out = [BASE]
  let R = BASE
  for (let g = 1; g <= N; g++) {
    const truth = g <= JUMP_AT ? BASE : after
    const s = r.uniform() < eloExpected(truth, BASE) ? 1 : 0
    R += K * (s - eloExpected(R, BASE))
    out.push(R)
  }
  return out
}

/** Standard deviation of the rating over a long run at a fixed true rating equal to the opponents'. */
function stationarySd(K: number, seed: number, games = 5000): number {
  const r = rng(seed + 1000)
  let R = BASE
  let sum = 0
  let sum2 = 0
  for (let g = 0; g < games; g++) {
    R += K * ((r.uniform() < 0.5 ? 1 : 0) - eloExpected(R, BASE))
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
  const K = useParam(32, { min: 2, max: 64, step: 1 })
  const after = useParam(1700, { min: 1300, max: 1900, step: 10 })
  const seed = useParam(1, { min: 1, max: 30, step: 1 })

  const res = useMemo(() => {
    const path = simulate(K.value, after.value, seed.value)
    const slow = simulate(10, after.value, seed.value)
    const p = eloExpected(after.value, BASE)
    return {
      path,
      slow,
      reach: after.value === BASE ? 0 : meanPathTime(K.value, after.value),
      noise: stationarySd(K.value, seed.value),
      theoryNoise: Math.sqrt((K.value * SCALE) / 2),
      theoryTime: SCALE / (K.value * p * (1 - p)),
    }
  }, [K.value, after.value, seed.value])

  const series = useMemo<XYSeries[]>(
    () => [
      {
        name: 'true rating',
        type: 'line',
        x: [0, JUMP_AT, JUMP_AT, N],
        y: [BASE, BASE, after.value, after.value],
        emphasis: true,
        dashed: true,
      },
      { name: 'K = 10', type: 'line', x: GAMES, y: res.slow, muted: true },
      { name: `K = ${K.value}`, type: 'line', x: GAMES, y: res.path, slot: 0 },
    ],
    [res, after.value, K.value],
  )
  const handles: Handle[] = [{ kind: 'y', at: after.value, onDrag: after.set }]

  return (
    <Interactive
      title="The K-factor is a learning rate"
      caption="A player meets opponents rated at their true 1500. The player's true rating jumps at game 150; drag the dashed line after the jump to set the new level. A large K follows the jump quickly but jitters around the truth; a small K is smooth but slow. The rating noise grows like the square root of K, and the time to adapt shrinks like 1/K."
      controls={
        <>
          <ParamSlider label="K" param={K} format={(v) => String(v)} />
          <ParamSlider label="true rating after game 150" param={after} format={(v) => String(v)} />
          <ParamSlider label="seed" param={seed} format={(v) => String(v)} withArrows />
        </>
      }
      readout={
        <>
          <Readout label="rating sd over 5000 games at a fixed truth" value={formatNumber(res.noise)} />
          <Readout label="theory √(K s / 2)" value={formatNumber(res.theoryNoise)} />
          <Readout label="expected path: games to cover 63% of the jump" value={String(res.reach)} />
          <Readout label="theory s / (K p(1 − p))" value={formatNumber(res.theoryTime)} />
        </>
      }
    >
      <XYChart
        series={series}
        xLabel="game"
        yLabel="rating"
        xRange={[0, N]}
        yRange={[1200, 2000]}
        height={300}
        handles={handles}
      />
    </Interactive>
  )
}
