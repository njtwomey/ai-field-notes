import { useMemo } from 'react'
import { choice, Curve, Figure, formatNumber, int, Plot, Readout, useAxis, useFigureState } from 'aifn-render'
import { stream, uniform as drawUniform } from 'aifn/foundation/random'

type Host = 'knows' | 'ignorant'

/**
 * Plays the game many times. The host opens every door except the player's and one other, so a switch always has one
 * door to go to. A host who knows where the car is never reveals it; an ignorant host opens doors at random, and games
 * in which he reveals the car are discarded.
 */
export function MontyHallSimulator() {
  const state = useFigureState({
    doors: int(3, { min: 3, max: 20, step: 1, suggestions: [3, 5, 10, 20], label: 'doors' }),
    host: choice<Host>(
      [
        { value: 'knows', label: 'knows' },
        { value: 'ignorant', label: 'ignorant' },
      ],
      'knows',
      { label: 'host' },
    ),
    games: int(1000, { min: 50, max: 5000, step: 50, suggestions: [100, 1000, 5000], label: 'games' }),
    seed: int(1, { ge: 0, label: 'seed' }),
  })
  const { doors, host, games, seed } = state

  const result = useMemo(() => {
    const draws = stream(seed)
    const uniform = () => drawUniform(draws)
    const pick = () => Math.floor(uniform() * doors)
    const x: number[] = []
    const stay: number[] = []
    const swap: number[] = []
    let played = 0
    let stayWins = 0
    let swapWins = 0
    let discarded = 0
    for (let g = 0; g < games; g++) {
      const car = pick()
      const first = pick()
      // The one other door left closed.
      let closed: number
      if (host === 'knows') {
        closed = car !== first ? car : (first + 1 + Math.floor(uniform() * (doors - 1))) % doors
      } else {
        closed = (first + 1 + Math.floor(uniform() * (doors - 1))) % doors
        if (car !== first && car !== closed) {
          discarded++
          continue
        }
      }
      played++
      if (car === first) stayWins++
      if (car === closed) swapWins++
      x.push(played)
      stay.push(stayWins / played)
      swap.push(swapWins / played)
    }
    const exactSwap = host === 'knows' ? (doors - 1) / doors : 0.5
    const exactStay = host === 'knows' ? 1 / doors : 0.5
    return { x, stay, swap, played, discarded, exactSwap, exactStay }
  }, [doors, host, games, seed])

  const series = useMemo(() => {
    const ends = [1, Math.max(result.played, 1)]
    return [
      { name: 'switch', x: result.x, y: result.swap, slot: 0 },
      { name: 'stay', x: result.x, y: result.stay, slot: 1 },
      { name: 'exact, switch', x: ends, y: [result.exactSwap, result.exactSwap], dashed: true, slot: 0 },
      { name: 'exact, stay', x: ends, y: [result.exactStay, result.exactStay], dashed: true, slot: 1 },
    ] as const
  }, [result])

  const xAxis = useAxis({ label: 'games played', hold: 'union' })
  const yAxis = useAxis({ label: 'share won', range: [0, 1] })
  return (
    <Figure
      title="Switch or stay, played many times"
      caption="Running share of games won by each strategy, against the number of games played. Dashed lines are the exact probabilities. With a host who knows, switching wins whenever the first pick was wrong. With an ignorant host, the games he spoils are discarded, and the remaining games favour neither strategy."
      state={state}
      readouts={
        <>
          <Readout label="switch wins, exact" value={formatNumber(result.exactSwap)} />
          <Readout label="switch wins, simulated" value={formatNumber(result.swap.at(-1) ?? 0)} />
          <Readout label="stay wins, simulated" value={formatNumber(result.stay.at(-1) ?? 0)} />
          <Readout label="games discarded" value={result.discarded} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={300}>
        <Curve {...series[0]} />
        <Curve {...series[1]} />
        <Curve {...series[2]} />
        <Curve {...series[3]} />
      </Plot>
    </Figure>
  )
}
