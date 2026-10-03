import { useMemo, useState } from 'react'
import { Interactive, ParamChoice, ParamSlider, Readout, XYChart, formatNumber, type XYSeries } from 'aifn-render'
import { rng } from '@/lib/math'

type Host = 'knows' | 'ignorant'

/**
 * Plays the game many times. The host opens every door except the player's and one other, so a switch always has one
 * door to go to. A host who knows where the car is never reveals it; an ignorant host opens doors at random, and games
 * in which he reveals the car are discarded.
 */
export function MontyHallSimulator() {
  const [doors, setDoors] = useState(3)
  const [host, setHost] = useState<Host>('knows')
  const [games, setGames] = useState(1000)
  const [seed, setSeed] = useState(1)

  const result = useMemo(() => {
    const { uniform } = rng(seed)
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
      const choice = pick()
      // The one other door left closed.
      let closed: number
      if (host === 'knows') {
        closed = car !== choice ? car : (choice + 1 + Math.floor(uniform() * (doors - 1))) % doors
      } else {
        closed = (choice + 1 + Math.floor(uniform() * (doors - 1))) % doors
        if (car !== choice && car !== closed) {
          discarded++
          continue
        }
      }
      played++
      if (car === choice) stayWins++
      if (car === closed) swapWins++
      x.push(played)
      stay.push(stayWins / played)
      swap.push(swapWins / played)
    }
    const exactSwap = host === 'knows' ? (doors - 1) / doors : 0.5
    const exactStay = host === 'knows' ? 1 / doors : 0.5
    return { x, stay, swap, played, discarded, exactSwap, exactStay }
  }, [doors, host, games, seed])

  const series = useMemo((): XYSeries[] => {
    const ends = [1, Math.max(result.played, 1)]
    return [
      { name: 'switch', type: 'line', x: result.x, y: result.swap, slot: 0 },
      { name: 'stay', type: 'line', x: result.x, y: result.stay, slot: 1 },
      { name: 'exact, switch', type: 'line', x: ends, y: [result.exactSwap, result.exactSwap], dashed: true, slot: 0 },
      { name: 'exact, stay', type: 'line', x: ends, y: [result.exactStay, result.exactStay], dashed: true, slot: 1 },
    ]
  }, [result])

  return (
    <Interactive
      title="Switch or stay, played many times"
      caption="Running share of games won by each strategy, against the number of games played. Dashed lines are the exact probabilities. With a host who knows, switching wins whenever the first pick was wrong. With an ignorant host, the games he spoils are discarded, and the remaining games favour neither strategy."
      controls={
        <>
          <ParamSlider label="doors" value={doors} onChange={setDoors} min={3} max={20} step={1} />
          <ParamChoice
            label="host"
            value={host}
            onChange={setHost}
            options={[
              { value: 'knows', label: 'knows' },
              { value: 'ignorant', label: 'ignorant' },
            ]}
          />
          <ParamSlider label="games" value={games} onChange={setGames} min={50} max={5000} step={50} />
          <ParamSlider label="seed" value={seed} onChange={setSeed} min={0} max={30} step={1} />
        </>
      }
      readout={
        <>
          <Readout label="switch wins, exact" value={formatNumber(result.exactSwap)} />
          <Readout label="switch wins, simulated" value={formatNumber(result.swap.at(-1) ?? 0)} />
          <Readout label="stay wins, simulated" value={formatNumber(result.stay.at(-1) ?? 0)} />
          <Readout label="games discarded" value={result.discarded} />
        </>
      }
    >
      <XYChart height={300} xLabel="games played" yLabel="share won" series={series} yRange={[0, 1]} />
    </Interactive>
  )
}
