import { useMemo } from 'react'
import {
  Figure,
  formatNumber,
  Handle,
  int,
  Plot,
  Readout,
  seriesLayers,
  type SeriesSpec,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { stream, uniform as drawUniform } from 'aifn/foundation/random'

/** P(at least two of n share a value) when each is uniform on d values, exactly. */
function exactMatch(n: number, d: number): number {
  let none = 1
  for (let k = 0; k < n; k++) none *= (d - k) / d
  return 1 - none
}

const approxMatch = (n: number, d: number) => 1 - Math.exp((-n * (n - 1)) / (2 * d))

/** Exact and approximate collision probability against group size, checked by simulation at the chosen size. */
export function BirthdayCurve() {
  const state = useFigureState({
    n: int(23, { min: 2, max: 200, step: 1, label: 'people n' }),
    days: int(365, { min: 20, max: 2000, step: 5, label: 'days d' }),
    trials: int(2000, { min: 200, max: 20000, step: 200, label: 'simulated groups' }),
    seed: int(1, { min: 0, max: 30, step: 1, label: 'seed' }),
  })
  const nMax = Math.min(state.days + 1, Math.ceil(4 * Math.sqrt(state.days)))
  const people = Math.min(state.n, nMax)

  const curves = useMemo((): SeriesSpec[] => {
    const ns = Array.from({ length: nMax }, (_, i) => i + 1)
    return [
      { name: 'exact', type: 'line', x: ns, y: ns.map((k) => exactMatch(k, state.days)), slot: 0 },
      {
        name: '1 − exp(−n(n − 1)/2d)',
        type: 'line',
        x: ns,
        y: ns.map((k) => approxMatch(k, state.days)),
        dashed: true,
        slot: 1,
      },
    ]
  }, [state.days, nMax])

  const simulated = useMemo(() => {
    const draws = stream(state.seed)
    const uniform = () => drawUniform(draws)
    let hits = 0
    const seen = new Uint32Array(state.days)
    for (let t = 1; t <= state.trials; t++) {
      for (let k = 0; k < people; k++) {
        const day = Math.floor(uniform() * state.days)
        // Stamping with the trial number avoids clearing the array between trials.
        if (seen[day] === t) {
          hits++
          break
        }
        seen[day] = t
      }
    }
    return hits / state.trials
  }, [state.days, people, state.trials, state.seed])

  const series = useMemo(
    (): SeriesSpec[] => [
      ...curves,
      { name: 'simulated', type: 'scatter', x: [people], y: [simulated], emphasis: true },
    ],
    [curves, people, simulated],
  )

  const xAxis = useAxis({ label: 'people n', range: [1, nMax] })
  const yAxis = useAxis({ label: 'P(some pair shares)', range: [0, 1] })
  return (
    <Figure
      title="How many people until a shared birthday?"
      state={state}
      caption="Probability that at least two of n people share a birthday, when each birthday is uniform on d days. Drag the line labelled n, or use its slider, to pick a group size; the marker is the share of simulated groups with a match. The curve passes one half near n = 1.18√d."

      readouts={
        <>
          <Readout label="exact" value={formatNumber(exactMatch(people, state.days))} />
          <Readout label="approximation" value={formatNumber(approxMatch(people, state.days))} />
          <Readout label="simulated" value={formatNumber(simulated)} />
          <Readout label="1.18√d" value={formatNumber(1.1774 * Math.sqrt(state.days))} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={300}>
        {seriesLayers(series)}
        <Handle kind="x" at={people} label="n" onDrag={(x) => state.set('n', Math.round(x))} />
      </Plot>
    </Figure>
  )
}
