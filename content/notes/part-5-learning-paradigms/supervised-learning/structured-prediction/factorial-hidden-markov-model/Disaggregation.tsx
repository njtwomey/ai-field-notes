import { useMemo } from 'react'
import {
  Figure,
  formatNumber,
  int,
  Plot,
  Readout,
  seriesLayers,
  useAxis,
  useFigureState,
  type SeriesSpec,
} from 'aifn-render'
import { APPLIANCES, exactPosterior, meanFieldPosterior, simulate } from './fhmm'

const N = 200

/** A 0/1 sequence as a step line, so each state covers the width of its time step. */
function steps(values: number[]): { x: number[]; y: number[] } {
  return { x: values.flatMap((_, t) => [t + 0.5, t + 1.5]), y: values.flatMap((v) => [v, v]) }
}

/**
 * Energy disaggregation with an additive factorial HMM: each appliance is a two-state chain, and the meter reads the
 * sum of their powers plus noise. Exact inference over all 2^M joint states is compared with structured mean field.
 */
export function Disaggregation() {
  const state = useFigureState({
    count: int(3, { min: 1, max: 4, step: 1, label: 'appliances M', format: (v) => String(v) }),
    sigma: int(100, { min: 10, max: 400, step: 10, label: 'meter noise σ (W)', format: (v) => `${v}` }),
    seed: int(1, { min: 1, max: 20, step: 1, label: 'seed', format: (v) => String(v) }),
  })

  const r = useMemo(() => {
    const apps = APPLIANCES.slice(0, state.count)
    const sim = simulate(apps, N, state.sigma, state.seed)
    const exact = exactPosterior(apps, sim.x, state.sigma)
    const mf = meanFieldPosterior(apps, sim.x, state.sigma)
    const accuracy = (q: number[][]) =>
      q.reduce((s, row, m) => s + row.filter((p, n) => (p > 0.5 ? 1 : 0) === sim.states[m][n]).length, 0) /
      (N * apps.length)
    return { apps, sim, exact, mf, exactAccuracy: accuracy(exact), mfAccuracy: accuracy(mf) }
  }, [state.count, state.sigma, state.seed])

  const t = Array.from({ length: N }, (_, n) => n + 1)
  const meter: SeriesSpec[] = [
    {
      name: 'true total power',
      type: 'line',
      ...steps(t.map((_, n) => r.apps.reduce((s, a, m) => s + a.power * r.sim.states[m][n], 0))),
      slot: 2,
      area: true,
    },
    { name: 'meter reading x', type: 'line', x: t, y: r.sim.x, slot: 0 },
  ]
  const M = r.apps.length
  const K = 2

  const xAxis = useAxis({ label: 'time step', hold: 'union' })
  const yAxis = useAxis({ label: 'power (W)', hold: 'union' })
  // Every appliance's panel shares the time axis of the meter and one probability axis.
  const pAxis = useAxis({ label: 'p(on)', range: [0, 1] })
  return (
    <Figure
      title="One meter, several appliances"
      state={state}
      caption="The meter reads only the total power. Each appliance is a two-state hidden chain, and the factorial HMM infers which appliances are on. Exact inference runs forward–backward over all 2^M combinations; structured mean field runs one forward–backward per appliance against the others' expected power. With more appliances and more noise, combinations with similar totals (a kettle against a heater plus a microwave) become hard to tell apart, and mean field can lock onto the wrong one."
      readouts={
        <>
          <Readout label="joint states Kᴹ" value={K ** M} />
          <Readout label="exact cost per step K²ᴹ" value={K ** (2 * M)} />
          <Readout label="mean-field cost per step and sweep M·K²" value={M * K * K} />
          <Readout label="exact accuracy" value={`${formatNumber(100 * r.exactAccuracy)}%`} />
          <Readout label="mean-field accuracy" value={`${formatNumber(100 * r.mfAccuracy)}%`} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={220}>
        {seriesLayers(meter)}
      </Plot>
      <div className="grid gap-3 md:grid-cols-2">
        {r.apps.map((a, m) => (
          <div key={a.name} className="min-w-0 space-y-1">
            <div className="text-center text-xs text-muted-foreground">
              {a.name}, {a.power} W
            </div>
            <Plot x={xAxis} y={pAxis} height={170}>
              {seriesLayers([
                { name: 'on (truth)', type: 'line', ...steps(r.sim.states[m]), slot: 2, area: true },
                { name: 'exact p(on | x)', type: 'line', x: t, y: r.exact[m], slot: 0 },
                { name: 'mean-field q(on)', type: 'line', x: t, y: r.mf[m], slot: 1, dashed: true },
              ])}
            </Plot>
          </div>
        ))}
      </div>
    </Figure>
  )
}
