import { useMemo } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, useParam, type XYSeries } from '@/components/viz'
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
  const count = useParam(3, { min: 1, max: 4, step: 1 })
  const sigma = useParam(100, { min: 10, max: 400, step: 10 })
  const seed = useParam(1, { min: 1, max: 20, step: 1 })

  const r = useMemo(() => {
    const apps = APPLIANCES.slice(0, count.value)
    const sim = simulate(apps, N, sigma.value, seed.value)
    const exact = exactPosterior(apps, sim.x, sigma.value)
    const mf = meanFieldPosterior(apps, sim.x, sigma.value)
    const accuracy = (q: number[][]) =>
      q.reduce((s, row, m) => s + row.filter((p, n) => (p > 0.5 ? 1 : 0) === sim.states[m][n]).length, 0) /
      (N * apps.length)
    return { apps, sim, exact, mf, exactAccuracy: accuracy(exact), mfAccuracy: accuracy(mf) }
  }, [count.value, sigma.value, seed.value])

  const t = Array.from({ length: N }, (_, n) => n + 1)
  const meter: XYSeries[] = [
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

  return (
    <Interactive
      title="One meter, several appliances"
      caption="The meter reads only the total power. Each appliance is a two-state hidden chain, and the factorial HMM infers which appliances are on. Exact inference runs forward–backward over all 2^M combinations; structured mean field runs one forward–backward per appliance against the others' expected power. With more appliances and more noise, combinations with similar totals (a kettle against a heater plus a microwave) become hard to tell apart, and mean field can lock onto the wrong one."
      controls={
        <>
          <ParamSlider label="appliances M" param={count} format={(v) => String(v)} withArrows />
          <ParamSlider label="meter noise σ (W)" param={sigma} format={(v) => `${v}`} />
          <ParamSlider label="seed" param={seed} format={(v) => String(v)} withArrows />
        </>
      }
      readout={
        <>
          <Readout label="joint states Kᴹ" value={K ** M} />
          <Readout label="exact cost per step K²ᴹ" value={K ** (2 * M)} />
          <Readout label="mean-field cost per step and sweep M·K²" value={M * K * K} />
          <Readout label="exact accuracy" value={`${formatNumber(100 * r.exactAccuracy)}%`} />
          <Readout label="mean-field accuracy" value={`${formatNumber(100 * r.mfAccuracy)}%`} />
        </>
      }
    >
      <XYChart series={meter} xLabel="time step" yLabel="power (W)" height={220} />
      <div className="grid gap-3 md:grid-cols-2">
        {r.apps.map((a, m) => (
          <div key={a.name} className="min-w-0 space-y-1">
            <div className="text-center text-xs text-muted-foreground">
              {a.name}, {a.power} W
            </div>
            <XYChart
              series={[
                { name: 'on (truth)', type: 'line', ...steps(r.sim.states[m]), slot: 2, area: true },
                { name: 'exact p(on | x)', type: 'line', x: t, y: r.exact[m], slot: 0 },
                { name: 'mean-field q(on)', type: 'line', x: t, y: r.mf[m], slot: 1, dashed: true },
              ]}
              xLabel="time step"
              yRange={[0, 1]}
              height={170}
            />
          </div>
        ))}
      </div>
    </Interactive>
  )
}
