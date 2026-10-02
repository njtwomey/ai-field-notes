import { useMemo, useState } from 'react'
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

/** P(at least two of n share a value) when each is uniform on d values, exactly. */
function exactMatch(n: number, d: number): number {
  let none = 1
  for (let k = 0; k < n; k++) none *= (d - k) / d
  return 1 - none
}

const approxMatch = (n: number, d: number) => 1 - Math.exp((-n * (n - 1)) / (2 * d))

/** Exact and approximate collision probability against group size, checked by simulation at the chosen size. */
export function BirthdayCurve() {
  const [days, setDays] = useState(365)
  const [trials, setTrials] = useState(2000)
  const [seed, setSeed] = useState(1)
  const nMax = Math.min(days + 1, Math.ceil(4 * Math.sqrt(days)))
  const n = useParam(23, { min: 2, max: 200, step: 1 })
  const people = Math.min(n.value, nMax)

  const curves = useMemo((): XYSeries[] => {
    const ns = Array.from({ length: nMax }, (_, i) => i + 1)
    return [
      { name: 'exact', type: 'line', x: ns, y: ns.map((k) => exactMatch(k, days)), slot: 0 },
      {
        name: '1 − exp(−n(n − 1)/2d)',
        type: 'line',
        x: ns,
        y: ns.map((k) => approxMatch(k, days)),
        dashed: true,
        slot: 1,
      },
    ]
  }, [days, nMax])

  const simulated = useMemo(() => {
    const { uniform } = rng(seed)
    let hits = 0
    const seen = new Uint32Array(days)
    for (let t = 1; t <= trials; t++) {
      for (let k = 0; k < people; k++) {
        const day = Math.floor(uniform() * days)
        // Stamping with the trial number avoids clearing the array between trials.
        if (seen[day] === t) {
          hits++
          break
        }
        seen[day] = t
      }
    }
    return hits / trials
  }, [days, people, trials, seed])

  const series = useMemo(
    (): XYSeries[] => [...curves, { name: 'simulated', type: 'scatter', x: [people], y: [simulated], emphasis: true }],
    [curves, people, simulated],
  )
  const handles: Handle[] = [{ kind: 'x', at: people, label: 'n', onDrag: (x) => n.set(Math.round(x)) }]

  return (
    <Interactive
      title="How many people until a shared birthday?"
      caption="Probability that at least two of n people share a birthday, when each birthday is uniform on d days. Drag the line labelled n, or use its slider, to pick a group size; the marker is the share of simulated groups with a match. The curve passes one half near n = 1.18√d."
      controls={
        <>
          <ParamSlider label="people n" param={n} />
          <ParamSlider label="days d" value={days} onChange={setDays} min={20} max={2000} step={5} />
          <ParamSlider label="simulated groups" value={trials} onChange={setTrials} min={200} max={20000} step={200} />
          <ParamSlider label="seed" value={seed} onChange={setSeed} min={0} max={30} step={1} />
        </>
      }
      readout={
        <>
          <Readout label="exact" value={formatNumber(exactMatch(people, days))} />
          <Readout label="approximation" value={formatNumber(approxMatch(people, days))} />
          <Readout label="simulated" value={formatNumber(simulated)} />
          <Readout label="1.18√d" value={formatNumber(1.1774 * Math.sqrt(days))} />
        </>
      }
    >
      <XYChart
        height={300}
        xLabel="people n"
        yLabel="P(some pair shares)"
        series={series}
        xRange={[1, nMax]}
        yRange={[0, 1]}
        handles={handles}
      />
    </Interactive>
  )
}
