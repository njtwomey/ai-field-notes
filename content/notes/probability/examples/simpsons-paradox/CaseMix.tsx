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
} from '@/components/viz'
import { rng } from '@/lib/math'

/** Success rates by stone size, from Charig et al. (1986). A is open surgery, B is percutaneous nephrolithotomy. */
const RATES = {
  A: { small: 81 / 87, large: 192 / 263 },
  B: { small: 234 / 270, large: 55 / 80 },
}

const overall = (t: keyof typeof RATES, largeShare: number) =>
  (1 - largeShare) * RATES[t].small + largeShare * RATES[t].large

/**
 * Each treatment's overall success rate is a weighted average of its two stratum rates, so it lies on the segment
 * between them, at the treatment's share of large stones. Dragging a point changes that case mix.
 */
export function CaseMix() {
  const mixA = useParam(263 / 350, { min: 0, max: 1, step: 0.001 })
  const mixB = useParam(80 / 350, { min: 0, max: 1, step: 0.001 })
  const [patients, setPatients] = useState(350)
  const [seed, setSeed] = useState(1)

  const series = useMemo(
    (): XYSeries[] => [
      { name: 'A: open surgery', type: 'line', x: [0, 1], y: [RATES.A.small, RATES.A.large], slot: 0 },
      { name: 'B: nephrolithotomy', type: 'line', x: [0, 1], y: [RATES.B.small, RATES.B.large], slot: 1 },
    ],
    [],
  )

  const simulated = useMemo(() => {
    const { uniform } = rng(seed)
    const arm = (t: keyof typeof RATES, share: number) => {
      let successes = 0
      for (let k = 0; k < patients; k++) {
        const rate = uniform() < share ? RATES[t].large : RATES[t].small
        if (uniform() < rate) successes++
      }
      return successes / patients
    }
    return { A: arm('A', mixA.value), B: arm('B', mixB.value) }
  }, [mixA.value, mixB.value, patients, seed])

  const a = overall('A', mixA.value)
  const b = overall('B', mixB.value)
  const handles: Handle[] = [
    { kind: 'point', at: [mixA.value, a], label: 'A overall', onDrag: ([x]) => mixA.set(x) },
    { kind: 'point', at: [mixB.value, b], label: 'B overall', onDrag: ([x]) => mixB.set(x) },
  ]

  return (
    <Interactive
      title="Success rate against case mix"
      caption="Each line joins a treatment's success rate for small stones (left) and for large stones (right). A is better in both groups, so its line is higher everywhere. Each dot is a treatment's overall success rate, placed at its share of large stones. In the study, A treated mostly large stones, so its dot sits low on a higher line. Drag the dots to change the case mix: with equal mixes, A wins overall too."
      controls={
        <>
          <ParamSlider label="A: share of large stones" param={mixA} />
          <ParamSlider label="B: share of large stones" param={mixB} />
          <ParamSlider
            label="simulated patients per arm"
            value={patients}
            onChange={setPatients}
            min={50}
            max={5000}
            step={50}
          />
          <ParamSlider label="seed" value={seed} onChange={setSeed} min={0} max={30} step={1} />
        </>
      }
      readout={
        <>
          <Readout label="A overall, exact" value={formatNumber(a)} />
          <Readout label="B overall, exact" value={formatNumber(b)} />
          <Readout label="A overall, simulated" value={formatNumber(simulated.A)} />
          <Readout label="B overall, simulated" value={formatNumber(simulated.B)} />
          <Readout label="better overall" value={a > b ? 'A' : a < b ? 'B' : 'tie'} />
        </>
      }
    >
      <XYChart
        height={320}
        xLabel="share of patients with large stones"
        yLabel="success rate"
        series={series}
        xRange={[0, 1]}
        yRange={[0.6, 1]}
        handles={handles}
      />
    </Interactive>
  )
}
