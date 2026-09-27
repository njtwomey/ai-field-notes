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
import { linspace } from '@/lib/math'
import { normalQuantile } from '@/lib/math/special'

const FRACTIONS = linspace(0.005, 0.5, 100)
// Two-sided 5% test with 80% power.
const Z = normalQuantile(0.975) + normalQuantile(0.8)

/** Smallest absolute increase in error rate detectable when a fraction f of N requests goes to the canary. */
const mde = (p: number, n: number, f: number) => Z * Math.sqrt((p * (1 - p)) / (n * f * (1 - f)))

export function CanarySizing() {
  const [base, setBase] = useState(1)
  const [thousands, setThousands] = useState(1000)
  const frac = useParam(0.05, { min: 0.005, max: 0.5, step: 0.005 })
  const p = base / 100
  const n = thousands * 1000

  const series = useMemo(
    (): XYSeries[] => [
      {
        name: 'detectable increase (percentage points)',
        type: 'line',
        x: FRACTIONS,
        y: FRACTIONS.map((f) => 100 * mde(p, n, f)),
        slot: 0,
      },
    ],
    [p, n],
  )
  const d = mde(p, n, frac.value)
  const handles: Handle[] = [{ kind: 'x', at: frac.value, label: 'canary share', onDrag: frac.set }]

  return (
    <Interactive
      title="How large a canary must be"
      caption="The smallest increase in error rate that a two-sided 5% test detects with 80% power, when a fraction f of the requests in the evaluation window goes to the canary and the rest to the control. Precision improves quickly up to a few per cent of traffic and slowly after that, while the number of requests exposed to a bad release grows in proportion to f. Drag the vertical line to change f."
      controls={
        <>
          <ParamSlider label="baseline error rate (%)" value={base} onChange={setBase} min={0.1} max={10} step={0.1} />
          <ParamSlider
            label="requests in window (thousands)"
            value={thousands}
            onChange={setThousands}
            min={10}
            max={5000}
            step={10}
          />
          <ParamSlider label="canary share f" param={frac} />
        </>
      }
      readout={
        <>
          <Readout label="detectable increase" value={`${formatNumber(100 * d)} pp`} />
          <Readout label="canary error rate detectable" value={`${formatNumber(100 * (p + d))}%`} />
          <Readout label="requests on the canary" value={formatNumber(frac.value * n)} />
        </>
      }
    >
      <XYChart
        series={series}
        xLabel="canary share f"
        yLabel="detectable increase (pp)"
        xRange={[0, 0.5]}
        yRange={[0, undefined]}
        handles={handles}
        height={280}
      />
    </Interactive>
  )
}
