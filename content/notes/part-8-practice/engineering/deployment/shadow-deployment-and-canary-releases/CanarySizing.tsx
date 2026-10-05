import { useMemo } from 'react'
import { Curve, Figure, float, formatNumber, Handle, int, Plot, Readout, useAxis, useFigureState } from 'aifn-render'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'
import { normalQuantile } from 'aifn-compute/numerics/special'

const FRACTIONS = toFlat(linspace(0.005, 0.5, 100))
// Two-sided 5% test with 80% power.
const Z = normalQuantile(0.975) + normalQuantile(0.8)

/** Smallest absolute increase in error rate detectable when a fraction f of N requests goes to the canary. */
const mde = (p: number, n: number, f: number) => Z * Math.sqrt((p * (1 - p)) / (n * f * (1 - f)))

export function CanarySizing() {
  const state = useFigureState({
    base: float(1, { min: 0.1, max: 10, step: 0.1, label: 'baseline error rate (%)' }),
    thousands: int(1000, { min: 10, max: 5000, step: 10, label: 'requests in window (thousands)' }),
    frac: float(0.05, { min: 0.005, max: 0.5, step: 0.005, label: 'canary share f' }),
  })
  const p = state.base / 100
  const n = state.thousands * 1000

  const series = useMemo(
    () =>
      [
        {
          name: 'detectable increase (percentage points)',
          x: FRACTIONS,
          y: FRACTIONS.map((f) => 100 * mde(p, n, f)),
          slot: 0,
        },
      ] as const,
    [p, n],
  )
  const d = mde(p, n, state.frac)

  const xAxis = useAxis({ label: 'canary share f', range: [0, 0.5] })
  const yAxis = useAxis({ label: 'detectable increase (pp)', range: [0, undefined], hold: 'union' })
  return (
    <Figure
      title="How large a canary must be"
      purpose="Change the baseline error rate, the traffic and the canary share to see the smallest increase in error rate a canary detects."
      state={state}
      caption="The smallest increase in error rate that a two-sided 5% test detects with 80% power, when a fraction f of the requests in the evaluation window goes to the canary and the rest to the control. Precision improves quickly up to a few per cent of traffic and slowly after that, while the number of requests exposed to a bad release grows in proportion to f. Drag the vertical line to change f."

      readouts={
        <>
          <Readout label="detectable increase" value={`${formatNumber(100 * d)} pp`} />
          <Readout label="canary error rate detectable" value={`${formatNumber(100 * (p + d))}%`} />
          <Readout label="requests on the canary" value={formatNumber(state.frac * n)} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={280}>
        <Curve {...series[0]} />
        <Handle {...state.handle('frac', { label: 'canary share' })} />
      </Plot>
    </Figure>
  )
}
