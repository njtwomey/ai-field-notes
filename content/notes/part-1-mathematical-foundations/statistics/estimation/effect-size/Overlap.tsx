import { useMemo } from 'react'
import { Area, Curve, Figure, float, formatNumber, Plot, Readout, useAxis, useFigureState } from 'aifn-render'
import { linspace, toFlat } from 'aifn/foundation/tensor'
import { normalCdf, normalPdf } from 'aifn/numerics/special'

/** Two unit-variance normal groups whose means differ by d, with three readings of the same d. */
export function Overlap() {
  const state = useFigureState({
    d: float(0.5, { min: 0, max: 3, step: 0.05, label: "Cohen's d" }),
  })

  const series = useMemo(() => {
    const xs = toFlat(linspace(-4, 4 + state.d, 300))
    const shared = xs.map((x) => Math.min(normalPdf(x), normalPdf(x - state.d)))
    return [
      { name: 'control', x: xs, y: xs.map((x) => normalPdf(x)), slot: 0 },
      { name: 'treatment', x: xs, y: xs.map((x) => normalPdf(x - state.d)), slot: 1 },
      { name: 'overlap', x: xs, y: shared, slot: 2 },
    ] as const
  }, [state.d])

  const xAxis = useAxis({ label: 'outcome (control standard deviations)', hold: 'union' })
  const yAxis = useAxis({ label: 'density', range: [0, undefined], hold: 'union' })
  return (
    <Figure
      title="One effect size, three readings"
      state={state}
      caption="Two groups with equal standard deviations whose means differ by d standard deviations. The shaded area is the overlap of the two distributions. Cohen's U₃ is the share of the treatment group above the control mean. The probability of superiority is the chance that a random treated unit exceeds a random control unit."

      readouts={
        <>
          <Readout label="overlap 2Φ(−d/2)" value={`${formatNumber(100 * 2 * normalCdf(-state.d / 2))}%`} />
          <Readout label="U₃ = Φ(d)" value={`${formatNumber(100 * normalCdf(state.d))}%`} />
          <Readout label="P(treated > control) = Φ(d/√2)" value={formatNumber(normalCdf(state.d / Math.SQRT2))} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={280}>
        <Curve {...series[0]} />
        <Curve {...series[1]} />
        <Area {...series[2]} />
      </Plot>
    </Figure>
  )
}
