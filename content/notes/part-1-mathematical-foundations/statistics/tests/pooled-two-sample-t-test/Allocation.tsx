import { useMemo } from 'react'
import { Curve, Figure, float, formatNumber, Handle, int, Plot, Readout, useAxis, useFigureState } from 'aifn-render'
import { tTestPower } from 'aifn-compute/probability/tests'

/** Power of the pooled two-sample t-test against a standardised difference d, as the total N is split n₁ + n₂. */
export function Allocation() {
  const state = useFigureState({
    total: int(40, { min: 10, max: 200, step: 2, label: 'total N', suggestions: [20, 40, 100, 200] }),
    n1: int(10, { min: 2, max: 198, step: 1, label: 'observations in group 1, n₁' }),
    d: float(0.8, { min: 0.1, max: 2, step: 0.05, label: 'true effect d' }),
    alpha: float(0.05, { min: 0.01, max: 0.1, step: 0.005, label: 'significance level α' }),
  })
  const total = state.total
  const split = Math.min(Math.max(state.n1, 2), total - 2)

  const result = useMemo(() => {
    const power = (a: number) => tTestPower(state.d / Math.sqrt(1 / a + 1 / (total - a)), total - 2, state.alpha)
    const xs = Array.from({ length: total - 3 }, (_, i) => i + 2)
    const series = [{ name: 'power', x: xs, y: xs.map(power), slot: 0 }] as const
    return { series, power }
  }, [total, state.d, state.alpha])

  const xAxis = useAxis({ label: 'observations in group 1', range: [0, total] })
  const yAxis = useAxis({ label: 'power', range: [0, 1] })
  return (
    <Figure
      title="Unequal groups waste observations"
      state={state}
      caption="Power of the pooled t-test for a fixed total number of observations, as a function of how many go to group 1. The standard error of the difference, σ√(1/n₁ + 1/n₂), is smallest for equal groups, so power peaks there. Mild imbalance costs little; strong imbalance costs a lot. Drag the line labelled n₁, or use its slider, to change the split."
      readouts={
        <>
          <Readout label="split" value={`${split} + ${total - split}`} />
          <Readout label="power" value={formatNumber(result.power(split))} />
          <Readout label="power, equal split" value={formatNumber(result.power(Math.floor(total / 2)))} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={280}>
        <Curve {...result.series[0]} />
        <Handle kind="x" at={split} label="n₁" onDrag={(x) => state.set('n1', Math.min(total - 2, Math.round(x)))} />
      </Plot>
    </Figure>
  )
}
