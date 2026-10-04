import { useMemo } from 'react'
import {
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Plot,
  Readout,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { tTestPower } from 'aifn/probability/tests'
import { linspace, toFlat } from 'aifn/foundation/tensor'

/**
 * Power against a mean difference δ (in units of the outcome's standard deviation σ) for two designs with the same
 * number of measurements per condition: n pairs measured twice, with within-pair correlation ρ, or two independent
 * groups of n. The paired difference has standard deviation σ√(2(1 − ρ)); the independent difference σ√(2/n).
 */
export function PairingPower() {
  const state = useFigureState({
    rho: slider(-0.5, 0.95, 0.6, { step: 0.01, label: 'within-pair correlation ρ' }),
    n: int(15, { min: 3, max: 100, step: 1, label: 'pairs n' }),
    effect: float(0.5, { min: 0.05, max: 1.5, step: 0.05, label: 'true difference δ/σ' }),
    alpha: float(0.05, { min: 0.01, max: 0.1, step: 0.005, label: 'significance level α' }),
  })

  const result = useMemo(() => {
    const paired = (r: number) =>
      tTestPower((state.effect * Math.sqrt(state.n)) / Math.sqrt(2 * (1 - r)), state.n - 1, state.alpha)
    const independent = tTestPower(state.effect / Math.sqrt(2 / state.n), 2 * state.n - 2, state.alpha)
    const rs = toFlat(linspace(-0.5, 0.95, 146))
    const series = [
      { name: 'paired design', x: rs, y: rs.map(paired), slot: 0 },
      {
        name: 'two independent groups',
        x: [-0.5, 0.95],
        y: [independent, independent],
        slot: 1,
        dashed: true,
      },
    ] as const
    return { series, paired, independent }
  }, [state.n, state.effect, state.alpha])

  const xAxis = useAxis({ label: 'within-pair correlation ρ', range: [-0.5, 0.95] })
  const yAxis = useAxis({ label: 'power', range: [0, 1] })
  return (
    <Figure
      title="Pairing pays when measurements correlate"
      state={state}
      caption="Power of a two-sided t-test to detect a mean difference, for n pairs measured under both conditions against two independent groups of n. The paired design removes the variation shared within a pair, so its power grows with the correlation ρ between the two measurements. Below ρ = 0 pairing hurts. Drag the line labelled ρ, or use its slider."

      readouts={
        <>
          <Readout label="power, paired" value={formatNumber(result.paired(state.rho))} />
          <Readout label="power, independent" value={formatNumber(result.independent)} />
          <Readout label="variance ratio 1 − ρ" value={formatNumber(1 - state.rho)} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={280}>
        <Curve {...result.series[0]} />
        <Curve {...result.series[1]} />
        <Handle {...state.handle('rho', { label: 'ρ' })} />
      </Plot>
    </Figure>
  )
}
