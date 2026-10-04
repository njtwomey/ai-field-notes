import { useMemo } from 'react'
import { Curve, Figure, float, formatNumber, Handle, int, Plot, Readout, useAxis, useFigureState } from 'aifn-render'
import { linspace, toFlat } from 'aifn/foundation/tensor'
import { normalCdf, normalQuantile } from 'aifn/numerics/special'

/** Power of the one-sided (H₁: μ > μ₀) and two-sided z-tests as functions of the true standardised effect δ. */
export function SidedPower() {
  const state = useFigureState({
    effect: float(0.3, { min: -1, max: 1, step: 0.01, label: 'true effect δ (standard deviations)' }),
    n: int(25, { min: 5, max: 200, step: 1, label: 'observations n' }),
    alpha: float(0.05, { min: 0.01, max: 0.2, step: 0.01, label: 'significance level α' }),
  })

  const result = useMemo(() => {
    const one = normalQuantile(1 - state.alpha)
    const two = normalQuantile(1 - state.alpha / 2)
    const oneSided = (d: number) => 1 - normalCdf(one - d * Math.sqrt(state.n))
    const twoSided = (d: number) => {
      const s = d * Math.sqrt(state.n)
      return 1 - normalCdf(two - s) + normalCdf(-two - s)
    }
    const ds = toFlat(linspace(-1, 1, 201))
    const series = [
      { name: 'one-sided, H₁: μ > μ₀', x: ds, y: ds.map(oneSided), slot: 0 },
      { name: 'two-sided, H₁: μ ≠ μ₀', x: ds, y: ds.map(twoSided), slot: 1 },
      { name: 'α', x: [-1, 1], y: [state.alpha, state.alpha], slot: 2, dashed: true },
    ] as const
    return { series, one, two, oneSided, twoSided }
  }, [state.n, state.alpha])

  const xAxis = useAxis({ label: 'true effect δ', range: [-1, 1] })
  const yAxis = useAxis({ label: 'P(reject H₀)', range: [0, 1] })
  return (
    <Figure
      title="Where each test spends its α"
      state={state}
      caption="Power of a z-test against a true standardised effect δ. The one-sided test is more powerful for δ > 0 and has almost no power for δ < 0. The two-sided test is symmetric. Both reject a true null (δ = 0) with probability α. Drag the line labelled δ, or use its slider, to read both powers."

      readouts={
        <>
          <Readout label="critical z, one-sided" value={formatNumber(result.one)} />
          <Readout label="critical |z|, two-sided" value={formatNumber(result.two)} />
          <Readout label="power, one-sided" value={formatNumber(result.oneSided(state.effect))} />
          <Readout label="power, two-sided" value={formatNumber(result.twoSided(state.effect))} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={300}>
        <Curve {...result.series[0]} />
        <Curve {...result.series[1]} />
        <Curve {...result.series[2]} />
        <Handle {...state.handle('effect', { label: 'δ' })} />
      </Plot>
    </Figure>
  )
}
