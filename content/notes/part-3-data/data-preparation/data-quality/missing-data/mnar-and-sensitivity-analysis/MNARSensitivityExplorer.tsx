import { useMemo } from 'react'
import { Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'

export function MNARSensitivityExplorer() {
  const state = useFigureState({
    delta: slider(-3.0, 1.0, -1.0, { step: 0.1, label: 'dropout offset δ' }),
    dropoutRate: slider(0.1, 0.5, 0.3, { step: 0.05, label: 'treatment dropout rate' }),
  })

  const results = useMemo(() => {
    const baseEffect = 1.6
    const seBase = 0.35
    const pMis = state.dropoutRate

    // Delta-adjusted effect: Delta(delta) = baseEffect + pMis * delta
    // Standard error slightly expands under MNAR uncertainty
    const seAdjusted = seBase * Math.sqrt(1 + 0.5 * pMis)

    // Tipping point: delta where lower bound of 95% CI crosses 0:
    // baseEffect + pMis * delta_tip - 1.96 * seAdjusted = 0
    // => delta_tip = (1.96 * seAdjusted - baseEffect) / pMis
    const deltaTipSig = (1.96 * seAdjusted - baseEffect) / pMis
    // Tipping point where point estimate itself reaches 0:
    const deltaTipPoint = -baseEffect / pMis

    // Generate curve across delta in [-3.0, 1.0]
    const deltaGrid: number[] = []
    const effectCurve: number[] = []
    const upperCiCurve: number[] = []
    const lowerCiCurve: number[] = []
    const zeroLine: number[] = []

    for (let d = -3.0; d <= 1.05; d += 0.05) {
      const eff = baseEffect + pMis * d
      deltaGrid.push(d)
      effectCurve.push(eff)
      upperCiCurve.push(eff + 1.96 * seAdjusted)
      lowerCiCurve.push(eff - 1.96 * seAdjusted)
      zeroLine.push(0)
    }

    const currentEffect = baseEffect + pMis * state.delta
    const currentLower = currentEffect - 1.96 * seAdjusted
    const currentUpper = currentEffect + 1.96 * seAdjusted

    const markerX = [state.delta]
    const markerY = [currentEffect]

    return {
      baseEffect,
      currentEffect,
      currentLower,
      currentUpper,
      deltaTipSig,
      deltaTipPoint,
      deltaGrid,
      effectCurve,
      upperCiCurve,
      lowerCiCurve,
      zeroLine,
      markerX,
      markerY,
    }
  }, [state.delta, state.dropoutRate])

  const xAxis = useAxis({ label: 'unobserved dropout shift δ (units)', range: [-3.0, 1.0] })
  const yAxis = useAxis({ label: 'adjusted treatment effect Δ(δ)', range: [-1.5, 3.0] })

  const isSignificantlyPositive = results.currentLower > 0

  return (
    <Figure
      title="Tipping-point sensitivity analysis under MNAR departures"
      state={state}
      hoverReadout={false}
      caption="Evaluating the robustness of a treatment effect under non-ignorable dropout. Under baseline MAR (δ = 0), the effect is positive and statistically significant (p < 0.05). As the hypothetical deficit δ among dropouts grows more severe, the adjusted effect shrinks linearly: Δ(δ) = Δ_MAR + p_mis · δ. The tipping point marks the exact degree of negative selection required before the confidence interval covers zero."
      readouts={
        <>
          <Readout label="baseline MAR effect" value={`+${results.baseEffect.toFixed(2)}`} />
          <Readout label="adjusted effect Δ(δ)" value={`${results.currentEffect.toFixed(2)}`} />
          <Readout
            label="95% CI bound"
            value={`[${results.currentLower.toFixed(2)}, ${results.currentUpper.toFixed(2)}]`}
          />
          <Readout
            label="statistical significance"
            value={isSignificantlyPositive ? 'p < 0.05 (significant)' : 'not significant (covers 0)'}
          />
          <Readout label="tipping point δ*" value={`${results.deltaTipSig.toFixed(2)} units`} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} title="Adjusted treatment effect vs dropout penalty">
        <Curve name="null effect line (Δ = 0)" x={results.deltaGrid} y={results.zeroLine} slot={2} dashed width={1.5} />
        <Curve
          name="upper 95% confidence bound"
          x={results.deltaGrid}
          y={results.upperCiCurve}
          slot={1}
          thin
          width={1}
        />
        <Curve
          name="lower 95% confidence bound"
          x={results.deltaGrid}
          y={results.lowerCiCurve}
          slot={1}
          thin
          width={1}
        />
        <Curve
          name="delta-adjusted point estimate Δ(δ)"
          x={results.deltaGrid}
          y={results.effectCurve}
          slot={0}
          width={2.5}
        />
        <Points name="current sensitivity setting" x={results.markerX} y={results.markerY} slot={0} size={7} />
      </Plot>
    </Figure>
  )
}
