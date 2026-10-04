import { useMemo } from 'react'
import { formatNumber, Readout, slider, useFigureState } from 'aifn-render'
import { ConstrainedExplorer, tField, type ConstrainedProblem } from './ConstrainedExplorer'

/**
 * Minimise the variance wᵀΣw of a two-asset portfolio subject to w₁ + w₂ = 1, with Σ built from the two volatilities
 * and their correlation. Short positions (negative weights) are allowed.
 */
function problem(s1: number, s2: number, rho: number): ConstrainedProblem {
  const c = rho * s1 * s2
  return {
    f: (w1, w2) => s1 * s1 * w1 * w1 + 2 * c * w1 * w2 + s2 * s2 * w2 * w2,
    gradF: (w1, w2) => [2 * (s1 * s1 * w1 + c * w2), 2 * (c * w1 + s2 * s2 * w2)],
    gradG: () => [1, 1],
    curve: (w1) => [w1, 1 - w1],
    tRange: [-0.5, 1.5],
    xRange: [-0.5, 1.5],
    yRange: [-0.5, 1.5],
    goal: 'min',
  }
}

/** The closed form w* = Σ⁻¹1 / 1ᵀΣ⁻¹1 and its variance 1 / 1ᵀΣ⁻¹1. */
function solution(s1: number, s2: number, rho: number) {
  const c = rho * s1 * s2
  const denom = s1 * s1 + s2 * s2 - 2 * c
  const w1 = (s2 * s2 - c) / denom
  const variance = (s1 * s1 * s2 * s2 - c * c) / denom
  return { w1, w2: 1 - w1, variance }
}

export function MinimumVariancePortfolio() {
  const state = useFigureState({
    t: tField([-0.5, 1.5], 0.2, 0.005, 'weight w₁ in asset 1'),
    s1: slider(0.05, 0.4, 0.2, { step: 0.01, label: 'volatility σ₁' }),
    s2: slider(0.05, 0.4, 0.3, { step: 0.01, label: 'volatility σ₂' }),
    rho: slider(-0.95, 0.95, 0.2, { step: 0.05, label: 'correlation ρ' }),
  })
  const p = useMemo(() => problem(state.s1, state.s2, state.rho), [state.s1, state.s2, state.rho])
  const sol = solution(state.s1, state.s2, state.rho)
  return (
    <ConstrainedExplorer
      state={state}
      problem={p}
      title="The minimum-variance portfolio of two assets"
      caption="Left: contours of the portfolio variance wᵀΣw over the weights (grey), the budget line w₁ + w₂ = 1, and the arrow of ∇f = 2Σw against the line's normal (1, 1). Drag the point along the line; weights outside [0, 1] are short positions. Set the two volatilities and their correlation with the sliders. Right: the variance along the line against w₁."
      xLabel="w₁"
      yLabel="w₂"
      tLabel="weight w₁ in asset 1"
      tSymbol="w₁"
      fLabel="variance"
      readout={({ value }) => (
        <>
          <Readout label="portfolio volatility" value={formatNumber(Math.sqrt(value))} />
          <Readout label="w* = Σ⁻¹1 / 1ᵀΣ⁻¹1" value={`(${formatNumber(sol.w1)}, ${formatNumber(sol.w2)})`} />
          <Readout label="σ* = (1ᵀΣ⁻¹1)^(−1/2)" value={formatNumber(Math.sqrt(sol.variance))} />
          <Readout label="λ* = 2σ*²" value={formatNumber(2 * sol.variance)} />
        </>
      )}
    />
  )
}
