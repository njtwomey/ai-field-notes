import { useMemo } from 'react'
import { Curve, Figure, float, formatNumber, Handle, Plot, Points, Readout, useAxis, useFigureState } from 'aifn-render'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'

const LOGITS = [0, 0.5, 1]
const BASE_MEANS = [1, 2, 3]
const NOISE_SD = 0.5
const B = toFlat(linspace(-4, 10, 281))

const PI = (() => {
  const e = LOGITS.map(Math.exp)
  const z = e.reduce((a, b) => a + b, 0)
  return e.map((v) => v / z)
})()
/** ‖∇θ log π(a)‖² = ‖e_a − π‖² for a softmax policy. */
const SCORE_SQ = PI.map((p) => 1 - 2 * p + PI.reduce((acc, q) => acc + q * q, 0))

/**
 * Exact variance of the one-sample REINFORCE estimate (R − b) ∇θ log π(a) for a three-action softmax policy, summed
 * over the three components of θ, as a function of the baseline b.
 */
function totalVariance(means: number[], b: number): number {
  const second = PI.reduce((acc, p, a) => acc + p * ((means[a] - b) ** 2 + NOISE_SD ** 2) * SCORE_SQ[a], 0)
  // The mean gradient Σ π_a μ_a (e_a − π) does not depend on b.
  const grad = PI.map((pj, j) => PI.reduce((acc, p, a) => acc + p * means[a] * ((a === j ? 1 : 0) - pj), 0))
  return second - grad.reduce((acc, g) => acc + g * g, 0)
}

/** How a baseline changes the variance, but not the mean, of the REINFORCE gradient estimate. */
export function BaselineVariance() {
  const state = useFigureState({
    baseline: float(0, { min: -4, max: 10, step: 0.05, label: 'baseline b' }),
    offset: float(0, { min: 0, max: 5, step: 0.1, label: 'constant added to every reward' }),
  })
  const means = useMemo(() => BASE_MEANS.map((m) => m + state.offset), [state.offset])
  const value = PI.reduce((acc, p, a) => acc + p * means[a], 0)
  // Setting d var / db = 0 gives the variance-minimising constant baseline: an average of rewards weighted by ‖score‖².
  const best =
    PI.reduce((acc, p, a) => acc + p * SCORE_SQ[a] * means[a], 0) / PI.reduce((acc, p, a) => acc + p * SCORE_SQ[a], 0)

  const series = useMemo(
    () =>
      [
        { name: 'variance of the estimate', x: B, y: B.map((b) => totalVariance(means, b)), slot: 0 },
        {
          name: 'b = V(π)',
          x: [value],
          y: [totalVariance(means, value)],
          slot: 1,
        },
        { name: 'variance-minimising b', x: [best], y: [totalVariance(means, best)], emphasis: true },
      ] as const,
    [means, value, best],
  )

  const xAxis = useAxis({ label: 'baseline b', range: [-4, 10] })
  const yAxis = useAxis({ label: 'variance (sum over θ)', range: [0, undefined], hold: 'union' })
  return (
    <Figure
      title="A baseline changes the variance, not the mean"
      state={state}
      caption="A softmax policy over three actions with probabilities 0.19, 0.31 and 0.51 (logits 0, 0.5, 1). Action a pays its mean reward plus Gaussian noise of standard deviation 0.5. The curve is the exact variance of the one-sample REINFORCE estimate (R − b) ∇θ log π(a), summed over the three parameters, as a function of the baseline b; its mean is the same for every b. Drag the vertical line to move the baseline. Adding a constant to every reward leaves the true gradient unchanged but, with b = 0, inflates the variance; a baseline near V(π) removes the offset. The diamond marks the variance-minimising constant, a weighted average of the rewards close to V(π)."

      readouts={
        <>
          <Readout label="variance at b" value={formatNumber(totalVariance(means, state.baseline))} />
          <Readout label="variance at b = 0" value={formatNumber(totalVariance(means, 0))} />
          <Readout label="V(π)" value={formatNumber(value)} />
          <Readout label="best constant b" value={formatNumber(best)} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis}>
        <Curve {...series[0]} />
        <Points {...series[1]} />
        <Points {...series[2]} />
        <Handle {...state.handle('baseline', { label: 'b' })} />
      </Plot>
    </Figure>
  )
}
