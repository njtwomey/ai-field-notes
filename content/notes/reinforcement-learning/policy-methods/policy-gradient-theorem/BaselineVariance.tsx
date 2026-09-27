import { useMemo } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, useParam, type XYSeries } from '@/components/viz'
import { linspace } from '@/lib/math'

const LOGITS = [0, 0.5, 1]
const BASE_MEANS = [1, 2, 3]
const NOISE_SD = 0.5
const B = linspace(-4, 10, 281)

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
  const baseline = useParam(0, { min: -4, max: 10, step: 0.05 })
  const offset = useParam(0, { min: 0, max: 5, step: 0.1 })
  const means = useMemo(() => BASE_MEANS.map((m) => m + offset.value), [offset.value])
  const value = PI.reduce((acc, p, a) => acc + p * means[a], 0)
  // Setting d var / db = 0 gives the variance-minimising constant baseline: an average of rewards weighted by ‖score‖².
  const best =
    PI.reduce((acc, p, a) => acc + p * SCORE_SQ[a] * means[a], 0) / PI.reduce((acc, p, a) => acc + p * SCORE_SQ[a], 0)

  const series: XYSeries[] = useMemo(
    () => [
      { name: 'variance of the estimate', type: 'line', x: B, y: B.map((b) => totalVariance(means, b)), slot: 0 },
      {
        name: 'b = V(π)',
        type: 'scatter',
        x: [value],
        y: [totalVariance(means, value)],
        slot: 1,
      },
      { name: 'variance-minimising b', type: 'scatter', x: [best], y: [totalVariance(means, best)], emphasis: true },
    ],
    [means, value, best],
  )

  return (
    <Interactive
      title="A baseline changes the variance, not the mean"
      caption="A softmax policy over three actions with probabilities 0.19, 0.31 and 0.51 (logits 0, 0.5, 1). Action a pays its mean reward plus Gaussian noise of standard deviation 0.5. The curve is the exact variance of the one-sample REINFORCE estimate (R − b) ∇θ log π(a), summed over the three parameters, as a function of the baseline b; its mean is the same for every b. Drag the vertical line to move the baseline. Adding a constant to every reward leaves the true gradient unchanged but, with b = 0, inflates the variance; a baseline near V(π) removes the offset. The diamond marks the variance-minimising constant, a weighted average of the rewards close to V(π)."
      controls={
        <>
          <ParamSlider label="baseline b" param={baseline} />
          <ParamSlider label="constant added to every reward" param={offset} />
        </>
      }
      readout={
        <>
          <Readout label="variance at b" value={formatNumber(totalVariance(means, baseline.value))} />
          <Readout label="variance at b = 0" value={formatNumber(totalVariance(means, 0))} />
          <Readout label="V(π)" value={formatNumber(value)} />
          <Readout label="best constant b" value={formatNumber(best)} />
        </>
      }
    >
      <XYChart
        series={series}
        xLabel="baseline b"
        yLabel="variance (sum over θ)"
        xRange={[-4, 10]}
        yRange={[0, undefined]}
        handles={[{ kind: 'x', at: baseline.value, label: 'b', onDrag: (x) => baseline.set(x) }]}
      />
    </Interactive>
  )
}
