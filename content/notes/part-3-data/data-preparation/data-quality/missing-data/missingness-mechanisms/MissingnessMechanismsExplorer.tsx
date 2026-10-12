import { useMemo } from 'react'
import { choice, Curve, Figure, int, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'
import { normal, stream, uniform } from 'aifn-compute/foundation/random'

const SAMPLE_SIZE = 160

export function MissingnessMechanismsExplorer() {
  const state = useFigureState({
    mechanism: choice<'mcar' | 'mar' | 'mnar'>(
      [
        { value: 'mcar', label: 'MCAR (independent coin flip)' },
        { value: 'mar', label: 'MAR (missingness depends on X)' },
        { value: 'mnar', label: 'MNAR (missingness depends on Y)' },
      ],
      'mcar',
      { label: 'mechanism' },
    ),
    missingRate: slider(0.1, 0.7, 0.4, { step: 0.05, label: 'missingness fraction' }),
    seed: int(3, { min: 1, max: 20, label: 'sample seed' }),
  })

  const data = useMemo(() => {
    const rng = stream(state.seed)
    const xs: number[] = []
    const ys: number[] = []

    // Generate bivariate normal data: X ~ N(0, 1), Y = 0.75 * X + eps, eps ~ N(0, 0.6^2)
    for (let i = 0; i < SAMPLE_SIZE; i++) {
      const x = normal(rng, 0, 1)
      const eps = normal(rng, 0, 0.6)
      const y = 0.75 * x + eps
      xs.push(x)
      ys.push(y)
    }

    // Determine missingness indicator for Y
    // R = 1 means observed, R = 0 means missing
    const isObserved: boolean[] = []

    if (state.mechanism === 'mcar') {
      for (let i = 0; i < SAMPLE_SIZE; i++) {
        isObserved.push(uniform(rng) >= state.missingRate)
      }
    } else if (state.mechanism === 'mar') {
      // Missingness in Y depends on X: sort or score by X
      // Sigmoid probability of missingness: P(missing) = sigmoid(k * (X - c))
      // Adjust shift so expected missing rate aligns with target
      const scores = xs.map((x) => x)
      const sorted = [...scores].sort((a, b) => a - b)
      const cutIdx = Math.floor((1 - state.missingRate) * SAMPLE_SIZE)
      const cutoff = sorted[Math.min(SAMPLE_SIZE - 1, Math.max(0, cutIdx))]
      for (let i = 0; i < SAMPLE_SIZE; i++) {
        // Points with X > cutoff have higher probability of being missing
        const probMis =
          xs[i] > cutoff ? Math.min(0.95, state.missingRate * 1.8) : Math.max(0.05, state.missingRate * 0.2)
        isObserved.push(uniform(rng) >= probMis)
      }
    } else {
      // MNAR: Missingness in Y depends on Y itself!
      // Points with high Y are hidden (e.g. high earners omitting salary)
      const sorted = [...ys].sort((a, b) => a - b)
      const cutIdx = Math.floor((1 - state.missingRate) * SAMPLE_SIZE)
      const cutoff = sorted[Math.min(SAMPLE_SIZE - 1, Math.max(0, cutIdx))]
      for (let i = 0; i < SAMPLE_SIZE; i++) {
        const probMis =
          ys[i] > cutoff ? Math.min(0.95, state.missingRate * 1.8) : Math.max(0.05, state.missingRate * 0.2)
        isObserved.push(uniform(rng) >= probMis)
      }
    }

    const obsX: number[] = []
    const obsY: number[] = []
    const misX: number[] = []
    const misY: number[] = []

    for (let i = 0; i < SAMPLE_SIZE; i++) {
      if (isObserved[i]) {
        obsX.push(xs[i])
        obsY.push(ys[i])
      } else {
        misX.push(xs[i])
        misY.push(ys[i])
      }
    }

    // Population true stats
    const trueMeanY = ys.reduce((a, b) => a + b, 0) / ys.length

    // Observed sample stats (complete case analysis)
    const nObs = obsY.length
    const obsMeanY = nObs > 0 ? obsY.reduce((a, b) => a + b, 0) / nObs : 0
    const obsMeanX = nObs > 0 ? obsX.reduce((a, b) => a + b, 0) / nObs : 0

    // OLS slope on observed cases: beta1 = sum((x - meanX) * (y - meanY)) / sum((x - meanX)^2)
    let sxy = 0
    let sxx = 0
    for (let i = 0; i < nObs; i++) {
      const dx = obsX[i] - obsMeanX
      const dy = obsY[i] - obsMeanY
      sxy += dx * dy
      sxx += dx * dx
    }
    const obsSlope = sxx > 1e-6 ? sxy / sxx : 0.75
    const obsIntercept = obsMeanY - obsSlope * obsMeanX

    // Regression curves from -3 to 3
    const curveX = [-3, -2, -1, 0, 1, 2, 3]
    const trueLineY = curveX.map((x) => 0.75 * x)
    const obsLineY = curveX.map((x) => obsIntercept + obsSlope * x)

    return {
      obsX,
      obsY,
      misX,
      misY,
      trueMeanY,
      obsMeanY,
      obsSlope,
      trueSlope: 0.75,
      actualMisRate: SAMPLE_SIZE > 0 ? misX.length / SAMPLE_SIZE : 0,
      curveX,
      trueLineY,
      obsLineY,
    }
  }, [state.mechanism, state.missingRate, state.seed])

  const xAxis = useAxis({ label: 'feature X (fully observed)', range: [-3, 3] })
  const yAxis = useAxis({ label: 'target Y (subject to missingness)', range: [-3, 3] })

  const meanBias = Math.abs(data.obsMeanY - data.trueMeanY)
  const slopeBias = Math.abs(data.obsSlope - data.trueSlope)

  return (
    <Figure
      title="Rubin missingness mechanisms and estimation bias"
      state={state}
      hoverReadout={false}
      caption="Simulated bivariate Gaussian sample with missingness introduced in Y. Under MCAR, missingness is independent of all variables: observed cases form an unbiased sample of the population. Under MAR, missingness depends only on observed X: the marginal mean of Y is shifted, but the conditional regression slope beta_1 remains asymptotically unbiased. Under MNAR, missingness depends directly on unobserved Y: truncation induces severe attenuation bias in both the marginal mean and the regression slope."
      readouts={
        <>
          <Readout label="missing fraction" value={`${(data.actualMisRate * 100).toFixed(0)}%`} />
          <Readout
            label="marginal mean Y (obs vs true)"
            value={`${data.obsMeanY.toFixed(2)} vs ${data.trueMeanY.toFixed(2)}`}
          />
          <Readout label="mean bias |Δμ|" value={meanBias.toFixed(2)} />
          <Readout label="OLS slope β₁ (obs vs true)" value={`${data.obsSlope.toFixed(2)} vs 0.75`} />
          <Readout label="slope bias |Δβ|" value={slopeBias.toFixed(2)} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} title="Observed vs unobserved observations">
        <Points name="observed points (complete cases)" x={data.obsX} y={data.obsY} slot={0} size={5} />
        <Points name="missing values in Y (unobserved)" x={data.misX} y={data.misY} slot={1} size={5} />
        <Curve name="true population line (β₁ = 0.75)" x={data.curveX} y={data.trueLineY} dashed slot={2} width={2} />
        <Curve name="complete-case OLS fit" x={data.curveX} y={data.obsLineY} slot={0} width={2.5} />
      </Plot>
    </Figure>
  )
}
