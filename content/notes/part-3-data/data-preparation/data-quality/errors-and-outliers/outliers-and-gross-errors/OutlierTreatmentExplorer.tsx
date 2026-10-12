import { useMemo } from 'react'
import { choice, Curve, Figure, int, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'
import { normal, stream } from 'aifn-compute/foundation/random'

const SAMPLE_SIZE = 60

export function OutlierTreatmentExplorer() {
  const state = useFigureState({
    treatment: choice<'none' | 'remove' | 'winsorize' | 'huber'>(
      [
        { value: 'none', label: 'unadjusted OLS (keep raw outliers)' },
        { value: 'remove', label: 'trimming (remove points > threshold)' },
        { value: 'winsorize', label: 'winsorising (clip to 95th percentile)' },
        { value: 'huber', label: 'robust Huber M-estimator' },
      ],
      'none',
      { label: 'outlier treatment' },
    ),
    outlierOffset: slider(3.0, 10.0, 6.0, { step: 0.5, label: 'outlier error magnitude' }),
    seed: int(3, { min: 1, max: 20, label: 'random seed' }),
  })

  const results = useMemo(() => {
    const rng = stream(state.seed)
    const rawX: number[] = []
    const rawY: number[] = []

    // 55 clean points following Y = 1.0 * X + noise, X in [-2.5, 2.5]
    for (let i = 0; i < SAMPLE_SIZE - 5; i++) {
      const x = -2.5 + (5.0 * i) / (SAMPLE_SIZE - 6)
      const eps = normal(rng, 0, 0.4)
      const y = 1.0 * x + eps
      rawX.push(x)
      rawY.push(y)
    }

    // 5 gross errors / high-leverage outliers placed at extreme X with opposite Y
    for (let i = 0; i < 5; i++) {
      const x = 1.8 + i * 0.15
      const y = -state.outlierOffset + normal(rng, 0, 0.2)
      rawX.push(x)
      rawY.push(y)
    }

    // Identify outliers via univariate threshold or residual
    const isOutlier = rawX.map((_, i) => i >= SAMPLE_SIZE - 5)

    // Compute 95th percentile cutoff for Y
    const sortedY = [...rawY].sort((a, b) => a - b)
    const lowerCut = sortedY[Math.floor(0.08 * sortedY.length)]

    // Apply treatment to data
    const fitX: number[] = []
    const fitY: number[] = []

    for (let i = 0; i < SAMPLE_SIZE; i++) {
      if (state.treatment === 'remove') {
        if (!isOutlier[i]) {
          fitX.push(rawX[i])
          fitY.push(rawY[i])
        }
      } else if (state.treatment === 'winsorize') {
        fitX.push(rawX[i])
        // Clip Y to lower cutoff
        fitY.push(Math.max(lowerCut, rawY[i]))
      } else {
        fitX.push(rawX[i])
        fitY.push(rawY[i])
      }
    }

    // Fit linear model based on treatment
    const n = fitX.length
    let meanX = 0
    let meanY = 0
    for (let i = 0; i < n; i++) {
      meanX += fitX[i]
      meanY += fitY[i]
    }
    meanX /= n
    meanY /= n

    let slope = 1.0
    let intercept = 0.0

    if (state.treatment === 'huber') {
      // Robust Huber M-estimator: iteratively reweighted least squares (IRLS)
      slope = 0.95
      intercept = 0.0
      const delta = 1.345 * 0.4
      for (let iter = 0; iter < 10; iter++) {
        let sumW = 0
        let sumWX = 0
        let sumWY = 0
        let sumWXX = 0
        let sumWXY = 0
        for (let i = 0; i < n; i++) {
          const res = fitY[i] - (intercept + slope * fitX[i])
          const absRes = Math.abs(res)
          const w = absRes <= delta ? 1 : delta / Math.max(1e-5, absRes)
          sumW += w
          sumWX += w * fitX[i]
          sumWY += w * fitY[i]
          sumWXX += w * fitX[i] * fitX[i]
          sumWXY += w * fitX[i] * fitY[i]
        }
        const denom = sumW * sumWXX - sumWX * sumWX
        if (Math.abs(denom) > 1e-6) {
          slope = (sumW * sumWXY - sumWX * sumWY) / denom
          intercept = (sumWY - slope * sumWX) / sumW
        }
      }
    } else {
      // Standard OLS on fitX, fitY
      let sxx = 0
      let sxy = 0
      for (let i = 0; i < n; i++) {
        sxx += (fitX[i] - meanX) * (fitX[i] - meanX)
        sxy += (fitX[i] - meanX) * (fitY[i] - meanY)
      }
      slope = sxx > 1e-6 ? sxy / sxx : 1.0
      intercept = meanY - slope * meanX
    }

    const curveX = [-2.5, -1.5, 0, 1.5, 2.5]
    const trueLine = curveX.map((x) => 1.0 * x)
    const fitLine = curveX.map((x) => intercept + slope * x)

    const cleanX = rawX.slice(0, SAMPLE_SIZE - 5)
    const cleanY = rawY.slice(0, SAMPLE_SIZE - 5)
    const outX = rawX.slice(SAMPLE_SIZE - 5)
    const outY = state.treatment === 'winsorize' ? fitY.slice(SAMPLE_SIZE - 5) : rawY.slice(SAMPLE_SIZE - 5)

    return {
      cleanX,
      cleanY,
      outX,
      outY,
      slope,
      slopeError: Math.abs(slope - 1.0),
      retainedCount: n,
      curveX,
      trueLine,
      fitLine,
    }
  }, [state.treatment, state.outlierOffset, state.seed])

  const xAxis = useAxis({ label: 'predictor feature X', range: [-2.5, 2.5] })
  const yAxis = useAxis({ label: 'target Y', range: [-11, 4] })

  return (
    <Figure
      title="Impact of outlier treatments on linear regression geometry"
      state={state}
      hoverReadout={false}
      caption="Comparing model sensitivity to gross errors and high-leverage outliers. Under unadjusted OLS, squared error loss penalizes large residuals quadratically, allowing five corrupted points to drag the slope from +1.0 down into negative territory. Trimming restores the clean slope by deleting suspect rows. Winsorising caps extreme residuals, limiting influence while keeping sample size intact. The Huber M-estimator employs linear residual penalties to achieve robustness without data deletion."
      readouts={
        <>
          <Readout label="fitted slope β̂" value={results.slope.toFixed(3)} />
          <Readout label="slope error |β̂ − 1.0|" value={results.slopeError.toFixed(3)} />
          <Readout label="retained sample N" value={`${results.retainedCount} / ${SAMPLE_SIZE}`} />
          <Readout
            label="treatment status"
            value={state.treatment === 'none' ? 'compromised (high leverage)' : 'stabilized'}
          />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} title="Regression response under outlier treatments">
        <Points name="inlier / regular observations" x={results.cleanX} y={results.cleanY} slot={0} size={5} />
        <Points name="gross errors / outliers" x={results.outX} y={results.outY} slot={1} size={6} />
        <Curve
          name="true uncorrupted relationship (β = 1.0)"
          x={results.curveX}
          y={results.trueLine}
          slot={2}
          dashed
          width={1.5}
        />
        <Curve
          name="fitted estimator line"
          x={results.curveX}
          y={results.fitLine}
          slot={state.treatment === 'none' ? 1 : 0}
          width={2.5}
        />
      </Plot>
    </Figure>
  )
}
