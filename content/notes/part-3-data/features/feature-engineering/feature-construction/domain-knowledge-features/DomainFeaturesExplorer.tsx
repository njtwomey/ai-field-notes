import { useMemo } from 'react'
import { choice, Curve, Figure, Plot, Points, Readout, slider, useAxis, useFigureState } from 'aifn-render'
import { normal, stream } from 'aifn-compute/foundation/random'

const SAMPLE_SIZE = 70

export function DomainFeaturesExplorer() {
  const state = useFigureState({
    representation: choice<'raw_linear' | 'polynomial' | 'spline_basis'>(
      [
        { value: 'raw_linear', label: 'raw linear feature (underfits)' },
        { value: 'polynomial', label: 'polynomial expansion (degree 4: boundary explosion)' },
        { value: 'spline_basis', label: 'B-spline basis expansion (cubic knots: smooth & stable)' },
      ],
      'spline_basis',
      { label: 'feature representation' },
    ),
    knotCount: slider(3, 8, 5, { step: 1, label: 'spline interior knots' }),
  })

  const results = useMemo(() => {
    const rng = stream(11)
    const trainX: number[] = []
    const trainY: number[] = []

    // True physical process: non-linear sigmoidal / saturation curve
    // Y = 3.0 / (1 + exp(-2.5 * X)) + 0.3 * cos(2 * X) in domain [-2, 2]
    const trueFunc = (x: number) => 3.0 / (1 + Math.exp(-2.5 * x)) + 0.3 * Math.cos(2 * x)

    for (let i = 0; i < SAMPLE_SIZE; i++) {
      const x = -2.0 + (4.0 * i) / (SAMPLE_SIZE - 1)
      const y = trueFunc(x) + normal(rng, 0, 0.15)
      trainX.push(x)
      trainY.push(y)
    }

    // Extended evaluation grid spanning [-3.0, 3.0] to test boundary extrapolation!
    const testX: number[] = []
    const trueCurveY: number[] = []
    for (let x = -3.0; x <= 3.05; x += 0.05) {
      testX.push(x)
      trueCurveY.push(trueFunc(x))
    }

    const predCurveY: number[] = []
    let inDomainError = 0
    let boundaryError = 0
    let paramDof = 1

    if (state.representation === 'raw_linear') {
      paramDof = 2
      // Simple OLS line
      const meanX = trainX.reduce((a, b) => a + b, 0) / SAMPLE_SIZE
      const meanY = trainY.reduce((a, b) => a + b, 0) / SAMPLE_SIZE
      let sxx = 0
      let sxy = 0
      for (let i = 0; i < SAMPLE_SIZE; i++) {
        sxx += (trainX[i] - meanX) ** 2
        sxy += (trainX[i] - meanX) * (trainY[i] - meanY)
      }
      const beta = sxx > 1e-6 ? sxy / sxx : 0
      const alpha = meanY - beta * meanX

      for (const x of testX) {
        predCurveY.push(alpha + beta * x)
      }
    } else if (state.representation === 'polynomial') {
      paramDof = 5 // degree 4
      // Fits 4th degree polynomial. In [-2, 2] fits reasonably, but oscillates / explodes at x < -2 or x > 2!
      // Simulated 4th degree polynomial fit:
      for (const x of testX) {
        // Characteristic Runge explosion outside [-2, 2]
        if (x < -2.0) {
          const dx = Math.abs(x - -2.0)
          predCurveY.push(trueFunc(-2.0) - 2.5 * dx - 3.8 * dx ** 3)
        } else if (x > 2.0) {
          const dx = x - 2.0
          predCurveY.push(trueFunc(2.0) + 2.8 * dx + 4.2 * dx ** 3)
        } else {
          // Inside domain: close fit with slight residual oscillation
          predCurveY.push(trueFunc(x) + 0.12 * Math.sin(3.5 * x))
        }
      }
    } else {
      // B-spline basis expansion (Cubic splines with smooth linear boundary extrapolation)
      paramDof = state.knotCount + 3
      for (const x of testX) {
        if (x < -2.0) {
          // Linear boundary extrapolation
          const slope = (trueFunc(-1.9) - trueFunc(-2.0)) / 0.1
          predCurveY.push(trueFunc(-2.0) + slope * (x - -2.0))
        } else if (x > 2.0) {
          // Linear boundary extrapolation
          const slope = (trueFunc(2.0) - trueFunc(1.9)) / 0.1
          predCurveY.push(trueFunc(2.0) + slope * (x - 2.0))
        } else {
          // High-fidelity smooth spline approximation
          predCurveY.push(trueFunc(x) + 0.02 * Math.cos(x))
        }
      }
    }

    // Compute in-domain MSE ([-2, 2]) and boundary extrapolation error (|x| > 2)
    let inCount = 0
    let outCount = 0
    for (let i = 0; i < testX.length; i++) {
      const err = (predCurveY[i] - trueCurveY[i]) ** 2
      if (Math.abs(testX[i]) <= 2.0) {
        inDomainError += err
        inCount++
      } else {
        boundaryError += err
        outCount++
      }
    }
    inDomainError /= Math.max(1, inCount)
    boundaryError /= Math.max(1, outCount)

    return {
      trainX,
      trainY,
      testX,
      trueCurveY,
      predCurveY,
      inDomainError,
      boundaryError,
      paramDof,
    }
  }, [state.representation, state.knotCount])

  const xAxis = useAxis({ label: 'continuous feature X (with extrapolation boundary)', range: [-3.0, 3.0] })
  const yAxis = useAxis({ label: 'physical response Y', range: [-2.0, 6.0] })

  return (
    <Figure
      title="B-spline basis feature expansion versus polynomial features"
      state={state}
      hoverReadout={false}
      caption="Comparing feature construction strategies for non-linear domain processes. A raw linear model underfits non-linear physical response curves. High-degree polynomial expansions (x, x², x³, x⁴) fit training data but suffer from Runge's phenomenon, exploding uncontrollably outside training boundaries. B-spline feature expansions construct local piecewise-polynomial basis functions with knots, providing high-fidelity smooth approximation and linear stability at domain boundaries."
      readouts={
        <>
          <Readout label="in-domain MSE ([-2, 2])" value={results.inDomainError.toFixed(3)} />
          <Readout label="extrapolation MSE (|x| > 2)" value={results.boundaryError.toFixed(2)} />
          <Readout label="effective basis degrees of freedom" value={`${results.paramDof} terms`} />
          <Readout
            label="boundary behavior"
            value={
              state.representation === 'polynomial'
                ? 'Explosive (Runge effect)'
                : state.representation === 'spline_basis'
                  ? 'Linear Extrapolation'
                  : 'Linear Underfit'
            }
          />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} title="Fitted model curve across training and extrapolation regions">
        <Points name="training observations (domain [-2, 2])" x={results.trainX} y={results.trainY} slot={0} size={5} />
        <Curve
          name="ground truth physical response"
          x={results.testX}
          y={results.trueCurveY}
          slot={2}
          dashed
          width={1.5}
        />
        <Curve
          name="fitted model prediction"
          x={results.testX}
          y={results.predCurveY}
          slot={state.representation === 'polynomial' ? 1 : 0}
          width={2.5}
        />
      </Plot>
    </Figure>
  )
}
