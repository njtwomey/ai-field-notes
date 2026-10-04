import { useMemo } from 'react'
import { Curve, Figure, float, formatNumber, Handle, Plot, Readout, useAxis, useFigureState } from 'aifn-render'
import { linspace, toFlat } from 'aifn/foundation/tensor'

const S = toFlat(linspace(0.001, 0.999, 250))

/**
 * The beta calibration family μ(s) = 1 / (1 + 1 / (e^c s^a / (1 − s)^b)), parametrised by a, b and the midpoint m
 * where μ(m) = 1/2, so that c = b ln(1 − m) − a ln m. a = b = 1, m = 1/2 is the identity.
 */
export function BetaFamily() {
  const state = useFigureState({
    a: float(1, { min: 0, max: 5, step: 0.05, label: 'a' }),
    b: float(1, { min: 0, max: 5, step: 0.05, label: 'b' }),
    m: float(0.5, { min: 0.02, max: 0.98, step: 0.005, label: 'midpoint m' }),
  })

  const c = state.b * Math.log(1 - state.m) - state.a * Math.log(state.m)
  const curve = useMemo(
    () => S.map((s) => 1 / (1 + 1 / (Math.exp(c) * s ** state.a * (1 - s) ** -state.b))),
    [state.a, state.b, c],
  )
  const shape =
    Math.abs(state.a - 1) < 1e-9 && Math.abs(state.b - 1) < 1e-9 && Math.abs(state.m - 0.5) < 1e-9
      ? 'identity'
      : state.a === state.b
        ? state.a > 1
          ? 'sigmoid'
          : state.a < 1
            ? 'inverse sigmoid'
            : 'shifted identity'
        : 'asymmetric'
  // The midpoint is a point on the curve with an obvious place on the chart: drag it along the line μ = 1/2.

  const xAxis = useAxis({ label: 'score s', range: [0, 1] })
  const yAxis = useAxis({ label: 'calibrated probability μ(s)', range: [0, 1], equal: xAxis })
  return (
    <Figure
      title="The beta calibration family"
      state={state}
      caption="Each curve maps a classifier's score s to a calibrated probability. The parameters a and b weight ln s and −ln(1 − s); the midpoint m is where the curve crosses 1/2 (drag the vertical line). a = b > 1 gives sigmoids, a = b < 1 inverse sigmoids, a ≠ b skewed curves. At a = b = 1 and m = 1/2 the map is the identity, which the logistic family cannot produce."

      readouts={
        <>
          <Readout label="c" value={formatNumber(c)} />
          <Readout label="shape" value={shape} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis}>
        <Curve name="identity" x={[0, 1]} y={[0, 1]} dashed muted />
        <Curve name="μ = 1/2" x={[0, 1]} y={[0.5, 0.5]} dashed muted />
        <Curve name="beta calibration map" x={S} y={curve} slot={0} />
        <Handle {...state.handle('m', { label: 'midpoint m' })} />
      </Plot>
    </Figure>
  )
}
