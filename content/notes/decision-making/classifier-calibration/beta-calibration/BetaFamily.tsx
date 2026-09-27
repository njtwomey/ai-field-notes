import { useMemo } from 'react'
import { Interactive, ParamSlider, Readout, XYChart, formatNumber, useParam, type Handle } from '@/components/viz'
import { linspace } from '@/lib/math'

const S = linspace(0.001, 0.999, 250)

/**
 * The beta calibration family μ(s) = 1 / (1 + 1 / (e^c s^a / (1 − s)^b)), parametrised by a, b and the midpoint m
 * where μ(m) = 1/2, so that c = b ln(1 − m) − a ln m. a = b = 1, m = 1/2 is the identity.
 */
export function BetaFamily() {
  const a = useParam(1, { min: 0, max: 5, step: 0.05 })
  const b = useParam(1, { min: 0, max: 5, step: 0.05 })
  const m = useParam(0.5, { min: 0.02, max: 0.98, step: 0.005 })

  const c = b.value * Math.log(1 - m.value) - a.value * Math.log(m.value)
  const curve = useMemo(
    () => S.map((s) => 1 / (1 + 1 / (Math.exp(c) * s ** a.value * (1 - s) ** -b.value))),
    [a.value, b.value, c],
  )
  const shape =
    Math.abs(a.value - 1) < 1e-9 && Math.abs(b.value - 1) < 1e-9 && Math.abs(m.value - 0.5) < 1e-9
      ? 'identity'
      : a.value === b.value
        ? a.value > 1
          ? 'sigmoid'
          : a.value < 1
            ? 'inverse sigmoid'
            : 'shifted identity'
        : 'asymmetric'
  // The midpoint is a point on the curve with an obvious place on the chart: drag it along the line μ = 1/2.
  const handles: Handle[] = [{ kind: 'x', at: m.value, label: 'midpoint m', onDrag: (x) => m.set(x) }]

  return (
    <Interactive
      title="The beta calibration family"
      caption="Each curve maps a classifier's score s to a calibrated probability. The parameters a and b weight ln s and −ln(1 − s); the midpoint m is where the curve crosses 1/2 (drag the vertical line). a = b > 1 gives sigmoids, a = b < 1 inverse sigmoids, a ≠ b skewed curves. At a = b = 1 and m = 1/2 the map is the identity, which the logistic family cannot produce."
      controls={
        <>
          <ParamSlider label="a" param={a} />
          <ParamSlider label="b" param={b} />
          <ParamSlider label="midpoint m" param={m} />
        </>
      }
      readout={
        <>
          <Readout label="c" value={formatNumber(c)} />
          <Readout label="shape" value={shape} />
        </>
      }
    >
      <XYChart
        equalAspect
        xLabel="score s"
        yLabel="calibrated probability μ(s)"
        xRange={[0, 1]}
        yRange={[0, 1]}
        handles={handles}
        series={[
          { name: 'identity', type: 'line', x: [0, 1], y: [0, 1], dashed: true, muted: true },
          { name: 'μ = 1/2', type: 'line', x: [0, 1], y: [0.5, 0.5], dashed: true, muted: true },
          { name: 'beta calibration map', type: 'line', x: S, y: curve, slot: 0 },
        ]}
      />
    </Interactive>
  )
}
