import { useMemo } from 'react'
import {
  Annotation,
  Curve,
  Figure,
  formatNumber,
  Plot,
  Points,
  Readout,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { tensor, toFlat } from 'aifn-compute/foundation/tensor'
import { forrester } from 'aifn-methods/data/objectives'

const XS = Array.from({ length: 401 }, (_, i) => i / 400)

/**
 * The Forrester function and its cheap approximation a f(x) + b (x − 0.5) − c: correlated, but with the minimum in
 * the wrong place, so optimising the cheap version alone finds the wrong configuration.
 */
export function ForresterFidelities() {
  const state = useFigureState({
    a: slider(0.1, 1.5, 0.5, { step: 0.05, label: 'scale a' }),
    b: slider(-20, 20, 10, { step: 0.5, label: 'tilt b' }),
    c: slider(-10, 10, -5, { step: 0.5, label: 'offset c' }),
  })
  const { a, b, c } = state

  const result = useMemo(() => {
    const high = forrester()
    const low = forrester({ fidelity: 'low', a, b, c })
    const xHigh = toFlat(high.minima[0])[0]
    const xLow = toFlat(low.minima[0])[0]
    return {
      high: XS.map((x) => high.value(tensor([x]))),
      low: XS.map((x) => low.value(tensor([x]))),
      xHigh,
      xLow,
      fHigh: high.minimumValue,
      fLow: low.minimumValue,
      highAtLow: high.value(tensor([xLow])),
    }
  }, [a, b, c])

  const xAxis = useAxis({ label: 'hyperparameter x', range: [0, 1] })
  const yAxis = useAxis({ label: 'loss', range: [-10, 20] })
  return (
    <Figure
      title="A cheap approximation with its minimum in the wrong place"
      state={state}
      caption="The Forrester function f(x) = (6x − 2)² sin(12x − 4) stands for the full-fidelity validation loss (ink). Its low-fidelity version a·f(x) + b·(x − 0.5) − c (orange) is what a cheap evaluation returns: correlated with f, but scaled, tilted and shifted. With the default a = 0.5, b = 10, c = −5 its minimum sits at x = 0.092, far from the full-fidelity minimum at x = 0.757, and evaluating the cheap minimiser at full fidelity gives a loss of −0.52 against the optimum −6.02. Set b = 0 and the two minima coincide: a scale and an offset do not move the minimiser, a tilt does. A multi-fidelity model learns the relation between the fidelities from a few expensive evaluations instead of trusting the cheap one."
      readouts={
        <>
          <Readout
            label="full-fidelity minimum"
            value={`${formatNumber(result.fHigh)} at x = ${formatNumber(result.xHigh)}`}
          />
          <Readout
            label="low-fidelity minimum"
            value={`${formatNumber(result.fLow)} at x = ${formatNumber(result.xLow)}`}
          />
          <Readout label="full-fidelity loss at the cheap minimiser" value={formatNumber(result.highAtLow)} />
          <Readout label="regret of trusting the cheap version" value={formatNumber(result.highAtLow - result.fHigh)} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis}>
        <Curve name="full fidelity f" x={XS} y={result.high} emphasis />
        <Curve name="low fidelity" x={XS} y={result.low} slot={1} />
        <Points name="full-fidelity minimum" x={[result.xHigh]} y={[result.fHigh]} emphasis size={8} />
        <Points name="low-fidelity minimum" x={[result.xLow]} y={[result.fLow]} slot={1} size={8} />
        <Points name="f at the cheap minimiser" x={[result.xLow]} y={[result.highAtLow]} slot={1} size={8} />
        <Annotation x={result.xLow} text="cheap minimiser" dashed muted />
      </Plot>
    </Figure>
  )
}
