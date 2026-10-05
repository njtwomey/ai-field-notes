import { Curve, Figure, formatNumber, Handle, Plot, Readout, slider, useAxis, useFigureState } from 'aifn-render'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'

const P = toFlat(linspace(0.005, 1, 200))

/**
 * The penalty each score gives one case, as a function of the probability q assigned to the class that occurred: the
 * log score −log q is unbounded as q → 0; the (binary) Brier score (1 − q)² is at most 1.
 */
export function PenaltyCurves() {
  const state = useFigureState({
    q: slider(0.005, 1, 0.1, { step: 0.005, onChart: true }),
  })
  const series = [
    { name: 'log loss −ln q', x: P, y: P.map((p) => -Math.log(p)), slot: 0 },
    { name: 'Brier (1 − q)²', x: P, y: P.map((p) => (1 - p) ** 2), slot: 1 },
  ] as const
  const xAxis = useAxis({ label: 'probability q given to the true outcome', range: [0, 1] })
  const yAxis = useAxis({ label: 'penalty', range: [0, 5] })
  return (
    <Figure
      title="How much one prediction costs"
      state={state}
      caption="Each curve is the penalty for one case as a function of q, the probability the model gave to the outcome that happened. Drag q towards 0: a confident wrong prediction costs a bounded 1 under the Brier score, and an unbounded amount under log loss."
      readouts={
        <>
          <Readout label="q" value={formatNumber(state.q)} />
          <Readout label="log loss −ln q" value={formatNumber(-Math.log(state.q))} />
          <Readout label="Brier (1 − q)²" value={formatNumber((1 - state.q) ** 2)} />
        </>
      }
    >
      <div className="mx-auto w-full max-w-xl">
        <Plot x={xAxis} y={yAxis} height={300}>
          <Curve {...series[0]} />
          <Curve {...series[1]} />
          <Handle {...state.handle('q', { label: 'q' })} />
        </Plot>
      </div>
    </Figure>
  )
}
