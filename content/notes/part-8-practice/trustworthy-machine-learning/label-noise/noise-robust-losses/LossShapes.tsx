import { useMemo } from 'react'
import { choice, Curve, Figure, float, formatNumber, Handle, Plot, Readout, useAxis, useFigureState } from 'aifn-render'
import { linspace, toFlat } from 'aifn/foundation/tensor'

const P = toFlat(linspace(0.01, 1, 200))
const VIEW = [
  { value: 'loss', label: 'loss' },
  { value: 'weight', label: 'gradient weight' },
] as const
type View = (typeof VIEW)[number]['value']

const gce = (p: number, q: number) => (1 - p ** q) / q

/**
 * Cross-entropy, MAE and generalised cross-entropy as functions of the probability p assigned to the given label, and
 * the weight each puts on −∇p relative to cross-entropy's 1/p.
 */
export function LossShapes() {
  const state = useFigureState({
    view: choice<View>(VIEW, 'loss', { label: 'show' }),
    q: float(0.7, { min: 0.05, max: 1, step: 0.05, label: 'GCE exponent q' }),
    p: float(0.05, { min: 0.01, max: 1, step: 0.01, label: 'p for readout' }),
  })

  const series = useMemo(() => {
    const f =
      state.view === 'loss'
        ? { ce: (x: number) => -Math.log(x), mae: (x: number) => 2 * (1 - x), gce: (x: number) => gce(x, state.q) }
        : { ce: (x: number) => 1 / x, mae: () => 2, gce: (x: number) => x ** (state.q - 1) }
    return [
      { name: 'cross-entropy', x: P, y: P.map(f.ce), slot: 0 },
      { name: 'MAE', x: P, y: P.map(f.mae), slot: 1 },
      { name: `GCE, q = ${formatNumber(state.q)}`, x: P, y: P.map(f.gce), slot: 2 },
    ] as const
  }, [state.view, state.q])

  const xAxis = useAxis({ label: 'probability of the observed label, p', range: [0, 1] })
  const yAxis = useAxis({
    label: state.view === 'loss' ? 'loss' : 'weight on −∇p',
    range: [0, state.view === 'loss' ? 5 : 25],
  })
  return (
    <Figure
      title="How much a doubtful label pulls"
      purpose="Compare how cross-entropy, mean absolute error and generalised cross-entropy weight an example whose observed label the model doubts."
      state={state}
      caption="The horizontal axis is the probability p the model gives to the observed label. A mislabelled example that the model fits well under its true class has small p. Cross-entropy's loss and gradient weight 1/p grow without bound there, so such examples dominate training. MAE weights every example equally and generalised cross-entropy (GCE) weights by p to the power q − 1, between the two. Drag the vertical line to read off values at one p."

      readouts={
        <>
          <Readout
            label="cross-entropy"
            value={formatNumber(state.view === 'loss' ? -Math.log(state.p) : 1 / state.p)}
          />
          <Readout label="MAE" value={formatNumber(state.view === 'loss' ? 2 * (1 - state.p) : 2)} />
          <Readout
            label="GCE"
            value={formatNumber(state.view === 'loss' ? gce(state.p, state.q) : state.p ** (state.q - 1))}
          />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} ariaLabel={'Loss or gradient weight against the probability of the observed label'}>
        <Curve {...series[0]} />
        <Curve {...series[1]} />
        <Curve {...series[2]} />
        <Handle {...state.handle('p', { label: 'p' })} />
      </Plot>
    </Figure>
  )
}
