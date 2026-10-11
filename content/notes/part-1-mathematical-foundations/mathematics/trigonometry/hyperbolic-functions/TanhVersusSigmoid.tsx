import { useMemo } from 'react'
import {
  Curve,
  Figure,
  formatNumber,
  Handle,
  int,
  Plot,
  Plots,
  Points,
  Readout,
  setting,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { sigmoid } from 'aifn-compute/numerics/special'
import { grid } from '../_shared/geometry'

const X = grid(-6, 6, 481)
const TANH = X.map(Math.tanh)
const SIG = X.map((x) => sigmoid(x))
const RESCALED = X.map((x) => 2 * sigmoid(2 * x) - 1)
const SOFTSIGN = X.map((x) => x / (1 + Math.abs(x)))
const dTanh = (x: number) => 1 - Math.tanh(x) ** 2
const dSig = (x: number) => sigmoid(x) * (1 - sigmoid(x))
const dSoft = (x: number) => 1 / (1 + Math.abs(x)) ** 2
const D_TANH = X.map(dTanh)
const D_SIG = X.map(dSig)
const D_SOFT = X.map(dSoft)
const SLOT = { tanh: 0, sigmoid: 1, softsign: 2 } as const

/**
 * tanh against the logistic sigmoid σ, with their derivatives below. tanh is σ stretched to (−1, 1) and squeezed
 * horizontally by 2; its slope at 0 is 1, four times σ's, and both slopes decay exponentially away from 0.
 */
export function TanhVersusSigmoid() {
  const state = useFigureState({
    x0: slider(-6, 6, 2, { step: 0.05, onChart: true }),
    depth: int(10, { min: 1, max: 50, step: 1, label: 'layers L' }),
    softsign: setting(false, { label: 'show softsign x / (1 + |x|)' }),
  })
  const { x0, depth, softsign } = state

  const at = useMemo(
    () => ({
      tanh: Math.tanh(x0),
      sig: sigmoid(x0),
      soft: x0 / (1 + Math.abs(x0)),
      dTanh: dTanh(x0),
      dSig: dSig(x0),
      dSoft: dSoft(x0),
    }),
    [x0],
  )

  const x = useAxis({ label: 'x', range: [-6, 6] })
  const fy = useAxis({ label: 'f(x)', range: [-1.1, 1.1] })
  const dy = useAxis({ label: 'f′(x)', range: [0, 1.05] })
  const sci = (v: number) => (v < 1e-3 ? v.toExponential(2) : formatNumber(v))
  return (
    <Figure
      title="tanh, the sigmoid and saturation"
      state={state}
      caption="Top: tanh x (blue), the logistic sigmoid σ(x) (orange), and 2σ(2x) − 1 (dashed), which lies exactly on tanh. Bottom: the derivatives. tanh′(0) = 1 and σ′(0) = 1/4, and both fall off exponentially: at |x| = 3, tanh′ is below 0.01. Drag the vertical line to a pre-activation x. Backpropagation through L units all at that input multiplies the gradient by f′(x) once per unit; the readouts give that factor. Away from 0 it vanishes quickly for both functions, faster for the sigmoid. Softsign (green) saturates polynomially, so its derivative decays more slowly."
      readouts={{
        [`at x = ${formatNumber(x0)}`]: (
          <>
            <Readout label="tanh x" value={formatNumber(at.tanh)} />
            <Readout label="σ(x)" value={formatNumber(at.sig)} />
            <Readout label="tanh′(x) = 1 − tanh²x" value={formatNumber(at.dTanh)} />
            <Readout label="σ′(x) = σ(1 − σ)" value={formatNumber(at.dSig)} />
            {softsign && <Readout label="softsign′(x)" value={formatNumber(at.dSoft)} />}
          </>
        ),
        [`gradient factor over ${depth} layers`]: (
          <>
            <Readout label="tanh′(x)ᴸ" value={sci(at.dTanh ** depth)} />
            <Readout label="σ′(x)ᴸ" value={sci(at.dSig ** depth)} />
            {softsign && <Readout label="softsign′(x)ᴸ" value={sci(at.dSoft ** depth)} />}
          </>
        ),
      }}
    >
      <Plots rows={2} heights={[1, 1]} hoverGroup>
        <Plot x={x} y={fy}>
          <Curve name="tanh x" x={X} y={TANH} slot={SLOT.tanh} />
          <Curve name="σ(x)" x={X} y={SIG} slot={SLOT.sigmoid} />
          <Curve name="2σ(2x) − 1" x={X} y={RESCALED} emphasis dashed width={1.5} />
          {softsign && <Curve name="softsign" x={X} y={SOFTSIGN} slot={SLOT.softsign} />}
          <Points
            name="values at x"
            x={softsign ? [x0, x0, x0] : [x0, x0]}
            y={softsign ? [at.tanh, at.sig, at.soft] : [at.tanh, at.sig]}
            emphasis
            size={7}
            live
          />
          <Handle {...state.handle('x0', { label: 'x' })} />
        </Plot>
        <Plot x={x} y={dy}>
          <Curve name="tanh′" x={X} y={D_TANH} slot={SLOT.tanh} />
          <Curve name="σ′" x={X} y={D_SIG} slot={SLOT.sigmoid} />
          {softsign && <Curve name="softsign′" x={X} y={D_SOFT} slot={SLOT.softsign} />}
          <Handle {...state.handle('x0', { label: 'x' })} />
        </Plot>
      </Plots>
    </Figure>
  )
}
