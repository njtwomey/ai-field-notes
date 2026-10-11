import { useMemo } from 'react'
import {
  Curve,
  Figure,
  formatNumber,
  Handle,
  Plot,
  Plots,
  Points,
  Readout,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'

const SPAN = 2 * Math.PI
const X = toFlat(linspace(-SPAN, SPAN, 481))
const SIN = X.map(Math.sin)
const COS = X.map(Math.cos)
const REACH = 1.6

/**
 * The slope of sin x at each point is cos x. Top: sin x with its tangent line at x₀ and a secant to x₀ + h. Bottom:
 * cos x, with the tangent's slope and the secant's slope (the difference quotient) marked at x₀.
 */
export function SlopeOfSine() {
  const state = useFigureState({
    x0: slider(-SPAN, SPAN, 0.8, { step: 0.01, onChart: true }),
    h: slider(0.01, 2, 1, { step: 0.01, label: 'step h' }),
  })
  const { x0, h } = state
  const slope = Math.cos(x0)
  const dq = (Math.sin(x0 + h) - Math.sin(x0)) / h

  const lines = useMemo(() => {
    const tx = [x0 - REACH, x0 + REACH]
    const sx = [x0 - REACH, x0 + h + REACH]
    return {
      tangent: { x: tx, y: tx.map((u) => Math.sin(x0) + slope * (u - x0)) },
      secant: { x: sx, y: sx.map((u) => Math.sin(x0) + dq * (u - x0)) },
    }
  }, [x0, h, slope, dq])

  const x = useAxis({ label: 'x (rad)', range: [-SPAN, SPAN] })
  const fy = useAxis({ label: 'sin x', range: [-1.6, 1.6] })
  const dy = useAxis({ label: 'slope', range: [-1.6, 1.6] })
  return (
    <Figure
      title="The slope of sine is cosine"
      state={state}
      caption="Drag the vertical line to move x₀. Top: sin x with its tangent line at x₀ (orange) and the secant line through x₀ and x₀ + h (dashed green). Bottom: cos x, with two marks at x₀: the tangent's slope (black), which always lies on the cosine curve, and the secant's slope (green), the difference quotient (sin(x₀ + h) − sin x₀)/h. Shrink h and the green mark closes in on the black one. The slope of sin is largest, 1, where sin crosses zero going up, and zero at its peaks and troughs, which is where cos is 1 and 0."
      readouts={
        <>
          <Readout label="x₀" value={formatNumber(x0)} />
          <Readout label="tangent slope cos x₀" value={formatNumber(slope)} />
          <Readout label="secant slope" value={formatNumber(dq)} />
          <Readout label="difference" value={formatNumber(dq - slope)} />
        </>
      }
    >
      <Plots rows={2} heights={[1, 1]} hoverGroup>
        <Plot x={x} y={fy}>
          <Curve name="sin x" x={X} y={SIN} emphasis />
          <Curve name="tangent at x₀" x={lines.tangent.x} y={lines.tangent.y} slot={1} />
          <Curve name="secant x₀ to x₀ + h" x={lines.secant.x} y={lines.secant.y} slot={2} dashed />
          <Points name="points" x={[x0, x0 + h]} y={[Math.sin(x0), Math.sin(x0 + h)]} slot={2} size={7} live />
          <Handle {...state.handle('x0', { label: 'x₀' })} />
        </Plot>
        <Plot x={x} y={dy}>
          <Curve name="cos x" x={X} y={COS} slot={0} />
          <Points name="secant slope" x={[x0]} y={[dq]} slot={2} size={8} live />
          <Points name="tangent slope" x={[x0]} y={[slope]} emphasis size={8} live />
          <Handle {...state.handle('x0', { label: 'x₀' })} />
        </Plot>
      </Plots>
    </Figure>
  )
}
