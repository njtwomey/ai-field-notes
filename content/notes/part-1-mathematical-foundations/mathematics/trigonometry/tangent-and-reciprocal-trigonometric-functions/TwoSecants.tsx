import { useMemo } from 'react'
import {
  Curve,
  Figure,
  formatNumber,
  Handle,
  Plot,
  Points,
  Readout,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { FixedHeight } from '../_shared/FixedHeight'
import { grid, toDegrees } from '../_shared/geometry'

const PANEL = 300
const CAP = 6
const TH = grid(-1.5 * Math.PI, 1.5 * Math.PI, 721)
const COS = TH.map(Math.cos)
// sec θ = 1/cos θ, broken (NaN) where it leaves the view so the asymptotes are not joined.
const SEC = TH.map((t) => {
  const v = 1 / Math.cos(t)
  return Math.abs(v) > CAP ? NaN : v
})
const X = grid(-2, 2, 201)
const f = (x: number) => 0.25 * x * x * x - x + 1
const df = (x: number) => 0.75 * x * x - 1
const FX = X.map(f)

/**
 * The two meanings of "secant". Left: the secant function sec θ = 1/cos θ, a number attached to an angle. Right: a
 * secant line, the line through two points of a curve, whose slope is the difference quotient.
 */
export function TwoSecants() {
  const state = useFigureState({
    theta: slider(-1.5 * Math.PI, 1.5 * Math.PI, 1, { step: 0.01, label: 'angle θ (rad)' }),
    a: slider(-2, 2, -0.5, { step: 0.01, label: 'first point a' }),
    b: slider(-2, 2, 1.5, { step: 0.01, label: 'second point b' }),
  })
  const { theta, a, b } = state
  const sec = 1 / Math.cos(theta)

  const line = useMemo(() => {
    const h = b - a
    const slope = Math.abs(h) > 1e-9 ? (f(b) - f(a)) / h : df(a)
    return { slope, y: X.map((x) => f(a) + slope * (x - a)), tangent: X.map((x) => f(a) + df(a) * (x - a)) }
  }, [a, b])
  const alpha = Math.atan(line.slope)

  const tx = useAxis({ label: 'θ (rad)', range: [-1.5 * Math.PI, 1.5 * Math.PI] })
  const ty = useAxis({ label: 'value', range: [-CAP, CAP] })
  const xx = useAxis({ label: 'x', range: [-2, 2] })
  const xy = useAxis({ label: 'y', range: [-1, 3] })

  return (
    <Figure
      title="Two meanings of secant"
      state={state}
      caption="Left: the secant function sec θ = 1/cos θ (blue) with cos θ (dashed). Drag θ: sec θ is large where cos θ is small and undefined where cos θ = 0, at odd multiples of π/2. Right: a secant line of the curve y = x³/4 − x + 1, the line through the points above a and b; drag either vertical line. Its slope is the difference quotient (f(b) − f(a))/(b − a), and that slope is the tangent of the angle α the line makes with the x-axis. Bring b towards a and the secant line turns into the dashed tangent line at a, whose slope is the derivative f′(a). Nothing on the right involves 1/cos: the two uses share only the Latin secare, to cut."
      readouts={{
        'secant function': (
          <>
            <Readout label="cos θ" value={formatNumber(Math.cos(theta))} />
            <Readout label="sec θ" value={Math.abs(sec) < 1e6 ? formatNumber(sec) : 'undefined'} />
          </>
        ),
        'secant line': (
          <>
            <Readout label="slope (f(b) − f(a))/(b − a)" value={formatNumber(line.slope)} />
            <Readout label="inclination α" value={`${formatNumber(toDegrees(alpha))}°`} />
            <Readout label="tan α" value={formatNumber(Math.tan(alpha))} />
            <Readout label="tangent slope f′(a)" value={formatNumber(df(a))} />
          </>
        ),
      }}
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <FixedHeight height={PANEL}>
          <Plot x={tx} y={ty} title="secant function sec θ = 1/cos θ">
            <Curve name="cos θ" x={TH} y={COS} muted dashed />
            <Curve name="sec θ" x={TH} y={SEC} slot={0} />
            <Points name="(θ, sec θ)" x={[theta]} y={[Math.abs(sec) <= CAP ? sec : NaN]} emphasis size={8} live />
            <Handle {...state.handle('theta', { label: 'θ' })} />
          </Plot>
        </FixedHeight>
        <FixedHeight height={PANEL}>
          <Plot x={xx} y={xy} title="secant line of a curve">
            <Curve name="y = f(x)" x={X} y={FX} emphasis />
            <Curve name="tangent at a" x={X} y={line.tangent} muted dashed />
            <Curve name="secant line" x={X} y={line.y} slot={1} />
            <Points name="the two points" x={[a, b]} y={[f(a), f(b)]} slot={1} size={8} live />
            <Handle {...state.handle('a', { label: 'a' })} />
            <Handle {...state.handle('b', { label: 'b' })} />
          </Plot>
        </FixedHeight>
      </div>
    </Figure>
  )
}
