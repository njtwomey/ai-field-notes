import { useMemo } from 'react'
import {
  Curve,
  Figure,
  formatNumber,
  Handle,
  Plot,
  Readout,
  slider,
  useAxis,
  useFigureState,
  Vectors,
  type Vector,
} from 'aifn-render'
import { arc, circle, toDegrees, wrapSigned } from '../_shared/geometry'

const R = 3
const UNIT = circle(1)
const coord = (initial: number) => slider(-R, R, initial, { step: 0.05, onChart: true })
const deg = (a: number) => `${formatNumber(toDegrees(a))}°`

/**
 * Multiplying complex numbers in polar form: with z = r e^{iα} and w = s e^{iβ}, the product is zw = rs e^{i(α + β)}.
 * Moduli multiply and arguments add.
 */
export function ComplexMultiplication() {
  const state = useFigureState({ zr: coord(1.2), zi: coord(0.6), wr: coord(0.4), wi: coord(1.1) })
  const { zr, zi, wr, wi } = state

  const g = useMemo(() => {
    const pr = zr * wr - zi * wi
    const pi = zr * wi + zi * wr
    const a = Math.atan2(zi, zr)
    const b = Math.atan2(wi, wr)
    const vectors: Vector[] = [
      { from: [0, 0], to: [zr, zi], slot: 0, label: 'z' },
      { from: [0, 0], to: [wr, wi], slot: 1, label: 'w' },
      { from: [0, 0], to: [pr, pi], label: 'zw' },
    ]
    return {
      pr,
      pi,
      a,
      b,
      vectors,
      arcA: arc(0.35, 0, a, 48),
      arcB: arc(0.5, a, a + b, 48),
    }
  }, [zr, zi, wr, wi])

  const x = useAxis({ label: 'real part', range: [-R, R] })
  const y = useAxis({ label: 'imaginary part', range: [-R, R], equal: x })
  return (
    <Figure
      title="Multiplication rotates and scales"
      state={state}
      caption="Drag the tips of z (blue) and w (orange). The black arrow is the product zw. Its length is |z| |w| and its angle is arg z + arg w: the inner arc is arg z, and the outer arc continues from it by arg w to reach the product. Put w on the unit circle and multiplication by w is a pure rotation; put it on the positive real axis and it is a pure scaling. With w = i, the product is z turned a quarter turn anticlockwise."
      readouts={{
        moduli: (
          <>
            <Readout label="|z|" value={formatNumber(Math.hypot(zr, zi))} />
            <Readout label="|w|" value={formatNumber(Math.hypot(wr, wi))} />
            <Readout label="|z| |w|" value={formatNumber(Math.hypot(zr, zi) * Math.hypot(wr, wi))} />
            <Readout label="|zw|" value={formatNumber(Math.hypot(g.pr, g.pi))} />
          </>
        ),
        arguments: (
          <>
            <Readout label="arg z" value={deg(g.a)} />
            <Readout label="arg w" value={deg(g.b)} />
            <Readout label="arg z + arg w (wrapped)" value={deg(wrapSigned(g.a + g.b))} />
            <Readout label="arg zw" value={deg(Math.atan2(g.pi, g.pr))} />
          </>
        ),
      }}
    >
      <div className="mx-auto w-full max-w-lg">
        <Plot x={x} y={y} fitHeight legend={false}>
          <Curve name="unit circle" x={UNIT.x} y={UNIT.y} muted dashed silent />
          <Curve name="arg z" x={g.arcA.x} y={g.arcA.y} slot={0} width={2} silent />
          <Curve name="arg w" x={g.arcB.x} y={g.arcB.y} slot={1} width={2} silent />
          <Vectors vectors={g.vectors} />
          <Handle {...state.handle(['zr', 'zi'], { label: 'z' })} />
          <Handle {...state.handle(['wr', 'wi'], { label: 'w' })} />
        </Plot>
      </div>
    </Figure>
  )
}
