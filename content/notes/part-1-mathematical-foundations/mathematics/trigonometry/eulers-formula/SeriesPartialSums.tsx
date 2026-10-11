import { useMemo } from 'react'
import {
  Curve,
  Figure,
  formatNumber,
  Handle,
  int,
  Plot,
  Points,
  Readout,
  slider,
  useAxis,
  useFigureState,
  Vectors,
  type Vector,
} from 'aifn-render'
import { circle, toDegrees } from '../_shared/geometry'

const VIEW = 4.5
const UNIT = circle(1)
const fmtC = (re: number, im: number) => `${formatNumber(re)} ${im < 0 ? '−' : '+'} ${formatNumber(Math.abs(im))}i`

/**
 * The exponential series of iθ, one term at a time. Term k is (iθ)ᵏ/k!: its direction cycles through 1, i, −1, −i,
 * so the partial sums turn round a square spiral that closes in on cos θ + i sin θ on the unit circle.
 */
export function SeriesPartialSums() {
  const state = useFigureState({
    theta: slider(-Math.PI, Math.PI, 2, { step: 0.01, label: 'θ (rad)' }),
    n: int(5, { min: 1, max: 20, step: 1, label: 'terms n' }),
  })
  const { theta, n } = state

  const s = useMemo(() => {
    const vectors: Vector[] = []
    const px: number[] = [0]
    const py: number[] = [0]
    let re = 0
    let im = 0
    let term = 1 // θᵏ / k!; the factor iᵏ is the direction cycle below
    for (let k = 0; k < n; k++) {
      if (k > 0) term *= theta / k
      // iᵏ cycles 1, i, −1, −i.
      const dir = [
        [1, 0],
        [0, 1],
        [-1, 0],
        [0, -1],
      ][k % 4]
      const next: [number, number] = [re + term * dir[0], im + term * dir[1]]
      vectors.push({ from: [re, im], to: next, slot: k % 4 })
      re = next[0]
      im = next[1]
      px.push(re)
      py.push(im)
    }
    const err = Math.hypot(re - Math.cos(theta), im - Math.sin(theta))
    return { vectors, px, py, re, im, err }
  }, [theta, n])

  const x = useAxis({ label: 'real part', range: [-VIEW, VIEW] })
  const y = useAxis({ label: 'imaginary part', range: [-VIEW, VIEW], equal: x })
  return (
    <Figure
      title="Summing the exponential series of iθ"
      state={state}
      caption="Each arrow is one term (iθ)ᵏ/k! of the exponential series, placed tip to tail from 0. Powers of i turn each term a quarter turn from the last: real terms (blue, k = 0, 4, 8, …), imaginary terms (orange), negative real (green) and negative imaginary (yellow). The real terms add up to the series of cos θ and the imaginary terms to the series of sin θ. Increase the number of terms n and the end of the chain closes in on the black point cos θ + i sin θ on the unit circle; drag that point to change θ. Larger |θ| needs more terms, because the terms grow until k passes |θ| and only then shrink."
      readouts={
        <>
          <Readout label="θ" value={`${formatNumber(theta)} rad = ${formatNumber(toDegrees(theta))}°`} />
          <Readout label="partial sum of n terms" value={fmtC(s.re, s.im)} />
          <Readout label="exp(iθ) = cos θ + i sin θ" value={fmtC(Math.cos(theta), Math.sin(theta))} />
          <Readout label="distance between them" value={formatNumber(s.err)} />
        </>
      }
    >
      <div className="mx-auto w-full max-w-lg">
        <Plot x={x} y={y} fitHeight legend={false}>
          <Curve name="unit circle" x={UNIT.x} y={UNIT.y} muted silent />
          <Vectors vectors={s.vectors} />
          <Points name="partial sums" x={s.px} y={s.py} muted size={5} />
          <Handle
            kind="point"
            at={[Math.cos(theta), Math.sin(theta)]}
            label="exp(iθ)"
            onDrag={([a, b]) => state.set('theta', Math.atan2(b, a))}
          />
        </Plot>
      </div>
    </Figure>
  )
}
