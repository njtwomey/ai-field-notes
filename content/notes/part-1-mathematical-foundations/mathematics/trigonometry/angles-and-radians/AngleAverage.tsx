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
import { arc, circle, TAU, toDegrees, wrapPositive, wrapSigned } from '../_shared/geometry'

const VIEW: [number, number] = [-1.4, 1.4]
const UNIT = circle(1)
const deg = (a: number) => `${formatNumber(toDegrees(a))}°`

/**
 * Two headings stored as angles in [0, 2π). Their arithmetic mean can point the wrong way when they straddle 0; the
 * mean of the two unit vectors, converted back to an angle, does not.
 */
export function AngleAverage() {
  const state = useFigureState({
    a: slider(0, TAU, 5.93, { step: 0.01, onChart: true }),
    b: slider(0, TAU, 0.52, { step: 0.01, onChart: true }),
  })
  const { a, b } = state

  const r = useMemo(() => {
    const naive = (a + b) / 2
    const sx = Math.cos(a) + Math.cos(b)
    const sy = Math.sin(a) + Math.sin(b)
    const resultant = Math.hypot(sx, sy) / 2
    const circular = resultant > 1e-9 ? wrapPositive(Math.atan2(sy, sx)) : NaN
    const diff = wrapSigned(b - a)
    // The short way round from a to b, at radius 0.35.
    const between = arc(0.35, a, a + diff, 48)
    return { naive, circular, resultant, diff, between }
  }, [a, b])

  const vectors = useMemo((): Vector[] => {
    const v: Vector[] = [
      { from: [0, 0], to: [Math.cos(a), Math.sin(a)], slot: 0, label: 'a' },
      { from: [0, 0], to: [Math.cos(b), Math.sin(b)], slot: 1, label: 'b' },
      {
        from: [0, 0],
        to: [1.2 * Math.cos(r.naive), 1.2 * Math.sin(r.naive)],
        muted: true,
        label: '(a + b)/2',
      },
    ]
    if (Number.isFinite(r.circular)) {
      v.push({
        from: [0, 0],
        to: [1.2 * Math.cos(r.circular), 1.2 * Math.sin(r.circular)],
        label: 'circular mean',
      })
    }
    return v
  }, [a, b, r.naive, r.circular])

  const x = useAxis({ label: 'x', range: VIEW })
  const y = useAxis({ label: 'y', range: VIEW, equal: x })
  return (
    <Figure
      title="Averaging angles"
      state={state}
      caption="Drag the two headings a and b around the circle. Each is stored as an angle in [0°, 360°). The grey arrow is the arithmetic mean of the two stored numbers; the black arrow is the circular mean, the direction of the average of the two unit vectors. At the start, a ≈ 340° and b ≈ 30° are 50° apart across 0°: the circular mean is about 5° and the arithmetic mean about 185°, pointing the opposite way. The short arc marks the wrapped difference b − a in (−180°, 180°]."
      readouts={
        <>
          <Readout label="a" value={deg(a)} />
          <Readout label="b" value={deg(b)} />
          <Readout label="wrapped difference b − a" value={deg(r.diff)} />
          <Readout label="arithmetic mean" value={deg(r.naive)} />
          <Readout
            label="circular mean"
            value={Number.isFinite(r.circular) ? deg(r.circular) : 'undefined (opposite headings)'}
          />
          <Readout label="length of mean vector" value={formatNumber(r.resultant)} />
        </>
      }
    >
      <div className="mx-auto w-full max-w-md">
        <Plot x={x} y={y} fitHeight legend={false}>
          <Curve name="unit circle" x={UNIT.x} y={UNIT.y} muted silent />
          <Curve name="b − a" x={r.between.x} y={r.between.y} slot={2} width={3} silent />
          <Vectors vectors={vectors} />
          <Handle
            kind="point"
            at={[Math.cos(a), Math.sin(a)]}
            label="a"
            onDrag={([u, v]) => state.set('a', wrapPositive(Math.atan2(v, u)))}
          />
          <Handle
            kind="point"
            at={[Math.cos(b), Math.sin(b)]}
            label="b"
            onDrag={([u, v]) => state.set('b', wrapPositive(Math.atan2(v, u)))}
          />
        </Plot>
      </div>
    </Figure>
  )
}
