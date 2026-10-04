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
} from 'aifn-render'

type Vec = [number, number]
const R = 3
const entry = (initial: number, label: string) => slider(-2.5, 2.5, initial, { step: 0.05, label })

/** An asymmetric letter F inside the unit square, so that a reflection shows as a mirrored letter. */
const F: Vec[] = [
  [0.2, 0.1],
  [0.2, 0.9],
  [0.75, 0.9],
  [0.75, 0.75],
  [0.35, 0.75],
  [0.35, 0.55],
  [0.65, 0.55],
  [0.65, 0.4],
  [0.35, 0.4],
  [0.35, 0.1],
  [0.2, 0.1],
]

/** The unit square under a 2 × 2 matrix: its area scales by |det A| and its orientation flips when det A < 0. */
export function SignedArea() {
  const state = useFigureState({
    a: entry(1.5, 'a'),
    b: entry(0.5, 'b'),
    c: entry(0.25, 'c'),
    d: entry(1.25, 'd'),
  })
  const [a, b, c, d] = [state.a, state.b, state.c, state.d]

  const r = useMemo(() => {
    const map = (p: Vec): Vec => [a * p[0] + b * p[1], c * p[0] + d * p[1]]
    const square: Vec[] = [
      [0, 0],
      [1, 0],
      [1, 1],
      [0, 1],
      [0, 0],
    ]
    const image = square.map(map)
    const letter = F.map(map)
    const series = [
      {
        name: 'unit square',
        x: square.map((p) => p[0]),
        y: square.map((p) => p[1]),
        slot: 0,
        dashed: true,
      },
      { name: 'image of the square', x: image.map((p) => p[0]), y: image.map((p) => p[1]), slot: 1 },
      { name: 'image of the letter F', x: letter.map((p) => p[0]), y: letter.map((p) => p[1]), slot: 2 },
    ] as const
    return { det: a * d - b * c, series }
  }, [a, b, c, d])

  // The columns of A are the images of e₁ and e₂, the two sides of the parallelogram.
  const orientation = Math.abs(r.det) < 1e-9 ? 'collapsed' : r.det > 0 ? 'kept' : 'reversed'

  const xAxis = useAxis({ label: 'x₁', range: [-R, R] })
  const yAxis = useAxis({ label: 'x₂', range: [-R, R], equal: xAxis })
  return (
    <Figure
      title="Area and orientation"
      caption="The matrix A = [[a, b], [c, d]] sends the dashed unit square to the solid parallelogram, whose sides are the columns A e₁ and A e₂. Drag their tips. The area of the parallelogram is |det A|. Swing A e₂ past the line through A e₁ and the determinant changes sign: the letter F comes out mirrored. With the two columns on one line the square collapses to a segment and det A = 0."
      state={state}
      readouts={
        <>
          <Readout label="det A = ad − bc" value={formatNumber(r.det)} />
          <Readout label="area" value={formatNumber(Math.abs(r.det))} />
          <Readout label="orientation" value={orientation} />
        </>
      }
    >
      {/* Equal-aspect charts take their height from their width; keep square plots a readable size. */}
      <div className="mx-auto w-full max-w-lg">
        <Plot x={xAxis} y={yAxis}>
          <Curve {...r.series[0]} />
          <Curve {...r.series[1]} />
          <Curve {...r.series[2]} />
          <Vectors
            vectors={[
              { from: [0, 0], to: [a, c] },
              { from: [0, 0], to: [b, d] },
            ]}
          />
          <Handle {...state.handle(['a', 'c'], { label: 'A e₁' })} />
          <Handle {...state.handle(['b', 'd'], { label: 'A e₂' })} />
        </Plot>
      </div>
    </Figure>
  )
}
