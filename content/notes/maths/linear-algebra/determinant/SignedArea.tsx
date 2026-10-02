import { useMemo } from 'react'
import {
  Interactive,
  ParamSlider,
  Readout,
  XYChart,
  formatNumber,
  useParam,
  type Handle,
  type ParamSpec,
  type XYSeries,
} from 'aifn-render'

type Vec = [number, number]
const R = 3
const ENTRY: ParamSpec = { min: -2.5, max: 2.5, step: 0.05 }

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
  const pa = useParam(1.5, ENTRY)
  const pb = useParam(0.5, ENTRY)
  const pc = useParam(0.25, ENTRY)
  const pd = useParam(1.25, ENTRY)
  const [a, b, c, d] = [pa.value, pb.value, pc.value, pd.value]

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
    const series: XYSeries[] = [
      {
        name: 'unit square',
        type: 'line',
        x: square.map((p) => p[0]),
        y: square.map((p) => p[1]),
        slot: 0,
        dashed: true,
      },
      { name: 'image of the square', type: 'line', x: image.map((p) => p[0]), y: image.map((p) => p[1]), slot: 1 },
      { name: 'image of the letter F', type: 'line', x: letter.map((p) => p[0]), y: letter.map((p) => p[1]), slot: 2 },
    ]
    return { det: a * d - b * c, series }
  }, [a, b, c, d])

  // The columns of A are the images of e₁ and e₂, the two sides of the parallelogram.
  const handles: Handle[] = [
    {
      kind: 'point',
      at: [a, c],
      label: 'A e₁',
      onDrag: ([x, y]) => {
        pa.set(x)
        pc.set(y)
      },
    },
    {
      kind: 'point',
      at: [b, d],
      label: 'A e₂',
      onDrag: ([x, y]) => {
        pb.set(x)
        pd.set(y)
      },
    },
  ]
  const orientation = Math.abs(r.det) < 1e-9 ? 'collapsed' : r.det > 0 ? 'kept' : 'reversed'

  return (
    <Interactive
      title="Area and orientation"
      caption="The matrix A = [[a, b], [c, d]] sends the dashed unit square to the solid parallelogram, whose sides are the columns A e₁ and A e₂. Drag their tips. The area of the parallelogram is |det A|. Swing A e₂ past the line through A e₁ and the determinant changes sign: the letter F comes out mirrored. With the two columns on one line the square collapses to a segment and det A = 0."
      controls={
        <>
          <ParamSlider label="a" param={pa} />
          <ParamSlider label="b" param={pb} />
          <ParamSlider label="c" param={pc} />
          <ParamSlider label="d" param={pd} />
        </>
      }
      readout={
        <>
          <Readout label="det A = ad − bc" value={formatNumber(r.det)} />
          <Readout label="area" value={formatNumber(Math.abs(r.det))} />
          <Readout label="orientation" value={orientation} />
        </>
      }
    >
      {/* Equal-aspect charts take their height from their width; keep square plots a readable size. */}
      <div className="mx-auto w-full max-w-lg">
        <XYChart
          equalAspect
          xRange={[-R, R]}
          yRange={[-R, R]}
          xLabel="x₁"
          yLabel="x₂"
          series={r.series}
          vectors={[
            { from: [0, 0], to: [a, c] },
            { from: [0, 0], to: [b, d] },
          ]}
          handles={handles}
        />
      </div>
    </Interactive>
  )
}
