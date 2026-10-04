import { useMemo, useState } from 'react'
import { Curve, Figure, formatNumber, Handle, Plot, Points, Readout, useAxis, Vectors } from 'aifn-render'

type Vec = [number, number]
const R = 4
const clamp = (v: number) => Math.round(Math.min(Math.max(v, -R), R) * 20) / 20
const f = formatNumber

/** A x built as x₁ c₁ + x₂ c₂ (the column view), with the row view's dot products in the readout. */
export function TwoViews() {
  const [c1, setC1] = useState<Vec>([1.5, 0.5])
  const [c2, setC2] = useState<Vec>([-0.5, 1])
  const [x, setX] = useState<Vec>([1.5, 1.5])

  const r = useMemo(() => {
    const first: Vec = [x[0] * c1[0], x[0] * c1[1]]
    const ax: Vec = [first[0] + x[1] * c2[0], first[1] + x[1] * c2[1]]
    const series = [
      { name: 'x', x: [x[0]], y: [x[1]], slot: 0 },
      {
        name: 'x₁ c₁, then + x₂ c₂',
        x: [0, first[0], ax[0]],
        y: [0, first[1], ax[1]],
        slot: 1,
        dashed: true,
      },
      { name: 'A x', x: [ax[0]], y: [ax[1]], emphasis: true },
    ] as const
    // Rows of A = [c₁ c₂] are (c₁[0], c₂[0]) and (c₁[1], c₂[1]).
    const rows: [Vec, Vec] = [
      [c1[0], c2[0]],
      [c1[1], c2[1]],
    ]
    return { ax, rows, series }
  }, [c1, c2, x])

  const [row1, row2] = r.rows

  const xAxis = useAxis({ label: 'first coordinate', range: [-R, R] })
  const yAxis = useAxis({ label: 'second coordinate', range: [-R, R], equal: xAxis })
  return (
    <Figure
      title="Two readings of A x"
      caption="The arrows are the columns c₁ and c₂ of a 2 × 2 matrix A. Drag their tips, and drag the point x. The dashed route builds A x in the column view: go x₁ times along c₁, then x₂ times along c₂. The readout computes the same point in the row view, one dot product per row. Both give the same answer for every A and x."
      readouts={
        <>
          <Readout label="A" value={`[[${f(row1[0])}, ${f(row1[1])}], [${f(row2[0])}, ${f(row2[1])}]]`} />
          <Readout label="x" value={`(${f(x[0])}, ${f(x[1])})`} />
          <Readout label="row 1 · x" value={f(row1[0] * x[0] + row1[1] * x[1])} />
          <Readout label="row 2 · x" value={f(row2[0] * x[0] + row2[1] * x[1])} />
          <Readout label="A x" value={`(${f(r.ax[0])}, ${f(r.ax[1])})`} />
        </>
      }
    >
      {/* Equal-aspect charts take their height from their width; keep square plots a readable size. */}
      <div className="mx-auto w-full max-w-lg">
        <Plot x={xAxis} y={yAxis}>
          <Points {...r.series[0]} />
          <Curve {...r.series[1]} />
          <Points {...r.series[2]} />
          <Vectors
            vectors={[
              { from: [0, 0], to: c1 },
              { from: [0, 0], to: c2 },
            ]}
          />
          <Handle kind="point" at={c1} label="c₁" onDrag={([a, b]) => setC1([clamp(a), clamp(b)])} />
          <Handle kind="point" at={c2} label="c₂" onDrag={([a, b]) => setC2([clamp(a), clamp(b)])} />
          <Handle kind="point" at={x} label="x" onDrag={([a, b]) => setX([clamp(a), clamp(b)])} />
        </Plot>
      </div>
    </Figure>
  )
}
