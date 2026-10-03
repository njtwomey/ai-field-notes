import { useMemo, useState } from 'react'
import { Interactive, Readout, XYChart, formatNumber, type Handle, type XYSeries } from 'aifn-render'

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
    const series: XYSeries[] = [
      { name: 'x', type: 'scatter', x: [x[0]], y: [x[1]], slot: 0 },
      {
        name: 'x₁ c₁, then + x₂ c₂',
        type: 'line',
        x: [0, first[0], ax[0]],
        y: [0, first[1], ax[1]],
        slot: 1,
        dashed: true,
      },
      { name: 'A x', type: 'scatter', x: [ax[0]], y: [ax[1]], emphasis: true },
    ]
    // Rows of A = [c₁ c₂] are (c₁[0], c₂[0]) and (c₁[1], c₂[1]).
    const rows: [Vec, Vec] = [
      [c1[0], c2[0]],
      [c1[1], c2[1]],
    ]
    return { ax, rows, series }
  }, [c1, c2, x])

  const handles: Handle[] = [
    { kind: 'point', at: c1, label: 'c₁', onDrag: ([a, b]) => setC1([clamp(a), clamp(b)]) },
    { kind: 'point', at: c2, label: 'c₂', onDrag: ([a, b]) => setC2([clamp(a), clamp(b)]) },
    { kind: 'point', at: x, label: 'x', onDrag: ([a, b]) => setX([clamp(a), clamp(b)]) },
  ]
  const [row1, row2] = r.rows

  return (
    <Interactive
      title="Two readings of A x"
      caption="The arrows are the columns c₁ and c₂ of a 2 × 2 matrix A. Drag their tips, and drag the point x. The dashed route builds A x in the column view: go x₁ times along c₁, then x₂ times along c₂. The readout computes the same point in the row view, one dot product per row. Both give the same answer for every A and x."
      readout={
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
        <XYChart
          equalAspect
          xRange={[-R, R]}
          yRange={[-R, R]}
          xLabel="first coordinate"
          yLabel="second coordinate"
          series={r.series}
          vectors={[
            { from: [0, 0], to: c1 },
            { from: [0, 0], to: c2 },
          ]}
          handles={handles}
        />
      </div>
    </Interactive>
  )
}
