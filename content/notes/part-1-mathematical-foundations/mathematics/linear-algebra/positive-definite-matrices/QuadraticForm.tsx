import { useMemo } from 'react'
import { Figure, formatNumber, Plot, Raster, Readout, slider, useAxis, useFigureState, Vectors } from 'aifn-render'
import { linspace, toFlat } from 'aifn-compute/foundation/tensor'
import { eigh2 } from 'aifn-compute/numerics/linalg'

const GRID = toFlat(linspace(-2, 2, 41))
const entry = (initial: number, label: string) => slider(-2, 2, initial, { step: 0.05, label })

function classify(l1: number, l2: number): string {
  const eps = 1e-9
  if (l2 > eps) return 'positive definite'
  if (l2 > -eps && l1 > eps) return 'positive semi-definite'
  if (l1 < -eps) return 'negative definite'
  if (l1 < eps && l2 < -eps) return 'negative semi-definite'
  if (Math.abs(l1) < eps && Math.abs(l2) < eps) return 'zero'
  return 'indefinite'
}

/** The quadratic form xᵀAx of a symmetric 2×2 matrix: a bowl, a trough, a saddle or an upside-down bowl. */
export function QuadraticForm() {
  const state = useFigureState({
    a: entry(1.5, 'a'),
    b: entry(0.5, 'b (off-diagonal)'),
    c: entry(1, 'c'),
  })
  const { a, b, c } = state

  const r = useMemo(() => {
    const z = GRID.map((y) => GRID.map((x) => a * x * x + 2 * b * x * y + c * y * y))
    const peak = Math.max(...z.flat().map(Math.abs), 1e-9)
    const eig = eigh2([
      [a, b],
      [b, c],
    ])
    return { z, peak, eig }
  }, [a, b, c])

  const [l1, l2] = r.eig.values
  const [v1, v2] = r.eig.vectors
  const xAxis = useAxis({ label: 'x₁' })
  const yAxis = useAxis({ label: 'x₂' })
  return (
    <Figure
      title="The quadratic form xᵀAx"
      caption="Set the entries of the symmetric matrix A = [[a, b], [b, c]]. Colour shows xᵀAx: red is positive, blue is negative. The arrows are the eigenvectors. Along each one the form grows like λᵢ times the squared distance, so the signs of the two eigenvalues decide the shape. Both positive gives a bowl; one zero gives a trough; opposite signs give a saddle."
      state={state}
      readouts={
        <>
          <Readout label="eigenvalues" value={`${formatNumber(l1)}, ${formatNumber(l2)}`} />
          <Readout label="leading minors" value={`${formatNumber(a)}, ${formatNumber(a * c - b ** 2)}`} />
          <Readout label="A is" value={classify(l1, l2)} />
        </>
      }
    >
      <div className="mx-auto w-full max-w-lg">
        <Plot x={xAxis} y={yAxis} height={420}>
          <Raster x={GRID} y={GRID} z={r.z} scale={'diverging'} range={[-r.peak, r.peak]} valueLabel={'xᵀAx'} />
          <Vectors
            vectors={[
              { from: [0, 0], to: [1.5 * v1[0], 1.5 * v1[1]] },
              { from: [0, 0], to: [1.5 * v2[0], 1.5 * v2[1]] },
            ]}
          />
        </Plot>
      </div>
    </Figure>
  )
}
