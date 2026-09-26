import { useMemo } from 'react'
import { Heatmap, Interactive, ParamSlider, Readout, formatNumber, useParam } from '@/components/viz'
import { linspace } from '@/lib/math'
import { eigSym } from '@/lib/math/mat2'

const GRID = linspace(-2, 2, 41)
const ENTRY = { min: -2, max: 2, step: 0.05 }

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
  const a = useParam(1.5, ENTRY)
  const b = useParam(0.5, ENTRY)
  const c = useParam(1, ENTRY)

  const r = useMemo(() => {
    const z = GRID.map((y) => GRID.map((x) => a.value * x * x + 2 * b.value * x * y + c.value * y * y))
    const peak = Math.max(...z.flat().map(Math.abs), 1e-9)
    const eig = eigSym(a.value, b.value, c.value)
    return { z, peak, eig }
  }, [a.value, b.value, c.value])

  const [l1, l2] = r.eig.values
  const [v1, v2] = r.eig.vectors
  return (
    <Interactive
      title="The quadratic form xᵀAx"
      caption="Set the entries of the symmetric matrix A = [[a, b], [b, c]]. Colour shows xᵀAx: red is positive, blue is negative. The arrows are the eigenvectors. Along each one the form grows like λᵢ times the squared distance, so the signs of the two eigenvalues decide the shape. Both positive gives a bowl; one zero gives a trough; opposite signs give a saddle."
      controls={
        <>
          <ParamSlider label="a" param={a} />
          <ParamSlider label="b (off-diagonal)" param={b} />
          <ParamSlider label="c" param={c} />
        </>
      }
      readout={
        <>
          <Readout label="eigenvalues" value={`${formatNumber(l1)}, ${formatNumber(l2)}`} />
          <Readout
            label="leading minors"
            value={`${formatNumber(a.value)}, ${formatNumber(a.value * c.value - b.value ** 2)}`}
          />
          <Readout label="A is" value={classify(l1, l2)} />
        </>
      }
    >
      <div className="mx-auto w-full max-w-lg">
        <Heatmap
          x={GRID}
          y={GRID}
          z={r.z}
          scale="diverging"
          range={[-r.peak, r.peak]}
          xLabel="x₁"
          yLabel="x₂"
          valueLabel="xᵀAx"
          height={420}
          vectors={[
            { from: [0, 0], to: [1.5 * v1[0], 1.5 * v1[1]] },
            { from: [0, 0], to: [1.5 * v2[0], 1.5 * v2[1]] },
          ]}
        />
      </div>
    </Interactive>
  )
}
