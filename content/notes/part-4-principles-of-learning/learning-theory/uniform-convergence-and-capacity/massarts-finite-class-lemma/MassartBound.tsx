import { useMemo } from 'react'
import { empiricalRademacher, shatteringTable, type ShatterClass } from 'aifn-methods/theory/capacity'
import { child, stream, uniform } from 'aifn-compute/foundation/random'
import { toRows, type Tensor } from 'aifn-compute/foundation/tensor'
import { choice, Curve, Figure, formatNumber, int, Plot, Points, Readout, useAxis, useFigureState } from 'aifn-render'

const DRAWS = 500
/** Half-planes need one linear program per labelling, so they stop at 10 points (1,024 programs). */
const HALF_PLANE_LIMIT = 10
const fmt = (v: number) => formatNumber(Number(v.toPrecision(3)))

/**
 * The empirical Rademacher complexity of the projection of a class on the first k of n random points, measured by
 * Monte Carlo over sign vectors, beside Massart's bound √(2 ln |A| / k) for k = 1, …, n.
 */
export function MassartBound() {
  const state = useFigureState({
    family: choice(
      [
        { value: 'intervals', label: 'intervals on x₁ (VC 2)' },
        { value: 'half-planes', label: 'half-planes (VC 3)' },
        { value: 'rectangles', label: 'axis-aligned rectangles (VC 4)' },
      ],
      'intervals',
      { label: 'class' },
    ),
    n: int(12, { ge: 2, le: 12, suggestions: [6, 10, 12], label: 'points n' }),
    seed: int(1, { ge: 0, le: 999, label: 'seed' }),
  })
  const { family, seed } = state
  const n = family === 'half-planes' ? Math.min(state.n, HALF_PLANE_LIMIT) : state.n
  const rows = useMemo(
    () => toRows(uniform(child(stream('massart'), 'points', seed), -1, 1, { shape: [n, 2] }) as Tensor),
    [n, seed],
  )
  const curve = useMemo(() => {
    const sizes: number[] = []
    const estimate: number[] = []
    const lower: number[] = []
    const upper: number[] = []
    const massart: number[] = []
    const projection: number[] = []
    rows.forEach((_, i) => {
      const table = shatteringTable(rows.slice(0, i + 1), family as ShatterClass)
      const realised = table.labellings.filter((_, j) => table.realised[j])
      const r = empiricalRademacher(child(stream('massart'), 'signs', seed, i), realised, DRAWS)
      sizes.push(i + 1)
      estimate.push(r.estimate)
      lower.push(r.estimate - 2 * r.standardError)
      upper.push(r.estimate + 2 * r.standardError)
      massart.push(r.massart)
      projection.push(realised.length)
    })
    return { sizes, estimate, lower, upper, massart, projection }
  }, [rows, family, seed])
  const last = curve.sizes.length - 1
  const kAxis = useAxis({ label: 'points k', range: [0.5, 12.5] })
  const rAxis = useAxis({ label: 'empirical Rademacher complexity', range: [0, 1.3] })
  return (
    <Figure
      title="Massart's bound against the measured complexity"
      state={state}
      readouts={
        <>
          <Readout label="labellings realised, |A|" value={`${curve.projection[last]} of ${2 ** n}`} />
          <Readout label="measured" value={fmt(curve.estimate[last])} />
          <Readout label="Massart √(2 ln|A|/n)" value={fmt(curve.massart[last])} />
        </>
      }
      caption={`n points are drawn uniformly on the square [−1, 1]²; for each k the engine's shatteringTable finds the labellings A the class realises on the first k points, and empiricalRademacher estimates E max over A of (1/k) Σ σᵢ aᵢ from ${DRAWS} random sign vectors (the band is ± 2 standard errors). The dashed curve is Massart's bound √(2 ln|A|/k) for vectors of norm √k. A class that realises every labelling has complexity exactly 1 (the thin dashed line), and Massart's bound is then √(2 ln 2) ≈ 1.18. Below the VC dimension the bound exceeds 1 and says nothing; it falls below 1 only once ln|A| is small next to k. Half-planes stop at ${HALF_PLANE_LIMIT} points.`}
    >
      <Plot x={kAxis} y={rAxis}>
        <Curve name="every labelling" x={[0.5, 12.5]} y={[1, 1]} thin dashed silent />
        <Curve name="measured − 2 s.e." x={curve.sizes} y={curve.lower} slot={0} thin />
        <Curve name="measured + 2 s.e." x={curve.sizes} y={curve.upper} slot={0} thin />
        <Curve name="measured" x={curve.sizes} y={curve.estimate} slot={0} />
        <Points name="measured" x={curve.sizes} y={curve.estimate} slot={0} size={10} />
        <Curve name="Massart's bound" x={curve.sizes} y={curve.massart} emphasis dashed />
      </Plot>
    </Figure>
  )
}
