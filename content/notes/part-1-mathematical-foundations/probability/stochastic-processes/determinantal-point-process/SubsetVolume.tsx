import { useMemo } from 'react'
import {
  Bars,
  Figure,
  formatNumber,
  Handle,
  Plot,
  Readout,
  Shapes,
  slider,
  useAxis,
  useFigureState,
  Vectors,
} from 'aifn-render'

const SUBSETS = ['∅', '{1}', '{2}', '{1, 2}']
const AT = [0, 1, 2, 3]
const LEFT = AT.map((k) => k - 0.2)
const RIGHT = AT.map((k) => k + 0.2)

/**
 * A ground set of two items, each a vector b_i whose length is its quality and whose direction is its features. The
 * L-ensemble L = BᵀB gives each subset probability det(L_S) / det(L + I): the empty set 1, a single item its squared
 * length, the pair the squared area of the parallelogram the two vectors span.
 */
export function SubsetVolume() {
  const state = useFigureState({
    b1x: slider(-1.5, 1.5, 1.1, { step: 0.01, onChart: true }),
    b1y: slider(-1.5, 1.5, 0.2, { step: 0.01, onChart: true }),
    b2x: slider(-1.5, 1.5, 0.5, { step: 0.01, onChart: true }),
    b2y: slider(-1.5, 1.5, 1.0, { step: 0.01, onChart: true }),
  })
  const { b1x, b1y, b2x, b2y } = state

  const l11 = b1x * b1x + b1y * b1y
  const l22 = b2x * b2x + b2y * b2y
  const l12 = b1x * b2x + b1y * b2y
  const area = b1x * b2y - b1y * b2x
  const z = (l11 + 1) * (l22 + 1) - l12 * l12
  const dpp = [1 / z, l11 / z, l22 / z, (area * area) / z]
  const m1 = dpp[1] + dpp[3]
  const m2 = dpp[2] + dpp[3]
  const indep = [(1 - m1) * (1 - m2), m1 * (1 - m2), (1 - m1) * m2, m1 * m2]
  const cos = l12 / Math.sqrt(l11 * l22)

  const shapes = useMemo(
    () => [
      {
        contours: [
          [
            [0, 0],
            [b1x, b1y],
            [b1x + b2x, b1y + b2y],
            [b2x, b2y],
          ] as [number, number][],
        ],
        tone: 0,
        opacity: 0.18,
      },
    ],
    [b1x, b1y, b2x, b2y],
  )
  const vectors = useMemo(
    () => [
      { from: [0, 0] as [number, number], to: [b1x, b1y] as [number, number], slot: 0, label: 'b₁' },
      { from: [0, 0] as [number, number], to: [b2x, b2y] as [number, number], slot: 1, label: 'b₂' },
    ],
    [b1x, b1y, b2x, b2y],
  )

  const vx = useAxis({ label: 'feature 1', range: [-1.6, 1.6] })
  const vy = useAxis({ label: 'feature 2', range: [-1.6, 1.6], equal: vx })
  const sx = useAxis({ label: 'subset S', categories: SUBSETS })
  const sy = useAxis({ label: 'P(Y = S)', range: [0, 1] })
  return (
    <Figure
      title="Probability as squared volume"
      state={state}
      caption="Two items, each a vector: its length is the item's quality and its direction its features. Drag either tip. A determinantal point process with L = BᵀB gives the pair {1, 2} probability proportional to the squared area of the shaded parallelogram, so two similar items (nearly parallel vectors) rarely appear together however good each is. The second bars are independent items with the same inclusion probabilities: the pair is always less likely under the DPP, never more."
      readouts={
        <>
          <Readout label="cos angle" value={formatNumber(cos)} />
          <Readout label="det L = area²" value={formatNumber(area * area)} />
          <Readout label="P(both) DPP" value={formatNumber(dpp[3])} />
          <Readout label="P(1) P(2)" value={formatNumber(m1 * m2)} />
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2">
        <Plot x={vx} y={vy} height={300}>
          <Shapes shapes={shapes} />
          <Vectors vectors={vectors} />
          <Handle {...state.handle(['b1x', 'b1y'], { label: 'b₁' })} />
          <Handle {...state.handle(['b2x', 'b2y'], { label: 'b₂' })} />
        </Plot>
        <Plot x={sx} y={sy} height={300}>
          <Bars name="DPP" x={LEFT} y={dpp} width={0.38} slot={0} />
          <Bars name="independent, same marginals" x={RIGHT} y={indep} width={0.38} muted />
        </Plot>
      </div>
    </Figure>
  )
}
