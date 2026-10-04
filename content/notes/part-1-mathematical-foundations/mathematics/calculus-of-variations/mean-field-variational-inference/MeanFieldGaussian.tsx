import { useMemo, useState } from 'react'
import {
  Figure,
  formatNumber,
  Handle,
  int,
  Plot,
  Readout,
  seriesLayers,
  type SeriesSpec,
  slider,
  useAxis,
  useFigureState,
  type Vec2,
} from 'aifn-render'
import { linspace, toFlat } from 'aifn/foundation/tensor'

const RANGE: [number, number] = [-3, 3]
const ANGLES = toFlat(linspace(0, 2 * Math.PI, 97))

/** The set {z : (z − c)ᵀ Σ⁻¹ (z − c) = k²} for Σ = [[a, b], [b, d]], via its Cholesky factor. */
function ellipse(c: Vec2, a: number, b: number, d: number, k: number) {
  const l11 = Math.sqrt(a)
  const l21 = b / l11
  const l22 = Math.sqrt(Math.max(d - l21 * l21, 0))
  return {
    x: ANGLES.map((t) => c[0] + k * l11 * Math.cos(t)),
    y: ANGLES.map((t) => c[1] + k * (l21 * Math.cos(t) + l22 * Math.sin(t))),
  }
}

/**
 * Mean-field variational inference for a correlated Gaussian target N(0, Σ) with unit variances and correlation ρ.
 * Coordinate ascent updates one factor mean at a time; each factor's variance is 1 − ρ², the conditional variance.
 */
export function MeanFieldGaussian() {
  const state = useFigureState({
    rho: slider(-0.95, 0.95, 0.8, { step: 0.05, label: 'correlation ρ' }),
    sweeps: int(3, { min: 0, max: 12, step: 1, label: 'sweeps' }),
  })
  const [start, setStart] = useState<Vec2>([2.2, -1.6])
  const r = state.rho
  const v = 1 - r * r

  // Half-steps of coordinate ascent: m₁ ← ρ m₂, then m₂ ← ρ m₁ (the optimal factor means for this target).
  const path = useMemo(() => {
    const points: Vec2[] = [start]
    let [m1, m2] = start
    for (let i = 0; i < state.sweeps; i++) {
      m1 = r * m2
      points.push([m1, m2])
      m2 = r * m1
      points.push([m1, m2])
    }
    return points
  }, [start, state.sweeps, r])
  const current = path[path.length - 1]
  // KL(q ‖ p) for q = N(m, (1 − ρ²) I) and p = N(0, Σ): the mean term plus −½ log(1 − ρ²).
  const precisionQuadratic = (current[0] ** 2 - 2 * r * current[0] * current[1] + current[1] ** 2) / v
  const kl = 0.5 * precisionQuadratic - 0.5 * Math.log(v)

  const series: SeriesSpec[] = useMemo(() => {
    const target = [1, 2].map((k): SeriesSpec => ({
      name: 'target p',
      type: 'line',
      ...ellipse([0, 0], 1, r, 1, k),
      slot: 0,
      dashed: k === 2,
    }))
    const forward = ellipse([0, 0], 1, 0, 1, 1)
    const reverse = ellipse(current, v, 0, v, 1)
    return [
      ...target,
      { name: 'forward-KL fit (matches marginals)', type: 'line', ...forward, slot: 2, dashed: true },
      { name: 'mean-field q (reverse KL)', type: 'line', ...reverse, slot: 1 },
      { name: 'coordinate ascent path', type: 'line', x: path.map((p) => p[0]), y: path.map((p) => p[1]), muted: true },
    ]
  }, [r, v, current, path])

  const xAxis = useAxis({ label: 'z₁', range: RANGE })
  const yAxis = useAxis({ label: 'z₂', range: RANGE, equal: xAxis })
  return (
    <Figure
      title="Mean-field fit to a correlated Gaussian"
      state={state}
      caption="Blue: the target's one- and two-standard-deviation contours. Orange: the factorised q, one standard deviation, after the chosen number of coordinate-ascent sweeps from the draggable start point; the grey path alternates between updating q₁ and q₂. Each factor has variance 1 − ρ², the target's conditional variance, so q is too narrow along the correlated direction. The dashed green circle is the factorised fit that minimises the forward KL instead: it matches the marginal variances of 1 and is too wide across it."

      readouts={
        <>
          <Readout label="factor variance 1 − ρ²" value={formatNumber(v)} />
          <Readout label="true marginal variance" value="1" />
          <Readout label="mean of q" value={`(${formatNumber(current[0])}, ${formatNumber(current[1])})`} />
          <Readout label="KL(q ‖ p)" value={formatNumber(kl)} />
          <Readout label="KL at convergence, −½ log(1 − ρ²)" value={formatNumber(-0.5 * Math.log(v))} />
        </>
      }
    >
      <div className="mx-auto w-full max-w-xl">
        <Plot x={xAxis} y={yAxis}>
          {seriesLayers(series)}
          <Handle
            kind="point"
            at={start}
            label="start"
            onDrag={([x, y]) => setStart([Math.max(-2.8, Math.min(2.8, x)), Math.max(-2.8, Math.min(2.8, y))])}
          />
        </Plot>
      </div>
    </Figure>
  )
}
