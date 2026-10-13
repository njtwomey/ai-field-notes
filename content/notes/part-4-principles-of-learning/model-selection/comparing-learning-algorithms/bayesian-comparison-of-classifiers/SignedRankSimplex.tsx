import { useMemo } from 'react'
import {
  Annotation,
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Plot,
  Plots,
  Points,
  Readout,
  Segments,
  useAxis,
  useComputed,
  useFigureState,
} from 'aifn-render'
import { normals, stream } from 'aifn-compute/foundation/random'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { Normal } from 'aifn-compute/probability/distributions'
import { bayesianSignedRankTest, wilcoxonSignedRank } from 'aifn-compute/probability/tests'

const fmt = (v: number) => (Number.isFinite(v) ? formatNumber(Number(v.toPrecision(3))) : '—')
/** Posterior draws of the Dirichlet weights, and how many of them are drawn on the triangle. */
const SAMPLES = 10000
const SHOWN = 3000
const MAX_ROPE = 0.05
const H = Math.sqrt(3) / 2
/**
 * A point (θ_left, θ_rope, θ_right) of the simplex drawn in the plane: B practically better at the bottom left, A at
 * the bottom right, practical equivalence at the top.
 */
const place = (_left: number, rope: number, right: number): [number, number] => [right + rope / 2, H * rope]
const CENTRE = place(1 / 3, 1 / 3, 1 / 3)
const EDGES = [
  { from: place(1, 0, 0), to: place(0, 0, 1) },
  { from: place(0, 0, 1), to: place(0, 1, 0) },
  { from: place(0, 1, 0), to: place(1, 0, 0) },
]
/** The borders between the three regions, where two of the θs are equal and largest. */
const BORDERS = [place(0.5, 0, 0.5), place(0.5, 0.5, 0), place(0, 0.5, 0.5)].map((to) => ({ from: CENTRE, to }))

/**
 * The Bayesian signed-rank test (`bayesianSignedRankTest`) on accuracy differences over q data sets: the differences
 * with a draggable rope, and the posterior draws of (θ_left, θ_rope, θ_right) on the simplex beside the true point.
 */
export function SignedRankSimplex() {
  const state = useFigureState({
    q: int(30, { ge: 5, le: 100, suggestions: [10, 30, 54], label: 'data sets q' }),
    delta: float(0.005, { min: -0.03, max: 0.03, step: 0.001, label: 'mean difference δ, A minus B' }),
    tau: float(0.01, { min: 0.002, max: 0.05, step: 0.001, label: 'spread τ over data sets' }),
    rope: float(0.01, { ge: 0, le: MAX_ROPE, suggestions: [0, 0.01, 0.02], label: 'rope half-width r' }),
    seed: int(1, { ge: 1, le: 999, label: 'draw' }),
  })
  const { q, delta, tau, rope, seed } = state
  const z = useMemo(
    () => Array.from(toFlat(normals(stream(`comparing/signed-rank/${seed}`), [q], delta, tau))),
    [q, delta, tau, seed],
  )
  const run = useComputed(() => {
    const test = bayesianSignedRankTest(z, {
      rope,
      stream: stream(`comparing/signed-rank/${seed}/posterior`),
      samples: SAMPLES,
    })
    const draws = toFlat(test.samples)
    const points = Array.from({ length: SHOWN }, (_, k) => place(draws[3 * k], draws[3 * k + 1], draws[3 * k + 2]))
    return {
      pLeft: test.pLeft,
      pRope: test.pRope,
      pRight: test.pRight,
      x: points.map((p) => p[0]),
      y: points.map((p) => p[1]),
    }
  }, [z, rope, seed])
  const frequentist = useMemo(() => wilcoxonSignedRank(z).pValue, [z])
  // The truth: θ for the distribution the differences are drawn from. The mean of two independent draws from
  // N(δ, τ²) is N(δ, τ²/2).
  const truth = useMemo(() => {
    const pairMean = Normal(delta, tau / Math.SQRT2)
    const left = pairMean.cdf(-rope) as number
    const right = 1 - (pairMean.cdf(rope) as number)
    return { left, rope: 1 - left - right, right }
  }, [delta, tau, rope])
  const truePoint = place(truth.left, truth.rope, truth.right)
  const r = run.value
  const dx = useAxis({ label: 'accuracy difference z_i, A minus B', hold: 'union', key: `${q}/${seed}` })
  const dy = useAxis({ label: 'data set i', range: [0, q + 1], nice: false, integer: true })
  const sx = useAxis({ label: '', range: [-0.05, 1.25], nice: false })
  const sy = useAxis({ label: '', range: [-0.1, H + 0.1], nice: false, equal: sx })
  return (
    <Figure
      title="The Bayesian signed-rank test on the simplex"
      state={state}
      hoverReadout={false}
      caption={`Left: the difference in accuracy between learners A and B on each of q data sets, drawn from a normal distribution with mean δ and standard deviation τ, with the rope [−r, r] between the dashed lines; drag the right line to resize the rope. Right: ${SHOWN} of ${SAMPLES} posterior draws of (θ_left, θ_rope, θ_right) on the triangle, where θ_left is the probability that the mean of two differences falls below −r, θ_rope that it falls inside the rope and θ_right that it falls above r. A corner is certainty about one outcome; the three lines from the centre split the triangle by which θ is largest, and the posterior probability of each outcome is the share of draws in its region. The large point is the true θ of the normal distribution the differences come from. The defaults are a small consistent advantage for A: the frequentist signed-rank test rejects equality, while the Bayesian test finds the two practically equivalent.`}
      readouts={{
        'posterior probability': (
          <>
            <Readout label="B practically better" value={fmt(r.pLeft)} />
            <Readout label="practically equivalent" value={fmt(r.pRope)} />
            <Readout label="A practically better" value={fmt(r.pRight)} />
          </>
        ),
        'true θ (left, rope, right)': (
          <Readout label="from N(δ, τ²)" value={`${fmt(truth.left)}, ${fmt(truth.rope)}, ${fmt(truth.right)}`} />
        ),
        'frequentist signed-rank test': <Readout label="p" value={fmt(frequentist)} />,
      }}
    >
      <Plots cols={2}>
        <Plot x={dx} y={dy} title="differences over data sets">
          <Points name="difference z_i" x={z} y={z.map((_, i) => i + 1)} slot={0} size={5} />
          <Annotation x={-rope} dashed />
          <Annotation x={0} muted />
          <Handle
            kind="x"
            at={rope}
            label="rope"
            onDrag={(v) => state.set('rope', Math.min(MAX_ROPE, Math.max(0, v)))}
          />
        </Plot>
        <Plot x={sx} y={sy} title="posterior on the simplex">
          <Points name="posterior draw" x={r.x} y={r.y} slot={0} dense stale={run.stale} />
          <Segments name="simplex" segments={EDGES} width={1.5} />
          <Segments name="region borders" segments={BORDERS} dashed />
          <Points name="true θ" x={[truePoint[0]]} y={[truePoint[1]]} emphasis size={9} />
          <Annotation at={place(1, 0, 0)} text="B better" />
          <Annotation at={place(0, 0, 1)} text="A better" />
          <Annotation at={place(0, 1, 0)} text="equivalent" />
        </Plot>
      </Plots>
    </Figure>
  )
}
