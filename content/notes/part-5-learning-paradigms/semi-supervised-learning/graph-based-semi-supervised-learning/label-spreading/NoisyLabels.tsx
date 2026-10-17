import { useMemo } from 'react'
import {
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Plot,
  Plots,
  Points,
  Readout,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { moons } from 'aifn-methods/data'
import { harmonicLabels, labelSpreading, pointAffinity } from 'aifn-compute/graph/propagation'
import { child, stream } from 'aifn-compute/foundation/random'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { semiSupervisedSplit } from 'aifn-methods/data/synthetic'

const N = 200
const SIGMA = 0.2
const K = 10
const X_RANGE: [number, number] = [-1.5, 2.5]
const Y_RANGE: [number, number] = [-1, 1.5]
const ALPHAS = [0.05, 0.2, 0.4, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.96, 0.98, 0.99, 0.995, 0.999]

/** The share of the points `rows` whose predicted class equals the true one (0 for no rows). */
function accuracyOf(pred: number[], truth: number[], rows: readonly number[]): number {
  return rows.length ? rows.filter((i) => pred[i] === truth[i]).length / rows.length : 0
}

/**
 * Harmonic label propagation against label spreading on two moons when some of the given labels are wrong: the
 * harmonic solution keeps every given label, label spreading may overrule one when α is near 1.
 */
export function NoisyLabels() {
  const fig = useFigureState({
    alpha: float(0.99, { min: 0.01, max: 0.999, step: 0.001, label: 'α', onChart: true }),
    perClass: int(10, { min: 1, max: 30, label: 'labels per class', suggestions: [3, 10, 20] }),
    flips: int(4, { min: 0, max: 10, label: 'wrong labels', suggestions: [0, 2, 4, 8] }),
    seed: int(2, { min: 1, max: 999, label: 'data seed' }),
  })
  const { alpha, perClass, flips, seed } = fig

  const data = useMemo(() => {
    const d = moons(stream(seed), { n: N, noise: 0.1, spacing: 'random' })
    const y = toFlat(d.y)
    const xy = toFlat(d.x)
    const W = pointAffinity(d.x, { gamma: 1 / (2 * SIGMA * SIGMA), neighbours: K })
    return { y, W, px: y.map((_, i) => xy[2 * i]), py: y.map((_, i) => xy[2 * i + 1]) }
  }, [seed])

  // The given labels: perClass per class, then `flips` of them (spread over the labelled list) set to the other class.
  const given = useMemo(() => {
    const split = semiSupervisedSplit(child(stream(seed), 'split'), data.y, perClass)
    const labels = Array.from(split.labels)
    const count = Math.min(flips, split.labelled.length)
    const stride = count > 0 ? Math.floor(split.labelled.length / count) : 1
    const flipped = Array.from({ length: count }, (_, r) => split.labelled[r * stride])
    for (const i of flipped) labels[i] = 1 - labels[i]
    return { labels, labelled: split.labelled, unlabelled: split.unlabelled, flipped }
  }, [data, seed, perClass, flips])

  const accuracy = (pred: number[]) => accuracyOf(pred, data.y, given.unlabelled)
  const corrected = (pred: number[]) => given.flipped.filter((i) => pred[i] === data.y[i]).length

  const harmonic = useMemo(() => toFlat(harmonicLabels(data.W, given.labels, { classes: 2 }).labels), [data, given])
  const sweep = useMemo(
    () =>
      ALPHAS.map((a) => {
        const p = toFlat(labelSpreading(data.W, given.labels, { alpha: a, classes: 2 }).labels)
        return accuracyOf(p, data.y, given.unlabelled)
      }),
    [data, given],
  )
  const spread = useMemo(
    () => toFlat(labelSpreading(data.W, given.labels, { alpha, classes: 2 }).labels),
    [data, given, alpha],
  )

  const xa = useAxis({ label: 'x₁', range: X_RANGE })
  const ya = useAxis({ label: 'x₂', range: Y_RANGE, equal: xa })
  const xb = useAxis({ label: 'α', range: [0, 1] })
  const yb = useAxis({ label: 'accuracy on unlabelled points', range: [0.5, 1.01] })

  const panel = (pred: number[], title: string) => {
    const byClass = [0, 1].map((k) => given.unlabelled.filter((i) => pred[i] === k))
    const right = given.labelled.filter((i) => !given.flipped.includes(i))
    const fixedNow = given.flipped.filter((i) => pred[i] === data.y[i])
    const stillWrong = given.flipped.filter((i) => pred[i] !== data.y[i])
    return (
      <Plot x={xa} y={ya} title={title}>
        {byClass.map(
          (rows, k) =>
            rows.length > 0 && (
              <Points
                key={k}
                name={`predicted ${k}`}
                x={rows.map((i) => data.px[i])}
                y={rows.map((i) => data.py[i])}
                slot={k}
                thin
              />
            ),
        )}
        {right.length > 0 && (
          <Points
            name="correct given label"
            x={right.map((i) => data.px[i])}
            y={right.map((i) => data.py[i])}
            emphasis
            size={8}
          />
        )}
        {fixedNow.length > 0 && (
          <Points
            name="wrong label, overruled"
            x={fixedNow.map((i) => data.px[i])}
            y={fixedNow.map((i) => data.py[i])}
            tone="success"
            size={11}
          />
        )}
        {stillWrong.length > 0 && (
          <Points
            name="wrong label, kept"
            x={stillWrong.map((i) => data.px[i])}
            y={stillWrong.map((i) => data.py[i])}
            tone="destructive"
            size={11}
          />
        )}
      </Plot>
    )
  }

  const harmonicAccuracy = accuracy(harmonic)
  return (
    <Figure
      title="Label spreading overrules wrong labels"
      state={fig}
      caption={`Two hundred points on two moons, joined by a ${K}-nearest-neighbour graph with Gaussian weights (σ = ${SIGMA}). A few labels per class are given, and some of them are set to the wrong class (large red or green marks). Left: the harmonic solution clamps every given label, so each wrong label claims the points around it. Middle: label spreading with the chosen α; a wrong label turns green when the scores spread from its correctly labelled neighbours outweigh its own label. Right: accuracy on the unlabelled points against α, with the harmonic solution dashed. Drag the vertical line to set α. With α near 0.99, as Zhou et al. used, the wrong labels are overruled; small α keeps every label and spreads the scores only a few steps.`}
      readouts={
        <>
          <Readout label="harmonic accuracy" value={formatNumber(harmonicAccuracy)} />
          <Readout label="spreading accuracy" value={formatNumber(accuracy(spread))} />
          <Readout label="wrong labels overruled" value={`${corrected(spread)} of ${given.flipped.length}`} />
        </>
      }
    >
      <Plots cols={3}>
        {panel(harmonic, 'harmonic (labels clamped)')}
        {panel(spread, `label spreading, α = ${alpha}`)}
        <Plot x={xb} y={yb} title="accuracy against α">
          <Curve name="label spreading" x={ALPHAS} y={sweep} slot={2} showPoints />
          <Curve name="harmonic" x={[0, 1]} y={[harmonicAccuracy, harmonicAccuracy]} dashed muted />
          <Handle
            kind="x"
            at={alpha}
            label="α"
            onDrag={(v) => fig.set('alpha', Math.round(Math.min(0.999, Math.max(0.01, v)) * 1000) / 1000)}
          />
        </Plot>
      </Plots>
    </Figure>
  )
}
