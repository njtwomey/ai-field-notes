import { useMemo, useState } from 'react'
import {
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Player,
  Plot,
  Points,
  Readout,
  Segments,
  useAxis,
  useComputed,
  useFigureState,
} from 'aifn-render'
import { blobs } from 'aifn-methods/data'
import { logisticRegression } from 'aifn-methods/learning/generalised/glm'
import { stream } from 'aifn-compute/foundation/random'
import { concat, linspace, tensor, toFlat } from 'aifn-compute/foundation/tensor'
import { trace } from 'aifn-compute/foundation/trace'
import { selfTrainingSteps } from 'aifn-methods/learning/semi-supervised'

const X_RANGE: [number, number] = [-5.5, 5.5]
const GRID = toFlat(linspace(X_RANGE[0], X_RANGE[1], 221))
const LABELLED = [
  [-0.5, 0],
  [2.5, 1],
] as const

/** Where a curve of P(y = 1 | x) on the grid crosses ½ (linear interpolation; NaN if it never does). */
function crossing(p1: number[]): number {
  for (let i = 1; i < p1.length; i++)
    if ((p1[i - 1] - 0.5) * (p1[i] - 0.5) <= 0 && p1[i] !== p1[i - 1])
      return GRID[i - 1] + ((0.5 - p1[i - 1]) / (p1[i] - p1[i - 1])) * (GRID[i] - GRID[i - 1])
  return NaN
}

/**
 * Self-training of a logistic regression in one dimension, round by round: 1000 unlabelled points from two unit
 * Gaussians at ∓2 and two labelled points, as in the note's worked example.
 */
export function SelfTrainingRounds() {
  const fig = useFigureState({
    tau: float(0.9, { min: 0.5, max: 0.999, step: 0.001, label: 'threshold τ', onChart: true }),
    seed: int(4, { min: 1, max: 999, label: 'data seed' }),
  })
  const { tau, seed } = fig

  const data = useMemo(() => {
    const d = blobs(stream(seed), { n: 1000, centers: [[-2], [2]], sd: 1 })
    const x = concat([tensor(LABELLED.map(([v]) => [v])), d.x], 0)
    const truth = [...LABELLED.map(([, y]) => y), ...toFlat(d.y)]
    return { x, xs: toFlat(x), truth, labels: truth.map((v, i) => (i < LABELLED.length ? v : -1)) }
  }, [seed])

  const run = useComputed(
    () => {
      const alg = selfTrainingSteps(
        { x: data.x, labels: data.labels, classes: 2, truth: data.truth, evaluate: tensor(GRID.map((v) => [v])) },
        { base: logisticRegression({ l2: 1 }), threshold: tau },
      )
      return trace(alg, undefined, 40, { keep: 'all' }).steps.map((s) => {
        const p1 = GRID.map((_, i) => s.evaluateProbs[2 * i + 1])
        return { state: s, p1, boundary: crossing(p1) }
      })
    },
    [data, tau],
    { mode: 'release' },
  )
  const rounds = run.value

  const settings = `${seed}/${tau}`
  const [pos, setPos] = useState({ settings, at: 0 })
  const at = pos.settings === settings ? Math.min(pos.at, rounds.length - 1) : 0
  const r = rounds[at]

  const xAxis = useAxis({ label: 'x', range: X_RANGE })
  const yAxis = useAxis({ label: 'P(y = 1 | x)', range: [-0.05, 1.05] })
  if (!r) return null
  const s = r.state
  const p1At = (i: number) => s.probs[2 * i + 1]
  const groups = [0, 1].map((k) => {
    const rows = data.xs.map((_, i) => i).filter((i) => i >= LABELLED.length && s.proposed[i] === k)
    return { x: rows.map((i) => data.xs[i]), y: rows.map(p1At) }
  })
  const rest = data.xs.map((_, i) => i).filter((i) => i >= LABELLED.length && s.proposed[i] < 0)

  return (
    <Figure
      title="Self-training round by round"
      state={fig}
      caption="Each point is an unlabelled x placed at the current model's P(y = 1 | x), so the points lie on the fitted curve. Points above the threshold τ or below 1 − τ (the two horizontal lines) are pseudo-labelled for the next round; the rest stay unlabelled. The two black points are the labelled points. Drag the upper line to change τ, and step through the rounds with the player. With τ = 0.9 the boundary first moves away from the Bayes boundary x = 0 (grey line) and then settles near it. With τ close to 1 no point qualifies and the model never changes; with τ near 0.5 nearly every point is pseudo-labelled at once."
      controls={<Player value={at} onChange={(v) => setPos({ settings, at: v })} count={rounds.length} label="round" />}
      readouts={
        <>
          <Readout label="round" value={s.t} />
          <Readout label="boundary" value={formatNumber(r.boundary)} />
          <Readout label="accuracy on unlabelled" value={formatNumber(s.unlabelledAccuracy)} />
          <Readout label="pseudo-labels (class 0 / 1)" value={s.proposedPerClass.join(' / ')} />
          <Readout label="wrong pseudo-labels" value={formatNumber(s.pseudoLabelError)} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={360}>
        <Segments name="Bayes boundary" segments={[{ from: [0, -0.05], to: [0, 1.05] }]} muted />
        <Segments
          name="threshold 1 − τ"
          segments={[{ from: [X_RANGE[0], 1 - tau], to: [X_RANGE[1], 1 - tau] }]}
          emphasis
        />
        {Number.isFinite(r.boundary) && (
          <Segments
            name="model boundary"
            segments={[{ from: [r.boundary, -0.05], to: [r.boundary, 1.05] }]}
            emphasis
            width={2}
          />
        )}
        <Curve name="fitted P(y = 1 | x)" x={GRID} y={r.p1} slot={2} stale={run.stale} />
        {rest.length > 0 && (
          <Points name="not pseudo-labelled" x={rest.map((i) => data.xs[i])} y={rest.map(p1At)} muted thin />
        )}
        {groups[0].x.length > 0 && <Points name="pseudo-label 0" x={groups[0].x} y={groups[0].y} slot={0} thin />}
        {groups[1].x.length > 0 && <Points name="pseudo-label 1" x={groups[1].x} y={groups[1].y} slot={1} thin />}
        <Points name="labelled" x={LABELLED.map(([v]) => v)} y={LABELLED.map(([, y]) => y)} size={10} emphasis />
        <Handle
          kind="y"
          at={tau}
          label="threshold τ"
          onDrag={(v) => fig.set('tau', Math.round(Math.min(0.999, Math.max(0.5, v)) * 1000) / 1000)}
        />
      </Plot>
    </Figure>
  )
}
