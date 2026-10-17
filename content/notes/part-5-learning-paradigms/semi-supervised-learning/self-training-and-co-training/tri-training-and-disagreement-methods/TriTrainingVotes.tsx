import { useMemo, useState } from 'react'
import {
  Curve,
  Figure,
  formatNumber,
  int,
  Player,
  Plot,
  Points,
  Raster,
  Readout,
  useAxis,
  useComputed,
  useFigureState,
} from 'aifn-render'
import { moons } from 'aifn-methods/data'
import { child, stream } from 'aifn-compute/foundation/random'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { trace } from 'aifn-compute/foundation/trace'
import { triTrainingSteps } from 'aifn-methods/learning/semi-supervised'
import { semiSupervisedSplit } from 'aifn-methods/data/synthetic'
import { predictionGrid, randomFeatureLogistic } from '../_shared/models'

const N = 300
const NX = 41
const NY = 31
const X_RANGE: [number, number] = [-1.5, 2.5]
const Y_RANGE: [number, number] = [-1, 1.5]
const GRID = predictionGrid(X_RANGE, Y_RANGE, NX, NY)
// Each learner: logistic regression on 40 fixed random Fourier features; diversity comes from the bootstrap samples.
const BASE = randomFeatureLogistic({ components: 40, lengthscale: 0.6, l2: 0.01, seed: 1 })

/**
 * Tri-training on two moons: three learners fitted on bootstrap samples of the labels; each round, a learner is
 * refitted on the points the other two agree on, when the other two's estimated error has fallen enough.
 */
export function TriTrainingVotes() {
  const fig = useFigureState({
    perClass: int(4, { min: 1, max: 20, label: 'labels per class', suggestions: [2, 4, 8] }),
    rounds: int(10, { min: 1, max: 30, label: 'most rounds', suggestions: [5, 10, 30] }),
    seed: int(4, { min: 1, max: 999, label: 'data seed' }),
  })
  const { perClass, rounds, seed } = fig

  const data = useMemo(() => {
    const d = moons(stream(seed), { n: N, noise: 0.15, spacing: 'random' })
    const y = toFlat(d.y)
    const split = semiSupervisedSplit(child(stream(seed), 'split'), y, perClass)
    const xy = toFlat(d.x)
    return { x: d.x, y, split, px: y.map((_, i) => xy[2 * i]), py: y.map((_, i) => xy[2 * i + 1]) }
  }, [seed, perClass])

  const run = useComputed(
    () =>
      trace(
        triTrainingSteps(
          { x: data.x, labels: data.split.labels, classes: 2, truth: data.y, evaluate: GRID.points },
          { base: BASE },
        ),
        undefined,
        rounds,
        { keep: 'all' },
      ).steps,
    [data, rounds],
    { mode: 'release' },
  )

  const settings = `${seed}/${perClass}/${rounds}`
  const [pos, setPos] = useState({ settings, at: 0 })
  const states = run.value
  const at = pos.settings === settings ? Math.min(pos.at, states.length - 1) : 0
  const s = states[at]

  const xa = useAxis({ label: 'x₁', range: X_RANGE })
  const ya = useAxis({ label: 'x₂', range: Y_RANGE, equal: xa })
  const xb = useAxis({ label: 'round', range: [0, Math.max(1, states.length - 1)] })
  const yb = useAxis({ label: 'accuracy on unlabelled', hold: 'union' })
  if (!s) return null

  const U = data.split.unlabelled
  const L = data.split.labelled
  const accuracyOf = (d: Int32Array) => U.filter((i) => d[i] === data.y[i]).length / U.length
  // Votes for class 1 on the grid, centred: −1.5 (all three say 0) to +1.5 (all three say 1).
  const votes = Array.from({ length: NY }, (_, r) =>
    Array.from({ length: NX }, (_, c) => s.evaluateDecisions.reduce((a, d) => a + d[r * NX + c], 0) - 1.5),
  )
  // A point used as a pseudo-label by at least one learner in its last refit.
  const used = U.filter((i) => s.pseudo.some((p) => p[i] >= 0))
  const label = (i: number) => s.pseudo.map((p) => p[i]).find((v) => v >= 0) ?? -1
  const right = [0, 1].map((k) => used.filter((i) => label(i) === k && data.y[i] === k))
  const wrong = used.filter((i) => label(i) !== data.y[i])
  const unused = U.filter((i) => !s.pseudo.some((p) => p[i] >= 0))
  const rx = states.map((st) => st.t)

  return (
    <Figure
      title="Tri-training round by round"
      state={fig}
      caption="Left: the shading counts the three learners voting for class 1, from all three for class 0 (one end) to all three for class 1 (the other); the ink line is the majority boundary. Black points are the labels; coloured points were given to some learner as pseudo-labels because the other two agreed on them, red ones wrongly; grey points were not used. Right: accuracy on the unlabelled points of each learner and of the majority vote. Step through the rounds with the player. With few labels the estimated error of a pair is often 0 on the labelled set; a learner whose pair has error 0 takes every point the pair agrees on, and once every estimate is 0 none can fall further and the loop stops. A pair with a non-zero error, such as h2's in round 1 with data seed 4, admits only a small random subsample."
      controls={<Player value={at} onChange={(v) => setPos({ settings, at: v })} count={states.length} label="round" />}
      readouts={
        <>
          <Readout label="round" value={s.t} />
          <Readout label="majority vote accuracy" value={formatNumber(s.unlabelledAccuracy)} />
          <Readout
            label="refitted this round"
            value={
              s.updated
                .map((u, j) => (u ? `h${j + 1}` : ''))
                .filter(Boolean)
                .join(', ') || 'none'
            }
          />
          <Readout label="e′ of h1, h2, h3" value={s.errorBound.map((e) => formatNumber(e)).join(', ')} />
          <Readout label="l′ of h1, h2, h3" value={s.lastSize.join(', ')} />
          <Readout label="wrong pseudo-labels" value={formatNumber(s.pseudoLabelError)} />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <Plot x={xa} y={ya}>
          <Raster
            x={GRID.gx}
            y={GRID.gy}
            z={votes}
            scale="diverging"
            range={[-1.5, 1.5]}
            scaleTicks={[-1.5, -0.5, 0.5, 1.5]}
            valueLabel="votes for class 1 − 1.5"
            boundary
            fillOpacity={0.5}
            stale={run.stale}
          />
          {unused.length > 0 && (
            <Points name="not used" x={unused.map((i) => data.px[i])} y={unused.map((i) => data.py[i])} muted thin />
          )}
          {right.map(
            (rows, k) =>
              rows.length > 0 && (
                <Points
                  key={k}
                  name={`pseudo-label ${k}`}
                  x={rows.map((i) => data.px[i])}
                  y={rows.map((i) => data.py[i])}
                  slot={k}
                  thin
                />
              ),
          )}
          {wrong.length > 0 && (
            <Points
              name="wrong pseudo-label"
              x={wrong.map((i) => data.px[i])}
              y={wrong.map((i) => data.py[i])}
              tone="destructive"
            />
          )}
          <Points name="labelled" x={L.map((i) => data.px[i])} y={L.map((i) => data.py[i])} emphasis size={9} />
        </Plot>
        <Plot x={xb} y={yb}>
          {[0, 1, 2].map((j) => (
            <Curve
              key={j}
              name={`learner h${j + 1}`}
              x={rx}
              y={states.map((st) => accuracyOf(st.decisions[j]))}
              slot={j}
              thin
              showPoints
            />
          ))}
          <Curve name="majority vote" x={rx} y={states.map((st) => st.unlabelledAccuracy)} emphasis showPoints />
        </Plot>
      </div>
    </Figure>
  )
}
