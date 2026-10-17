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
import { selfTrainingSteps } from 'aifn-methods/learning/semi-supervised'
import { semiSupervisedSplit } from 'aifn-methods/data/synthetic'
import { centredField, predictionGrid, randomFeatureLogistic } from '../_shared/models'

const N = 300
const NX = 41
const NY = 31
const X_RANGE: [number, number] = [-1.5, 2.5]
const Y_RANGE: [number, number] = [-1, 1.5]
const GRID = predictionGrid(X_RANGE, Y_RANGE, NX, NY)
const SWEEP = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.925, 0.95, 0.975, 0.99]
// A smooth nonlinear classifier: logistic regression on 40 fixed random Fourier features (lengthscale 0.6).
const BASE = randomFeatureLogistic({ components: 40, lengthscale: 0.6, l2: 0.01, seed: 1 })

/**
 * Pseudo-labelling on two moons with a few labels per class: the confidence threshold against the number and the
 * error rate of the pseudo-labels, and the accuracy the run reaches.
 */
export function ThresholdSweep() {
  const fig = useFigureState({
    tau: float(0.9, { min: 0.5, max: 0.995, step: 0.005, label: 'threshold τ', onChart: true }),
    perClass: int(4, { min: 1, max: 30, label: 'labels per class', suggestions: [1, 2, 4, 8, 16] }),
    rounds: int(10, { min: 1, max: 40, label: 'rounds', suggestions: [1, 5, 10, 20] }),
    seed: int(3, { min: 1, max: 999, label: 'data seed' }),
  })
  const { tau, perClass, rounds, seed } = fig

  const data = useMemo(() => {
    const d = moons(stream(seed), { n: N, noise: 0.12, spacing: 'random' })
    const y = toFlat(d.y)
    const split = semiSupervisedSplit(child(stream(seed), 'split'), y, perClass)
    const xy = toFlat(d.x)
    return { x: d.x, y, split, px: y.map((_, i) => xy[2 * i]), py: y.map((_, i) => xy[2 * i + 1]) }
  }, [seed, perClass])

  const problem = useMemo(() => ({ x: data.x, labels: data.split.labels, classes: 2, truth: data.y }), [data])

  // The run at the chosen threshold, with the grid probabilities of every round.
  const run = useComputed(
    () =>
      trace(
        selfTrainingSteps({ ...problem, evaluate: GRID.points }, { base: BASE, threshold: tau }),
        undefined,
        rounds,
        {
          keep: 'all',
        },
      ).steps,
    [problem, tau, rounds],
    { mode: 'release' },
  )

  // The sweep: round-0 pseudo-labels and the accuracy after the full run, for each threshold.
  const sweep = useComputed(
    () =>
      SWEEP.map((t) => {
        const tr = trace(selfTrainingSteps(problem, { base: BASE, threshold: t }), undefined, rounds, { keep: 'all' })
        const first = tr.steps[0]
        return {
          share: first.proposedCount / data.split.unlabelled.length,
          error: first.proposedCount > 0 ? first.pseudoLabelError : 0,
          accuracy: tr.final.unlabelledAccuracy,
          baseline: first.unlabelledAccuracy,
        }
      }),
    [problem, rounds, data],
    { mode: 'release' },
  )

  const settings = `${seed}/${perClass}/${rounds}/${tau}`
  const [pos, setPos] = useState({ settings, at: 0 })
  const states = run.value
  const at = pos.settings === settings ? Math.min(pos.at, states.length - 1) : 0
  const s = states[at]

  const xa = useAxis({ label: 'x₁', range: X_RANGE })
  const ya = useAxis({ label: 'x₂', range: Y_RANGE, equal: xa })
  const xb = useAxis({ label: 'threshold τ', range: [0.5, 1] })
  const yb = useAxis({ label: 'fraction', range: [0, 1.02] })
  if (!s) return null

  const U = data.split.unlabelled
  const right = [0, 1].map((k) => U.filter((i) => s.proposed[i] === k && data.y[i] === k))
  const wrong = U.filter((i) => s.proposed[i] >= 0 && s.proposed[i] !== data.y[i])
  const rest = U.filter((i) => s.proposed[i] < 0)
  const L = data.split.labelled
  const sw = sweep.value
  const baseline = sw[0]?.baseline ?? NaN

  return (
    <Figure
      title="The confidence threshold of pseudo-labelling"
      state={fig}
      caption="Left: two moons with a few labelled points per class (black). The shading is the model's P(class 1) − ½ in the current round, with its boundary in ink. Coloured points carry a pseudo-label for the next round, red ones a wrong pseudo-label; grey points are below the threshold. Right: for each threshold, the share of unlabelled points pseudo-labelled in round 0, the share of those that are wrong, and the accuracy on the unlabelled points after the last round; the dashed line is the model fitted to the labels alone. Drag the vertical line to set τ and step through the rounds with the player. At τ = 0.5 every point is pseudo-labelled at once, mistakes included, and the refit reproduces them; a threshold near 0.9 admits only correct labels at first and lets the boundary move in later rounds; near 0.99 almost nothing qualifies."
      controls={<Player value={at} onChange={(v) => setPos({ settings, at: v })} count={states.length} label="round" />}
      readouts={
        <>
          <Readout label="round" value={s.t} />
          <Readout label="pseudo-labels" value={s.proposedCount} />
          <Readout label="wrong pseudo-labels" value={formatNumber(s.pseudoLabelError)} />
          <Readout label="accuracy on unlabelled" value={formatNumber(s.unlabelledAccuracy)} />
          <Readout label="labels alone" value={formatNumber(baseline)} />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <Plot x={xa} y={ya}>
          <Raster
            x={GRID.gx}
            y={GRID.gy}
            z={centredField(s.evaluateProbs, NX, NY)}
            scale="diverging"
            range={[-0.5, 0.5]}
            valueLabel="P(class 1) − ½"
            boundary
            fillOpacity={0.5}
            stale={run.stale}
          />
          {rest.length > 0 && (
            <Points name="below threshold" x={rest.map((i) => data.px[i])} y={rest.map((i) => data.py[i])} muted thin />
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
          <Curve
            name="share pseudo-labelled, round 0"
            x={SWEEP}
            y={sw.map((v) => v.share)}
            slot={0}
            showPoints
            stale={sweep.stale}
          />
          <Curve
            name="wrong among them, round 0"
            x={SWEEP}
            y={sw.map((v) => v.error)}
            slot={1}
            showPoints
            stale={sweep.stale}
          />
          <Curve
            name="accuracy after the last round"
            x={SWEEP}
            y={sw.map((v) => v.accuracy)}
            slot={2}
            showPoints
            stale={sweep.stale}
          />
          {Number.isFinite(baseline) && (
            <Curve name="labels alone" x={[0.5, 1]} y={[baseline, baseline]} dashed muted />
          )}
          <Handle
            kind="x"
            at={tau}
            label="τ"
            onDrag={(v) => fig.set('tau', Math.round(Math.min(0.995, Math.max(0.5, v)) * 200) / 200)}
          />
        </Plot>
      </div>
    </Figure>
  )
}
