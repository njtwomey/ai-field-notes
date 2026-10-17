import { useMemo, useState } from 'react'
import {
  Curve,
  Figure,
  float,
  formatNumber,
  int,
  Player,
  Plot,
  Points,
  Raster,
  Readout,
  Segments,
  useAxis,
  useComputed,
  useFigureState,
} from 'aifn-render'
import { moons } from 'aifn-methods/data'
import { child, stream } from 'aifn-compute/foundation/random'
import { concat, toFlat } from 'aifn-compute/foundation/tensor'
import { trace } from 'aifn-compute/foundation/trace'
import { semiSupervisedSplit } from 'aifn-methods/data/synthetic'
import { featureDropout, inputJitter, selfTrainingSteps } from 'aifn-methods/learning/semi-supervised'
import { centredField, predictionGrid, randomFeatureLogistic } from '../_shared/models'

const N = 200
const N_TEST = 1000
const NX = 41
const NY = 31
const X_RANGE: [number, number] = [-1.5, 2.5]
const Y_RANGE: [number, number] = [-1, 1.5]
const GRID = predictionGrid(X_RANGE, Y_RANGE, NX, NY)
// Teacher: logistic regression on 40 random Fourier features. Students: 80 features at a shorter lengthscale.
const TEACHER = randomFeatureLogistic({ components: 40, lengthscale: 0.6, l2: 0.01, seed: 1 })
const STUDENT = { components: 80, lengthscale: 0.45, l2: 0.01, seed: 2 }
const PLAIN_STUDENT = randomFeatureLogistic(STUDENT)

/** Accuracy of the arg max over rows `from` onwards of row-major two-class probabilities. */
function tailAccuracy(probs: Float64Array, from: number, truth: number[]): number {
  let right = 0
  for (let i = 0; i < truth.length; i++) {
    const r = from + i
    if ((probs[2 * r + 1] > probs[2 * r] ? 1 : 0) === truth[i]) right++
  }
  return right / truth.length
}

/**
 * Noisy Student generations on two moons: a teacher fitted to a few labels pseudo-labels the rest; a larger student is
 * fitted to labels and pseudo-labels with input jitter and feature dropout, and becomes the next teacher. The same
 * student without noise runs beside it.
 */
export function NoisyStudentGenerations() {
  const fig = useFigureState({
    jitter: float(0.15, { min: 0, max: 0.5, step: 0.01, label: 'input noise sd' }),
    dropout: float(0.3, { min: 0, max: 0.8, step: 0.05, label: 'feature dropout rate' }),
    generations: int(5, { min: 1, max: 10, label: 'generations', suggestions: [1, 3, 5, 10] }),
    perClass: int(4, { min: 1, max: 20, label: 'labels per class', suggestions: [2, 4, 8] }),
    seed: int(7, { min: 1, max: 999, label: 'data seed' }),
  })
  const { jitter, dropout, generations, perClass, seed } = fig

  const data = useMemo(() => {
    const d = moons(stream(seed), { n: N, noise: 0.15, spacing: 'random' })
    const test = moons(child(stream(seed), 'test'), { n: N_TEST, noise: 0.15, spacing: 'random' })
    const y = toFlat(d.y)
    const split = semiSupervisedSplit(child(stream(seed), 'split'), y, perClass)
    const xy = toFlat(d.x)
    return {
      x: d.x,
      y,
      split,
      // The grid rows first, then the test set: one prediction pass per generation gives the field and the accuracy.
      evaluate: concat([GRID.points, test.x], 0),
      testY: toFlat(test.y),
      px: y.map((_, i) => xy[2 * i]),
      py: y.map((_, i) => xy[2 * i + 1]),
    }
  }, [seed, perClass])

  const run = useComputed(
    () => {
      const problem = { x: data.x, labels: data.split.labels, classes: 2, truth: data.y, evaluate: data.evaluate }
      const accuracy = (p: Float64Array) => tailAccuracy(p, NX * NY, data.testY)
      const plain = trace(
        selfTrainingSteps(problem, { base: TEACHER, student: PLAIN_STUDENT, threshold: 0 }),
        undefined,
        generations,
        { keep: 'all' },
      ).steps
      const noisy = trace(
        selfTrainingSteps(problem, {
          base: TEACHER,
          student: randomFeatureLogistic({ ...STUDENT, featureNoise: featureDropout(dropout, 2, { mask: 'uniform' }) }),
          augment: jitter > 0 ? inputJitter(jitter, 3) : undefined,
          threshold: 0,
        }),
        undefined,
        generations,
        { keep: 'all' },
      ).steps
      // A run that stops early (its pseudo-labels no longer change) keeps its last accuracy.
      const padded = (acc: number[]) =>
        Array.from({ length: generations + 1 }, (_, g) => acc[Math.min(g, acc.length - 1)])
      return {
        noisy,
        noisyAcc: padded(noisy.map((s) => accuracy(s.evaluateProbs))),
        plainAcc: padded(plain.map((s) => accuracy(s.evaluateProbs))),
        plainStopped: plain[plain.length - 1].t,
      }
    },
    [data, jitter, dropout, generations],
    { mode: 'release' },
  )

  const settings = `${seed}/${perClass}/${jitter}/${dropout}/${generations}`
  const [pos, setPos] = useState({ settings, at: 0 })
  const { noisy, noisyAcc, plainAcc, plainStopped } = run.value
  const at = pos.settings === settings ? Math.min(pos.at, noisy.length - 1) : 0
  const s = noisy[at]

  const xa = useAxis({ label: 'x₁', range: X_RANGE })
  const ya = useAxis({ label: 'x₂', range: Y_RANGE, equal: xa })
  const xb = useAxis({ label: 'generation', range: [0, generations] })
  const yb = useAxis({ label: 'test accuracy', hold: 'union' })
  if (!s) return null

  const U = data.split.unlabelled
  const L = data.split.labelled
  const right = [0, 1].map((k) => U.filter((i) => s.proposed[i] === k && data.y[i] === k))
  const wrong = U.filter((i) => s.proposed[i] >= 0 && s.proposed[i] !== data.y[i])
  const gens = Array.from({ length: generations + 1 }, (_, g) => g)
  const lo = Math.min(...noisyAcc, ...plainAcc)
  const hi = Math.max(...noisyAcc, ...plainAcc)

  return (
    <Figure
      title="Noisy Student, generation by generation"
      state={fig}
      caption="Left: two moons with a few labelled points (black). Generation 0 is the teacher, fitted to the labels alone; each later generation is a larger student fitted to the labels and to the previous model's pseudo-labels on every unlabelled point, with input noise (jittered copies) and dropout on its features, and it then labels the data for the next student. The shading is the current model's P(class 1) − ½ with its boundary in ink; coloured points show the pseudo-labels it hands on, red ones wrong. Right: test accuracy by generation for the noisy student and for the same student trained without noise. Step through the generations with the player. Without noise the student reproduces the teacher's pseudo-labels and the loop stops after one generation; with moderate noise the boundary moves into the gap between the moons; too much noise blurs it."
      controls={
        <Player value={at} onChange={(v) => setPos({ settings, at: v })} count={noisy.length} label="generation" />
      }
      readouts={
        <>
          <Readout label="generation" value={s.t} />
          <Readout label="test accuracy, noisy student" value={formatNumber(noisyAcc[at])} />
          <Readout label="without noise" value={formatNumber(plainAcc[at])} />
          <Readout label="teacher" value={formatNumber(noisyAcc[0])} />
          <Readout label="wrong pseudo-labels" value={formatNumber(s.pseudoLabelError)} />
          <Readout label="noiseless run stops at" value={`generation ${plainStopped}`} />
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
          <Curve name="noisy student" x={gens} y={noisyAcc} slot={0} showPoints stale={run.stale} />
          <Curve name="student without noise" x={gens} y={plainAcc} slot={1} showPoints dashed stale={run.stale} />
          <Segments name="current generation" segments={[{ from: [at, lo - 0.01], to: [at, hi + 0.01] }]} muted />
        </Plot>
      </div>
    </Figure>
  )
}
