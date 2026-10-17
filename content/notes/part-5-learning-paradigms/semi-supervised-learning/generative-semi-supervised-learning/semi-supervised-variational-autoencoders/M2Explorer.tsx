import { useMemo } from 'react'
import {
  Curve,
  Figure,
  float,
  Handle,
  int,
  Player,
  Plot,
  Points,
  Raster,
  Readout,
  Segments,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { stream } from 'aifn-compute/foundation/random'
import { float64Data, fromData, toFlat } from 'aifn-compute/foundation/tensor'
import { live } from 'aifn-compute/foundation/trace'
import { adamRule } from 'aifn-compute/optim/first-order'
import {
  m2ClassProbabilities,
  m2Decode,
  m2Model,
  semiSupervisedVaeTraining,
  type M2Model,
  type M2Params,
  type M2Problem,
} from 'aifn-methods/generative/autoencoders'
import { parallelWaves, semiSupervisedSplit } from 'aifn-methods/data/synthetic'
import { centredField, grid, percent, pointsOf, usePlayerPosition, useSlicedRun } from '../_shared/figure-helpers'

const X_RANGE: [number, number] = [-2.6, 2.6]
const Y_RANGE: [number, number] = [-1.7, 1.7]
const GRID = grid(X_RANGE, Y_RANGE, 53, 35)
const Z_LINE = fromData(
  Float64Array.from({ length: 61 }, (_, i) => -2.5 + i / 12),
  [61, 1],
)
const EVERY = 10

type Problem = M2Problem & { truth: Int32Array; rows: Float64Array; cls: number[] }

type Setup = {
  problem: Problem
  mode: 'm2' | 'classifier'
  steps: number
  rate: number
  classWeight: number
  seed: number
}

type Checkpoint = {
  step: number
  model: M2Model
  params: M2Params
  accuracy: number
  labelled: number
  unlabelled: number
  ce: number
}

function accuracy(model: M2Model, params: M2Params, p: Problem): number {
  const q = m2ClassProbabilities(model, params, p.unlabelledX)
  let right = 0
  p.truth.forEach((c, i) => {
    if ((q[2 * i + 1] > 0.5 ? 1 : 0) === c) right++
  })
  return right / p.truth.length
}

/** Trains M2 (or the classifier alone) and yields a checkpoint every EVERY steps, from step 0. */
function* m2Run(s: Setup): Generator<Checkpoint> {
  const model = m2Model(2, 2)
  const alg = semiSupervisedVaeTraining(s.problem, {
    model,
    mode: s.mode,
    classWeight: s.classWeight,
    optimizer: adamRule({ stepSize: s.rate }),
  })
  const it = live(alg, undefined)
  for (let k = 0; k <= s.steps; k++) {
    const next = it.next()
    if (next.done) return
    const st = next.value.state
    if (k % EVERY === 0 || k === s.steps || st.diverged)
      yield {
        step: k,
        model,
        params: st.params,
        accuracy: accuracy(model, st.params, s.problem),
        labelled: st.labelled,
        unlabelled: st.unlabelled,
        ce: st.classification,
      }
    if (st.diverged) return
  }
}

const decoded = (cp: Checkpoint, y: number) => {
  const v = m2Decode(cp.model, cp.params, Z_LINE, y)
  return { x: Array.from({ length: 61 }, (_, i) => v[2 * i]), y: Array.from({ length: 61 }, (_, i) => v[2 * i + 1]) }
}

/**
 * Kingma et al.'s M2 on two parallel waves with a few labels: the classifier q(y | x) against the same classifier
 * trained on the labels alone, and the decoder's curve x(z, y) for each class.
 */
export function M2Explorer() {
  const state = useFigureState({
    perClass: int(4, { ge: 1, le: 20, label: 'labels per class', suggestions: [1, 2, 4, 10] }),
    classWeight: float(5, { ge: 0, le: 50, label: 'classification weight α / N', suggestions: [0.1, 1, 5, 20] }),
    z: float(0.8, { ge: -2.5, le: 2.5, label: 'code z of the analogy pair', suggestions: [-2, -1, 0, 1, 2] }),
    steps: int(400, { ge: 10, le: 2000, label: 'training steps', suggestions: [200, 400, 800] }),
    rate: float(0.003, { gt: 0, le: 0.05, label: 'Adam step size', suggestions: [0.001, 0.003, 0.01] }),
    seed: int(1, { ge: 1, le: 999, label: 'data seed' }),
  })
  const { perClass, classWeight, z, steps, rate, seed } = state

  const problem = useMemo<Problem>(() => {
    const d = parallelWaves(stream(`gssl/m2/waves/${seed}`), {})
    const y = toFlat(d.y)
    const sp = semiSupervisedSplit(stream(`gssl/m2/split/${seed}`), y, perClass, {
      x: d.x,
      draw: 'global',
      strict: true,
    })
    const cls = Array.from(y)
    sp.unlabelled.forEach((i) => (cls[i] = -1 - y[i]))
    return {
      labelledX: sp.labelledX,
      labelledY: sp.labelledY,
      unlabelledX: sp.unlabelledX,
      classes: 2,
      truth: sp.unlabelledTruth,
      rows: Float64Array.from(float64Data(d.x)),
      cls,
    }
  }, [seed, perClass])

  const setup = useMemo<Setup>(
    () => ({ problem, mode: 'm2', steps, rate, classWeight, seed }),
    [problem, steps, rate, classWeight, seed],
  )
  const baseSetup = useMemo<Setup>(
    () => ({ problem, mode: 'classifier', steps, rate, classWeight: 0, seed }),
    [problem, steps, rate, seed],
  )
  const run = useSlicedRun(setup, m2Run)
  const base = useSlicedRun(baseSetup, m2Run)
  const [pos, setPos] = usePlayerPosition(setup)
  const count = run.checkpoints.length
  const at = Math.min(pos, Math.max(0, count - 1))
  const cp = run.checkpoints[at]
  const baseCp = base.checkpoints[Math.min(at, base.checkpoints.length - 1)]

  const field = useMemo(
    () => (cp ? centredField(m2ClassProbabilities(cp.model, cp.params, GRID.points), GRID) : null),
    [cp],
  )
  const curves = useMemo(() => (cp ? [decoded(cp, 0), decoded(cp, 1)] : null), [cp])
  const pair = useMemo(() => {
    if (!cp) return null
    const zt = fromData(Float64Array.from([z]), [1, 1])
    const a = m2Decode(cp.model, cp.params, zt, 0)
    const b = m2Decode(cp.model, cp.params, zt, 1)
    return { x0: a[0], y0: a[1], x1: b[0], y1: b[1] }
  }, [cp, z])

  const xa = useAxis({ label: 'x₁', range: X_RANGE })
  const ya = useAxis({ label: 'x₂', range: Y_RANGE, equal: xa })

  return (
    <Figure
      title="M2 on two waves"
      state={state}
      defaultSize="L"
      caption={`Two hundred points on two parallel sine waves (the class sets the offset, a shared style variable t the position along the wave), with ${perClass} labelled point${perClass === 1 ? '' : 's'} per class (large marks); small marks are unlabelled points coloured by their true class. M2 has a classifier q(y | x), an encoder q(z | x, y) and a decoder p(x | y, z) with a one-dimensional code z, all with two hidden layers of 32 units, trained with Adam on the labelled bound, the unlabelled bound (summed over both classes, weighted by q) and α times the labelled cross-entropy. The shading is the classifier's q(class 1 | x) − ½ with its boundary in ink. The two coloured curves are the decoder's means for z from −2.5 to 2.5, one per class: the model has learned one wave per label, and z moves along it. The two ink points joined by a segment decode the same z with each label, Kingma et al.'s analogy: the same style, either class. Drag the point on the class-0 curve to move z; its partner on the class-1 curve follows. The baseline is the same classifier network trained on the labelled points alone. With α / N = 0.1, Kingma et al.'s setting, the labelled term is too weak on this small problem and the run can attach the classes the wrong way round (try seed 5).`}
      controls={
        <Player
          value={at}
          onChange={setPos}
          count={Math.max(1, count)}
          label="checkpoint"
          format={(a) => `step ${run.checkpoints[a]?.step ?? 0}`}
        />
      }
      readouts={
        <>
          <Readout label="step" value={cp ? `${cp.step}${run.done ? '' : ', training…'}` : '…'} />
          <Readout label="M2 accuracy on unlabelled points" value={percent(cp?.accuracy)} />
          <Readout label="classifier on labels alone" value={percent(baseCp?.accuracy)} />
          <Readout
            label="mean labelled bound"
            value={cp && Number.isFinite(cp.labelled) ? cp.labelled.toFixed(2) : '…'}
          />
          <Readout
            label="mean unlabelled bound"
            value={cp && Number.isFinite(cp.unlabelled) ? cp.unlabelled.toFixed(2) : '…'}
          />
        </>
      }
    >
      <Plot x={xa} y={ya}>
        {field && (
          <Raster
            x={GRID.xs}
            y={GRID.ys}
            z={field}
            scale="diverging"
            range={[-0.5, 0.5]}
            valueLabel="q(class 1 | x) − ½"
            boundary
            fillOpacity={0.45}
            stale={!run.done}
          />
        )}
        {[0, 1].map((c) => {
          const p = pointsOf(problem.rows, problem.cls, -1 - c)
          return (
            p.x.length > 0 && <Points key={`u${c}`} name={`class ${c}, unlabelled`} x={p.x} y={p.y} slot={c} thin />
          )
        })}
        {[0, 1].map((c) => {
          const p = pointsOf(problem.rows, problem.cls, c)
          return (
            p.x.length > 0 && (
              <Points key={`l${c}`} name={`class ${c}, labelled`} x={p.x} y={p.y} slot={c} emphasis size={11} />
            )
          )
        })}
        {curves?.map((c, k) => (
          <Curve key={`d${k}`} name={`decoder, y = ${k}`} x={c.x} y={c.y} slot={k} />
        ))}
        {pair && (
          <Segments
            name={`same z = ${z.toFixed(1)}, both labels`}
            segments={[{ from: [pair.x0, pair.y0], to: [pair.x1, pair.y1] }]}
            emphasis
            width={2}
          />
        )}
        {pair && <Points name="decoded pair" x={[pair.x0, pair.x1]} y={[pair.y0, pair.y1]} emphasis size={9} />}
        {pair && curves && (
          <Handle
            kind="point"
            at={[pair.x0, pair.y0]}
            slot={0}
            label="z"
            onDrag={([px, py]) => {
              // Snap to the nearest decoded point of the class-0 curve and take its code.
              const c = curves[0]
              let best = 0
              for (let i = 1; i < c.x.length; i++)
                if ((c.x[i] - px) ** 2 + (c.y[i] - py) ** 2 < (c.x[best] - px) ** 2 + (c.y[best] - py) ** 2) best = i
              state.set('z', Math.round((-2.5 + best / 12) * 100) / 100)
            }}
          />
        )}
      </Plot>
    </Figure>
  )
}
