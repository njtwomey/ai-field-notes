import { useMemo, useState } from 'react'
import {
  choice,
  Contours,
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
  useAxis,
  useFigureState,
} from 'aifn-render'
import { child, stream } from 'aifn-compute/foundation/random'
import { float64Data, fromData, tensor, toArray } from 'aifn-compute/foundation/tensor'
import { live } from 'aifn-compute/foundation/trace'
import { covarianceEllipse } from 'aifn-compute/numerics/geometry'
import { multivariateNormal } from 'aifn-compute/probability/samplers'
import {
  classPosterior,
  semiSupervisedMixtureSteps,
  type SemiSupervisedMixtureState,
} from 'aifn-methods/unsupervised/clustering'
import { semiSupervisedSplit } from 'aifn-methods/data/synthetic'
import { centredField, grid, percent, pointsOf, usePlayerPosition } from '../_shared/figure-helpers'

const X_RANGE: [number, number] = [-3.5, 8.5]
const Y_RANGE: [number, number] = [-3, 3.5]
const GRID = grid(X_RANGE, Y_RANGE, 73, 40)
const SHARED = [
  [1, 0.4],
  [0.4, 0.6],
]
const MAX_STEPS = 60

const DATA = [
  { value: 'right', label: 'model right: two Gaussians, one covariance' },
  { value: 'wrong', label: 'model wrong: class 1 has a second, far cluster' },
] as const
type DataKey = (typeof DATA)[number]['value']

/**
 * The two data sets: class 0 is 150 points from N((0, 0), Σ). In the first, class 1 is 150 points from N(c, Σ) with
 * the same Σ, so the tied-covariance model is correct. In the second, class 1 is 100 tight points around c and 50
 * around c + (3.5, 0), a skewed class that one Gaussian describes badly.
 */
function makeData(key: DataKey, centre: readonly [number, number], seed: number) {
  const s = stream(`gssl/em/${seed}`)
  const draw = (tag: string, mean: readonly number[], cov: number[][], n: number) =>
    float64Data(multivariateNormal(child(s, tag), [...mean], { covariance: tensor(cov) }, { shape: [n] }))
  const parts =
    key === 'right'
      ? [
          { c: 0, x: draw('a', [0, 0], SHARED, 150) },
          { c: 1, x: draw('b', centre, SHARED, 150) },
        ]
      : [
          { c: 0, x: draw('a', [0, 0], SHARED, 150) },
          {
            c: 1,
            x: draw(
              'b',
              centre,
              [
                [0.15, 0],
                [0, 0.15],
              ],
              100,
            ),
          },
          {
            c: 1,
            x: draw(
              'c',
              [centre[0] + 3.5, centre[1]],
              [
                [0.3, 0],
                [0, 0.3],
              ],
              50,
            ),
          },
        ]
  const rows = new Float64Array(parts.reduce((a, p) => a + p.x.length, 0))
  const y: number[] = []
  let o = 0
  for (const p of parts) {
    rows.set(p.x, o)
    o += p.x.length
    for (let i = 0; i < p.x.length / 2; i++) y.push(p.c)
  }
  return { rows, y, x: fromData(rows, [y.length, 2]) }
}

/**
 * Semi-supervised EM for two Gaussian classes with a shared covariance, stepped from the labelled-only fit, on a data
 * set where the model is right and one where it is wrong.
 */
export function SemiSupervisedEm() {
  const state = useFigureState({
    data: choice(DATA, 'right', { label: 'data' }),
    perClass: int(2, { ge: 1, le: 20, label: 'labels per class', suggestions: [1, 2, 5, 10] }),
    lambda: float(1, { ge: 0, le: 1, label: 'unlabelled weight λ', suggestions: [0, 0.1, 0.5, 1] }),
    seed: int(1, { ge: 1, le: 999, label: 'data seed' }),
  })
  const key = state.data as DataKey
  const { perClass, lambda, seed } = state
  const [centre, setCentre] = useState<[number, number]>([2.5, 0.5])

  const data = useMemo(() => makeData(key, centre, seed), [key, centre, seed])
  const split = useMemo(
    () =>
      semiSupervisedSplit(stream(`gssl/em/split/${seed}`), data.y, perClass, {
        x: data.x,
        draw: 'global',
        strict: true,
      }),
    [data, perClass, seed],
  )
  const labels = useMemo(() => {
    const l = data.y.map(() => -1)
    split.labelled.forEach((i) => (l[i] = data.y[i]))
    return l
  }, [data, split])

  // The whole EM trace, from the labelled-only fit (step 0) until it converges or reaches MAX_STEPS.
  const steps = useMemo(() => {
    const out: SemiSupervisedMixtureState[] = []
    const alg = semiSupervisedMixtureSteps({ x: data.x, labels, classes: 2 }, { unlabelledWeight: lambda })
    for (const { state: s } of live(alg, undefined)) {
      out.push(s)
      if (s.t >= MAX_STEPS) break
    }
    return out
  }, [data, labels, lambda])

  const setup = steps
  const [pos, setPos] = usePlayerPosition(setup)
  const at = Math.min(pos, steps.length - 1)
  const cur = steps[at]

  const accuracy = (s: SemiSupervisedMixtureState) => {
    const p = classPosterior(s, split.unlabelledX)
    let right = 0
    split.unlabelledTruth.forEach((c, i) => {
      if ((p[2 * i + 1] > 0.5 ? 1 : 0) === c) right++
    })
    return right / split.unlabelledTruth.length
  }
  const field = useMemo(() => centredField(classPosterior(cur, GRID.points), GRID), [cur])
  const startField = useMemo(() => centredField(classPosterior(steps[0], GRID.points), GRID), [steps])
  const ellipses = [0, 1].map((k) => {
    const e = covarianceEllipse(
      [cur.means[2 * k], cur.means[2 * k + 1]],
      [
        [cur.covariances[4 * k], cur.covariances[4 * k + 1]],
        [cur.covariances[4 * k + 2], cur.covariances[4 * k + 3]],
      ],
      { k: 2, points: 60 },
    )
    const pts = toArray(e.points) as number[][]
    return { x: pts.map((p) => p[0]), y: pts.map((p) => p[1]) }
  })

  const unlabelledCls = labels.map((l, i) => (l < 0 ? data.y[i] : -1))
  const labelledCls = labels
  const xa = useAxis({ label: 'x₁', range: X_RANGE })
  const ya = useAxis({ label: 'x₂', range: Y_RANGE, equal: xa })

  return (
    <Figure
      title="Semi-supervised EM, step by step"
      state={state}
      defaultSize="L"
      caption={`Two classes in the plane, ${data.y.length} points, of which ${perClass} per class are labelled (large marks); small marks are unlabelled points coloured by their true class, which EM never sees. The model gives each class a Gaussian with one shared covariance. Step 0 is the fit to the labelled points alone; its boundary is the dashed line. Each step is an M-step (refit the means, weights and covariance from all points, an unlabelled point counted with weight λ and split between the classes by its posterior) and an E-step (recompute the posteriors). The shading is the current P(class 1) − ½, with its boundary in ink, and the ellipses enclose two standard deviations of each fitted Gaussian. Drag the handle to move the centre of class 1. With the model right, EM moves the boundary towards the Bayes boundary; with class 1 split into a near and a far cluster, the second Gaussian chases the far cluster, the first swallows the near one, and accuracy falls below the labelled-only fit. Lowering λ limits the damage.`}
      controls={
        <Player value={at} onChange={setPos} count={steps.length} label="EM step" format={(a) => `step ${a}`} />
      }
      readouts={
        <>
          <Readout label="step" value={`${at} of ${steps.length - 1}`} />
          <Readout label="accuracy on unlabelled points" value={percent(accuracy(cur))} />
          <Readout label="labelled-only fit (step 0)" value={percent(accuracy(steps[0]))} />
          <Readout label="objective per point" value={cur.objective.toFixed(3)} />
        </>
      }
    >
      <Plot x={xa} y={ya}>
        <Raster
          x={GRID.xs}
          y={GRID.ys}
          z={field}
          scale="diverging"
          range={[-0.5, 0.5]}
          valueLabel="P(class 1) − ½"
          boundary
          fillOpacity={0.45}
        />
        <Contours
          name="labelled-only boundary"
          x={GRID.xs}
          y={GRID.ys}
          z={startField}
          levels={[0]}
          labels={false}
          dashed
        />
        {[0, 1].map((c) => {
          const p = pointsOf(data.rows, unlabelledCls, c)
          return (
            p.x.length > 0 && <Points key={`u${c}`} name={`class ${c}, unlabelled`} x={p.x} y={p.y} slot={c} thin />
          )
        })}
        {[0, 1].map((c) => {
          const p = pointsOf(data.rows, labelledCls, c)
          return (
            p.x.length > 0 && (
              <Points key={`l${c}`} name={`class ${c}, labelled`} x={p.x} y={p.y} slot={c} emphasis size={11} />
            )
          )
        })}
        {ellipses.map((e, k) => (
          <Curve key={`e${k}`} name={`class ${k} Gaussian, 2 sd`} x={e.x} y={e.y} slot={k} />
        ))}
        <Handle
          kind="point"
          at={centre}
          slot={1}
          label="class 1 centre"
          onDrag={([a, b]) =>
            setCentre([
              Math.max(0.5, Math.min(4.5, Math.round(a * 10) / 10)),
              Math.max(-2, Math.min(2.5, Math.round(b * 10) / 10)),
            ])
          }
        />
      </Plot>
    </Figure>
  )
}
