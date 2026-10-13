import { useMemo } from 'react'
import { stream } from 'aifn-compute/foundation/random'
import { fromRows, tensor, toFlat, toRows } from 'aifn-compute/foundation/tensor'
import { dataset } from 'aifn-compute/learning/estimators'
import { rbf } from 'aifn-compute/learning/kernels'
import { moons } from 'aifn-methods/data/synthetic'
import { supportVectorMachine } from 'aifn-methods/learning/kernel-methods'
import {
  Contours,
  Figure,
  float,
  formatNumber,
  int,
  Plot,
  Points,
  Readout,
  slider,
  useAxis,
  useComputed,
  useFigureState,
} from 'aifn-render'

const TEST = 2000
const GRID = 70
const X_RANGE: [number, number] = [-1.6, 2.6]
const Y_RANGE: [number, number] = [-1.2, 1.7]

const fmt = (v: number) => (Number.isFinite(v) ? formatNumber(Number(v.toPrecision(3))) : '—')
const spaced = ([a, b]: [number, number]) => Array.from({ length: GRID }, (_, i) => a + ((b - a) * i) / (GRID - 1))

/**
 * A kernel SVM is a compression scheme: retraining it on its support vectors alone returns the same classifier, and
 * that classifier is correct on every training point outside the support set. The figure fits the SVM on all the
 * points, refits it on the support vectors, and compares the two.
 */
export function SupportVectorCompression() {
  const state = useFigureState({
    noise: slider(0, 0.4, 0.2, { step: 0.01, label: 'input noise' }),
    C: float(10, {
      min: 0.01,
      max: 1000,
      scale: 'log10',
      suggestions: [0.1, 1, 10, 100, 1000],
      label: 'SVM penalty C',
    }),
    lengthscale: float(0.5, {
      min: 0.1,
      max: 3,
      scale: 'log10',
      suggestions: [0.2, 0.5, 1, 2],
      label: 'RBF lengthscale',
    }),
    m: int(200, { ge: 20, le: 600, suggestions: [50, 200, 500], label: 'training points m' }),
    seed: int(1, { ge: 0, le: 9999, label: 'seed' }),
  })
  const { noise, C, lengthscale, m, seed } = state
  const grid = useMemo(() => {
    const xs = spaced(X_RANGE)
    const ys = spaced(Y_RANGE)
    const rows: number[][] = []
    for (const y of ys) for (const x of xs) rows.push([x, y])
    return { xs, ys, points: fromRows(rows) }
  }, [])
  const run = useComputed(
    () => {
      const train = moons(stream(`compression/${seed}/train`), { n: m, noise })
      const test = moons(stream(`compression/${seed}/test`), { n: TEST, noise })
      const kernel = rbf({ lengthscale })
      const full = supportVectorMachine({ C, kernel }).fit(dataset(train.x, train.y!))
      const kept = toFlat(full.supportVectors)
      const rows = toRows(train.x) as number[][]
      const labels = toFlat(train.y!)
      const refit = supportVectorMachine({ C, kernel }).fit(
        dataset(fromRows(kept.map((i) => rows[i])), tensor(kept.map((i) => labels[i]))),
      )
      const inSet = new Set(kept)
      const testLabels = toFlat(test.y!)
      const fullTest = toFlat(full.decide(test.x))
      const refitTest = toFlat(refit.decide(test.x))
      const refitTrain = toFlat(refit.decide(train.x))
      const fullTrain = toFlat(full.decide(train.x))
      const surface = (scores: number[]) => grid.ys.map((_, i) => scores.slice(i * GRID, (i + 1) * GRID))
      return {
        rows,
        labels,
        kept,
        inSet,
        k: kept.length,
        errorsOutside: labels.filter((y, i) => !inSet.has(i) && refitTrain[i] !== y).length,
        errorsInside: labels.filter((y, i) => inSet.has(i) && fullTrain[i] !== y).length,
        fullTestError: fullTest.filter((v, i) => v !== testLabels[i]).length / TEST,
        refitTestError: refitTest.filter((v, i) => v !== testLabels[i]).length / TEST,
        disagreements: fullTest.filter((v, i) => v !== refitTest[i]).length,
        fullSurface: surface(toFlat(full.score(grid.points))),
        refitSurface: surface(toFlat(refit.score(grid.points))),
      }
    },
    [noise, C, lengthscale, m, seed, grid],
    { mode: 'release' },
  )
  const r = run.value
  const outside = r.rows.filter((_, i) => !r.inSet.has(i))
  const outsideLabels = r.labels.filter((_, i) => !r.inSet.has(i))
  const inside = r.kept.map((i) => r.rows[i])
  const x1 = useAxis({ label: 'x₁', range: X_RANGE })
  const x2 = useAxis({ label: 'x₂', range: Y_RANGE, equal: x1 })
  return (
    <Figure
      title="Support vectors as a compression set"
      purpose="The SVM refitted on its k support vectors alone draws the same boundary as the SVM fitted on all m points, and it classifies every point outside the support set correctly. The classifier is therefore described by k of the m examples, which is what a compression bound needs."
      state={state}
      defaultSize="M"
      readouts={
        <>
          <Readout label="compression size k (support vectors)" value={`${r.k} of ${m}`} />
          <Readout label="errors outside the set (refit)" value={String(r.errorsOutside)} />
          <Readout label="training errors inside the set" value={String(r.errorsInside)} />
          <Readout
            label="test error: all points / support vectors only"
            value={`${fmt(r.fullTestError)} / ${fmt(r.refitTestError)}`}
          />
          <Readout label="test points where the two disagree" value={`${r.disagreements} of ${TEST}`} />
        </>
      }
      caption="A kernel SVM (aifn supportVectorMachine, Gaussian kernel) on two moons, fitted to all m training points and then refitted to its support vectors alone, the points with αᵢ > 0. The solid contour is the boundary f(x) = 0 of the full fit and the dashed contour that of the refit; they coincide. Support vectors are marked in ink. Every point outside the support set has αᵢ = 0 and margin at least 1, so the refit classifies it correctly, even when the classes overlap and some support vectors are training errors. The compression bound with k support vectors and no errors outside the set then applies. At the opening values k = 32 of m = 200, the test error is 0.035 for both fits, and the realisable compression bound with a union over k is (ln C(200, 32) + ln(200/0.05))/168 = 0.56. Lower the noise to 0.1: k falls to 13 and the bound to 0.29. Shorten the lengthscale to 0.1: the boundary wraps around individual points, k rises to 141 of 200, the test error to 0.072, and the bound passes 1. A classifier that needs most of its sample to describe it gets no guarantee from compression."
    >
      <Plot x={x1} y={x2}>
        {outside.length > 0 && (
          <Points
            name="outside the compression set"
            x={outside.map((p) => p[0])}
            y={outside.map((p) => p[1])}
            group={outsideLabels}
            groupNames={['moon 0', 'moon 1']}
            thin
          />
        )}
        {inside.length > 0 && (
          <Points name="support vectors" x={inside.map((p) => p[0])} y={inside.map((p) => p[1])} emphasis />
        )}
        <Contours
          name="boundary, all points"
          x={grid.xs}
          y={grid.ys}
          z={r.fullSurface}
          levels={[0]}
          labels={false}
          width={2}
          stale={run.stale}
        />
        <Contours
          name="boundary, support vectors only"
          x={grid.xs}
          y={grid.ys}
          z={r.refitSurface}
          levels={[0]}
          labels={false}
          width={2}
          dashed
          slot={2}
        />
      </Plot>
    </Figure>
  )
}
