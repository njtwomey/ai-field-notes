import { useMemo } from 'react'
import {
  Area,
  Curve,
  Figure,
  formatNumber,
  Handle,
  int,
  Plot,
  Readout,
  useAxis,
  useComputed,
  useFigureState,
} from 'aifn-render'
import { child, integers, normals, stream, uniform, type Stream } from 'aifn-compute/foundation/random'
import { fromData, mean, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { dataset } from 'aifn-compute/learning/estimators'
import { rbf } from 'aifn-compute/learning/kernels'
import { errorRate } from 'aifn-compute/learning/metrics'
import { crossValidate, kFold, type Fittable } from 'aifn-compute/learning/validate'
import { quantile } from 'aifn-compute/probability/stats'
import { expectedMaxOfDraws } from 'aifn-compute/probability/tests'
import { supportVectorMachine } from 'aifn-methods/learning/kernel-methods'
import { kNearestNeighbours } from 'aifn-methods/learning/neighbours'

const fmt = (v: number) => (Number.isFinite(v) ? formatNumber(Number(v.toPrecision(3))) : '—')
const P = 5
const POOL = 30
const FRESH = 1000
const SEARCHES = 300

/** n rows: five standard-normal inputs, class 1 when the radius in the first two plus noise exceeds its median. */
function draw(s: Stream, n: number) {
  const x = toFlat(normals(child(s, 'x'), [n, P]))
  const e = toFlat(normals(child(s, 'e'), [n]))
  const y = Float64Array.from({ length: n }, (_, i) =>
    Math.hypot(x[P * i], x[P * i + 1]) + 0.3 * e[i] > 1.1774 ? 1 : 0,
  )
  return dataset(fromData(Float64Array.from(x), [n, P]), fromData(y))
}

/** A log-uniform draw on [10^lo, 10^hi]. */
const logUniform = (s: Stream, lo: number, hi: number) => 10 ** (uniform(s, lo, hi) as number)

type Trial = { cv: number; test: number }

/**
 * Random search on two learners with the same data: the test error of the configuration a search of b trials selects,
 * as a function of b, from a pool of trials scored once. The two budgets are draggable.
 */
export function BudgetCurves() {
  const state = useFigureState({
    n: int(150, { ge: 60, le: 400, suggestions: [100, 150, 300], label: 'training rows n' }),
    knnBudget: int(30, { ge: 1, le: POOL, suggestions: [1, 5, 30], label: 'trials for k-NN' }),
    svmBudget: int(2, { ge: 1, le: POOL, suggestions: [1, 5, 30], label: 'trials for the SVM' }),
  })
  const { n, knnBudget, svmBudget } = state
  const computed = useComputed(
    () => {
      const root = stream('comparing/tuning-budgets')
      const data = draw(child(root, 'data'), n)
      const fresh = draw(child(root, 'fresh'), FRESH)
      const score = (est: Fittable<Tensor, { decide: (x: Tensor) => Tensor }>): Trial => ({
        cv: crossValidate(est, data, kFold({ k: 5, shuffle: true }), [errorRate], { stream: child(root, 'cv') }).mean
          .errorRate,
        test: errorRate(fresh.y, est.fit(data).decide(fresh.x)),
      })
      // k-NN: one hyperparameter, k, log-uniform on [1, 100]; equal values of k are scored once.
      const byK = new Map<number, Trial>()
      const knn = Array.from({ length: POOL }, (_, i) => {
        const k = Math.max(1, Math.round(logUniform(child(root, 'knn', i), 0, 2)))
        if (!byK.has(k)) byK.set(k, score(kNearestNeighbours({ k })))
        return byK.get(k)!
      })
      // RBF SVM: C log-uniform on [0.01, 1000], kernel lengthscale log-uniform on [0.03, 30].
      const svm = Array.from({ length: POOL }, (_, i) =>
        score(
          supportVectorMachine({
            C: logUniform(child(root, 'svm', i, 'C'), -2, 3),
            kernel: rbf({ lengthscale: logUniform(child(root, 'svm', i, 'l'), -1.5, 1.5) }),
          }),
        ),
      )
      return { knn, svm }
    },
    [n],
    { mode: 'release' },
  )
  const { knn, svm } = computed.value
  const view = useMemo(() => {
    const budgets = Array.from({ length: POOL }, (_, i) => i + 1)
    // A search of b trials draws b configurations from the pool and keeps the one with the lowest CV error.
    const curve = (pool: Trial[], key: string) => {
      const s = stream(`comparing/tuning-budgets/searches/${key}`)
      const rows = budgets.map((b) => {
        const picked = Array.from({ length: SEARCHES }, (_, m) => {
          const idx = toFlat(integers(child(s, b, m), POOL, { shape: [b] }))
          let best = pool[idx[0]]
          for (const i of idx) if (pool[i].cv < best.cv) best = pool[i]
          return best.test
        })
        return {
          mean: mean(fromData(Float64Array.from(picked))) as number,
          lo: quantile(picked, 0.1),
          hi: quantile(picked, 0.9),
        }
      })
      return {
        mean: rows.map((r) => r.mean),
        lo: rows.map((r) => r.lo),
        hi: rows.map((r) => r.hi),
      }
    }
    // Dodge et al.'s expected best validation score of b random trials, for a loss: the expected maximum of the
    // negated cross-validation errors, negated back.
    const expectedBest = (pool: Trial[]) =>
      Array.from(
        toFlat(
          expectedMaxOfDraws(
            pool.map((t) => -t.cv),
            budgets,
          ),
        ),
        (v) => -v,
      )
    return {
      budgets,
      knn: curve(knn, 'knn'),
      svm: curve(svm, 'svm'),
      dodge: { knn: expectedBest(knn), svm: expectedBest(svm) },
    }
  }, [knn, svm])
  const x = useAxis({ label: 'random-search trials b', integer: true, range: [1, POOL] })
  const y = useAxis({ label: 'test error of the selected configuration' })
  const atKnn = view.knn.mean[knnBudget - 1]
  const atSvm = view.svm.mean[svmBudget - 1]
  return (
    <Figure
      title="The winner depends on the tuning budget"
      state={state}
      caption={`Two learners tuned by random search on the same n training rows (five standard-normal inputs; the class depends on the radius in the first two). k-nearest neighbours tunes k on a log scale from 1 to 100; the RBF support vector machine tunes C and the kernel lengthscale on log scales. Each learner has a pool of ${POOL} random configurations, scored once by 5-fold cross-validation and once on ${FRESH} fresh rows. A search with b trials draws b configurations from the pool and keeps the one with the lowest cross-validation error; the curve is the mean of its fresh-data error over ${SEARCHES} searches, the band its 10th to 90th percentiles. The dashed curves are Dodge and colleagues' expected best cross-validation error of b trials, computed from the same pools by the formula above. Drag the two vertical lines to give each learner its budget.`}
      readouts={{
        'expected test error at the chosen budgets': (
          <>
            <Readout label={`k-NN, ${knnBudget} trials`} value={fmt(atKnn)} />
            <Readout label={`SVM, ${svmBudget} trials`} value={fmt(atSvm)} />
          </>
        ),
        verdict: <Readout label="looks better" value={atKnn < atSvm ? 'k-NN' : 'SVM'} />,
        'expected best CV error at 30 trials': (
          <>
            <Readout label="k-NN" value={fmt(view.dodge.knn[POOL - 1])} />
            <Readout label="SVM" value={fmt(view.dodge.svm[POOL - 1])} />
          </>
        ),
        'equal budgets': (
          <>
            <Readout label="k-NN, 30 trials" value={fmt(view.knn.mean[POOL - 1])} />
            <Readout label="SVM, 30 trials" value={fmt(view.svm.mean[POOL - 1])} />
          </>
        ),
      }}
    >
      <Plot x={x} y={y}>
        <Area
          name="k-NN, 10–90%"
          x={view.budgets}
          y={view.knn.hi}
          base={view.knn.lo}
          slot={0}
          opacity={0.15}
          line={false}
        />
        <Area
          name="SVM, 10–90%"
          x={view.budgets}
          y={view.svm.hi}
          base={view.svm.lo}
          slot={1}
          opacity={0.15}
          line={false}
        />
        <Curve name="k-NN" x={view.budgets} y={view.knn.mean} slot={0} stale={computed.stale} />
        <Curve name="SVM" x={view.budgets} y={view.svm.mean} slot={1} stale={computed.stale} />
        <Curve
          name="k-NN, expected best CV error"
          x={view.budgets}
          y={view.dodge.knn}
          slot={0}
          dashed
          stale={computed.stale}
        />
        <Curve
          name="SVM, expected best CV error"
          x={view.budgets}
          y={view.dodge.svm}
          slot={1}
          dashed
          stale={computed.stale}
        />
        <Handle
          kind="x"
          at={knnBudget}
          label="k-NN"
          slot={0}
          onDrag={(v) => state.set('knnBudget', Math.min(POOL, Math.max(1, Math.round(v))))}
        />
        <Handle
          kind="x"
          at={svmBudget}
          label="SVM"
          slot={1}
          onDrag={(v) => state.set('svmBudget', Math.min(POOL, Math.max(1, Math.round(v))))}
        />
      </Plot>
    </Figure>
  )
}
