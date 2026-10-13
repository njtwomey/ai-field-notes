import { useContext, useMemo, type ReactNode } from 'react'
import {
  Curve,
  Figure,
  formatNumber,
  FrameContext,
  int,
  Plot,
  Points,
  Raster,
  Readout,
  slider,
  useAxis,
  useComputed,
  useFigureState,
} from 'aifn-render'
import { child, stream } from 'aifn-compute/foundation/random'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { nested, stratifiedKFold } from 'aifn-compute/learning/validate'
import { featurePairs, pairSearch, trueAccuracy } from '../_shared/featurePairs'

const TEST_ROWS = 4000
const TRAINING = 0
const VALIDATION = 1
const OUTER_TEST = 2

/** Plots inside a Figure take the frame's height; this gives the plots inside it a fixed height instead. */
function FixedHeight({ height, children }: { height: number; children: ReactNode }) {
  const frame = useContext(FrameContext)
  return <FrameContext.Provider value={{ ...frame, height }}>{children}</FrameContext.Provider>
}

/** Nested cross-validation of a search over feature pairs: the two levels of folds, and the scores each level gives. */
export function NestedFolds() {
  const state = useFigureState({
    n: int(60, { ge: 30, le: 200, suggestions: [40, 60, 100], label: 'rows n' }),
    pairs: int(10, { ge: 2, le: 50, suggestions: [5, 10, 30], label: 'candidates J' }),
    separation: slider(0, 2.5, 0, { step: 0.1, label: 'signal in pair 0 (separation)' }),
    outerK: int(5, { ge: 2, le: 10, suggestions: [3, 5, 10], label: 'outer folds' }),
    innerK: int(5, { ge: 2, le: 10, suggestions: [3, 5, 10], label: 'inner folds' }),
    shown: int(1, { ge: 1, le: 10, label: 'outer fold shown' }),
    seed: int(2, { ge: 1, le: 999, label: 'seed' }),
  })
  const { n, pairs, separation, outerK, innerK, seed } = state
  const shown = Math.min(state.shown, outerK) - 1
  const computed = useComputed(
    () => {
      const s = stream(`nested-folds/${seed}`)
      const train = featurePairs(child(s, 'train'), { n, pairs, separation })
      const test = featurePairs(child(s, 'test'), { n: TEST_ROWS, pairs, separation })
      const result = nested(
        stratifiedKFold({ k: outerK, shuffle: true }),
        stratifiedKFold({ k: innerK, shuffle: true }),
        pairSearch(pairs),
        train as never,
        { stream: child(s, 'cv') },
      )
      const chosen = (result.unnested.best.params as { j: number }).j
      return { result, chosen, truth: trueAccuracy(chosen, train, test) }
    },
    [n, pairs, separation, outerK, innerK, seed],
    { mode: 'release' },
  )
  const { result, chosen, truth } = computed.value
  const rows = useMemo(() => Array.from({ length: n }, (_, i) => i), [n])
  const grids = useMemo(() => {
    const folds = result.outer.folds
    // Outer level: each outer split's training rows and outer test rows.
    const outer = folds.map((f) => {
      const line = new Array<number>(n).fill(TRAINING)
      for (const i of toFlat(f.test)) line[i] = OUTER_TEST
      return line
    })
    // Inner level, for the outer fold shown: the inner splits index that fold's training rows.
    const fold = folds[Math.min(shown, folds.length - 1)]
    const trainRows = Array.from(toFlat(fold.train))
    const inner = fold.model.search.validations[0].splits.map((split) => {
      const line = new Array<number>(n).fill(OUTER_TEST)
      for (const i of toFlat(split.train)) line[trainRows[i]] = TRAINING
      for (const i of toFlat(split.test)) line[trainRows[i]] = VALIDATION
      return line
    })
    return { outer, inner }
  }, [result, n, shown])
  const perFold = result.perFold
  const foldIndex = perFold.map((_, f) => f + 1)
  const ox = useAxis({ label: 'row', range: [-0.5, n - 0.5], nice: false })
  const oy = useAxis({ label: 'outer split', range: [0.5, outerK + 0.5], nice: false, integer: true })
  const ix = useAxis({ label: 'row', range: [-0.5, n - 0.5], nice: false })
  const iy = useAxis({
    label: `inner split of outer fold ${shown + 1}`,
    range: [0.5, innerK + 0.5],
    nice: false,
    integer: true,
  })
  const sx = useAxis({ label: 'outer fold', range: [0.5, outerK + 0.5], nice: false, integer: true })
  const sy = useAxis({ label: 'accuracy', range: [0, 1] })
  const ends = [0.5, outerK + 0.5]
  return (
    <Figure
      title="The two loops of nested cross-validation"
      state={state}
      hoverReadout={false}
      caption="A search picks one of J candidate classifiers (5-nearest neighbours on one feature pair each) by stratified cross-validation. Top: the outer splits; each outer test fold (green) is held out from the whole search. Middle: inside the outer fold shown, the search runs its own inner splits over that fold's training rows only, training (blue) and validating (orange) every candidate; the outer test rows are never touched. Bottom, per outer fold: the best inner cross-validation accuracy (orange), which chose the candidate, and the accuracy of the chosen, refitted candidate on the outer test fold (green). The dashed lines are their averages: the unnested estimate (the search run once on all rows) and the nested estimate. The solid line in the text colour is the true accuracy of the final model, the search refitted on all rows, measured on 4,000 fresh rows. At separation 0 the labels are noise, the true accuracy is 0.5, the inner scores sit well above it and the outer scores scatter around it. With signal, the outer folds that choose pair 0 score near the truth."
      readouts={
        <>
          <Readout
            label="candidate chosen per outer fold"
            value={perFold.map((f) => String((f.params as { j: number }).j)).join(', ')}
          />
          <Readout label="unnested (best inner mean, all rows)" value={formatNumber(result.unnestedScore)} />
          <Readout label="nested (mean outer score)" value={formatNumber(result.nestedScore)} />
          <Readout label={`final model (pair ${chosen}), true accuracy`} value={formatNumber(truth)} />
        </>
      }
    >
      <FixedHeight height={60 + 22 * outerK}>
        <Plot x={ox} y={oy} title="outer splits">
          <Raster
            name="outer assignment"
            x={rows}
            y={foldIndex}
            z={grids.outer}
            scale="categorical"
            categoryNames={['training', 'validation (inner)', 'outer test']}
          />
        </Plot>
      </FixedHeight>
      <FixedHeight height={60 + 22 * innerK}>
        <Plot x={ix} y={iy} title={`inner splits inside outer fold ${shown + 1}`}>
          <Raster
            name="inner assignment"
            x={rows}
            y={Array.from({ length: grids.inner.length }, (_, k) => k + 1)}
            z={grids.inner}
            scale="categorical"
            categoryNames={['training', 'validation (inner)', 'outer test']}
          />
        </Plot>
      </FixedHeight>
      <FixedHeight height={240}>
        <Plot x={sx} y={sy} title="scores per outer fold">
          <Curve name="chance" x={ends} y={[0.5, 0.5]} muted dashed />
          <Curve name="unnested estimate" x={ends} y={[result.unnestedScore, result.unnestedScore]} slot={1} dashed />
          <Curve name="nested estimate" x={ends} y={[result.nestedScore, result.nestedScore]} slot={2} dashed />
          <Curve name="final model, true accuracy" x={ends} y={[truth, truth]} emphasis />
          <Points
            name="best inner CV accuracy"
            x={foldIndex}
            y={perFold.map((f) => f.innerScore)}
            slot={1}
            stale={computed.stale}
          />
          <Points
            name="outer test accuracy"
            x={foldIndex}
            y={perFold.map((f) => f.outerScore)}
            slot={2}
            stale={computed.stale}
          />
        </Plot>
      </FixedHeight>
    </Figure>
  )
}
