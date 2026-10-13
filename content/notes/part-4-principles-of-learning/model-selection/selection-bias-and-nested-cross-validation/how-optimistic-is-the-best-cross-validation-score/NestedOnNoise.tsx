import { useMemo } from 'react'
import {
  Curve,
  Figure,
  formatNumber,
  int,
  Plot,
  Points,
  Readout,
  Segments,
  useAxis,
  useComputed,
  useFigureState,
} from 'aifn-render'
import { child, stream } from 'aifn-compute/foundation/random'
import { nested, stratifiedKFold } from 'aifn-compute/learning/validate'
import { featurePairs, pairSearch } from '../_shared/featurePairs'

/** Unnested and nested cross-validation of a search over J candidates, on replicate data sets with pure-noise labels. */
export function NestedOnNoise() {
  const state = useFigureState({
    candidates: int(10, { ge: 1, le: 100, suggestions: [1, 10, 30, 100], label: 'candidates J' }),
    n: int(50, { ge: 20, le: 200, suggestions: [30, 50, 100, 200], label: 'rows n' }),
    replicates: int(10, { ge: 2, le: 30, suggestions: [5, 10, 20], label: 'replicate data sets' }),
    seed: int(4, { ge: 1, le: 999, label: 'seed' }),
  })
  const { candidates, n, replicates, seed } = state
  const computed = useComputed(
    () => {
      const search = pairSearch(candidates)
      return Array.from({ length: replicates }, (_, r) => {
        const s = child(stream(`nested-on-noise/${seed}`), 'replicate', r)
        const data = featurePairs(child(s, 'train'), { n, pairs: candidates, separation: 0 })
        const result = nested(
          stratifiedKFold({ k: 5, shuffle: true }),
          stratifiedKFold({ k: 5, shuffle: true }),
          search,
          data as never,
          { stream: child(s, 'cv') },
        )
        return { unnested: result.unnestedScore, nested: result.nestedScore }
      })
    },
    [candidates, n, replicates, seed],
    { mode: 'release' },
  )
  const runs = computed.value
  const view = useMemo(() => {
    const index = runs.map((_, r) => r + 1)
    const mean = (v: number[]) => v.reduce((a, b) => a + b, 0) / v.length
    const unnested = runs.map((r) => r.unnested)
    const nestedScores = runs.map((r) => r.nested)
    return {
      index,
      unnested,
      nested: nestedScores,
      pairs: runs.map((r, i) => ({ from: [i + 1, r.nested] as const, to: [i + 1, r.unnested] as const })),
      meanUnnested: mean(unnested),
      meanNested: mean(nestedScores),
    }
  }, [runs])
  const xAxis = useAxis({ label: 'replicate data set', range: [0.5, replicates + 0.5], nice: false, integer: true })
  const yAxis = useAxis({ label: 'estimated accuracy', range: [0.2, 0.9] })
  const ends = [0.5, replicates + 0.5]
  return (
    <Figure
      title="Unnested and nested estimates on noise"
      state={state}
      caption="Each replicate draws n rows with labels that are a fair coin, exactly half of each class, independent of 2J standard normal features. The search scores J candidates (5-nearest neighbours on one feature pair each) by stratified 5-fold cross-validation and keeps the best. Orange: the best cross-validation accuracy, the number a tuning run reports. Green: the nested estimate, whose outer 5-fold loop scores the whole search on rows it never saw. The thin segment joins the two estimates from the same data set. Every classifier has true accuracy 0.5 (dashed). The orange points rise with J and stay above 0.5; the green points scatter around 0.5 at every J. With J = 1 there is nothing to select and the two estimates agree on average."
      readouts={
        <>
          <Readout label="mean best CV accuracy (unnested)" value={formatNumber(view.meanUnnested)} />
          <Readout label="mean nested estimate" value={formatNumber(view.meanNested)} />
          <Readout label="true accuracy" value="0.5" />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis}>
        <Curve name="true accuracy" x={ends} y={[0.5, 0.5]} emphasis dashed />
        <Segments name="same data set" segments={view.pairs} />
        <Points name="best CV accuracy (unnested)" x={view.index} y={view.unnested} slot={1} stale={computed.stale} />
        <Points name="nested estimate" x={view.index} y={view.nested} slot={2} stale={computed.stale} />
        <Curve name="mean, unnested" x={ends} y={[view.meanUnnested, view.meanUnnested]} slot={1} thin />
        <Curve name="mean, nested" x={ends} y={[view.meanNested, view.meanNested]} slot={2} thin />
      </Plot>
    </Figure>
  )
}
