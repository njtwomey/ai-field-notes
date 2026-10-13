import { useMemo } from 'react'
import {
  Curve,
  Figure,
  formatNumber,
  int,
  Plot,
  Readout,
  slider,
  useAxis,
  useComputed,
  useFigureState,
} from 'aifn-render'
import { child, stream } from 'aifn-compute/foundation/random'
import { stratifiedKFold } from 'aifn-compute/learning/validate'
import { standardDeviation } from 'aifn-compute/probability/stats'
import { expectedGaussianMaximum } from 'aifn-methods/theory/adaptive-holdout'
import { featurePairs, pairSearch, trueAccuracy } from '../_shared/featurePairs'

const TEST_ROWS = 4000

/**
 * The best of J cross-validation scores against J. Each replicate draws a data set, scores every candidate by
 * stratified 5-fold cross-validation once, and reads off the best score among the first J candidates for every J. The
 * true accuracy of that best candidate comes from refitting it on all rows and scoring it on a large fresh sample.
 */
export function BestOfJScores() {
  const state = useFigureState({
    n: int(50, { ge: 20, le: 400, suggestions: [30, 50, 100, 200], label: 'rows n' }),
    candidates: int(100, { ge: 2, le: 200, suggestions: [10, 50, 100, 200], label: 'candidates J' }),
    separation: slider(0, 2, 0, { step: 0.1, label: 'signal in pair 0 (separation)' }),
    replicates: int(8, { ge: 1, le: 20, suggestions: [4, 8, 16], label: 'replicate data sets' }),
    seed: int(1, { ge: 1, le: 999, label: 'seed' }),
  })
  const { n, candidates, separation, replicates, seed } = state
  const computed = useComputed(
    () => {
      const search = pairSearch(candidates)
      return Array.from({ length: replicates }, (_, r) => {
        const s = child(stream(`best-of-j-scores/${seed}`), 'replicate', r)
        const train = featurePairs(child(s, 'train'), { n, pairs: candidates, separation })
        const test = featurePairs(child(s, 'test'), { n: TEST_ROWS, pairs: candidates, separation })
        const result = search.run(train as never, stratifiedKFold({ k: 5, shuffle: true }), child(s, 'cv'))
        const scores = result.rows.map((row) => row.mean)
        // The best of the first J scores, and which candidate it is, for J = 1, 2, …
        const best: number[] = []
        const winner: number[] = []
        scores.forEach((v, j) => {
          const better = j === 0 || v > best[j - 1]
          best.push(better ? v : best[j - 1])
          winner.push(better ? j : winner[j - 1])
        })
        // The true accuracy of each distinct winner, refitted on all rows.
        const truth = new Map<number, number>()
        for (const j of new Set(winner)) truth.set(j, trueAccuracy(j, train, test))
        return { scores, best, truthOfWinner: winner.map((j) => truth.get(j) ?? NaN) }
      })
    },
    [n, candidates, separation, replicates, seed],
    { mode: 'release' },
  )
  const runs = computed.value
  const view = useMemo(() => {
    const js = Array.from({ length: candidates }, (_, j) => j + 1)
    const average = (pick: (r: (typeof runs)[number]) => number[]) =>
      js.map((_, j) => runs.reduce((a, r) => a + pick(r)[j], 0) / runs.length)
    // The average single candidate: the mean of all scores over candidates and replicates.
    const all = runs.flatMap((r) => r.scores)
    const typical = all.reduce((a, b) => a + b, 0) / all.length
    // σ̂: the spread of a single candidate's score over candidates and replicates, the noise of one estimate when
    // every candidate has the same true accuracy.
    const sigma = all.length > 1 ? standardDeviation(all, { sample: true }) : NaN
    return {
      js,
      best: average((r) => r.best),
      truth: average((r) => r.truthOfWinner),
      typical,
      sigma,
      gaussian: js.map((J) => typical + sigma * expectedGaussianMaximum(J)),
      bound: js.map((J) => typical + sigma * Math.sqrt(2 * Math.log(J))),
    }
  }, [runs, candidates])
  const last = candidates - 1
  const xAxis = useAxis({ label: 'number of candidates J', log: true })
  const yAxis = useAxis({ label: 'accuracy', range: [0.35, 1] })
  return (
    <Figure
      title="The best of J cross-validation scores"
      state={state}
      caption="J candidate classifiers, each a 5-nearest-neighbour rule on its own pair of features, scored by stratified 5-fold cross-validation on n rows. Only pair 0 can carry signal; at separation 0 the labels are pure noise and every candidate's true accuracy is 0.5. Thin lines: the best cross-validation accuracy among the first J candidates, one line per replicate data set. Thick orange: their average. The line in the text colour: the average true accuracy of the candidate that won, measured on 4,000 fresh rows. The gap between the two is the optimism of the reported score. It grows with J and shrinks as n grows. Dashed: the average single score plus σ̂√(2 ln J), the maximal-inequality bound, and plus σ̂ times the expected maximum of J independent standard normals, where σ̂ is the standard deviation of one candidate's score over candidates and data sets. At separation 0 the average best score follows the Gaussian curve closely and stays below the bound. Raise the separation to give pair 0 real signal: the winner is then usually pair 0, the overlays no longer apply, and the gap narrows but does not close."
      readouts={
        <>
          <Readout label="average single candidate, CV" value={formatNumber(view.typical)} />
          <Readout label={`best of ${candidates}, CV`} value={formatNumber(view.best[last])} />
          <Readout label="winner's true accuracy" value={formatNumber(view.truth[last])} />
          <Readout label="optimism" value={formatNumber(view.best[last] - view.truth[last])} />
          <Readout label="σ̂, spread of one score" value={formatNumber(view.sigma)} />
          <Readout
            label={`equal independent candidates, σ̂ · E max of ${candidates} normals`}
            value={formatNumber(view.gaussian[last] - view.typical)}
          />
          <Readout label={`bound σ̂√(2 ln ${candidates})`} value={formatNumber(view.bound[last] - view.typical)} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis}>
        {runs.map((r, i) => (
          <Curve key={i} name="best CV accuracy, one data set" x={view.js} y={r.best} slot={1} thin silent />
        ))}
        <Curve name="best CV accuracy, average" x={view.js} y={view.best} slot={1} stale={computed.stale} />
        <Curve name="winner's true accuracy, average" x={view.js} y={view.truth} emphasis stale={computed.stale} />
        <Curve name="chance" x={[1, candidates]} y={[0.5, 0.5]} muted dashed />
        <Curve name="mean + σ̂√(2 ln J), the maximal-inequality bound" x={view.js} y={view.bound} slot={0} dashed />
        <Curve name="mean + σ̂ · E max of J normals" x={view.js} y={view.gaussian} slot={2} dashed />
      </Plot>
    </Figure>
  )
}
