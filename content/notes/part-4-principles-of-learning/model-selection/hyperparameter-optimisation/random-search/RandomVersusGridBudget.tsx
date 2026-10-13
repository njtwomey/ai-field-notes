import { useMemo } from 'react'
import { Curve, Figure, formatNumber, int, Plot, Points, Readout, useAxis, useFigureState } from 'aifn-render'
import { child, stream } from 'aifn-compute/foundation/random'
import { grid, real, sample, space } from 'aifn-compute/foundation/space'
import { bestSoFar, SVM_BEST, svmError } from '../_shared/svmSurface'

const MAX_TRIALS = 256

/** The SVM's two hyperparameters plus `extra` that the error ignores, all searched uniformly. */
function searchSpace(extra: number) {
  const dims: Record<string, ReturnType<typeof real>> = { logC: real(-2, 3), logGamma: real(-3, 2) }
  for (let i = 0; i < extra; i++) dims[`unused${i}`] = real(0, 1)
  return space(dims)
}

/**
 * Best error found against the number of trials, for random search (one thin line per run, their average thick) and
 * for grids with r values per axis (one point per r), when the space holds the SVM's two hyperparameters and some that
 * do not affect the error.
 */
export function RandomVersusGridBudget() {
  const state = useFigureState({
    extra: int(0, { ge: 0, le: 4, suggestions: [0, 1, 2, 3], label: 'hyperparameters that do not matter' }),
    runs: int(8, { ge: 1, le: 30, suggestions: [4, 8, 16], label: 'random-search runs' }),
    seed: int(1, { ge: 1, le: 999, label: 'seed' }),
  })
  const { extra, runs, seed } = state

  const result = useMemo(() => {
    const sp = searchSpace(extra)
    const error = (p: Record<string, number>) => svmError(p.logC, p.logGamma)
    const trials = Array.from({ length: MAX_TRIALS }, (_, t) => t + 1)
    const random = Array.from({ length: runs }, (_, r) => {
      const s = child(stream(`random-versus-grid/${seed}`), 'run', r)
      return bestSoFar(trials.map((t) => error(sample(child(s, 'trial', t), sp) as Record<string, number>)))
    })
    const average = trials.map((_, t) => random.reduce((a, run) => a + run[t], 0) / runs)
    const grids: { n: number; best: number; perAxis: number }[] = []
    for (let r = 2; r ** (2 + extra) <= MAX_TRIALS; r++) {
      const points = grid(sp, { points: r }) as Record<string, number>[]
      grids.push({ n: points.length, best: Math.min(...points.map(error)), perAxis: r })
    }
    return { trials, random, average, grids }
  }, [extra, runs, seed])

  const lastGrid = result.grids[result.grids.length - 1]
  const xAxis = useAxis({ label: 'trials N', log: true, range: [1, MAX_TRIALS] })
  const yAxis = useAxis({ label: 'best error found', range: [0.035, 0.13] })
  return (
    <Figure
      title="Random search against grid search as the budget grows"
      state={state}
      caption="The best cross-validated error of an RBF support vector machine found within N trials, on the tabulated error over log C and log γ. Thin orange lines: random-search runs, each the running best of its own trials; thick orange: their average. Blue points: grids with r values on every axis, at their budgets r^D. The dashed line is the lowest error in the table. Add hyperparameters that do not affect the error: the grid's budget for r values per axis grows as r^D and it tests only r² distinct (C, γ) pairs, while the random-search curves do not move, because their projection onto (C, γ) is the same random search."
      readouts={
        <>
          <Readout label="dimensions D" value={String(2 + extra)} />
          <Readout
            label={`random search, average best of ${MAX_TRIALS}`}
            value={formatNumber(result.average[MAX_TRIALS - 1])}
          />
          <Readout
            label={`largest grid (${lastGrid.perAxis} per axis, ${lastGrid.n} trials)`}
            value={formatNumber(lastGrid.best)}
          />
          <Readout label="lowest error in the table" value={formatNumber(SVM_BEST)} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis}>
        {result.random.map((run, i) => (
          <Curve key={i} name="random search, one run" x={result.trials} y={run} slot={1} thin silent />
        ))}
        <Curve name="random search, average" x={result.trials} y={result.average} slot={1} />
        <Points name="grid search" x={result.grids.map((g) => g.n)} y={result.grids.map((g) => g.best)} slot={0} />
        <Curve name="lowest error" x={[1, MAX_TRIALS]} y={[SVM_BEST, SVM_BEST]} muted dashed />
      </Plot>
    </Figure>
  )
}
