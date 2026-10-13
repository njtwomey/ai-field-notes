import {
  call,
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Plot,
  Readout,
  setting,
  slider,
  useAxis,
  useComputed,
  useFigureState,
} from 'aifn-render'
import { type BayesianOptimisationStudy } from 'aifn-methods/optimisation/bayesian'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { SVM_BEST } from '../../_shared/svmSurface'

/** Evaluations in the Bayesian optimisation's initial design (a Latin hypercube, 2d + 1). */
const INIT = 5
const ARM_NAMES = ['expected improvement', 'random search']

type Curves = {
  /** The true error of the configuration with the best observed score so far: arm by seed by evaluation. */
  truth: number[][][]
  /** The best observed score so far, as the search reports it: arm by seed by evaluation. */
  observed: number[][][]
  /** Their means over the seeds: arm by evaluation. */
  meanTruth: number[][]
  meanObserved: number[][]
}

/**
 * Both arms of the study as a search returns them: at each evaluation the configuration with the lowest observed score
 * so far, its true (noise-free) error and the score itself, for the first `budget` evaluations.
 */
function returned(study: BayesianOptimisationStudy, budget: number): Curves {
  const [A, S, N] = study.observed.shape
  const observed = toFlat(study.observed)
  const values = toFlat(study.values)
  const truth: number[][][] = []
  const seen: number[][][] = []
  for (let a = 0; a < A; a++) {
    truth.push([])
    seen.push([])
    for (let s = 0; s < S; s++) {
      const row = (a * S + s) * N
      let best = row
      const t: number[] = []
      const o: number[] = []
      for (let n = 0; n < Math.min(budget, N); n++) {
        if (observed[row + n] < observed[best]) best = row + n
        t.push(values[best])
        o.push(observed[best])
      }
      truth[a].push(t)
      seen[a].push(o)
    }
  }
  const mean = (runs: number[][]) => runs[0].map((_, n) => runs.reduce((sum, r) => sum + r[n], 0) / runs.length)
  return { truth, observed: seen, meanTruth: truth.map(mean), meanObserved: seen.map(mean) }
}

/**
 * Gaussian-process expected improvement against random search on the SVM tuning surface over many seeds, run by the
 * engine's study in the compute worker: the mean true error of the configuration each method returns, by budget.
 */
export function SvmSearchStudy() {
  const state = useFigureState({
    seeds: int(20, { label: 'seeds (1 to S)', min: 1, max: 40, suggestions: [5, 10, 20] }),
    budget: int(30, { label: 'evaluations N', min: INIT + 1, max: 50, suggestions: [15, 30, 50] }),
    noise: float(0, { label: 'evaluation noise sd', ge: 0, le: 0.05, suggestions: [0, 0.01, 0.02] }),
    each: setting(false, { label: 'show each seed' }),
    t: slider(1, 50, 10, { step: 1, onChart: true, label: 'evaluations t' }),
  })
  const { seeds, budget, noise, each } = state
  const t = Math.min(Math.max(1, Math.round(state.t)), budget)

  const study = useComputed(
    () =>
      call<BayesianOptimisationStudy>('optimisation/bayesian/bayesianOptimisationStudy', {
        objective: noise > 0 ? { name: 'svm-tuning-surface', noise } : 'svm-tuning-surface',
        seeds: Array.from({ length: seeds }, (_, i) => i + 1),
        iterations: budget - INIT,
        noise: noise > 0 ? 'fit' : 1e-6,
        arms: [
          { label: ARM_NAMES[0], acquisition: 'ei' },
          // Random search: the whole budget as a uniform initial design. The loop's later steps (random grid points
          // on a 2 by 2 grid, no refits) are never used.
          {
            label: ARM_NAMES[1],
            init: budget,
            design: 'uniform',
            refit: false,
            noise: 1e-6,
            interleaveRandom: 1,
            grid: 2,
          },
        ],
      }),
    [seeds, budget, noise],
    { mode: 'worker', initial: null as Curves | null, then: (r) => returned(r, budget) },
  )
  const c = study.value
  const ready = c !== null && c.meanTruth[0].length === budget
  const evals = Array.from({ length: budget }, (_, i) => i + 1)
  const better = ready ? c.truth[0].filter((run, s) => run[t - 1] < c.truth[1][s][t - 1]).length : 0
  const thin = (a: number) => {
    const x: number[] = []
    const y: number[] = []
    if (ready)
      c.truth[a].forEach((run) => {
        x.push(...evals, NaN)
        y.push(...run, NaN)
      })
    return { x, y }
  }

  const nAxis = useAxis({ label: 'evaluations', range: [1, budget], integer: true })
  const eAxis = useAxis({ label: 'error', range: [noise > 0 ? 0 : 0.035, 0.12] })
  return (
    <Figure
      title="Expected improvement against random search over many seeds"
      state={state}
      caption="The mean over seeds of the true error of the configuration each search would return after t evaluations: the one with the lowest observed score so far. Expected improvement (blue) starts from a 5-point Latin hypercube; random search (orange) draws every configuration uniformly on the box. The dashed line is the lowest error in the table, 0.040. Drag the vertical line to read both methods at a budget t, and the number of seeds where expected improvement returned the better configuration. Switch on each seed to see the spread behind the means. With evaluation noise, the dashed curves are the best observed scores, which fall far below the true errors of the configurations that hold them. Computed in the background by the engine's study; 20 seeds take several seconds."
      readouts={
        ready && (
          <>
            <Readout label={`EI, true error at t = ${t}`} value={formatNumber(c.meanTruth[0][t - 1])} />
            <Readout label={`random, true error at t = ${t}`} value={formatNumber(c.meanTruth[1][t - 1])} />
            <Readout label="seeds where EI is better" value={`${better} of ${seeds}`} />
            {noise > 0 && (
              <>
                <Readout label="EI, observed best" value={formatNumber(c.meanObserved[0][t - 1])} />
                <Readout label="random, observed best" value={formatNumber(c.meanObserved[1][t - 1])} />
              </>
            )}
          </>
        )
      }
    >
      <Plot x={nAxis} y={eAxis} height={380}>
        <Curve name="lowest error" x={[1, budget]} y={[SVM_BEST, SVM_BEST]} muted dashed />
        {ready &&
          each &&
          [0, 1].map((a) => {
            const r = thin(a)
            return <Curve key={`seeds-${a}`} name={`${ARM_NAMES[a]}, each seed`} x={r.x} y={r.y} slot={a} thin silent />
          })}
        {ready &&
          [0, 1].map((a) => (
            <Curve
              key={`truth-${a}`}
              name={`${ARM_NAMES[a]}, true error`}
              x={evals}
              y={c.meanTruth[a]}
              slot={a}
              width={2.5}
              stale={study.stale}
            />
          ))}
        {ready &&
          noise > 0 &&
          [0, 1].map((a) => (
            <Curve
              key={`observed-${a}`}
              name={`${ARM_NAMES[a]}, observed best`}
              x={evals}
              y={c.meanObserved[a]}
              slot={a}
              dashed
              stale={study.stale}
            />
          ))}
        <Handle {...state.handle('t', { axis: 'x', label: 't' })} />
      </Plot>
    </Figure>
  )
}
