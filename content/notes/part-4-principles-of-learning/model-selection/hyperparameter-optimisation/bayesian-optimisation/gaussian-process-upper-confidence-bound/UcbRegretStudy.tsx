import {
  call,
  Curve,
  Figure,
  formatNumber,
  int,
  Plot,
  Readout,
  setting,
  useAxis,
  useComputed,
  useFigureState,
} from 'aifn-render'
import { type BayesianOptimisationStudy } from 'aifn-methods/optimisation/bayesian'
import { toFlat } from 'aifn-compute/foundation/tensor'

/** The arms: three fixed exploration weights and the Srinivas schedule with δ = 0.1. */
const ARMS = [
  { label: 'β = 0', beta: 0 },
  { label: 'β = 1', beta: 1 },
  { label: 'β = 4', beta: 4 },
  { label: 'schedule βₜ', beta: 'srinivas' as const },
]
/** A run whose best value stays above this never found the global well (the local minimum is −0.986). */
const STUCK = -1

type Curves = {
  /** Cumulative regret after the design, arm by seed by evaluation after the design. */
  regret: number[][][]
  /** Its mean over the seeds, arm by evaluation. */
  mean: number[][]
  /** Runs per arm whose best value found stays above STUCK. */
  stuck: number[]
}

/** Regret counted from the first evaluation after the initial design, as in the note's table. */
function afterDesign(study: BayesianOptimisationStudy): Curves {
  const [A, S, N] = study.cumulativeRegret.shape
  const cr = toFlat(study.cumulativeRegret)
  const best = toFlat(study.bestSoFar)
  const n0 = study.initial
  const regret = Array.from({ length: A }, (_, a) =>
    Array.from({ length: S }, (_, s) => {
      const row = (a * S + s) * N
      return Array.from({ length: N - n0 }, (_, i) => cr[row + n0 + i] - cr[row + n0 - 1])
    }),
  )
  const mean = regret.map((runs) => runs[0].map((_, i) => runs.reduce((sum, r) => sum + r[i], 0) / S))
  const stuck = regret.map(
    (_, a) => Array.from({ length: S }, (_, s) => best[(a * S + s) * N + N - 1]).filter((v) => v > STUCK).length,
  )
  return { regret, mean, stuck }
}

/**
 * GP-UCB on the Forrester function from several seeds, for three fixed exploration weights and the theoretical
 * schedule: the cumulative regret of the evaluations after the initial design, per seed and averaged. Run in the
 * compute worker by the engine's study.
 */
export function UcbRegretStudy() {
  const state = useFigureState({
    seeds: int(5, { label: 'seeds (1 to S)', min: 1, max: 30, suggestions: [5, 10, 20] }),
    iterations: int(20, { label: 'evaluations after the design', min: 2, max: 60, suggestions: [20, 40, 60] }),
    each: setting(true, { label: 'show each seed' }),
  })
  const { seeds, iterations, each } = state

  const study = useComputed(
    () =>
      call<BayesianOptimisationStudy>('optimisation/bayesian/bayesianOptimisationStudy', {
        objective: 'forrester',
        seeds: Array.from({ length: seeds }, (_, i) => i + 1),
        iterations,
        acquisition: 'ucb',
        arms: ARMS,
      }),
    [seeds, iterations],
    { mode: 'worker', initial: null as Curves | null, then: afterDesign },
  )
  const curves = study.value
  const steps = curves ? curves.mean[0].map((_, i) => i + 1) : []
  const thin = (a: number) => {
    const x: number[] = []
    const y: number[] = []
    curves?.regret[a].forEach((r) => {
      x.push(...steps, NaN)
      y.push(...r, NaN)
    })
    return { x, y }
  }

  const tAxis = useAxis({ label: 'evaluations after the design', range: [1, iterations], integer: true })
  const rAxis = useAxis({ label: 'cumulative regret', hold: 'union', key: `${seeds}|${iterations}` })
  return (
    <Figure
      title="Cumulative regret of GP-UCB over seeds"
      state={state}
      caption="The confidence bound on the Forrester function from seeds 1 to S, each run starting from a 3-point Latin hypercube and making the given number of evaluations; the curves are the cumulative regret of those evaluations, Σ (f(λₜ) − min f), with each seed thin and the mean over seeds thick. With the 5 default seeds, β = 0 and β = 1 each leave one run in the shallow local minimum near λ = 0.14, whose regret grows by about 5 per evaluation and passes 100. β = 4 never gets stuck and has the lowest mean. The Srinivas schedule never gets stuck either, but its regret grows steadily because it keeps evaluating the ends of the interval. Computed in the background by the engine's study."
      readouts={
        curves && (
          <>
            {ARMS.map((arm, a) => (
              <Readout
                key={arm.label}
                label={`${arm.label}: mean regret, stuck runs`}
                value={`${formatNumber(curves.mean[a][curves.mean[a].length - 1])}, ${curves.stuck[a]} of ${seeds}`}
              />
            ))}
          </>
        )
      }
    >
      <Plot x={tAxis} y={rAxis} height={380}>
        {curves &&
          ARMS.map((arm, a) => (
            <Curve
              key={arm.label}
              name={arm.label}
              x={steps}
              y={curves.mean[a]}
              slot={a}
              width={2.5}
              stale={study.stale}
            />
          ))}
        {curves &&
          each &&
          ARMS.map((arm, a) => {
            const t = thin(a)
            return (
              <Curve key={`${arm.label}-seeds`} name={`${arm.label}, each seed`} x={t.x} y={t.y} slot={a} thin silent />
            )
          })}
      </Plot>
    </Figure>
  )
}
