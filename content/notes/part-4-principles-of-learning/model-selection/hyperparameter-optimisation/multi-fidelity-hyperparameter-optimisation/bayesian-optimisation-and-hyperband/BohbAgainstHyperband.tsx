import { useMemo } from 'react'
import {
  Annotation,
  Curve,
  Figure,
  float,
  formatNumber,
  int,
  Plot,
  Plots,
  Points,
  Readout,
  slider,
  useAxis,
  useComputed,
  useFigureState,
} from 'aifn-render'
import { child, stream } from 'aifn-compute/foundation/random'
import { real, sample, space } from 'aifn-compute/foundation/space'
import { trace } from 'aifn-compute/foundation/trace'
import { learningCurveSurface } from 'aifn-methods/data/synthetic'
import { bohbSteps, hyperbandSchedule, type MultiFidelitySchedule } from 'aifn-methods/optimisation/multi-fidelity'

/** The maximum budget R (epochs) and the reduction factor, as in the worked example. */
const R = 81
const ETA = 3
/** Points on the budget axis at which the runs are compared. */
const GRID = 120
/** The budgets of the rungs of Hyperband's brackets for R = 81 and η = 3, the budgets a model can be fitted at. */
const RUNG_BUDGETS = [1, 3, 9, 27, 81]
const RUNG_LABELS = RUNG_BUDGETS.map(String)
/** The final loss of the best configuration of every surface. */
const OPTIMUM = 0.05

/** A space of d hyperparameters, each searched uniformly on [0, 1]. */
const spaceOf = (d: number) =>
  space(
    Object.fromEntries(Array.from({ length: d }, (_, j) => [`x${j + 1}`, real(0, 1)])) as Record<
      string,
      ReturnType<typeof real>
    >,
  )

/** Hyperband's brackets for R and η, repeated `passes` times one after another. */
function passesOf(passes: number): MultiFidelitySchedule {
  const one = hyperbandSchedule({ maxBudget: R, eta: ETA })
  const brackets = Array.from({ length: passes }, () => one.brackets)
    .flat()
    .map((b, i) => ({ ...b, bracket: i }))
  return {
    ...one,
    brackets,
    configurations: one.configurations * passes,
    budget: one.budget * passes,
    resumedBudget: one.resumedBudget * passes,
  }
}

/** The value of a step function of budget at b: the last (budget, value) pair at or before b, NaN before the first. */
function stepAt(curve: readonly (readonly [number, number])[], b: number) {
  let v = NaN
  for (const [at, value] of curve) {
    if (at > b) break
    v = value
  }
  return v
}

type Settings = { d: number; crossing: number; noise: number; passes: number; seeds: number; rho: number }

/** Random search, Hyperband (BOHB with every configuration random) and BOHB on the same seeded learning curves. */
function compare({ d, crossing, noise, passes, seeds, rho }: Settings) {
  const sp = spaceOf(d)
  const schedule = passesOf(passes)
  const steps = schedule.brackets.reduce((a, b) => a + b.rungs.length, 0)
  const total = schedule.budget
  const budgets = Array.from({ length: GRID }, (_, k) => ((k + 1) / GRID) * total)
  const sums = { random: budgets.map(() => 0), hyperband: budgets.map(() => 0), bohb: budgets.map(() => 0) }
  let modelBudgets: { spent: number; budget: number }[] = []
  let drawnRandom = 0
  let drawn = 0
  for (let s = 0; s < seeds; s++) {
    const surface = learningCurveSurface(child(stream('bohb-figure/surface'), s), {
      space: sp,
      crossing,
      noise,
      maxBudget: R,
    })
    for (const [name, fraction] of [
      ['hyperband', 1],
      ['bohb', rho],
    ] as const) {
      const run = trace(bohbSteps(surface, sp, schedule, { rho: fraction }), undefined, steps, {
        stream: child(stream('bohb-figure/run'), s),
        keep: 'all',
      })
      const curve = run.steps.filter((st) => st.incumbent).map((st) => [st.spent, st.incumbent!.final] as const)
      budgets.forEach((b, k) => (sums[name][k] += stepAt(curve, b)))
      if (name === 'bohb') {
        drawnRandom += run.final.random.filter(Boolean).length
        drawn += run.final.points.length
        if (s === 0) {
          modelBudgets = run.steps
            .filter((st) => st.sampled.length > 0 && st.modelBudget !== null)
            .map((st) => ({ spent: st.spent, budget: st.modelBudget! }))
        }
      }
    }
    // Random search: configurations drawn uniformly and each trained to R; it keeps the lowest observed loss.
    const draws = child(stream('bohb-figure/random'), s)
    const curve: [number, number][] = []
    let bestSeen = Infinity
    let bestFinal = NaN
    for (let i = 0; (i + 1) * R <= total; i++) {
      const point = sample(child(draws, 'point', i), sp)
      const seen = surface.at(point, R, child(draws, 'noise', i))
      if (seen < bestSeen) {
        bestSeen = seen
        bestFinal = surface.final(point)
      }
      curve.push([(i + 1) * R, bestFinal])
    }
    budgets.forEach((b, k) => (sums.random[k] += stepAt(curve, b)))
  }
  const mean = (v: number[]) => v.map((x) => x / seeds)
  return {
    budgets: budgets.map((b) => b / R),
    passBudget: schedule.budget / passes / R,
    random: mean(sums.random),
    hyperband: mean(sums.hyperband),
    bohb: mean(sums.bohb),
    modelBudgets: modelBudgets.map((m) => ({ spent: m.spent / R, budget: m.budget })),
    randomShare: drawn ? drawnRandom / drawn : NaN,
  }
}

/**
 * The final loss of the configuration each method returns, against the budget spent, for random search, Hyperband
 * and BOHB on seeded learning curves over d hyperparameters whose best final loss is known.
 */
export function BohbAgainstHyperband() {
  const state = useFigureState({
    d: int(6, { ge: 1, le: 10, suggestions: [2, 4, 6, 8], label: 'hyperparameters d' }),
    crossing: slider(0, 1, 0.5, { step: 0.05, label: 'crossing κ' }),
    noise: float(0, { ge: 0, le: 0.05, suggestions: [0, 0.005, 0.01], label: 'noise sd' }),
    rho: float(1 / 3, { ge: 0, le: 1, suggestions: [0, 1 / 3, 1], label: 'random fraction ρ' }),
    passes: int(3, { ge: 1, le: 5, suggestions: [1, 2, 3, 5], label: 'Hyperband passes' }),
    seeds: int(8, { ge: 1, le: 20, suggestions: [4, 8, 16], label: 'seeds' }),
  })
  const { d, crossing, noise, rho, passes, seeds } = state
  const computed = useComputed(
    () => compare({ d, crossing, noise, rho, passes, seeds }),
    [d, crossing, noise, rho, passes, seeds],
    { mode: 'release' },
  )
  const r = computed.value
  const last = GRID - 1
  const total = r.budgets[last]
  const passMarks = useMemo(
    () => Array.from({ length: passes - 1 }, (_, p) => (p + 1) * r.passBudget),
    [passes, r.passBudget],
  )

  const bx = useAxis({ label: 'budget spent (full trainings of 81 epochs)', range: [0, total] })
  const ly = useAxis({ label: 'final loss', hold: 'union', key: `${d}|${crossing}` })
  const my = useAxis({ label: 'epochs', categories: RUNG_LABELS })
  return (
    <Figure
      title="BOHB, Hyperband and random search at equal budget"
      state={state}
      caption="Seeded learning curves over d hyperparameters in [0, 1]: each configuration's loss falls as a power law from its value at 1 epoch to its final value at R = 81 epochs, which is a smooth bowl with its minimum 0.05 at a random centre. The crossing κ sets how often a configuration that starts well ends badly. Top: the true final loss of the configuration each method would return (the one with the lowest loss seen), against the budget spent, averaged over the seeds. Random search trains every configuration to 81 epochs; Hyperband runs its five brackets with η = 3, repeated for each pass; BOHB runs the same brackets with a fraction ρ of configurations drawn at random and the rest from its density model. ρ = 1 turns BOHB into Hyperband. Bottom: the budget whose results BOHB's model is fitted to when each bracket draws its configurations, for the first seed. The model starts on cheap budgets and moves up to 81 epochs as results accumulate there."
      readouts={{
        [`after ${formatNumber(total)} full trainings`]: (
          <>
            <Readout label="random search" value={formatNumber(r.random[last])} />
            <Readout label="Hyperband" value={formatNumber(r.hyperband[last])} />
            <Readout label="BOHB" value={formatNumber(r.bohb[last])} />
            <Readout label="best possible" value={formatNumber(OPTIMUM)} />
          </>
        ),
        BOHB: <Readout label="configurations drawn at random" value={`${Math.round(100 * r.randomShare)}%`} />,
      }}
    >
      <Plots rows={2} heights={[3, 1.4]}>
        <Plot x={bx} y={ly} title="final loss of the returned configuration">
          <Curve name="random search" x={r.budgets} y={r.random} slot={0} stale={computed.stale} />
          <Curve name="Hyperband" x={r.budgets} y={r.hyperband} slot={2} stale={computed.stale} />
          <Curve name="BOHB" x={r.budgets} y={r.bohb} slot={1} stale={computed.stale} />
          <Annotation y={OPTIMUM} text="best possible" dashed muted />
          {passMarks.map((b, k) => (
            <Annotation key={k} x={b} text={`pass ${k + 2}`} dashed muted />
          ))}
        </Plot>
        <Plot x={bx} y={my} title="BOHB's model budget (first seed)">
          {r.modelBudgets.length > 0 && (
            <Points
              name="model budget"
              x={r.modelBudgets.map((m) => m.spent)}
              y={r.modelBudgets.map((m) => RUNG_BUDGETS.indexOf(m.budget))}
              slot={1}
              stale={computed.stale}
            />
          )}
        </Plot>
      </Plots>
    </Figure>
  )
}
