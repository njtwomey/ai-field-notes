import { useMemo, useState } from 'react'
import {
  Annotation,
  Curve,
  Figure,
  float,
  formatNumber,
  int,
  Player,
  Plot,
  Points,
  Readout,
  setting,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { child, stream } from 'aifn-compute/foundation/random'
import { toArray, toFlat } from 'aifn-compute/foundation/tensor'
import { run, trace } from 'aifn-compute/foundation/trace'
import { learningCurves } from 'aifn-methods/data/synthetic'
import { multiFidelitySteps, successiveHalvingSchedule } from 'aifn-methods/optimisation/multi-fidelity'

const SEEDS = 100

/** Budgets 1, 2, …, R: the curves are tabulated at every integer budget. */
const budgetsTo = (R: number) => Array.from({ length: R }, (_, i) => i + 1)

/** One polyline through several curves, each cut at its own last budget, separated by NaN. */
function joined(rows: number[][], configs: number[], upTo: (i: number) => number) {
  const x: number[] = []
  const y: number[] = []
  for (const i of configs) {
    const last = upTo(i)
    if (last < 1) continue
    for (let t = 1; t <= last; t++) {
      x.push(t)
      y.push(rows[i][t - 1])
    }
    x.push(NaN)
    y.push(NaN)
  }
  return { x, y }
}

/**
 * Successive halving run over a seeded family of learning curves: which configurations each rung keeps, how far each
 * was trained, and whether the configuration with the best final loss survives. A second readout repeats the run
 * over 100 seeds.
 */
export function SuccessiveHalvingRun() {
  const state = useFigureState({
    n: int(9, { ge: 2, le: 243, suggestions: [9, 27, 81], label: 'configurations n' }),
    eta: int(3, { ge: 2, le: 6, suggestions: [2, 3, 4], label: 'reduction factor η' }),
    maxBudget: int(9, { ge: 2, le: 81, suggestions: [9, 27, 81], label: 'maximum budget R (epochs)' }),
    crossing: slider(0, 1, 0.5, { step: 0.05, label: 'crossing κ' }),
    noise: float(0, { ge: 0, le: 0.1, suggestions: [0, 0.01, 0.03], label: 'noise sd' }),
    seed: int(4, { ge: 1, le: 999, label: 'seed' }),
    hidden: setting(true, 'show the unseen rest of each curve'),
  })
  const { n, eta, maxBudget, crossing, noise, seed, hidden } = state

  const result = useMemo(() => {
    const budgets = budgetsTo(maxBudget)
    const curves = learningCurves(stream(seed), { n, budgets, crossing, noise })
    const schedule = successiveHalvingSchedule({ n, minBudget: 1, maxBudget, eta })
    const steps = trace(multiFidelitySteps(curves, schedule), undefined, schedule.rungs.length + 1, {
      keep: 'all',
    }).steps
    return {
      observed: toArray(curves.values) as number[][],
      clean: toArray(curves.clean) as number[][],
      final: Array.from(toFlat(curves.final)),
      best: curves.best,
      schedule,
      steps,
    }
  }, [n, eta, maxBudget, crossing, noise, seed])

  // The same schedule over many seeds: how often the best configuration reaches the last rung.
  const repeated = useMemo(() => {
    const budgets = budgetsTo(maxBudget)
    const schedule = successiveHalvingSchedule({ n, minBudget: 1, maxBudget, eta })
    const top = schedule.rungs.length - 1
    let survived = 0
    let regret = 0
    for (let s = 0; s < SEEDS; s++) {
      const curves = learningCurves(child(stream('successive-halving/repeat'), s), { n, budgets, crossing, noise })
      const end = run(multiFidelitySteps(curves, schedule), undefined, top + 2)
      if (end.evaluations.some((e) => e.config === curves.best && e.rung === top)) survived++
      const final = toFlat(curves.final)
      if (end.incumbent) regret += final[end.incumbent.config] - final[curves.best]
    }
    return { survived: survived / SEEDS, regret: regret / SEEDS }
  }, [n, eta, maxBudget, crossing, noise])

  const key = `${n}|${eta}|${maxBudget}|${crossing}|${noise}|${seed}`
  const [cursor, setCursor] = useState({ key, step: 0 })
  const count = result.steps.length
  const at = Math.min(cursor.key === key ? cursor.step : 0, count - 1)
  const now = result.steps[at]

  // How far each configuration has been trained by this step, and whether it is still in the race.
  const reached = new Array<number>(n).fill(0)
  for (const e of now.evaluations) reached[e.config] = Math.max(reached[e.config], e.budget)
  const alive = new Set<number>(now.queue ? now.queue.configurations : now.evaluated)
  const all = Array.from({ length: n }, (_, i) => i)
  const live = joined(
    result.observed,
    all.filter((i) => alive.has(i) && i !== result.best),
    (i) => reached[i],
  )
  const dropped = joined(
    result.observed,
    all.filter((i) => !alive.has(i) && i !== result.best),
    (i) => reached[i],
  )
  const unseen = joined(result.clean, all, () => maxBudget)
  const bestPath = joined(result.observed, [result.best], (i) => reached[i])
  const bestAlive = alive.has(result.best)
  const rungBudgets = result.schedule.rungs.map((g) => g.budget)
  const allFull = n * maxBudget
  const incumbent = now.incumbent

  const xAxis = useAxis({ label: 'budget (epochs)', log: true, range: [1, maxBudget] })
  const yAxis = useAxis({ label: 'validation loss', range: [0, 1.05] })
  const rung = now.rung
  return (
    <Figure
      title="Successive halving over learning curves"
      state={state}
      caption="Each line is one configuration's validation loss against its training budget, from a seeded family of power-law curves whose final losses are known. The crossing κ sets how often a curve that starts well ends badly. Step through the rungs: at each rung every surviving configuration is trained to the rung's budget (vertical lines), the best 1/η are kept (orange) and the rest stop (grey). The configuration with the lowest final loss is drawn in ink, dashed beyond the point it was trained to. With the opening settings (n = 9, η = 3, R = 9, κ = 0.5, no noise, seed 4) it ranks fourth at budget 1 and is dropped, as in the worked example. The last readouts repeat the run over 100 seeds."
      controls={
        <Player
          value={at}
          onChange={(v) => setCursor({ key, step: v })}
          count={count}
          label="rung"
          format={(p) => (p === 0 ? 'start' : `rung ${p - 1}`)}
        />
      }
      readouts={
        <>
          <Readout
            label="rung"
            value={at === 0 ? 'none yet' : `${rung}: ${now.evaluated.length} at budget ${now.budget}`}
          />
          <Readout
            label="budget spent"
            value={`${now.spent} of ${result.schedule.budget} (${allFull} to train all to R)`}
          />
          <Readout label="best configuration still in the race" value={bestAlive ? 'yes' : 'no'} />
          <Readout
            label="final loss: incumbent / best"
            value={
              incumbent
                ? `${formatNumber(result.final[incumbent.config])} / ${formatNumber(result.final[result.best])}`
                : `— / ${formatNumber(result.final[result.best])}`
            }
          />
          <Readout label={`best survives, over ${SEEDS} seeds`} value={`${Math.round(100 * repeated.survived)}%`} />
          <Readout label="mean regret of the result" value={formatNumber(repeated.regret)} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis}>
        {hidden && <Curve name="not yet seen" x={unseen.x} y={unseen.y} muted thin silent />}
        <Curve name="stopped" x={dropped.x} y={dropped.y} muted width={1.5} silent />
        <Curve name="still training" x={live.x} y={live.y} slot={1} width={1.5} silent />
        {hidden && (
          <Curve
            name="best final loss (unseen part)"
            x={budgetsTo(maxBudget)}
            y={result.clean[result.best]}
            emphasis
            dashed
            thin
            silent
          />
        )}
        {bestPath.x.length > 0 && <Curve name="best final loss" x={bestPath.x} y={bestPath.y} emphasis />}
        {at > 0 && (
          <Points
            name="earlier evaluations"
            x={now.evaluations.map((e) => e.budget)}
            y={now.evaluations.map((e) => e.loss)}
            muted
            thin
          />
        )}
        {at > 0 && (
          <Points
            name="evaluated at this rung"
            x={now.evaluated.map(() => now.budget)}
            y={now.losses}
            slot={1}
            size={7}
          />
        )}
        {rungBudgets.map((b, k) => (
          <Annotation key={k} x={b} text={`r${k} = ${b}`} dashed muted />
        ))}
      </Plot>
    </Figure>
  )
}
