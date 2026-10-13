import { useMemo } from 'react'
import {
  Curve,
  Figure,
  float,
  formatNumber,
  int,
  Plot,
  Points,
  Readout,
  slider,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { child, stream } from 'aifn-compute/foundation/random'
import { toArray, toFlat } from 'aifn-compute/foundation/tensor'
import { run } from 'aifn-compute/foundation/trace'
import { learningCurves, type LearningCurves } from 'aifn-methods/data/synthetic'
import {
  hyperbandSchedule,
  multiFidelitySteps,
  successiveHalvingSchedule,
  type MultiFidelitySchedule,
} from 'aifn-methods/optimisation/multi-fidelity'

const CROSSINGS = [0, 0.2, 0.4, 0.6, 0.8, 1]
const SEEDS = 20

const budgetsTo = (R: number) => Array.from({ length: R }, (_, i) => i + 1)

/** Curves read once: the budgets, the observed rows and the final losses. */
type CurveRows = { budgets: LearningCurves['budgets']; values: number[][]; final: ArrayLike<number> }

const tableOf = (curves: LearningCurves): CurveRows => ({
  budgets: curves.budgets,
  values: toArray(curves.values) as number[][],
  final: toFlat(curves.final),
})

/** The final loss of the configuration a run returns: the one with the lowest loss seen, as Hyperband returns. */
function returnedFinal(table: CurveRows, schedule: MultiFidelitySchedule, rows?: number[]) {
  const values = rows ? rows.map((i) => table.values[i]) : table.values
  const end = run(multiFidelitySteps({ budgets: table.budgets, values }, schedule), undefined, 10_000)
  const config = rows ? rows[end.incumbent!.config] : end.incumbent!.config
  return { loss: end.incumbent!.loss, final: table.final[config] }
}

/**
 * Hyperband's brackets as a table of n_i and r_i with their budgets, one run over seeded learning curves, and the
 * final loss each bracket and Hyperband return at equal budget as the curves cross more.
 */
export function HyperbandBrackets() {
  const state = useFigureState({
    maxBudget: int(27, { ge: 3, le: 81, suggestions: [9, 27, 81], label: 'maximum budget R' }),
    eta: int(3, { ge: 2, le: 4, suggestions: [2, 3, 4], label: 'reduction factor η' }),
    crossing: slider(0, 1, 0.8, { step: 0.05, label: 'crossing κ (this run)' }),
    noise: float(0.01, { ge: 0, le: 0.1, suggestions: [0, 0.01, 0.03], label: 'noise sd' }),
    seed: int(1, { ge: 1, le: 999, label: 'seed' }),
  })
  const { maxBudget, eta, crossing, noise, seed } = state

  const schedule = useMemo(() => hyperbandSchedule({ maxBudget, eta }), [maxBudget, eta])
  const sMax = schedule.brackets.length - 1

  // One Hyperband run on one family of curves: what each bracket returns.
  const single = useMemo(() => {
    const curves = learningCurves(stream(`hyperband/${seed}`), {
      n: schedule.configurations,
      budgets: budgetsTo(maxBudget),
      crossing,
      noise,
    })
    const end = run(multiFidelitySteps(curves, schedule), undefined, 10_000)
    const final = toFlat(curves.final)
    const brackets = schedule.brackets.map((b) => {
      const ev = end.evaluations.filter((e) => e.bracket === b.bracket)
      const top = Math.max(...ev.map((e) => e.rung))
      const atTop = ev.filter((e) => e.rung === top)
      const pick = atTop.reduce((a, e) => (e.loss < a.loss ? e : a), atTop[0])
      const members = ev.filter((e) => e.rung === 0).map((e) => e.config)
      const bestMember = members.reduce((a, i) => (final[i] < final[a] ? i : a), members[0])
      return {
        pick: final[pick.config],
        survived: atTop.some((e) => e.config === bestMember),
      }
    })
    return { brackets, hyperband: final[end.incumbent!.config], best: final[curves.best] }
  }, [schedule, maxBudget, crossing, noise, seed])

  // Over seeds and crossings: Hyperband against each bracket repeated s_max + 1 times (about the same budget).
  const study = useMemo(() => {
    const budgets = budgetsTo(maxBudget)
    const repeats = sMax + 1
    const need = Math.max(schedule.configurations, ...schedule.brackets.map((b) => b.rungs[0].configurations * repeats))
    const brackets = schedule.brackets.map((b) =>
      successiveHalvingSchedule({
        n: b.rungs[0].configurations,
        minBudget: b.rungs[0].budget,
        maxBudget,
        eta,
      }),
    )
    const hyperband: number[] = []
    const alone: number[][] = brackets.map(() => [])
    for (const kappa of CROSSINGS) {
      let hb = 0
      const sums = brackets.map(() => 0)
      for (let s = 0; s < SEEDS; s++) {
        const curves = tableOf(
          learningCurves(child(stream('hyperband/study'), s), { n: need, budgets, crossing: kappa, noise }),
        )
        hb += returnedFinal(curves, schedule).final
        brackets.forEach((sh, j) => {
          const n0 = sh.rungs[0].configurations
          let bestLoss = Infinity
          let bestFinal = 0
          for (let k = 0; k < repeats; k++) {
            const r = returnedFinal(
              curves,
              sh,
              Array.from({ length: n0 }, (_, i) => n0 * k + i),
            )
            if (r.loss < bestLoss) {
              bestLoss = r.loss
              bestFinal = r.final
            }
          }
          sums[j] += bestFinal
        })
      }
      hyperband.push(hb / SEEDS)
      sums.forEach((v, j) => alone[j].push(v / SEEDS))
    }
    return { hyperband, alone }
  }, [schedule, sMax, maxBudget, eta, noise])

  const xAxis = useAxis({ label: 'crossing κ', range: [0, 1] })
  const yAxis = useAxis({ label: 'mean final loss of the returned configuration' })
  return (
    <Figure
      title="Hyperband's brackets and the hedge they buy"
      state={state}
      caption="The table lists Hyperband's brackets for the maximum budget R and reduction factor η: bracket s starts n₀ configurations at budget R/η^s and runs successive halving, so each bracket spends about (s_max + 1)R. Its last two columns come from one run over a seeded family of learning curves at the chosen crossing κ: the final loss of the configuration each bracket returns, and whether the bracket's own best configuration reached its last rung. The chart averages over 20 seeds: each coloured line is one bracket repeated s_max + 1 times, which costs about as much as one Hyperband run; the ink line is Hyperband. The most aggressive bracket wins when early rankings hold (small κ) and collapses when curves cross; Hyperband is never the best and never far from it."
      readouts={
        <>
          <Readout label="brackets s_max + 1" value={String(sMax + 1)} />
          <Readout label="configurations started" value={String(schedule.configurations)} />
          <Readout label="total budget (resumed)" value={`${schedule.budget} (${schedule.resumedBudget})`} />
          <Readout label="this run: Hyperband returns" value={formatNumber(single.hyperband)} />
          <Readout label="this run: best final loss" value={formatNumber(single.best)} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis}>
        {study.alone.map((ys, j) => (
          <Curve
            key={j}
            name={`bracket s = ${schedule.brackets[j].s}, ×${sMax + 1}`}
            x={CROSSINGS}
            y={ys}
            slot={j}
            showPoints
          />
        ))}
        <Curve name="Hyperband" x={CROSSINGS} y={study.hyperband} emphasis width={3} />
        <Points name="Hyperband" x={CROSSINGS} y={study.hyperband} emphasis />
      </Plot>
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>bracket s</TableHead>
            <TableHead>rungs n_i @ r_i</TableHead>
            <TableHead className="text-right">budget</TableHead>
            <TableHead className="text-right">resumed</TableHead>
            <TableHead className="text-right">returns (this run)</TableHead>
            <TableHead className="text-center">own best survives</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {schedule.brackets.map((b, j) => (
            <TableRow key={b.bracket}>
              <TableCell className="py-1 font-mono text-xs">{b.s}</TableCell>
              <TableCell className="py-1 font-mono text-xs">
                {b.rungs.map((g) => `${g.configurations}@${g.budget}`).join(' → ')}
              </TableCell>
              <TableCell className="py-1 text-right font-mono text-xs tabular-nums">{b.budget}</TableCell>
              <TableCell className="py-1 text-right font-mono text-xs tabular-nums">{b.resumedBudget}</TableCell>
              <TableCell className="py-1 text-right font-mono text-xs tabular-nums">
                {formatNumber(single.brackets[j].pick)}
              </TableCell>
              <TableCell className="py-1 text-center text-xs">{single.brackets[j].survived ? '✓' : '✗'}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </Figure>
  )
}
