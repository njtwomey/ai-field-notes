import { useMemo } from 'react'
import {
  Figure,
  int,
  Readout,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
  useFigureState,
} from 'aifn-render'
import { hyperbandSchedule } from 'aifn-methods/optimisation/multi-fidelity'

/**
 * BOHB's choice of model budget over Hyperband's runs of successive halving: before each run samples its
 * configurations, the results available at each budget, and the largest budget with at least N_min + 2 of them.
 */
export function BohbModelBudget() {
  const state = useFigureState({
    maxBudget: int(81, { ge: 3, le: 243, suggestions: [9, 27, 81], label: 'maximum budget R' }),
    eta: int(3, { ge: 2, le: 4, suggestions: [2, 3, 4], label: 'reduction factor η' }),
    d: int(2, { ge: 1, le: 30, suggestions: [2, 6, 10], label: 'hyperparameters d' }),
    passes: int(2, { ge: 1, le: 3, label: 'Hyperband passes' }),
  })
  const { maxBudget, eta, d, passes } = state

  const table = useMemo(() => {
    const schedule = hyperbandSchedule({ maxBudget, eta })
    const budgets = Array.from(new Set(schedule.brackets.flatMap((b) => b.rungs.map((g) => g.budget)))).sort(
      (a, b) => a - b,
    )
    const need = d + 1 + 2
    const counts = new Map<number, number>(budgets.map((b) => [b, 0]))
    const rows: { pass: number; s: number; start: string; counts: number[]; model: number | null }[] = []
    for (let p = 0; p < passes; p++) {
      for (const b of schedule.brackets) {
        const ready = budgets.filter((r) => counts.get(r)! >= need)
        rows.push({
          pass: p + 1,
          s: b.s,
          start: `${b.rungs[0].configurations} at ${b.rungs[0].budget}`,
          counts: budgets.map((r) => counts.get(r)!),
          model: ready.length ? Math.max(...ready) : null,
        })
        for (const g of b.rungs) counts.set(g.budget, counts.get(g.budget)! + g.configurations)
      }
    }
    return { budgets, rows, need }
  }, [maxBudget, eta, d, passes])

  const firstModel = table.rows.findIndex((r) => r.model !== null)
  const firstFull = table.rows.findIndex((r) => r.model === maxBudget)
  return (
    <Figure
      title="Which budget BOHB models"
      state={state}
      caption="BOHB runs Hyperband's brackets in order. Before each run of successive halving draws its configurations, BOHB fits its density model to the results at one budget: the largest budget with at least N_min + 2 results, where N_min = d + 1. Each row is one run of successive halving; the columns n@r count the results available at budget r when the run starts (runs 1 to s_max + 1 are the first Hyperband pass), and the highlighted cell is the budget whose results the model uses (none: the configurations are drawn at random). Raise the number of hyperparameters d: the model needs more results, so it starts later and stays longer on cheap, less reliable budgets."
      readouts={
        <>
          <Readout label="results needed at a budget, N_min + 2" value={String(table.need)} />
          <Readout label="first run with a model" value={firstModel < 0 ? 'none' : `run ${firstModel + 1}`} />
          <Readout label="first run modelling budget R" value={firstFull < 0 ? 'none' : `run ${firstFull + 1}`} />
        </>
      }
    >
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>run</TableHead>
            <TableHead>s</TableHead>
            <TableHead>starts</TableHead>
            {table.budgets.map((b) => (
              <TableHead key={b} className="text-right">
                n@{b}
              </TableHead>
            ))}
            <TableHead className="text-right">model</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {table.rows.map((row, i) => (
            <TableRow key={i}>
              <TableCell className="py-1 font-mono text-xs">{i + 1}</TableCell>
              <TableCell className="py-1 font-mono text-xs">{row.s}</TableCell>
              <TableCell className="py-1 font-mono text-xs">{row.start}</TableCell>
              {row.counts.map((c, j) => (
                <TableCell
                  key={j}
                  className={`py-1 text-right font-mono text-xs tabular-nums ${
                    table.budgets[j] === row.model ? 'bg-muted font-semibold' : ''
                  }`}
                >
                  {c}
                </TableCell>
              ))}
              <TableCell className="py-1 text-right font-mono text-xs tabular-nums">{row.model ?? 'random'}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </Figure>
  )
}
