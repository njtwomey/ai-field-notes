import { useMemo } from 'react'
import {
  Annotation,
  choice,
  Curve,
  Figure,
  formatNumber,
  int,
  Plot,
  Plots,
  Points,
  Raster,
  Readout,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { toArray, toFlat } from 'aifn-compute/foundation/tensor'
import { tuningTables } from 'aifn-methods/data/objectives'
import { averageRank, greedyPortfolio, leaveOneTaskOut, portfolioLoss } from 'aifn-methods/optimisation/portfolio'

/** Seven tasks, each a 21 × 21 table of cross-validated SVM error over (log10 C, log10 γ). */
const TABLES = tuningTables()
const TASKS = TABLES.tasks
const LOG_C = Array.from(toFlat(TABLES.logC))
const LOG_GAMMA = Array.from(toFlat(TABLES.logGamma))
/** error[t][i][j]: task t at (log10 C_i, log10 γ_j). */
const ERROR = toArray(TABLES.error) as number[][][]
/** The configuration by task table, 441 × 7, and the configurations as (log10 C, log10 γ) rows. */
const TABLE = toArray(TABLES.table) as number[][]
const CONFIGS = toArray(TABLES.configurations) as number[][]
/** The mean error over the tasks at each grid point. */
const MEAN_ERROR = LOG_C.map((_, i) => LOG_GAMMA.map((_, j) => ERROR.reduce((a, t) => a + t[i][j], 0) / TASKS.length))
/** Each task's lowest error, and the configuration that reaches it. */
const TASK_BEST = TASKS.map((_, t) => {
  const column = TABLE.map((row) => row[t])
  const c = column.indexOf(Math.min(...column))
  return { config: c, error: column[c] }
})
/** The mean over tasks of each task's own best error: what one tuned configuration per task would reach. */
const ORACLE = TASK_BEST.reduce((a, b) => a + b.error, 0) / TASKS.length
/** Candidates ordered by their average rank over the tasks, the baseline ordering. */
const BY_RANK = averageRank(TABLE).order

type HeldOut = 'none' | (typeof TASKS)[number]

/**
 * A greedy portfolio of SVM configurations chosen on seven tuning tables: its mean best error on the tasks that chose
 * it, on each task left out in turn, and the average-rank ordering it is set against.
 */
export function PortfolioAcrossTasks() {
  const state = useFigureState({
    size: int(8, { ge: 1, le: 30, suggestions: [1, 2, 4, 8, 16], label: 'portfolio size k' }),
    heldOut: choice<HeldOut>(
      [{ value: 'none', label: 'none (all seven choose)' }, ...TASKS.map((t) => ({ value: t, label: t }))],
      'none',
      { label: 'task left out' },
    ),
  })
  const { size, heldOut } = state

  const result = useMemo(() => {
    const greedy = greedyPortfolio(TABLE, size, { exhaustive: 0 })
    const loto = leaveOneTaskOut(TABLE, size)
    const rank = Array.from({ length: size }, (_, s) => portfolioLoss(TABLE, BY_RANK.slice(0, s + 1)).loss)
    return {
      order: greedy.order,
      inSample: greedy.loss,
      heldOut: Array.from(toFlat(loto.mean)),
      perTask: toArray(loto.loss) as number[][],
      lotoOrder: loto.order,
      rank,
    }
  }, [size])

  const sizes = Array.from({ length: size }, (_, s) => s + 1)
  const task = heldOut === 'none' ? -1 : TASKS.indexOf(heldOut)
  const members = task < 0 ? result.order : result.lotoOrder[task]
  const field = task < 0 ? MEAN_ERROR : ERROR[task]
  const last = size - 1

  const gx = useAxis({ label: 'log₁₀ γ', range: [-3.125, 2.125] })
  const gy = useAxis({ label: 'log₁₀ C', range: [-2.125, 3.125], equal: gx })
  const sx = useAxis({ label: 'portfolio size s', range: [0.5, Math.max(size, 2) + 0.5], integer: true })
  const sy = useAxis({ label: 'mean over tasks of the best member’s error', range: [0.06, 0.2] })
  return (
    <Figure
      title="A portfolio chosen on six tasks, scored on the seventh"
      state={state}
      caption="Seven tuning tables: the cross-validated error of an RBF support vector machine on a 21 × 21 grid of log₁₀ C and log₁₀ γ, on seven small classification tasks, so 441 candidate configurations. Left: the mean error over the tasks, with the greedy portfolio's members numbered in the order they were added. Choose a task to leave out: the map then shows that task's error, the numbers are the portfolio built on the other six, and the green point marks the task's own best setting. Right: the mean over tasks of the best member's error, against the portfolio size. Orange is in-sample (the tasks that chose the portfolio score it); green leaves each task out in turn and scores it with the portfolio built on the other six; blue orders the candidates by average rank instead of greedily. The dashed line is the mean of each task's own best error."
      readouts={{
        [`at k = ${size}`]: (
          <>
            <Readout label="greedy, in-sample" value={formatNumber(result.inSample[last])} />
            <Readout label="greedy, task left out" value={formatNumber(result.heldOut[last])} />
            <Readout label="average rank, in-sample" value={formatNumber(result.rank[last])} />
            <Readout label="each task’s own best" value={formatNumber(ORACLE)} />
          </>
        ),
        ...(task >= 0
          ? {
              [heldOut]: (
                <>
                  <Readout label="portfolio’s best error" value={formatNumber(result.perTask[task][last])} />
                  <Readout label="task’s own best" value={formatNumber(TASK_BEST[task].error)} />
                </>
              ),
            }
          : {}),
      }}
    >
      <Plots cols={2}>
        <Plot x={gx} y={gy} title={task < 0 ? 'mean error, all tasks' : `error on ${heldOut}`}>
          <Raster x={LOG_GAMMA} y={LOG_C} z={field} range={[0, 0.5]} valueLabel="validation error" />
          <Points
            name="portfolio member"
            x={members.map((c) => CONFIGS[c][1])}
            y={members.map((c) => CONFIGS[c][0])}
            labels={members.map((_, s) => String(s + 1))}
            emphasis
          />
          {task >= 0 && (
            <Points
              name="the task’s best setting"
              x={[CONFIGS[TASK_BEST[task].config][1]]}
              y={[CONFIGS[TASK_BEST[task].config][0]]}
              slot={2}
              size={11}
            />
          )}
        </Plot>
        <Plot x={sx} y={sy} title="best member’s error against portfolio size">
          <Curve name="greedy, in-sample" x={sizes} y={result.inSample} slot={1} showPoints />
          <Curve name="greedy, task left out" x={sizes} y={result.heldOut} slot={2} showPoints />
          <Curve name="average rank, in-sample" x={sizes} y={result.rank} slot={0} dashed showPoints />
          <Annotation y={ORACLE} text="each task’s own best" dashed muted />
        </Plot>
      </Plots>
    </Figure>
  )
}
