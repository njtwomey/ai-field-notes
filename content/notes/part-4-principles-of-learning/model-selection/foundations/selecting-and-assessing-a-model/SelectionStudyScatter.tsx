import { type SelectionStudy } from 'aifn-methods/theory/model-selection'
import {
  call,
  choice,
  Curve,
  Figure,
  formatNumber,
  int,
  Plot,
  Plots,
  Points,
  Readout,
  Segments,
  useAxis,
  useComputed,
  useFigureState,
} from 'aifn-render'

/** Points measuring each data set's true error: one fresh sample shared by every data set. */
const FRESH = 20000
const RANGE: [number, number] = [0.03, 0.22]
type Estimate = 'trainingError' | 'bestScore' | 'nestedScore' | 'testScore'
/** The four estimates, in the order of the steps, with the colour slot each has in the per-data-set figure. */
const ESTIMATES: { key: Estimate; label: string; short: string; slot: number }[] = [
  { key: 'trainingError', label: 'training error', short: 'training', slot: 1 },
  { key: 'bestScore', label: 'best 10-fold CV score', short: 'best CV', slot: 0 },
  { key: 'nestedScore', label: 'nested 10 × 10-fold CV', short: 'nested CV', slot: 2 },
  { key: 'testScore', label: 'test error on 25 rows', short: 'test set', slot: 2 },
]

const fmt = (v: number | undefined) =>
  v !== undefined && Number.isFinite(v) ? formatNumber(Number(v.toPrecision(3))) : '—'

/**
 * The selection-and-assessment procedure of the note repeated over many data sets, one point per data set: each
 * estimate of the final model's error against its true error. Points on the diagonal are exact; the bias is the mean
 * offset below or above it, and the spread about the diagonal is each estimate's error on one data set.
 */
export function SelectionStudyScatter() {
  const state = useFigureState({
    shown: choice(
      ESTIMATES.map((e) => ({ value: e.key, label: e.label })),
      'bestScore',
      { label: 'estimate in the scatter' },
    ),
    repeats: int(200, { ge: 10, le: 1000, suggestions: [50, 100, 200, 500], label: 'data sets' }),
    learner: choice(
      [
        { value: 'pspline', label: 'P-spline (15 intervals)' },
        { value: 'ridge-polynomial', label: 'ridge on a degree-10 polynomial' },
      ],
      'pspline',
      { label: 'learner' },
    ),
  })
  const { shown, repeats, learner } = state
  const pick = ESTIMATES.find((e) => e.key === shown) ?? ESTIMATES[1]
  const run = useComputed(
    () =>
      call<SelectionStudy>(
        'theory/model-selection/selectionStudy',
        call('foundation/random/stream', 'selecting-and-assessing'),
        { generator: { target: 'sine', noise: 0.3 }, rows: 100, test: 25, fresh: FRESH, learner, repeats },
      ),
    [repeats, learner],
    { mode: 'worker', initial: null as SelectionStudy | null },
  )
  const r = run.value
  const x = useAxis({ label: 'true error of the final model', range: RANGE })
  const y = useAxis({ label: 'estimate', range: RANGE, equal: x })
  const cx = useAxis({ label: 'estimate', categories: ESTIMATES.map((e) => e.short) })
  const cy = useAxis({ label: 'estimate minus true error' })
  const truth = r?.summary.trueError.mean ?? NaN
  const at = ESTIMATES.map((_, i) => i)
  return (
    <Figure
      title="Four estimates against the true error over many data sets"
      state={state}
      defaultSize="L"
      readouts={
        r !== null ? (
          <>
            <Readout label="data sets" value={String(r.perRepeat.trueError.length)} />
            <Readout label="mean true error" value={fmt(truth)} />
            {ESTIMATES.map((e) => (
              <Readout
                key={e.key}
                label={`${e.label}: mean, bias, sd`}
                value={`${fmt(r.summary[e.key].mean)}, ${fmt(r.summary[e.key].bias)}, ${fmt(r.summary[e.key].sd)}`}
              />
            ))}
          </>
        ) : (
          <Readout label="status" value={run.error ?? 'computing…'} />
        )
      }
      caption={`Each data set has 100 points of y = sin(πx) + 0.3ε with x uniform on [−1, 1]: 75 development rows and 25 test rows. On the development rows a penalty is chosen from 25 values (10⁻⁴ to 10⁴) by shuffled 10-fold cross-validation and the fit at the minimum is the final model. Its true error is measured on ${FRESH.toLocaleString('en-GB')} fresh points. Left: the chosen estimate against the truth, one point per data set; the dashed diagonal is a perfect estimate, and the large point is the mean over data sets. Points below the diagonal are optimistic. The truth varies little across data sets; the estimates vary far more. Right: each estimate's bias, its mean minus the true error, with ±1 standard deviation of its error on one data set. The training error and the best cross-validation score sit below zero; nested cross-validation and the test set sit near it, the test set with the widest spread. The study takes about 15 seconds for 200 data sets and runs in the background.`}
    >
      <Plots cols={2} widths={[1, 1]}>
        <Plot x={x} y={y} title={pick.label}>
          <Curve name="estimate = truth" x={RANGE} y={RANGE} muted dashed />
          {r !== null && r.perRepeat.trueError.length > 0 && (
            <>
              <Points
                name="one data set"
                x={r.perRepeat.trueError}
                y={r.perRepeat[pick.key]}
                slot={pick.slot}
                thin
                stale={run.stale}
              />
              <Points
                name="mean over data sets"
                x={[truth]}
                y={[r.summary[pick.key].mean]}
                emphasis
                size={10}
                stale={run.stale}
              />
            </>
          )}
        </Plot>
        <Plot x={cx} y={cy} title="bias ± 1 sd over data sets">
          <Curve name="no error" x={[-0.5, ESTIMATES.length - 0.5]} y={[0, 0]} muted dashed />
          {r !== null &&
            ESTIMATES.map((e, i) => (
              <Segments
                key={e.key}
                name={`${e.label} ± 1 sd`}
                segments={[
                  {
                    from: [i, r.summary[e.key].bias - r.summary[e.key].sd],
                    to: [i, r.summary[e.key].bias + r.summary[e.key].sd],
                  },
                ]}
                slot={e.slot}
              />
            ))}
          {r !== null &&
            ESTIMATES.map((e, i) => (
              <Points
                key={e.key}
                name={`${e.label}, bias`}
                x={[at[i]]}
                y={[r.summary[e.key].bias]}
                slot={e.slot}
                emphasis={e.key === pick.key}
                stale={run.stale}
              />
            ))}
        </Plot>
      </Plots>
    </Figure>
  )
}
