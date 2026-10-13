import { useMemo } from 'react'
import {
  Annotation,
  Bars,
  Curve,
  Figure,
  float,
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
import { child, stream } from 'aifn-compute/foundation/random'
import { proportionInterval } from 'aifn-compute/probability/tests'
import { replicate, TESTS, type TestKey } from './equalLearners'

const ALPHA = 0.05
const fmt = (v: number) => (Number.isFinite(v) ? formatNumber(Number(v.toPrecision(3))) : '—')

type Props = {
  /** The tests drawn in the emphasis colour; the others are muted. */
  highlight?: readonly TestKey[]
  /** Add a second panel: the rejection rate of the naive and corrected tests against the number of repeats. */
  byRepeats?: boolean
}

/**
 * How often six tests reject the null hypothesis at the 5% level when comparing two learners on simulated data sets.
 * With no handicap the learners are equally good, so every rejection is a false positive.
 */
export function TypeIErrorRates({ highlight = [], byRepeats = false }: Props) {
  const state = useFigureState({
    n: int(100, { ge: 40, le: 400, suggestions: [60, 100, 200, 300], label: 'rows per data set n' }),
    repeats: int(10, { ge: 1, le: 10, suggestions: [1, 5, 10], label: 'repeats r of 10-fold CV' }),
    handicap: float(0, { ge: 0, le: 1, suggestions: [0, 0.3, 0.6], label: "noise on B's inputs" }),
    replicates: int(40, { ge: 20, le: 1000, suggestions: [40, 100, 300], label: 'simulated data sets' }),
  })
  const { n, repeats, handicap, replicates } = state
  const computed = useComputed(
    () => {
      const root = stream('comparing/type-i')
      return Array.from({ length: replicates }, (_, r) => replicate(child(root, r), n, handicap, repeats))
    },
    [n, repeats, handicap, replicates],
    { mode: 'release' },
  )
  const rows = computed.value
  const view = useMemo(() => {
    const count = rows.length
    const rate = (k: number) => {
      const ci = proportionInterval(k, count)
      return { rate: k / count, lower: ci.lower, upper: ci.upper }
    }
    const tests = TESTS.map((t) => ({ ...t, ...rate(rows.filter((r) => r.p[t.key] <= ALPHA).length) }))
    const rs = Array.from({ length: repeats }, (_, i) => i + 1)
    const naive = rs.map((r) => rows.filter((row) => row.naiveByRepeats[r - 1] <= ALPHA).length / count)
    const corrected = rs.map((r) => rows.filter((row) => row.correctedByRepeats[r - 1] <= ALPHA).length / count)
    const trueDifference = rows.reduce((a, r) => a + r.trueDifference, 0) / count
    return { tests, rs, naive, corrected, trueDifference }
  }, [rows, repeats])
  const emphasised = (key: TestKey) => highlight.length === 0 || highlight.includes(key)
  const position = (key: TestKey) => TESTS.findIndex((t) => t.key === key)
  const on = view.tests.filter((t) => emphasised(t.key))
  const off = view.tests.filter((t) => !emphasised(t.key))
  const tx = useAxis({ categories: TESTS.map((t) => t.label) })
  const ry = useAxis({ label: 'rejection rate at α = 0.05', range: [0, undefined] })
  const jx = useAxis({ label: 'repeats r of 10-fold CV (J = 10r splits)', integer: true, range: [1, 10] })
  const jy = useAxis({ label: 'rejection rate at α = 0.05', range: [0, undefined] })
  const intervals = view.tests.map((t) => ({
    from: [position(t.key), t.lower] as const,
    to: [position(t.key), t.upper] as const,
  }))
  const testsPlot = (
    <Plot x={tx} y={ry}>
      {on.length > 0 && (
        <Bars
          name="rejection rate"
          x={on.map((t) => position(t.key))}
          y={on.map((t) => t.rate)}
          slot={0}
          width={0.6}
          stale={computed.stale}
        />
      )}
      {off.length > 0 && (
        <Bars name="other tests" x={off.map((t) => position(t.key))} y={off.map((t) => t.rate)} muted width={0.6} />
      )}
      <Segments name="95% interval (Wilson)" segments={intervals} width={2} />
      <Annotation y={ALPHA} text="nominal 5%" dashed emphasis />
    </Plot>
  )
  return (
    <Figure
      title="False positives when comparing two learners"
      state={state}
      caption={`Each simulated data set has n rows of two standard-normal inputs, labelled 1 when the radius plus noise exceeds its median, so the class boundary is a circle. Learner A is a depth-4 decision tree on the inputs; learner B is the same tree on the inputs rotated by 45°. A circle looks the same after rotation, so the two learners have equal expected accuracy and every rejection is a false positive. The tests are McNemar's test on a random hold-out third; the paired t-test on one 10-fold cross-validation and on r repeats of it (r×10-fold); the corrected resampled t-test on the r repeats; and Dietterich's 5×2cv t-test and Alpaydın's combined F test. Each bar is the share of data sets on which a test rejects at the 5% level, with a 95% Wilson interval for that share; a valid test sits at or below the dashed line. ${byRepeats ? 'The second panel repeats 10-fold cross-validation r times and tests the J = 10r differences: the naive paired t-test rejects more often the more splits it is given, while the corrected test stays put. ' : ''}Adding noise to B's inputs makes B worse, so the bars then measure power.`}
      readouts={{
        'expected accuracy, A minus B': (
          <Readout label="average over data sets, on fresh data" value={fmt(view.trueDifference)} />
        ),
        'rejection rate': (
          <>
            {view.tests
              .filter((t) => emphasised(t.key))
              .map((t) => (
                <Readout key={t.key} label={t.label} value={fmt(t.rate)} />
              ))}
          </>
        ),
      }}
    >
      {byRepeats ? (
        <Plots cols={2}>
          {testsPlot}
          <Plot x={jx} y={jy}>
            <Curve name="paired t (naive)" x={view.rs} y={view.naive} slot={1} />
            <Points name="paired t (naive)" x={view.rs} y={view.naive} slot={1} size={5} />
            <Curve name="corrected resampled t" x={view.rs} y={view.corrected} slot={0} />
            <Points name="corrected resampled t" x={view.rs} y={view.corrected} slot={0} size={5} />
            <Annotation y={ALPHA} text="nominal 5%" dashed emphasis />
          </Plot>
        </Plots>
      ) : (
        testsPlot
      )}
    </Figure>
  )
}
