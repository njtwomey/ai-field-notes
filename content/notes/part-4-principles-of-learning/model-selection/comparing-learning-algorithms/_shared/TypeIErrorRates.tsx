import { useMemo } from 'react'
import {
  Annotation,
  Bars,
  call,
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
  useFigureState,
  useStreamed,
} from 'aifn-render'
import { proportionInterval } from 'aifn-compute/probability/tests'
import { type TypeIErrorStudy, type TypeITest } from 'aifn-methods/theory/algorithm-comparison'
import { TESTS } from './equalLearners'

const ALPHA = 0.05
const fmt = (v: number) => (Number.isFinite(v) ? formatNumber(Number(v.toPrecision(3))) : '—')
const position = (key: TypeITest) => TESTS.findIndex((t) => t.key === key)

type Props = {
  /** The tests drawn in the emphasis colour; the others are muted. */
  highlight?: readonly TypeITest[]
  /** Add a second panel: the rejection rate of the naive and corrected tests against the number of repeats. */
  byRepeats?: boolean
}

/**
 * How often six tests reject the null hypothesis at the 5% level when comparing two learners on simulated data sets
 * (`typeIErrorStudySteps`, streamed from the worker one data set at a time). With no handicap the learners are equally
 * good, so every rejection is a false positive.
 */
export function TypeIErrorRates({ highlight = [], byRepeats = false }: Props) {
  const state = useFigureState({
    n: int(100, { ge: 40, le: 400, suggestions: [60, 100, 200, 300], label: 'rows per data set n' }),
    repeats: int(10, { ge: 1, le: 10, suggestions: [1, 5, 10], label: 'repeats r of 10-fold CV' }),
    handicap: float(0, { ge: 0, le: 1, suggestions: [0, 0.3, 0.6], label: "noise on B's inputs" }),
    replicates: int(40, { ge: 20, le: 1000, suggestions: [40, 100, 300], label: 'simulated data sets' }),
  })
  const { n, repeats, handicap, replicates } = state
  const task = useMemo(
    () =>
      call<TypeIErrorStudy>(
        'theory/algorithm-comparison/typeIErrorStudySteps',
        call('foundation/random/stream', 'comparing/type-i'),
        { n, repeats, handicap, replicates, alpha: ALPHA },
      ),
    [n, repeats, handicap, replicates],
  )
  const run = useStreamed(task)
  const study = run.value
  const view = useMemo(() => {
    if (study === null) return null
    const tests = TESTS.map((t) => {
      const ci = proportionInterval(study.rejections[t.key], study.replicates)
      return { ...t, rate: study.rates[t.key], lower: ci.lower, upper: ci.upper }
    })
    return {
      tests,
      rs: study.byRepeats.repeats,
      naive: Array.from(study.byRepeats.naive),
      corrected: Array.from(study.byRepeats.corrected),
    }
  }, [study])
  const emphasised = (key: TypeITest) => highlight.length === 0 || highlight.includes(key)
  const tests = view?.tests ?? []
  const on = tests.filter((t) => emphasised(t.key))
  const off = tests.filter((t) => !emphasised(t.key))
  const tx = useAxis({ categories: TESTS.map((t) => t.axis) })
  const ry = useAxis({ label: 'rejection rate at α = 0.05', range: [0, 1] })
  const jx = useAxis({ label: 'repeats r of 10-fold CV (J = 10r splits)', integer: true, range: [1, 10] })
  const jy = useAxis({ label: 'rejection rate at α = 0.05', range: [0, 1] })
  const intervals = tests.map((t) => ({
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
          stale={run.running}
        />
      )}
      {off.length > 0 && (
        <Bars name="other tests" x={off.map((t) => position(t.key))} y={off.map((t) => t.rate)} muted width={0.6} />
      )}
      {intervals.length > 0 && <Segments name="95% interval (Wilson)" segments={intervals} width={2} />}
      <Annotation y={ALPHA} text="nominal 5%" dashed emphasis />
    </Plot>
  )
  const progress = study === null ? '—' : `${study.replicates} of ${replicates}${run.running ? ', running' : ''}`
  return (
    <Figure
      title="False positives when comparing two learners"
      state={state}
      caption={`Each simulated data set has n rows of two standard-normal inputs, labelled 1 when the radius plus noise exceeds its median, so the class boundary is a circle. Learner A is a depth-4 decision tree on the inputs; learner B is the same tree on the inputs rotated by 45°. A circle looks the same after rotation, so the two learners have equal expected accuracy and every rejection is a false positive. The tests are McNemar's test on a random hold-out third; the paired t-test on one 10-fold cross-validation and on r repeats of it (r×10-fold); the corrected resampled t-test on the r repeats; and Dietterich's 5×2cv t-test and Alpaydın's combined F test. Each bar is the share of data sets on which a test rejects at the 5% level, with a 95% Wilson interval for that share; a valid test sits at or below the dashed line. The simulation runs in the background and the bars fill in one data set at a time. ${byRepeats ? 'The second panel repeats 10-fold cross-validation r times and tests the J = 10r differences: the naive paired t-test rejects more often the more splits it is given, while the corrected test stays put. ' : ''}Adding noise to B's inputs makes B worse, so the bars then measure power.`}
      readouts={{
        'simulated data sets': <Readout label="done" value={progress} />,
        'expected accuracy, A minus B': (
          <Readout
            label="average over data sets, on fresh data"
            value={study === null ? '—' : fmt(study.trueDifference)}
          />
        ),
        'rejection rate': (
          <>
            {on.map((t) => (
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
            {view && (
              <>
                <Curve name="paired t (naive)" x={view.rs} y={view.naive} slot={1} stale={run.running} />
                <Points name="paired t (naive)" x={view.rs} y={view.naive} slot={1} size={5} />
                <Curve name="corrected resampled t" x={view.rs} y={view.corrected} slot={0} stale={run.running} />
                <Points name="corrected resampled t" x={view.rs} y={view.corrected} slot={0} size={5} />
              </>
            )}
            <Annotation y={ALPHA} text="nominal 5%" dashed emphasis />
          </Plot>
        </Plots>
      ) : (
        testsPlot
      )}
    </Figure>
  )
}
