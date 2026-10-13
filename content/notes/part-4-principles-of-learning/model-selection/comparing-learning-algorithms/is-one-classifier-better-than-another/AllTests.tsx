import { useMemo } from 'react'
import {
  Annotation,
  Bars,
  Figure,
  float,
  formatNumber,
  int,
  Plot,
  Readout,
  useAxis,
  useComputed,
  useFigureState,
} from 'aifn-render'
import { child, stream } from 'aifn-compute/foundation/random'
import { replicate, TESTS } from '../_shared/equalLearners'

const ALPHA = 0.05
const fmt = (v: number) => (Number.isFinite(v) ? formatNumber(Number(v.toPrecision(3))) : '—')

/** The six tests' p-values on one simulated data set, beside the true difference between the learners. */
export function AllTests() {
  const state = useFigureState({
    n: int(200, { ge: 60, le: 600, suggestions: [100, 200, 400], label: 'rows n' }),
    handicap: float(0, { ge: 0, le: 1, suggestions: [0, 0.3, 0.6], label: "noise on B's inputs" }),
    seed: int(3, { ge: 0, le: 999, label: 'data set' }),
  })
  const { n, handicap, seed } = state
  const computed = useComputed(
    () => replicate(child(stream('comparing/type-i'), seed), n, handicap, 10),
    [n, handicap, seed],
    { mode: 'release' },
  )
  const result = computed.value
  const view = useMemo(() => {
    const ps = TESTS.map((t) => Math.max(result.p[t.key], 1e-6))
    const index = TESTS.map((_, i) => i)
    return {
      reject: index.filter((i) => ps[i] <= ALPHA),
      keep: index.filter((i) => ps[i] > ALPHA),
      ps,
    }
  }, [result])
  const x = useAxis({ categories: TESTS.map((t) => t.label) })
  const y = useAxis({ label: 'p-value (log scale)', log: true, range: [1e-6, 1] })
  return (
    <Figure
      title="Six tests, one data set"
      state={state}
      caption="One simulated data set from the circle problem: learner A is a depth-4 decision tree on the inputs, B the same tree on the inputs rotated by 45°, with optional noise on B's inputs. Each bar is one test's p-value for the hypothesis that A and B are equally accurate, on a log scale; bars that reach below the dashed line reject at 5%. The resampled tests use 10 repeats of 10-fold cross-validation. Step through data sets with no noise: the learners are then equally good, and the naive paired t-test on the 100 resampled differences rejects on most of them."
      readouts={{
        truth: (
          <Readout label="A minus B on fresh data (models fitted to all n rows)" value={fmt(result.trueDifference)} />
        ),
        rejections: <Readout label="tests rejecting at 5%" value={`${view.reject.length} of ${TESTS.length}`} />,
      }}
    >
      <Plot x={x} y={y}>
        {view.reject.length > 0 && (
          <Bars
            name="rejects"
            x={view.reject}
            y={view.reject.map((i) => view.ps[i])}
            base={1}
            slot={1}
            width={0.6}
            stale={computed.stale}
          />
        )}
        {view.keep.length > 0 && (
          <Bars
            name="does not reject"
            x={view.keep}
            y={view.keep.map((i) => view.ps[i])}
            base={1}
            slot={0}
            width={0.6}
            stale={computed.stale}
          />
        )}
        <Annotation y={ALPHA} text="α = 0.05" dashed emphasis />
      </Plot>
    </Figure>
  )
}
