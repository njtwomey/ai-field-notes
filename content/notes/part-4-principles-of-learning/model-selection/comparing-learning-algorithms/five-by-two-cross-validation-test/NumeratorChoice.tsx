import { useMemo } from 'react'
import {
  Annotation,
  Figure,
  float,
  formatNumber,
  int,
  Plot,
  Points,
  Readout,
  useAxis,
  useComputed,
  useFigureState,
} from 'aifn-render'
import { child, stream } from 'aifn-compute/foundation/random'
import { kFold, repeated } from 'aifn-compute/learning/validate'
import { fiveByTwoCvTest } from 'aifn-compute/probability/tests'
import { differences, drawPair } from '../_shared/equalLearners'

const fmt = (v: number) => (Number.isFinite(v) ? formatNumber(Number(v.toPrecision(3))) : '—')
const LABELS = [1, 2, 3, 4, 5].flatMap((i) => [`${i}.1`, `${i}.2`])

/**
 * Dietterich's 5×2cv t statistic for each of the ten fold differences in the numerator, on one simulated data set,
 * beside Alpaydın's combined F, which uses all ten.
 */
export function NumeratorChoice() {
  const state = useFigureState({
    n: int(200, { ge: 60, le: 600, suggestions: [100, 200, 400], label: 'rows n' }),
    handicap: float(0.3, { ge: 0, le: 1, suggestions: [0, 0.3, 0.5], label: "noise on B's inputs" }),
    seed: int(6, { ge: 1, le: 999, label: 'data set' }),
  })
  const { n, handicap, seed } = state
  const computed = useComputed(
    () => {
      const s = stream(`comparing/five-by-two/${seed}`)
      const pair = drawPair(child(s, 'data'), n, handicap)
      const f = differences(pair, repeated(kFold({ k: 2, shuffle: true }), 5), child(s, '5x2'))
      const m = [0, 1, 2, 3, 4].map((i) => [f[2 * i], f[2 * i + 1]])
      // Reordering replications and folds puts each difference in Dietterich's numerator in turn.
      const ts = LABELS.map((_, c) => {
        const i = Math.floor(c / 2)
        const first = c % 2 === 0 ? m[i] : [m[i][1], m[i][0]]
        return fiveByTwoCvTest([first, ...m.filter((_, k) => k !== i)])
      })
      return { dietterich: ts[0], ts, combined: ts[0].combinedF }
    },
    [n, handicap, seed],
    { mode: 'release' },
  )
  const { dietterich, ts, combined } = computed.value
  const view = useMemo(() => {
    const critical = dietterich.null.isf(0.025) as number
    const index = LABELS.map((_, c) => c)
    const t = ts.map((r) => r.statistic)
    const rejects = t.filter((v) => Math.abs(v) >= critical).length
    return { critical, index, t, rejects }
  }, [dietterich, ts])
  const x = useAxis({ categories: LABELS, label: 'difference in the numerator (replication.fold)' })
  const y = useAxis({ label: 't statistic' })
  return (
    <Figure
      title="Ten numerators, ten verdicts"
      state={state}
      caption={`One simulated data set of n rows from the circle problem (learner A a depth-4 tree on the inputs, B the same tree on rotated inputs with added noise). Five replications of 2-fold cross-validation give ten differences in accuracy. Dietterich's statistic puts the first of them, 1.1, over a denominator built from all ten; each point puts a different difference in the numerator, which is the same test after relabelling the replications and folds. The dashed lines are the two-sided 5% critical values of t with 5 degrees of freedom, ±${fmt(view.critical)}. The combined F test sums all ten squared differences, so it gives one verdict.`}
      readouts={{
        "Dietterich's t (numerator 1.1)": (
          <>
            <Readout label="t" value={fmt(dietterich.statistic)} />
            <Readout label="p" value={fmt(dietterich.pValue)} />
          </>
        ),
        'all ten numerators': <Readout label="reject at 5%" value={`${view.rejects} of 10`} />,
        'combined F': (
          <>
            <Readout label="F (10, 5)" value={fmt(combined.statistic)} />
            <Readout label="p" value={fmt(combined.pValue)} />
          </>
        ),
      }}
    >
      <Plot x={x} y={y}>
        <Points name="t with this numerator" x={view.index} y={view.t} slot={0} size={7} stale={computed.stale} />
        <Points name="Dietterich's choice" x={[0]} y={[view.t[0]]} emphasis size={9} />
        <Annotation y={view.critical} text="reject above" dashed emphasis />
        <Annotation y={-view.critical} text="reject below" dashed emphasis />
      </Plot>
    </Figure>
  )
}
