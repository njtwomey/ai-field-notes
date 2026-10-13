import { useMemo } from 'react'
import {
  Annotation,
  Curve,
  Figure,
  formatNumber,
  int,
  Plot,
  Points,
  Readout,
  Segments,
  useAxis,
  useComputed,
  useFigureState,
} from 'aifn-render'
import { oneOf, space } from 'aifn-compute/foundation/space'
import { child, normals, stream, type Stream } from 'aifn-compute/foundation/random'
import { fromData, toFlat } from 'aifn-compute/foundation/tensor'
import { dataset } from 'aifn-compute/learning/estimators'
import { meanSquaredError } from 'aifn-compute/learning/metrics'
import { oneStandardError } from 'aifn-compute/learning/selection'
import { gridSearch, kFold } from 'aifn-compute/learning/validate'
import { linearRegression } from 'aifn-methods/learning/linear'

const D = 30
const TEST = 20000
/** True coefficients 0.85^(j−1): a few inputs matter, most only a little. */
const BETA = Array.from({ length: D }, (_, j) => 0.85 ** j)
/** Eleven penalties, half a decade apart, from 0.01 to 1000. */
const LAMBDAS = [0.01, 0.0316, 0.1, 0.316, 1, 3.16, 10, 31.6, 100, 316, 1000]
const PENALTIES = space({ l2: oneOf(LAMBDAS) })

/** `rows` rows of x ~ N(0, I_30) and y = xᵀβ + ε with ε ~ N(0, 1). */
function draw(s: Stream, rows: number) {
  const x = Float64Array.from(toFlat(normals(child(s, 'x'), [rows, D])))
  const e = toFlat(normals(child(s, 'e'), [rows]))
  const y = Float64Array.from({ length: rows }, (_, i) => {
    let v = e[i]
    for (let j = 0; j < D; j++) v += x[i * D + j] * BETA[j]
    return v
  })
  return dataset(fromData(x, [rows, D]), fromData(y))
}

/** The K-fold curve of a ridge penalty with standard-error bars, the minimum and the one-standard-error choice. */
export function OneStandardErrorCurve() {
  const state = useFigureState({
    seed: int(1, { ge: 1, le: 999, suggestions: [1, 2, 3, 4], label: 'data set (seed)' }),
    n: int(60, { ge: 40, le: 400, suggestions: [40, 60, 120, 400], label: 'training rows n' }),
    folds: int(10, { ge: 3, le: 20, suggestions: [3, 5, 10, 20], label: 'folds K' }),
  })
  const { seed, n, folds } = state
  const computed = useComputed(
    () => {
      const s = stream(`one-se/${seed}`)
      const train = draw(child(s, 'train'), n)
      const test = draw(child(stream('one-se'), 'test'), TEST)
      const search = gridSearch((p) => linearRegression({ l2: p.l2 }), PENALTIES, {
        metrics: [meanSquaredError],
        refit: false,
      }).run(train, kFold({ k: folds, shuffle: true }), child(s, 'folds'))
      const rule = oneStandardError(search, { simplicity: LAMBDAS.map((l) => -l) })
      const truth = LAMBDAS.map((l2) => meanSquaredError(test.y, linearRegression({ l2 }).fit(train).decide(test.x)))
      return {
        cv: Array.from(toFlat(rule.mean)),
        se: Array.from(toFlat(rule.se)),
        eligible: Array.from(toFlat(rule.eligible)),
        best: rule.best,
        chosen: rule.chosen,
        threshold: rule.threshold,
        truth,
      }
    },
    [seed, n, folds],
    { mode: 'release' },
  )
  const r = computed.value
  const view = useMemo(() => {
    const bars = LAMBDAS.map((l, i) => ({
      from: [l, r.cv[i] - r.se[i]] as [number, number],
      to: [l, r.cv[i] + r.se[i]] as [number, number],
    }))
    const inside = LAMBDAS.map((_, i) => i).filter((i) => r.eligible[i])
    const oracle = r.truth.indexOf(Math.min(...r.truth))
    return { bars, inside, oracle }
  }, [r])
  const lx = useAxis({ label: 'ridge penalty λ (larger is simpler)', log: true })
  const ey = useAxis({ label: 'mean squared error' })
  const at = (i: number) => `λ = ${formatNumber(LAMBDAS[i])}`
  return (
    <Figure
      title="The one-standard-error rule on a ridge path"
      state={state}
      caption={`Ridge regression on n rows of ${D} standard-normal inputs with true coefficients 0.85^(j−1) and unit noise, cross-validated over eleven penalties with K folds (the same folds for every penalty). Each K-fold error carries a bar of ±1 standard error, the standard deviation of its K fold errors over √K. The dashed horizontal line is the threshold, the minimum plus its standard error; the highlighted points lie under it, and the rule picks the largest penalty among them. The curve without bars or points is the true error of each ridge fit to all n rows, measured on ${TEST} fresh rows. The defaults are the worked example: the minimum is at λ = 3.16 and the rule moves one step to λ = 10, which costs 0.14 of true error. Change the seed to see the two choices move between data sets, and raise n to see the bars shrink and the choices draw together.`}
      readouts={{
        'K-fold minimum': (
          <>
            <Readout label="penalty" value={at(r.best)} />
            <Readout label="CV ± SE" value={`${formatNumber(r.cv[r.best])} ± ${formatNumber(r.se[r.best])}`} />
            <Readout label="true error" value={formatNumber(r.truth[r.best])} />
          </>
        ),
        'one-standard-error choice': (
          <>
            <Readout label="penalty" value={at(r.chosen)} />
            <Readout label="CV (threshold)" value={`${formatNumber(r.cv[r.chosen])} (${formatNumber(r.threshold)})`} />
            <Readout label="true error" value={formatNumber(r.truth[r.chosen])} />
          </>
        ),
        'best possible': (
          <>
            <Readout label="penalty" value={at(view.oracle)} />
            <Readout label="true error" value={formatNumber(r.truth[view.oracle])} />
          </>
        ),
      }}
    >
      <Plot x={lx} y={ey}>
        <Segments name="±1 SE" segments={view.bars} slot={0} />
        <Curve name={`${folds}-fold CV`} x={LAMBDAS} y={r.cv} slot={0} showPoints stale={computed.stale} />
        {view.inside.length > 0 && (
          <Points
            name="within one SE of the minimum"
            x={view.inside.map((i) => LAMBDAS[i])}
            y={view.inside.map((i) => r.cv[i])}
            slot={1}
            size={10}
            stale={computed.stale}
          />
        )}
        <Curve name="true error" x={LAMBDAS} y={r.truth} emphasis stale={computed.stale} />
        <Annotation y={r.threshold} text="minimum + 1 SE" dashed muted />
        <Annotation x={LAMBDAS[r.best]} text="CV min" dashed slot={0} />
        <Annotation x={LAMBDAS[r.chosen]} text="1-SE choice" dashed slot={1} />
      </Plot>
    </Figure>
  )
}
