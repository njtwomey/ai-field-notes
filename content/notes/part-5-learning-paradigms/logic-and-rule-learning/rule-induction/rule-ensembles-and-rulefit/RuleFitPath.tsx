import { useMemo, useState } from 'react'
import {
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Plot,
  Plots,
  Readout,
  Segments,
  toggle,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { stream } from 'aifn-compute/foundation/random'
import { dataset } from 'aifn-compute/learning/estimators'
import { toArray, toFlat } from 'aifn-compute/foundation/tensor'
import { friedman1 } from 'aifn-methods/data/synthetic'
import { gradientBoosting } from 'aifn-methods/learning/trees-and-ensembles/boosting'
import { ensembleRules, ruleText } from 'aifn-compute/learning/rules'
import { ruleFitPath, ruleFitPredict, ruleImportances } from 'aifn-methods/learning/linear'

const N_TRAIN = 400
const N_TEST = 1000

const mse = (p: readonly number[], y: readonly number[]) => p.reduce((a, v, i) => a + (v - y[i]) ** 2, 0) / y.length

/**
 * RuleFit on Friedman's first benchmark: rules from every node of a boosted ensemble, a lasso path over them and the
 * linear terms, held-out error against the number of terms kept, and the coefficient paths.
 */
export function RuleFitPath() {
  const fig = useFigureState({
    trees: int(50, { min: 5, max: 150, label: 'trees in the ensemble', suggestions: [20, 50, 100] }),
    leaves: int(4, { min: 2, max: 8, label: 'leaves per tree', suggestions: [2, 4, 6] }),
    rate: float(0.1, { min: 0.01, max: 1, step: 0.01, label: 'boosting learning rate', suggestions: [0.05, 0.1, 0.3] }),
    linear: toggle(true, { label: 'add the linear terms' }),
    seed: int(1, { min: 1, max: 999, label: 'seed' }),
  })
  const { trees, leaves, rate, linear, seed } = fig

  const data = useMemo(() => {
    const tr = friedman1(stream(`rulefit-train-${seed}`), { n: N_TRAIN, d: 10, noise: 1 })
    const te = friedman1(stream(`rulefit-test-${seed}`), { n: N_TEST, d: 10, noise: 1 })
    // `friedman1` always returns targets; the Dataset type marks them optional for unlabelled data.
    if (!tr.y || !te.y) throw new Error('friedman1 returned no targets')
    return {
      tr,
      te,
      trY: tr.y,
      xtr: toArray(tr.x) as number[][],
      ytr: toFlat(tr.y),
      xte: toArray(te.x) as number[][],
      yte: toFlat(te.y),
    }
  }, [seed])

  const linearOnly = useMemo(() => {
    const fit = ruleFitPath(data.xtr, data.ytr, [], { count: 30, span: 2.5, tolerance: 1e-5, maxSteps: 500 })
    return Math.min(...fit.alphas.map((_, t) => mse(ruleFitPredict(fit, t, data.xte), data.yte)))
  }, [data])

  const result = useMemo(() => {
    const model = gradientBoosting({
      stages: trees,
      learningRate: rate,
      tree: { maxLeaves: leaves },
      subsample: 0.5,
    }).fit(dataset(data.tr.x, data.trY), { stream: stream(`rulefit-boost-${seed}`) })
    const ensembleTest = mse(toFlat(model.decide(data.te.x)), data.yte)
    const rules = ensembleRules(model.stages.map((s) => s[0]))
    const fit = ruleFitPath(data.xtr, data.ytr, rules, {
      linear,
      count: 30,
      span: 2.5,
      tolerance: 1e-5,
      maxSteps: 500,
    })
    const test = fit.alphas.map((_, t) => mse(ruleFitPredict(fit, t, data.xte), data.yte))
    const train = fit.alphas.map((_, t) => mse(ruleFitPredict(fit, t, data.xtr), data.ytr))
    const best = test.indexOf(Math.min(...test))
    return { fit, test, train, best, ensembleTest, candidates: rules.length }
  }, [data, trees, leaves, rate, linear, seed])
  const { fit, test, train, best, ensembleTest, candidates } = result

  const settings = `${trees}/${leaves}/${rate}/${linear}/${seed}`
  const [pick, setPick] = useState({ settings, t: -1 })
  const t = pick.settings === settings && pick.t >= 0 ? Math.min(pick.t, fit.alphas.length - 1) : best
  const nearest = (count: number) => {
    let arg = 0
    fit.nonzero.forEach((v, k) => {
      if (Math.abs(v - count) < Math.abs(fit.nonzero[arg] - count)) arg = k
    })
    return arg
  }

  const terms = useMemo(() => ruleImportances(fit, t), [fit, t])
  const top = terms.slice(0, 3).map((r) => r.index)
  const describe = (r: (typeof terms)[number]) =>
    r.kind === 'rule' ? ruleText(r.conditions) : `linear term in x${r.feature + 1}`

  const paths = useMemo(() => {
    const segs: { from: [number, number]; to: [number, number] }[] = []
    const K = fit.coef[0].length
    for (let k = 0; k < K; k++) {
      if (top.includes(k) || fit.coef.every((c) => c[k] === 0)) continue
      for (let s = 1; s < fit.alphas.length; s++) {
        segs.push({
          from: [fit.alphas[s - 1], fit.coef[s - 1][k]],
          to: [fit.alphas[s], fit.coef[s][k]],
        })
      }
    }
    return segs
  }, [fit, top.join(',')]) // eslint-disable-line react-hooks/exhaustive-deps

  const maxTerms = Math.max(...fit.nonzero, 1)
  const coefRange = useMemo(() => {
    const all = fit.coef.flat()
    return [Math.min(0, ...all), Math.max(0, ...all)]
  }, [fit])
  const xa = useAxis({ label: 'terms with nonzero coefficient', range: [0, maxTerms], nice: false })
  const ya = useAxis({ label: 'mean squared error', range: [0, undefined] })
  const xb = useAxis({ label: 'lasso penalty α', log: true, inverse: true, nice: false })
  const yb = useAxis({ label: 'coefficient (scaled design)', hold: 'union' })

  const ruleCount = terms.filter((r) => r.kind === 'rule').length
  return (
    <Figure
      title="RuleFit: a lasso path over rules from a boosted ensemble"
      state={fig}
      caption={`Friedman's first benchmark: ${N_TRAIN} training rows of ten uniform inputs, y = 10 sin(πx₁x₂) + 20(x₃ − ½)² + 10x₄ + 5x₅ + noise of variance 1, and ${N_TEST} held-out rows; x₆ to x₁₀ are noise. Gradient boosting grows the trees (half of the rows per tree); every node except a root becomes a 0/1 rule, rules with identical coverage are merged, and the winsorised inputs, scaled to standard deviation 0.4, join as linear terms when switched on. Left: training and held-out error along the lasso path against the number of terms kept, with the held-out error of the boosted ensemble that supplied the rules (dashed ink) and of the best lasso on the linear terms alone (dashed grey). Right: the coefficient paths as the penalty falls; the three most important terms at the chosen penalty are coloured. Drag the vertical line to choose a penalty; it opens at the smallest held-out error.`}
      readouts={{
        [`at α = ${formatNumber(fit.alphas[t])}`]: (
          <>
            <Readout label="terms kept" value={`${fit.nonzero[t]} (${ruleCount} rules)`} />
            <Readout label="held-out MSE, training MSE" value={`${formatNumber(test[t])}, ${formatNumber(train[t])}`} />
            <Readout
              label="smallest held-out MSE on the path"
              value={`${formatNumber(test[best])} with ${fit.nonzero[best]} terms`}
            />
            <Readout label="boosted ensemble, held-out MSE" value={formatNumber(ensembleTest)} />
            <Readout label="lasso on the linear terms alone" value={formatNumber(linearOnly)} />
            <Readout
              label="candidate rules"
              value={`${candidates} from the trees, ${fit.rules.length} after merging equal coverage`}
            />
          </>
        ),
        'most important terms': (
          <>
            {terms.slice(0, 6).map((r, k) => (
              <Readout
                key={k}
                label={`${k + 1}. ${describe(r)}`}
                value={`${r.kind === 'rule' ? `s = ${r.support.toFixed(2)}, ` : ''}coef ${formatNumber(r.coefficient)}, importance ${formatNumber(r.importance)}`}
              />
            ))}
          </>
        ),
      }}
    >
      <Plots cols={2}>
        <Plot x={xa} y={ya}>
          <Curve name="held-out error" x={fit.nonzero} y={test} slot={0} />
          <Curve name="training error" x={fit.nonzero} y={train} slot={1} />
          <Curve name="boosted ensemble" x={[0, maxTerms]} y={[ensembleTest, ensembleTest]} dashed emphasis />
          <Curve name="linear terms alone" x={[0, maxTerms]} y={[linearOnly, linearOnly]} dashed muted />
          <Handle kind="x" at={fit.nonzero[t]} label="penalty" onDrag={(x) => setPick({ settings, t: nearest(x) })} />
        </Plot>
        <Plot x={xb} y={yb}>
          {paths.length > 0 && <Segments name="other terms" segments={paths} muted width={0.6} />}
          {top.map((k, j) => (
            <Curve key={k} name={describe(terms[j])} x={fit.alphas} y={fit.coef.map((c) => c[k])} slot={j} />
          ))}
          <Curve name="chosen penalty" x={[fit.alphas[t], fit.alphas[t]]} y={coefRange} dashed emphasis silent />
        </Plot>
      </Plots>
    </Figure>
  )
}
