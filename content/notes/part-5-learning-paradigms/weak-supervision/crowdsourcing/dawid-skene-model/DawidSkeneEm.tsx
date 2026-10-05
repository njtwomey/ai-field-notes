import { useMemo } from 'react'
import { Curve, Figure, float, formatNumber, int, Plot, Points, Readout, useAxis, useFigureState } from 'aifn-render'
import { stream, uniform } from 'aifn-compute/foundation/random'

const ITEMS = 300
const ITERATIONS = 50
const DIAGONAL = [0.4, 1]

/**
 * Binary Dawid–Skene EM. Each annotator has a 2 × 2 confusion matrix; items have a latent true class. Initialised from
 * soft majority vote, as in Dawid and Skene (1979).
 */
function dawidSkene(labels: number[][]) {
  const n = labels.length
  const m = labels[0].length
  let T = labels.map((row) => {
    const v = row.reduce((s, l) => s + l, 0) / m
    return [1 - v, v]
  })
  let conf: number[][][] = []
  for (let it = 0; it < ITERATIONS; it++) {
    // M-step: class prior and confusion matrices from soft counts, with a tiny pseudo-count against log 0.
    const prior = [0, 1].map((j) => T.reduce((s, t) => s + t[j], 0) / n)
    conf = Array.from({ length: m }, (_, k) =>
      [0, 1].map((j) => {
        const counts = [0, 1].map((l) => 1e-3 + T.reduce((s, t, i) => s + (labels[i][k] === l ? t[j] : 0), 0))
        const total = counts[0] + counts[1]
        return counts.map((c) => c / total)
      }),
    )
    // E-step: posterior over the true class of each item.
    T = labels.map((row) => {
      const logp = [0, 1].map((j) => Math.log(prior[j]) + row.reduce((s, l, k) => s + Math.log(conf[k][j][l]), 0))
      const mx = Math.max(...logp)
      const w = logp.map((v) => Math.exp(v - mx))
      return w.map((v) => v / (w[0] + w[1]))
    })
  }
  return { T, conf }
}

/**
 * Crowd of annotators, some reliable and some spammers who answer at random. Majority vote treats them equally;
 * Dawid–Skene estimates each annotator's confusion matrix and weights their votes accordingly.
 */
export function DawidSkeneEm() {
  const state = useFigureState({
    annotators: int(7, { min: 3, max: 15, step: 1, label: 'annotators', format: (v) => String(v) }),
    spam: float(0.4, { min: 0, max: 0.8, step: 0.1, label: 'share of spammers' }),
    seed: int(4, { min: 1, max: 20, step: 1, label: 'seed', format: (v) => String(v) }),
  })
  const m = state.annotators
  const q = state.spam
  const s = state.seed

  const r = useMemo(() => {
    const g = stream(s)
    const spammers = Math.round(q * m)
    // Accuracy of each annotator: spammers answer at random, the rest are between 0.6 and 0.9.
    const acc = Array.from({ length: m }, (_, k) => (k < spammers ? 0.5 : 0.6 + 0.3 * uniform(g)))
    const truth: number[] = Array.from({ length: ITEMS }, () => (uniform(g) < 0.5 ? 1 : 0))
    const labels: number[][] = truth.map((y) => acc.map((a) => (uniform(g) < a ? y : 1 - y)))
    const { T, conf } = dawidSkene(labels)
    const vote = labels.map((row) => row.reduce((t, l) => t + l, 0) / m)
    // Ties in majority vote are broken by a fair coin, so each tie scores one half.
    const mv = truth.reduce((t, y, i) => t + (vote[i] === 0.5 ? 0.5 : (vote[i] > 0.5 ? 1 : 0) === y ? 1 : 0), 0) / ITEMS
    const ds = truth.reduce((t, y, i) => t + ((T[i][1] > 0.5 ? 1 : 0) === y ? 1 : 0), 0) / ITEMS
    // Estimated accuracy: average of the two diagonal entries (the classes are balanced).
    const est = conf.map((c) => (c[0][0] + c[1][1]) / 2)
    return { acc, est, mv, ds, spammers }
  }, [m, q, s])

  const series = [
    { name: 'equal', x: DIAGONAL, y: DIAGONAL, muted: true, dashed: true },
    {
      name: 'annotators',
      x: r.acc,
      y: r.est,
      group: r.acc.map((_, k) => (k < r.spammers ? 0 : 1)),
      groupNames: ['spammer', 'reliable'],
    },
  ] as const

  const xAxis = useAxis({ label: 'true annotator accuracy', range: [0.4, 1] })
  const yAxis = useAxis({ label: 'estimated accuracy', range: [0.4, 1], equal: xAxis })
  return (
    <Figure
      title="Dawid–Skene against majority vote"
      state={state}
      caption="Three hundred binary items, each labelled by every annotator. A share of the annotators are spammers who answer at random; the others are right with probability between 0.6 and 0.9. Dawid–Skene EM estimates each annotator's confusion matrix without any gold labels (the chart compares each annotator's estimated accuracy with the true one) and weights votes by the log-odds of those estimates. Majority vote gives every annotator the same weight, so spammers dilute it. The gap is largest when spammers are numerous and the crowd is small."

      readouts={
        <>
          <Readout label="majority-vote accuracy" value={formatNumber(r.mv)} />
          <Readout label="Dawid–Skene accuracy" value={formatNumber(r.ds)} />
          <Readout label="spammers" value={`${r.spammers} of ${m}`} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis}>
        <Curve {...series[0]} />
        <Points {...series[1]} />
      </Plot>
    </Figure>
  )
}
