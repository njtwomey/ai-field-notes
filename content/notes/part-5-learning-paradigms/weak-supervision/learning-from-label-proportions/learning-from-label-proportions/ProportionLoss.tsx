import { useMemo } from 'react'
import {
  Figure,
  float,
  formatNumber,
  int,
  Plot,
  Readout,
  seriesLayers,
  type SeriesSpec,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { normal, stream, uniform } from 'aifn/foundation/random'
import { sigmoid } from 'aifn/numerics/special'

const TOTAL = 480
const MEAN: [number, number] = [0.9, 0.4]
const X_RANGE: [number, number] = [-4, 4]
const Y_RANGE: [number | undefined, number | undefined] = [-4, 4]
const STEPS = 400
const RATE = 0.5

type Data = { x1: number[]; x2: number[]; y: number[]; bag: number[]; pi: number[] }

/**
 * Bags of equal size whose positive counts are drawn around one half. Bag k has proportion 1/2 + δ or 1/2 − δ times a
 * uniform draw, alternating so the classes stay balanced overall.
 */
function makeBags(size: number, spread: number, seed: number): Data {
  const g = stream(seed)
  const bags = Math.max(2, Math.floor(TOTAL / size))
  const d: Data = { x1: [], x2: [], y: [], bag: [], pi: [] }
  for (let k = 0; k < bags; k++) {
    const offset = spread * uniform(g) * (k % 2 === 0 ? 1 : -1)
    const positives = Math.round((0.5 + offset) * size)
    d.pi.push(positives / size)
    for (let i = 0; i < size; i++) {
      const label = i < positives ? 1 : 0
      const sign = label === 1 ? 1 : -1
      d.x1.push(sign * MEAN[0] + normal(g))
      d.x2.push(sign * MEAN[1] + normal(g))
      d.y.push(label)
      d.bag.push(k)
    }
  }
  return d
}

type Weights = [number, number, number]

/**
 * Gradient descent on a logistic model. `target(bagMeans)` returns, for every instance, the derivative of the loss with
 * respect to that instance's predicted probability, so one routine fits all three variants.
 */
function fit(d: Data, grad: (p: number[]) => number[]): Weights {
  const w: Weights = [0, 0, 0]
  const n = d.y.length
  for (let s = 0; s < STEPS; s++) {
    const p = d.x1.map((_, i) => sigmoid(w[0] * d.x1[i] + w[1] * d.x2[i] + w[2]))
    const dp = grad(p)
    const gw: Weights = [0, 0, 0]
    for (let i = 0; i < n; i++) {
      const dz = dp[i] * p[i] * (1 - p[i])
      gw[0] += dz * d.x1[i]
      gw[1] += dz * d.x2[i]
      gw[2] += dz
    }
    for (let j = 0; j < 3; j++) w[j] -= (RATE * gw[j]) / n
  }
  return w
}

const brier = (d: Data, w: Weights) =>
  d.y.reduce((s, y, i) => s + (sigmoid(w[0] * d.x1[i] + w[1] * d.x2[i] + w[2]) - y) ** 2, 0) / d.y.length

const accuracy = (d: Data, w: Weights) =>
  d.y.filter((y, i) => (w[0] * d.x1[i] + w[1] * d.x2[i] + w[2] > 0 ? 1 : 0) === y).length / d.y.length

/** The line w₁x₁ + w₂x₂ + b = 0 across the plot, or nothing when the weights are all zero. */
function boundary(name: string, w: Weights, slot: number): SeriesSpec[] {
  if (Math.abs(w[1]) < 1e-9 && Math.abs(w[0]) < 1e-9) return []
  if (Math.abs(w[1]) < 1e-9) {
    const x = -w[2] / w[0]
    return [{ name, type: 'line', x: [x, x], y: [X_RANGE[0], X_RANGE[1]], slot }]
  }
  const y = (x: number) => -(w[0] * x + w[2]) / w[1]
  return [{ name, type: 'line', x: [...X_RANGE], y: X_RANGE.map(y), slot }]
}

/**
 * Logistic regression fitted three ways on the same bags: with the proportion loss on bag-mean predictions, with each
 * instance given its bag's proportion as a soft label, and with the true instance labels.
 */
export function ProportionLoss() {
  const state = useFigureState({
    size: int(16, { min: 4, max: 96, step: 4, label: 'bag size', format: (v) => String(v) }),
    spread: float(0.4, { min: 0, max: 0.5, step: 0.05, label: 'spread of proportions δ' }),
    seed: int(2, { min: 1, max: 20, step: 1, label: 'seed', format: (v) => String(v) }),
  })

  const data = useMemo(() => makeBags(state.size, state.spread, state.seed), [state.size, state.spread, state.seed])

  const r = useMemo(() => {
    const bags = data.pi.length
    const members: number[][] = Array.from({ length: bags }, () => [])
    data.bag.forEach((k, i) => members[k].push(i))
    const n = data.y.length
    // Proportion loss: −Σ_k [π_k log p̄_k + (1 − π_k) log(1 − p̄_k)], averaged over instances by the 1/n in fit().
    const proportion = fit(data, (p) => {
      const out = new Array<number>(n).fill(0)
      members.forEach((idx, k) => {
        const mean = Math.min(1 - 1e-9, Math.max(1e-9, idx.reduce((s, i) => s + p[i], 0) / idx.length))
        const dmean = -data.pi[k] / mean + (1 - data.pi[k]) / (1 - mean)
        // d p̄_k / d p_i = 1/|B_k|; multiplying by |B_k| instances per bag keeps the scale of the other losses.
        for (const i of idx) out[i] = dmean
      })
      return out
    })
    // Soft labels: every instance's target is its bag's proportion.
    const inherit = fit(data, (p) =>
      p.map((pi, i) => -data.pi[data.bag[i]] / pi + (1 - data.pi[data.bag[i]]) / (1 - pi)),
    )
    const oracle = fit(data, (p) => p.map((pi, i) => -data.y[i] / pi + (1 - data.y[i]) / (1 - pi)))
    return {
      bags,
      proportion,
      inherit,
      oracle,
      acc: { proportion: accuracy(data, proportion), inherit: accuracy(data, inherit), oracle: accuracy(data, oracle) },
      brier: { proportion: brier(data, proportion), inherit: brier(data, inherit), oracle: brier(data, oracle) },
    }
  }, [data])

  const pred = data.x1.map((_, i) =>
    r.proportion[0] * data.x1[i] + r.proportion[1] * data.x2[i] + r.proportion[2] > 0 ? 1 : 0,
  )
  const series: SeriesSpec[] = [
    {
      name: 'instances',
      type: 'scatter',
      x: data.x1,
      y: data.x2,
      group: data.y,
      groupNames: ['true class 0', 'true class 1'],
    },
    ...boundary('proportion loss', r.proportion, 2),
    ...boundary('bag proportion as soft label', r.inherit, 3),
    ...boundary('true labels (oracle)', r.oracle, 4),
  ]
  const errors = pred.filter((v, i) => v !== data.y[i]).length

  const xAxis = useAxis({ label: 'x₁', range: X_RANGE })
  const yAxis = useAxis({ label: 'x₂', range: Y_RANGE, equal: xAxis })
  return (
    <Figure
      title="Recovering instance labels from bag proportions"
      state={state}
      caption="Two Gaussian classes are split into bags of equal size. Each bag reveals only its fraction of class 1. Three logistic regressions are fitted by gradient descent: one matches each bag's mean predicted probability to its proportion (the proportion loss), one gives every instance its bag's proportion as a soft label, and one sees the true labels. When the proportions differ across bags, the proportion loss recovers the oracle's boundary and nearly its Brier score. The soft-label fit finds a similar direction here, because the classes are symmetric, but its probabilities are pulled towards the bag proportions, so its Brier score is worse. As the spread of proportions shrinks, the bags carry less information and the proportion-loss fit weakens; at zero spread every bag says one half, the loss is flat at the starting point and no boundary is learnt."

      readouts={
        <>
          <Readout label="bags" value={String(r.bags)} />
          <Readout
            label="accuracy (proportion loss / soft labels / oracle)"
            value={`${formatNumber(r.acc.proportion)} / ${formatNumber(r.acc.inherit)} / ${formatNumber(r.acc.oracle)}`}
          />
          <Readout
            label="Brier score"
            value={`${formatNumber(r.brier.proportion)} / ${formatNumber(r.brier.inherit)} / ${formatNumber(r.brier.oracle)}`}
          />
          <Readout label="misclassified by proportion loss" value={String(errors)} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} ariaLabel={'Scatter of instances by true class with three fitted decision boundaries'}>
        {seriesLayers(series)}
      </Plot>
    </Figure>
  )
}
