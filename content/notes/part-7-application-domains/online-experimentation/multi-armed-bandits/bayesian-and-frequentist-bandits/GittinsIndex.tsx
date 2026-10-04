import { useMemo } from 'react'
import {
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Plot,
  Points,
  Readout,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { regularisedBetaInverse } from 'aifn/numerics/special'

/** Pseudo-counts a + b at which the index is computed. */
const COUNTS = [2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 25, 32, 40, 50, 64, 80, 100]
const BISECTIONS = 26

/**
 * Gittins index of a Bernoulli arm with posterior Beta(a, b) and discount γ, by calibration: the retirement reward λ
 * at which retiring now (worth λ / (1 − γ)) and playing on optimally are equally good. The value of playing on is a
 * backward recursion over the posteriors reachable in `depth` more pulls; beyond that depth the arm's mean is treated
 * as known, an error of order γ^depth.
 */
function gittins(a: number, b: number, gamma: number, depth: number): number {
  let lo = a / (a + b)
  let hi = 1
  const values = new Float64Array(depth + 2)
  for (let k = 0; k < BISECTIONS; k++) {
    const lambda = (lo + hi) / 2
    const retire = lambda / (1 - gamma)
    for (let i = 0; i <= depth; i++) values[i] = Math.max(retire, (a + i) / (a + b + depth) / (1 - gamma))
    for (let d = depth - 1; d >= 0; d--) {
      for (let i = 0; i <= d; i++) {
        const p = (a + i) / (a + b + d)
        // Success moves to i + 1 successes, failure keeps i; both add one to the depth.
        const play = p * (1 + gamma * values[i + 1]) + (1 - p) * gamma * values[i]
        values[i] = Math.max(retire, play)
      }
    }
    if (values[0] > retire + 1e-12) lo = lambda
    else hi = lambda
  }
  return (lo + hi) / 2
}

/**
 * The Gittins index of a Bernoulli arm against its posterior mean and a posterior quantile, as the number of
 * observations grows at a fixed posterior mean.
 */
export function GittinsIndex() {
  const state = useFigureState({
    mean: float(0.4, { min: 0.05, max: 0.95, step: 0.01, label: 'posterior mean a / (a + b)' }),
    gamma: float(0.95, { min: 0.5, max: 0.98, step: 0.01, label: 'discount γ' }),
    count: int(10, { min: 2, max: 100, step: 1, label: 'observations a + b', format: (v) => String(v) }),
  })

  const g = state.gamma
  const p = state.mean
  // Look far enough ahead that γ^depth < 0.001.
  const depth = Math.min(400, Math.ceil(Math.log(1e-3) / Math.log(g)))

  const curve = useMemo(() => COUNTS.map((m) => gittins(p * m, (1 - p) * m, g, depth)), [p, g, depth])
  // Bayes-UCB-style comparison: the posterior quantile at level 1 − 1/H, with H = 1/(1 − γ) the effective horizon.
  const quantiles = useMemo(() => COUNTS.map((m) => regularisedBetaInverse(p * m, (1 - p) * m, g)), [p, g])

  const m = state.count
  const index = useMemo(() => gittins(p * m, (1 - p) * m, g, depth), [p, m, g, depth])

  const series = [
    { name: 'Gittins index', x: COUNTS, y: curve, slot: 0 },
    { name: 'posterior quantile at level γ', x: COUNTS, y: quantiles, slot: 1, dashed: true },
    { name: 'posterior mean', x: [2, 100], y: [p, p], emphasis: true },
    { name: 'index at a + b', x: [m], y: [index], emphasis: true },
  ] as const

  const xAxis = useAxis({ label: 'observations a + b', range: [0, 100] })
  const yAxis = useAxis({
    label: 'value per round',
    range: [Math.max(0, p - 0.1), Math.min(1, Math.max(...curve, ...quantiles) + 0.05)],
  })
  return (
    <Figure
      title="The Gittins index of a Bernoulli arm"
      state={state}
      caption="An arm whose success probability has posterior Beta(a, b), with posterior mean a / (a + b) held fixed while the number of observations a + b grows. The Gittins index is the constant reward λ at which a Bayesian who discounts by γ per round is indifferent between retiring to λ for ever and pulling the arm. It exceeds the posterior mean by an exploration bonus, the value of what the next pulls would reveal. The bonus shrinks as the posterior sharpens and grows with γ, which sets the effective horizon 1 / (1 − γ). Drag along the chart to read the index at a given a + b."

      readouts={
        <>
          <Readout label="Gittins index" value={formatNumber(index)} />
          <Readout label="exploration bonus" value={formatNumber(index - p)} />
          <Readout label="effective horizon 1 / (1 − γ)" value={formatNumber(1 / (1 - g))} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis}>
        <Curve {...series[0]} />
        <Curve {...series[1]} />
        <Curve {...series[2]} />
        <Points {...series[3]} />
        <Handle kind="x" at={m} label="a + b" onDrag={(x) => state.set('count', Math.round(x))} />
      </Plot>
    </Figure>
  )
}
