import { useMemo } from 'react'
import { bernoulliKl } from 'aifn-methods/theory/concentration'
import { Curve, Figure, formatNumber, Handle, int, Plot, Readout, slider, useAxis, useFigureState } from 'aifn-render'

const P_MAX = 0.3
const BUDGET_MAX = 200
const GRID = Array.from({ length: 600 }, (_, j) => ((j + 0.5) * P_MAX) / 600)
const fmt = (v: number) => formatNumber(Number(v.toPrecision(3)))

/**
 * The kl form of the PAC-Bayes bound reads: the true Gibbs risk p satisfies m·kl(q̂ ‖ p) ≤ KL(Q‖P) + ln(2√m/δ), where q̂
 * is the empirical Gibbs risk. The bound is the largest p where the scaled divergence curve stays below that budget.
 * The curve is steep near q̂ = 0, so small empirical risks give tight bounds. (The y axis is in nats rather than
 * nats per example because guide lines are placed to two decimals.)
 */
export function KlInversion() {
  const state = useFigureState({
    q: slider(0, 0.2, 0.05, { step: 0.01, onChart: true, label: 'empirical Gibbs risk q̂' }),
    budget: slider(1, BUDGET_MAX, 58.3, { step: 0.1, onChart: true, label: 'budget KL(Q‖P) + ln(2√m/δ) in nats' }),
    m: int(10000, { ge: 100, le: 1000000, suggestions: [1000, 10000, 100000], label: 'training examples m' }),
  })
  const { q, budget, m } = state
  const curve = useMemo(() => GRID.map((p) => m * bernoulliKl(q, p)), [q, m])
  const zero = useMemo(() => GRID.map((p) => m * bernoulliKl(0, p)), [m])
  const xAxis = useAxis({ label: 'true Gibbs risk p', range: [0, P_MAX] })
  const yAxis = useAxis({ label: 'm · kl(q̂ ‖ p) in nats', range: [0, BUDGET_MAX] })
  return (
    <Figure
      title="Reading a kl bound off the divergence curve"
      state={state}
      defaultSize="M"
      readouts={
        <>
          <Readout label="q̂" value={fmt(q)} />
          <Readout label="budget" value={`${fmt(budget)} nats`} />
          <Readout label="c = budget / m" value={fmt(budget / m)} />
        </>
      }
      caption="The solid curve is m · kl(q̂ ‖ p), where kl is the Kullback–Leibler divergence between Bernoulli laws with means q̂ and p, as a function of the unknown true Gibbs risk p. The dashed curve is the same for q̂ = 0. The kl form of the PAC-Bayes bound says that, with probability at least 1 − δ, the true risk p lies where the curve is at or below the horizontal budget KL(Q‖P) + ln(2√m/δ). The bound on p is where the curve crosses the budget to the right of q̂. Drag the vertical line to change q̂ and the horizontal line to change the budget; a larger m steepens both curves. At the opening values (q̂ = 0.05, m = 10,000, and a budget of 50 + 8.3 = 58.3 nats from KL(Q‖P) = 50 and δ = 0.05) the crossing is at p = 0.077, while the McAllester form q̂ + √(budget/(2m)) gives 0.104. For q̂ = 0 the crossing is at p = 0.0058, about budget/m itself rather than √(budget/(2m)) = 0.054: near zero empirical risk the kl form gives a rate of 1/m instead of 1/√m."
    >
      <Plot x={xAxis} y={yAxis} height={340}>
        <Curve name="m · kl(0 ‖ p)" x={GRID} y={zero} muted dashed />
        <Curve name="m · kl(q̂ ‖ p)" x={GRID} y={curve} slot={0} />
        <Handle kind="x" at={q} onDrag={(v) => state.set('q', Math.max(0, Math.min(0.2, v)))} label="q̂" />
        <Handle
          kind="y"
          at={budget}
          onDrag={(v) => state.set('budget', Math.max(1, Math.min(BUDGET_MAX, v)))}
          label="budget"
        />
      </Plot>
    </Figure>
  )
}
