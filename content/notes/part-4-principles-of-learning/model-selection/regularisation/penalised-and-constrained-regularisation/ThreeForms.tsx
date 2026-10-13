import { useMemo } from 'react'
import {
  Annotation,
  choice,
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Plot,
  Points,
  Readout,
  slider,
  useAxis,
  useComputed,
  useFigureState,
} from 'aifn-render'
import { stream } from 'aifn-compute/foundation/random'
import { toFlat, toRows } from 'aifn-compute/foundation/tensor'
import { elasticNetPath, ridgePath } from 'aifn-methods/learning/linear'
import { linearRegressionData } from 'aifn-methods/data/synthetic'

const N = 40
const TRUE_W = [2, 0.6]
const LOG_LAMBDAS = Array.from({ length: 101 }, (_, i) => -1 + i * 0.05)
const LAMBDAS = LOG_LAMBDAS.map((l) => 10 ** l)
const PENALTIES = [
  { value: 'ridge', label: 'ridge, Ω = ‖w‖₂²' },
  { value: 'lasso', label: 'lasso, Ω = ‖w‖₁' },
] as const

const fmt = (v: number) => (Number.isFinite(v) ? formatNumber(Number(v.toPrecision(3))) : '—')

/**
 * One regularisation path read three ways: the penalty λ, the budget t(λ) = Ω(ŵ_λ) and the training loss
 * δ(λ) = ‖y − Xŵ_λ − b‖², for ridge (`ridgePath`) or the lasso (`elasticNetPath`, rescaled to λ‖w‖₁ on the residual
 * sum of squares), on two correlated features.
 */
export function ThreeForms() {
  const state = useFigureState({
    penalty: choice(PENALTIES, 'lasso', { label: 'penalty' }),
    correlation: slider(-0.9, 0.9, 0.6, { step: 0.05, label: 'feature correlation ρ' }),
    noise: float(1, { min: 0.2, max: 3, step: 0.1, label: 'noise σ' }),
    seed: int(1, { ge: 1, le: 999, label: 'dataset' }),
    logLambda: float(1.5, { min: -1, max: 4, step: 0.05, label: 'log₁₀ penalty λ' }),
  })
  const { penalty, correlation, noise, seed } = state
  const data = useMemo(
    () =>
      linearRegressionData(stream(`three-forms/${seed}`), {
        n: N,
        weights: TRUE_W,
        noise,
        correlation,
      }),
    [seed, noise, correlation],
  )
  const run = useComputed(
    () => {
      const target = data.y!
      const x = toRows(data.x)
      const y = toFlat(target)
      const ols = toFlat(ridgePath(data.x, target, [0]).coef)
      let coef: number[][]
      let intercept: number[]
      if (penalty === 'ridge') {
        const path = ridgePath(data.x, target, LAMBDAS)
        coef = toRows(path.coef)
        intercept = path.intercept
      } else {
        // The engine's lasso is (1/2n)‖r‖² + α‖w‖₁; the note's is ‖r‖² + λ‖w‖₁, so α = λ / 2n.
        const path = elasticNetPath(data.x, target, { alphas: LAMBDAS.map((l) => l / (2 * N)) })
        // The path is fitted from the largest penalty down; put it back in the order of LAMBDAS.
        coef = toRows(path.coef).map((row) => [...row].reverse())
        intercept = [...path.intercept].reverse()
      }
      const w1 = coef[0]
      const w2 = coef[1]
      const budget = w1.map((a, k) => (penalty === 'ridge' ? a * a + w2[k] * w2[k] : Math.abs(a) + Math.abs(w2[k])))
      const loss = w1.map((a, k) =>
        x.reduce((s, row, i) => s + (y[i] - row[0] * a - row[1] * w2[k] - intercept[k]) ** 2, 0),
      )
      return { w1, w2, budget, loss, ols }
    },
    [data, penalty],
    { mode: 'release' },
  )
  const v = run.value
  const j = Math.max(0, Math.min(LOG_LAMBDAS.length - 1, Math.round((state.logLambda + 1) / 0.05)))
  const t = v.budget[j]
  const w = [v.w1[j], v.w2[j]]
  const level = N * noise * noise
  const morozov = v.loss.findIndex((d) => d >= level)
  // The multiplier as a price: the slope of the best loss against the budget, from the neighbouring grid points.
  const k0 = Math.max(0, j - 1)
  const k1 = Math.min(LOG_LAMBDAS.length - 1, j + 1)
  const dt = v.budget[k0] - v.budget[k1]
  const price = dt > 1e-9 ? (v.loss[k1] - v.loss[k0]) / dt : NaN

  const ball = useMemo(() => {
    if (penalty === 'ridge') {
      const r = Math.sqrt(t)
      const a = Array.from({ length: 121 }, (_, i) => (2 * Math.PI * i) / 120)
      return { x: a.map((s) => r * Math.cos(s)), y: a.map((s) => r * Math.sin(s)) }
    }
    return { x: [t, 0, -t, 0, t], y: [0, t, 0, -t, 0] }
  }, [penalty, t])

  const wx = useAxis({ label: 'w₁', range: [-1.5, 4], nice: false })
  const wy = useAxis({ label: 'w₂', range: [-2, 3], nice: false, equal: wx })
  const lx = useAxis({ label: 'log₁₀ penalty λ', range: [-1, 4], nice: false })
  const ty = useAxis({
    label: penalty === 'ridge' ? 'budget t = ‖ŵ‖₂²' : 'budget t = ‖ŵ‖₁',
    hold: 'union',
    key: penalty,
  })
  const dy = useAxis({
    label: 'training loss δ = RSS',
    hold: 'union',
    key: `${penalty}/${noise}/${correlation}/${seed}`,
  })
  const setLambda = (x: number) => state.set('logLambda', Math.max(-1, Math.min(4, Math.round(x * 20) / 20)))

  return (
    <Figure
      title="One path, three parameters"
      state={state}
      hoverReadout={false}
      caption={`A linear model y = 2x₁ + 0.6x₂ + σε with n = ${N} points and two standard normal features of correlation ρ, fitted by ridge or the lasso along a grid of penalties λ (loss written as the residual sum of squares). Left: the solution path in the coefficient plane, from the least-squares fit (λ near 0) to the origin, with the ball Ω(w) ≤ t(λ) whose surface the current solution lies on, and the true coefficients. Middle: the budget t(λ) = Ω(ŵ_λ) falls as λ grows. Right: the training loss δ(λ) rises, and the dashed line marks nσ², the level the discrepancy principle aims for. Each λ is one budget and one loss level: the Ivanov problem with budget t(λ) and the Morozov problem with level δ(λ) return the same point. The readout checks that the slope −Δδ/Δt between neighbouring grid points equals λ. Drag the vertical line in either right-hand panel to move λ.`}
      readouts={
        <>
          <Readout label="λ" value={fmt(LAMBDAS[j])} />
          <Readout label="solution ŵ_λ" value={`(${fmt(w[0])}, ${fmt(w[1])})`} />
          <Readout label="budget t(λ) / loss δ(λ)" value={`${fmt(t)} / ${fmt(v.loss[j])}`} />
          <Readout label="price −Δδ/Δt along the path" value={fmt(price)} />
          <Readout
            label="λ where δ reaches nσ² (Morozov)"
            value={morozov > 0 ? fmt(LAMBDAS[morozov]) : morozov === 0 ? `below ${fmt(LAMBDAS[0])}` : 'not reached'}
          />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        <Plot x={wx} y={wy} title="coefficients">
          <Curve name="solution path" x={v.w1} y={v.w2} slot={0} stale={run.stale} />
          <Curve name="ball Ω(w) ≤ t(λ)" x={ball.x} y={ball.y} slot={1} dashed stale={run.stale} />
          <Points name="least squares" x={[v.ols[0]]} y={[v.ols[1]]} slot={0} stale={run.stale} />
          <Points name="true w" x={[TRUE_W[0]]} y={[TRUE_W[1]]} muted />
          <Points name="ŵ_λ" x={[w[0]]} y={[w[1]]} emphasis size={9} stale={run.stale} />
        </Plot>
        <Plot x={lx} y={ty} title="budget t(λ)">
          <Curve name="t(λ)" x={LOG_LAMBDAS} y={v.budget} slot={1} stale={run.stale} />
          <Handle kind="x" at={LOG_LAMBDAS[j]} label="λ" onDrag={setLambda} />
        </Plot>
        <Plot x={lx} y={dy} title="training loss δ(λ)">
          <Curve name="δ(λ)" x={LOG_LAMBDAS} y={v.loss} slot={2} stale={run.stale} />
          <Annotation y={level} text="nσ²" dashed />
          <Handle kind="x" at={LOG_LAMBDAS[j]} label="λ" onDrag={setLambda} />
        </Plot>
      </div>
    </Figure>
  )
}
