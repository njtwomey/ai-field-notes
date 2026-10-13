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
  useComputed,
  useFigureState,
} from 'aifn-render'
import { normals, stream } from 'aifn-compute/foundation/random'
import { toFlat } from 'aifn-compute/foundation/tensor'
import { fusedLassoSignal } from 'aifn-compute/optim/proximal'
import { differenceOperator, tikhonov } from 'aifn-compute/numerics/regularisation'

const N = 100
const T = Array.from({ length: N }, (_, i) => i)
// A piecewise-constant truth: zero, a high block, zero, a low block.
const TRUTH: number[] = T.map((i) => (i >= 20 && i < 45 ? 2 : i >= 65 && i < 80 ? 1 : 0))
const LOG_LAMBDAS = Array.from({ length: 41 }, (_, i) => -1 + i * 0.1)
const LAMBDAS = LOG_LAMBDAS.map((l) => 10 ** l)
// The quadratic fits each solve a stacked least-squares problem, so they use a coarser grid.
const QUAD_LOG = LOG_LAMBDAS.filter((_, i) => i % 2 === 0)
const IDENTITY = T.map((i) => T.map((j) => (i === j ? 1 : 0)))
const D1 = differenceOperator(N)
const fmt = (v: number) => (Number.isFinite(v) ? formatNumber(Number(v.toPrecision(3))) : '—')
const rms = (x: ArrayLike<number>) => Math.sqrt(TRUTH.reduce((a, v, i) => a + (x[i] - v) ** 2, 0) / N)
const pieces = (x: ArrayLike<number>) => {
  let count = 1
  for (let i = 1; i < N; i++) if (Math.abs(x[i] - x[i - 1]) > 1e-9) count++
  return count
}

/**
 * Total variation and the fused lasso (`fusedLassoSignal`) against the quadratic first-difference penalty
 * (`tikhonov` with `differenceOperator`) on a noisy piecewise-constant signal, each scored against the truth.
 */
export function TotalVariationDenoising() {
  const state = useFigureState({
    noise: float(0.5, { min: 0.1, max: 1.5, step: 0.05, label: 'noise σ' }),
    seed: int(1, { ge: 1, le: 999, label: 'noise draw' }),
    lambda1: float(0, { min: 0, max: 1.5, step: 0.05, label: 'sparsity weight λ₁ (fused lasso)' }),
    logLambda: float(0, { min: -1, max: 3, step: 0.1, label: 'log₁₀ fusion weight λ' }),
  })
  const { noise, seed, lambda1 } = state
  const y = useMemo(() => {
    const e = toFlat(normals(stream(`tv/${seed}`), [N]))
    return TRUTH.map((v, i) => v + noise * e[i])
  }, [noise, seed])
  const run = useComputed(
    () => {
      const tv = LAMBDAS.map((lambda2) => toFlat(fusedLassoSignal(y, { lambda1, lambda2 })))
      return { tv, tvError: tv.map(rms) }
    },
    [y, lambda1],
    { mode: 'release' },
  )
  const quadRun = useComputed(
    () => {
      const quad = QUAD_LOG.map((l) => toFlat(tikhonov(IDENTITY, y, { lambda: 10 ** l, L: D1 }).x))
      return { quad, quadError: quad.map(rms) }
    },
    [y],
    { mode: 'release' },
  )
  const v = { ...run.value, ...quadRun.value }
  const j = Math.max(0, Math.min(LAMBDAS.length - 1, Math.round((state.logLambda + 1) / 0.1)))
  const argmin = (a: number[]) => a.reduce((k, e, i) => (e < a[k] ? i : k), 0)
  const bestQuad = argmin(v.quadError)
  const bestTv = argmin(v.tvError)

  const sx = useAxis({ label: 'position i', range: [0, N - 1], nice: false })
  const sy = useAxis({ label: 'value', hold: 'union', key: `${noise}/${seed}` })
  const lx = useAxis({ label: 'log₁₀ λ', range: [-1, 3], nice: false })
  const ey = useAxis({ label: 'RMS error against the truth', hold: 'union', key: `${noise}/${seed}/${lambda1}` })

  return (
    <Figure
      title="Total variation keeps the jumps"
      state={state}
      hoverReadout={false}
      caption={`A piecewise-constant signal of ${N} values with four jumps, observed with Gaussian noise of standard deviation σ. Left: the truth (dashed), the data, the total variation fit ½‖y − θ‖² + λ Σ|θᵢ₊₁ − θᵢ| at the chosen λ (exact, by Condat's algorithm), and the quadratic penalty ‖y − θ‖² + λ‖Dθ‖² at its own best λ. With λ₁ > 0 the total variation fit becomes the fused lasso, soft thresholded at λ₁, and its small pieces are set to zero. Right: the RMS error of both fits against the truth along their paths. Total variation returns flat pieces with sharp jumps, each jump shrunk by λ over the lengths of its neighbours; the quadratic penalty cannot have both flat pieces and sharp jumps, so its best fit is either noisy or ramped. Drag the vertical line on the right to move the total variation λ.`}
      readouts={
        <>
          <Readout label="total variation: RMS error / pieces" value={`${fmt(v.tvError[j])} / ${pieces(v.tv[j])}`} />
          <Readout
            label="total variation: best λ / its error"
            value={`${fmt(LAMBDAS[bestTv])} / ${fmt(v.tvError[bestTv])}`}
          />
          <Readout
            label="quadratic: best λ / its error"
            value={`${fmt(10 ** QUAD_LOG[bestQuad])} / ${fmt(v.quadError[bestQuad])}`}
          />
          <Readout label="true pieces" value="5" />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <Plot x={sx} y={sy} title="signal">
          <Points name="data y" x={T} y={y} muted size={4} />
          <Curve name="truth" x={T} y={TRUTH} emphasis dashed />
          <Curve name={lambda1 > 0 ? 'fused lasso' : 'total variation'} x={T} y={v.tv[j]} slot={0} stale={run.stale} />
          <Curve name="quadratic, best λ" x={T} y={v.quad[bestQuad]} slot={1} stale={quadRun.stale} />
        </Plot>
        <Plot x={lx} y={ey} title="error along the path">
          <Curve
            name={lambda1 > 0 ? 'fused lasso' : 'total variation'}
            x={LOG_LAMBDAS}
            y={v.tvError}
            slot={0}
            stale={run.stale}
          />
          <Curve name="quadratic" x={QUAD_LOG} y={v.quadError} slot={1} stale={quadRun.stale} />
          <Handle
            kind="x"
            at={LOG_LAMBDAS[j]}
            label="λ"
            onDrag={(x) => state.set('logLambda', Math.max(-1, Math.min(3, Math.round(x * 10) / 10)))}
          />
        </Plot>
      </div>
    </Figure>
  )
}
