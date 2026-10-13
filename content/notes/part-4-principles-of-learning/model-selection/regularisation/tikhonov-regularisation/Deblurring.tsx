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
  useAxis,
  useComputed,
  useFigureState,
} from 'aifn-render'
import { normals, stream } from 'aifn-compute/foundation/random'
import { add, matmul, mul, norm, sub, tensor, toFlat, toRows, type Tensor } from 'aifn-compute/foundation/tensor'
import { svd } from 'aifn-compute/numerics/linalg'
import {
  convolutionMatrix,
  differenceOperator,
  discrepancyPrinciple,
  tikhonov,
  tikhonovPath,
} from 'aifn-compute/numerics/regularisation'
import { gaussianKernel } from 'aifn-compute/signal/image'

const N = 64
const T = Array.from({ length: N }, (_, i) => (i + 0.5) / N)
const LOG_LAMBDAS = Array.from({ length: 41 }, (_, i) => -8 + i * 0.25)
const LAMBDAS = LOG_LAMBDAS.map((l) => 10 ** l)
const PENALTIES = [
  { value: '0', label: 'size, L = I' },
  { value: '1', label: 'slope, first differences' },
  { value: '2', label: 'curvature, second differences' },
] as const
const fmt = (v: number) => (Number.isFinite(v) ? formatNumber(Number(v.toPrecision(3))) : '—')

/** The signal to recover: a smooth bump plus a plateau, on [0, 1]. */
const truth = (t: number) => Math.exp(-((t - 0.3) ** 2) / 0.008) + (t > 0.6 && t < 0.85 ? 0.6 : 0)
const X_TRUE = T.map(truth)
const X_TRUE_T = tensor(X_TRUE)

/** ‖x − x_true‖ for each row of a matrix of solutions. */
const errorNorms = (x: Tensor) => Array.from(toFlat(norm(sub(x, X_TRUE_T), 1) as Tensor))

/**
 * Deblurring a one-dimensional signal by Tikhonov regularisation on the engine: the solution at a chosen λ, its error
 * and residual along the path, and the Picard plot of the blur matrix.
 */
export function Deblurring() {
  const state = useFigureState({
    width: float(2, { min: 0.5, max: 4, step: 0.25, label: 'blur width (samples)' }),
    noise: float(0.01, { min: 0.001, max: 0.1, step: 0.001, label: 'noise σ' }),
    order: choice(PENALTIES, '0', { label: 'penalty ‖Lx‖²' }),
    seed: int(1, { ge: 1, le: 999, label: 'noise draw' }),
    logLambda: float(-3, { min: -8, max: 2, step: 0.25, label: 'log₁₀ λ' }),
  })
  const { width, noise, order, seed } = state
  const problem = useMemo(() => {
    // The blur: a Gaussian kernel's taps as a convolution matrix, cut off at the ends.
    const A = convolutionMatrix(gaussianKernel(width), N)
    const e = normals(stream(`deblur/${seed}`), [N])
    const b = Array.from(toFlat(add(matmul(A, X_TRUE_T), mul(e, noise))))
    return { A, b }
  }, [width, noise, seed])
  const run = useComputed(
    () => {
      const { A, b } = problem
      const standard = tikhonovPath(A, b, LAMBDAS)
      const s = toFlat(standard.singularValues)
      // The Picard plot's data coefficients |u_iᵀb|: b in the left singular vectors of A.
      const beta = toFlat(matmul(tensor([b]), svd(A).U)).map(Math.abs)
      // A difference penalty: one transformation to standard form (Eldén), then a filter per λ.
      const path = order === '0' ? standard : tikhonovPath(A, b, LAMBDAS, { L: differenceOperator(N, Number(order)) })
      const solutions = toRows(path.x) as number[][]
      const residual = Array.from(toFlat(path.residualNorm))
      const error = errorNorms(path.x)
      const naiveError = norm(sub(tikhonov(A, b, { lambda: 0 }).x, X_TRUE_T)) as number
      const morozov = discrepancyPrinciple(A, b, { sigma: noise })
      return { s, beta, solutions, residual, error, naiveError, morozov }
    },
    [problem, order, noise],
    { mode: 'release' },
  )
  const v = run.value
  const j = Math.max(0, Math.min(LAMBDAS.length - 1, Math.round((state.logLambda + 8) / 0.25)))
  const best = v.error.reduce((k, e, i) => (e < v.error[k] ? i : k), 0)
  const index = Array.from({ length: v.s.length }, (_, i) => i + 1)

  const sx = useAxis({ label: 't', range: [0, 1], nice: false })
  const sy = useAxis({ label: 'signal', range: [-0.5, 1.5], nice: false })
  const lx = useAxis({ label: 'log₁₀ λ', range: [-8, 2], nice: false })
  const ly = useAxis({ label: 'norm (log scale)', log: true, hold: 'union', key: `${width}/${noise}/${order}/${seed}` })
  const ix = useAxis({ label: 'index i', range: [0, N + 1], nice: false })
  const iy = useAxis({ label: 'value (log scale)', log: true, hold: 'union', key: `${width}/${noise}` })
  const setLambda = (x: number) => state.set('logLambda', Math.max(-8, Math.min(2, Math.round(x * 4) / 4)))

  return (
    <Figure
      title="Deblurring a signal by Tikhonov regularisation"
      state={state}
      hoverReadout={false}
      caption={`A signal on ${N} samples (a bump and a plateau) blurred by a Gaussian kernel and observed with white noise of standard deviation σ. Left: the true signal (dashed), the data, and the Tikhonov solution at the chosen λ. Middle: along the path, the error ‖x_λ − x_true‖ and the residual ‖Ax_λ − b‖, with the noise level σ√n that the discrepancy principle matches (horizontal line). Right: the Picard plot. The singular values sᵢ of the blur matrix fall steadily; the data coefficients |uᵢᵀb| follow them down until they reach the noise floor σ and then level off, so components beyond that index carry only noise. The line √λ is Tikhonov's soft cutoff: components with sᵢ above it pass, those below are damped. Wider blur makes the singular values fall faster, so fewer components can be recovered at a given noise level. A size penalty pulls the plateau and the bump towards zero; difference penalties leave constants or lines free and smooth the plateau's edges into ramps. Least squares (λ = 0) is off the scale. Drag the vertical line in the middle panel to move λ.`}
      readouts={
        <>
          <Readout label="λ / cutoff √λ" value={`${fmt(LAMBDAS[j])} / ${fmt(Math.sqrt(LAMBDAS[j]))}`} />
          <Readout
            label="error at this λ / smallest on the path"
            value={`${fmt(v.error[j])} / ${fmt(v.error[best])}`}
          />
          <Readout label="λ with the smallest error" value={fmt(LAMBDAS[best])} />
          <Readout label="least-squares error (λ = 0)" value={fmt(v.naiveError)} />
          <Readout
            label="discrepancy principle λ (size penalty)"
            value={v.morozov.attained ? fmt(v.morozov.lambda) : 'not attained'}
          />
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        <Plot x={sx} y={sy} title="signal">
          <Points name="blurred, noisy data b" x={T} y={problem.b} muted size={4} />
          <Curve name="true signal" x={T} y={X_TRUE} emphasis dashed />
          <Curve name="Tikhonov solution" x={T} y={v.solutions[j]} slot={0} stale={run.stale} />
        </Plot>
        <Plot x={lx} y={ly} title="error and residual">
          <Curve name="error ‖x_λ − x_true‖" x={LOG_LAMBDAS} y={v.error} slot={0} stale={run.stale} />
          <Curve name="residual ‖Ax_λ − b‖" x={LOG_LAMBDAS} y={v.residual} slot={1} stale={run.stale} />
          <Annotation y={noise * Math.sqrt(N)} text="σ√n" dashed />
          <Handle kind="x" at={LOG_LAMBDAS[j]} label="λ" onDrag={setLambda} />
        </Plot>
        <Plot x={ix} y={iy} title="Picard plot">
          <Points
            name="singular value sᵢ"
            x={index}
            y={v.s.map((x) => Math.max(x, 1e-16))}
            slot={2}
            stale={run.stale}
          />
          <Points
            name="data coefficient |uᵢᵀb|"
            x={index}
            y={v.beta.map((x) => Math.max(x, 1e-16))}
            slot={1}
            stale={run.stale}
          />
          <Annotation y={Math.sqrt(LAMBDAS[j])} text="√λ" dashed />
          <Annotation y={noise} text="σ" />
        </Plot>
      </div>
    </Figure>
  )
}
