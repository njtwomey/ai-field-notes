/**
 * Linear systems x′ = Ax: the flow x(t) = e^{At} x₀, from the matrix exponential of `aifn/linalg`.
 */

import { expm } from 'aifn/linalg'
import { fromData, toFlat, type Matrix } from 'aifn/tensor'
import { matVec, toF64, toMatrixF64, type MatrixLike, type VectorLike } from './vector'

// TODO(consolidation WP1): remove; `expm` now lives in aifn/linalg.
export { expm, type MatrixExponential } from 'aifn/linalg'

/**
 * The flow of the linear system x′ = Ax: x(t) = e^{At} x₀ at each of the given times, as a [times, n] matrix. Each
 * time is an independent `expm`, so the times need not be evenly spaced.
 */
export function linearFlow(a: MatrixLike, x0: VectorLike, times: VectorLike): Matrix {
  const { data: A, n } = toMatrixF64(a, 'linearFlow')
  const x = toF64(x0, 'linearFlow')
  if (x.length !== n) throw new Error(`linearFlow: x0 has length ${x.length}, A is ${n}×${n}`)
  const ts = toF64(times, 'linearFlow')
  const out = new Float64Array(ts.length * n)
  ts.forEach((t, k) => {
    const E = toF64(
      toFlat(
        expm(
          fromData(
            A.map((v) => v * t),
            [n, n],
          ),
        ).value,
      ),
      'linearFlow',
    )
    out.set(matVec(E, x, n, n), k * n)
  })
  return fromData(out, [ts.length, n])
}
