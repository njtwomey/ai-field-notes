import { useMemo } from 'react'
import { gram as kernelGram, rbf } from 'aifn-compute/learning/kernels'
import { cholesky, conditionNumber, eigh, lu, qr, svd } from 'aifn-compute/numerics/linalg'
import {
  add,
  arange,
  cos,
  diag,
  exp,
  linspace,
  matmul,
  mul,
  norm,
  reshape,
  sin,
  slice,
  square,
  tensor,
  toFlat,
  transpose,
  type Tensor,
} from 'aifn-compute/foundation/tensor'
import { Dashboard, DashboardCell, DashboardRow, Equation, Figure, live, tex } from 'aifn-render/layout'
import { slider, useFigureState } from 'aifn-render/state'
import { Bars, Handle, Plot, Readout, useAxis } from 'aifn-render/viz'
import { formatValue, MatrixDecompositionPanel, TensorPanel } from '@lab/views'

/** A squared-exponential Gram matrix on n evenly spaced points in [0, 1], lengthscale ℓ. */
function gram(n: number, lengthscale: number): Tensor {
  return kernelGram(rbf({ lengthscale }), linspace(0, 1, n))
}

/** A positive number as TeX m × 10^e, with three significant digits. */
const powerTex = (v: number) => {
  const [m, e] = v.toExponential(2).split('e')
  return `${m} \\times 10^{${Number(e)}}`
}

/** Cholesky of a kernel matrix: long lengthscales make it numerically singular, and the jitter needed is reported. */
export function CholeskyJitterSpecimen() {
  const state = useFigureState({ lengthscale: slider(0.02, 1.5, 1.2, { label: 'lengthscale ℓ' }) })
  const lengthscale = state.lengthscale
  const n = 12
  const K = useMemo(() => gram(n, lengthscale), [lengthscale])
  const result = useMemo(() => cholesky(K), [K])
  const Lt = useMemo(() => transpose(result.L), [result])
  return (
    <Figure
      title="K + jitter·I = L Lᵀ"
      purpose="A long lengthscale makes a kernel matrix numerically singular; cholesky then adds the smallest diagonal jitter that lets it factor, and reports it."
      state={state}
      defaultSize="L"
      equation={
        <Equation>
          {tex`K + ${live(result.jitter, { digits: 3, strong: true })}\, I = L L^\top, \qquad \kappa(K) \approx ${powerTex(conditionNumber(K) as number)}`}
        </Equation>
      }
      readouts={
        <>
          <Readout label="jitter" value={formatValue(result.jitter)} />
          <Readout label="failed" value={String(result.failed)} />
        </>
      }
      caption="A squared-exponential Gram matrix on 12 evenly spaced points in [0, 1]. From about ℓ = 1 the factorisation needs jitter 10⁻¹²; below that K factors with jitter 0, even with a condition number near 10¹⁷. The rows of L below the first few are nearly zero: K has low numerical rank."
    >
      <MatrixDecompositionPanel
        input={{ name: 'K', tensor: K }}
        factors={[
          { name: 'L', tensor: result.L },
          { name: 'Lᵀ', tensor: Lt },
        ]}
      />
    </Figure>
  )
}

/** LU with partial pivoting of a matrix that becomes singular as ε → 0: `singular` is reported, never NaN. */
export function LuSpecimen() {
  const state = useFigureState({ epsilon: slider(0, 2, 0.5, { label: 'ε (A₃₃ = 9 + ε)' }) })
  const epsilon = state.epsilon
  const A = useMemo(
    () =>
      tensor([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9 + epsilon],
      ]),
    [epsilon],
  )
  const f = useMemo(() => lu(A), [A])
  return (
    <Figure
      title="P A = L U"
      purpose="Partial pivoting swaps rows so each pivot is the largest available; at ε = 0 the last pivot is zero and lu reports singular instead of dividing by it."
      state={state}
      defaultSize="L"
      equation={
        <Equation>{tex`P A = L U, \qquad U_{33} = ${live(f.U.data[8], { digits: 4, strong: true })}`}</Equation>
      }
      readouts={<Readout label="singular" value={String(f.singular)} />}
      caption="The rows of [[1, 2, 3], [4, 5, 6], [7, 8, 9]] are linearly dependent, so A is singular at ε = 0. U₃₃ is proportional to ε: slide ε to 0 and singular turns true."
    >
      <MatrixDecompositionPanel
        input={{ name: 'A', tensor: A }}
        factors={[
          { name: 'P', tensor: f.P },
          { name: 'L', tensor: f.L },
          { name: 'U', tensor: f.U },
        ]}
      />
    </Figure>
  )
}

/** Symmetric eigendecomposition: eigenvalues descending, eigenvectors as columns. */
export function EighSpecimen() {
  const state = useFigureState({ lengthscale: slider(0.03, 0.6, 0.2, { label: 'lengthscale ℓ' }) })
  const lengthscale = state.lengthscale
  const K = useMemo(() => gram(24, lengthscale), [lengthscale])
  const { values, vectors, sweeps } = useMemo(() => eigh(K), [K])
  const leading = useMemo(() => slice(vectors, null, [0, 4]) as Tensor, [vectors])
  const bars = useMemo(() => {
    const v = toFlat(values)
    return { x: v.map((_, i) => i), y: v.map((s) => Math.log10(Math.max(s, 1e-18))) }
  }, [values])
  const x = useAxis({ label: 'index', range: [-0.5, 23.5] })
  const y = useAxis({ label: 'log₁₀ eigenvalue', range: [-18, 2] })
  return (
    <Figure
      title="Spectrum and leading eigenvectors of K"
      purpose="A smooth kernel's eigenvalues fall off geometrically, faster for a longer lengthscale; its leading eigenvectors are smooth, each with one more sign change than the last."
      defaultSize="L"
      state={state}
      readouts={<Readout label="Jacobi sweeps" value={sweeps} />}
      caption="eigh of a 24 × 24 squared-exponential Gram matrix, by cyclic Jacobi rotations. Eigenvalues below about 10⁻¹⁶ of the largest are rounding noise (floored at 10⁻¹⁸ here). Raise ℓ and fewer eigenvalues stand above the noise."
    >
      <Dashboard>
        <DashboardRow ratio={1} minHeight={220}>
          <DashboardCell>
            <Plot x={x} y={y}>
              <Bars name="eigenvalues" x={bars.x} y={bars.y} base={-18} />
            </Plot>
          </DashboardCell>
        </DashboardRow>
        <DashboardRow ratio={1} minHeight={220}>
          <DashboardCell>
            <TensorPanel name="the four leading eigenvectors (columns)" tensor={leading} />
          </DashboardCell>
        </DashboardRow>
      </Dashboard>
    </Figure>
  )
}

/** Truncated SVD of a small pattern: the rank-k reconstruction and the singular values. */
export function SvdSpecimen() {
  const image = useMemo(() => {
    const y = reshape(linspace(-1, 1, 24), [24, 1])
    const x = reshape(linspace(-1, 1, 32), [1, 32])
    const r = add(square(x), square(y))
    return add(mul(cos(mul(6, r)), exp(mul(-1.5, r))), mul(0.3, sin(mul(4, add(x, y))))) as Tensor
  }, [])
  const { U, S, V } = useMemo(() => svd(image), [image])
  const last = S.shape[0]
  const state = useFigureState({ rank: slider(1, 24, 3, { step: 1, label: 'rank k' }) })
  const rank = Math.min(state.rank, last)
  const approx = useMemo(() => {
    const Uk = slice(U, null, [0, rank])
    const Vk = slice(V, null, [0, rank])
    return matmul(matmul(Uk, diag(slice(S, [0, rank]))), transpose(Vk)) as Tensor
  }, [U, S, V, rank])
  // Eckart–Young: the Frobenius error of the best rank-k approximation is the norm of the discarded singular values.
  const error = useMemo(() => (rank >= S.shape[0] ? 0 : (norm(slice(S, [rank, null])) as number)), [S, rank])
  const bars = useMemo(() => {
    const y = toFlat(S)
    return { x: toFlat(arange(S.shape[0])), y, base: Math.min(...y.filter((v) => v > 0)) / 10 }
  }, [S])
  const x = useAxis({ label: 'j', range: [-0.5, last - 0.5] })
  const y = useAxis({ label: 'σⱼ', log: true })
  return (
    <Figure
      title="A pattern and its rank-k reconstruction"
      purpose="Keeping the k largest singular values gives the best rank-k approximation, and its error is exactly the size of the singular values dropped (Eckart–Young)."
      defaultSize="L"
      state={state}
      equation={
        <Equation>
          {tex`\lVert A - A_{${rank}} \rVert_F = \sqrt{\textstyle\sum_{j > ${rank}} \sigma_j^2} = ${live(error, { digits: 4, strong: true })}`}
        </Equation>
      }
      caption="Drag the line on the singular values (log scale) to choose k: the values left of it are kept, and the error in the band is the root sum of squares of those right of it. The error falls fastest while k passes the few large values."
    >
      <Dashboard>
        <DashboardRow ratio={1.4} minHeight={240}>
          <DashboardCell>
            <TensorPanel name="A" tensor={image} />
          </DashboardCell>
          <DashboardCell>
            <TensorPanel name={`A_${rank}`} tensor={approx} />
          </DashboardCell>
        </DashboardRow>
        <DashboardRow ratio={1} minHeight={200}>
          <DashboardCell>
            <Plot x={x} y={y}>
              <Bars name="singular values" x={bars.x} y={bars.y} base={bars.base} />
              <Handle
                kind="x"
                at={rank - 0.5}
                label="k"
                onDrag={(v) => state.set('rank', Math.min(Math.max(Math.round(v + 0.5), 1), last))}
              />
            </Plot>
          </DashboardCell>
        </DashboardRow>
      </Dashboard>
    </Figure>
  )
}

/** Householder QR of a tall matrix: orthonormal Q and upper-triangular R. */
export function QrSpecimen() {
  const A = useMemo(() => reshape(sin(mul(0.7, arange(8 * 4))), [8, 4]) as Tensor, [])
  const { Q, R } = useMemo(() => qr(A), [A])
  const gram = useMemo(() => matmul(transpose(Q), Q) as Tensor, [Q])
  return (
    <Figure
      title="A = Q R (reduced), QᵀQ = I"
      purpose="Householder reflections turn a tall matrix into an orthonormal Q times an upper-triangular R; QᵀQ is the identity to rounding."
      defaultSize="L"
      hoverReadout={false}
    >
      <MatrixDecompositionPanel
        input={{ name: 'A', tensor: A }}
        factors={[
          { name: 'Q', tensor: Q },
          { name: 'R', tensor: R },
          { name: 'QᵀQ', tensor: gram },
        ]}
      />
    </Figure>
  )
}
