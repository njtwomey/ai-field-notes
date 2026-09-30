import { useMemo, useState } from 'react'
import { gram as kernelGram, rbf } from 'aifn/kernels'
import { cholesky, conditionNumber, eigh, lu, qr, svd } from 'aifn/linalg'
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
} from 'aifn/tensor'
import { Slider } from '@lab/controls'
import { Figure } from '@lab/layout'
import { ChartSize, Readout, XYChart } from '@lab/viz'
import { formatValue, MatrixDecompositionView, TensorPanel } from '@lab/views'

/** A squared-exponential Gram matrix on n evenly spaced points in [0, 1], lengthscale ℓ. */
function gram(n: number, lengthscale: number): Tensor {
  return kernelGram(rbf({ lengthscale }), linspace(0, 1, n))
}

/** Cholesky of a kernel matrix: long lengthscales make it numerically singular, and the jitter needed is reported. */
export function CholeskyJitterSpecimen() {
  const [lengthscale, setLengthscale] = useState(0.3)
  const n = 12
  const K = useMemo(() => gram(n, lengthscale), [lengthscale])
  const result = useMemo(() => cholesky(K), [K])
  const Lt = useMemo(() => transpose(result.L), [result])
  return (
    <MatrixDecompositionView
      controls={<Slider label="lengthscale ℓ" value={lengthscale} onChange={setLengthscale} min={0.02} max={1.5} />}
      input={{ name: 'K', tensor: K }}
      factors={[
        { name: 'L', tensor: result.L },
        { name: 'Lᵀ', tensor: Lt },
      ]}
      equation="K + jitter·I = L Lᵀ"
      notes={
        <>
          <Readout label="jitter" value={formatValue(result.jitter)} />
          <Readout label="failed" value={String(result.failed)} />
          <Readout label="condition number of K" value={formatValue(conditionNumber(K))} />
        </>
      }
    />
  )
}

/** LU with partial pivoting of a matrix that becomes singular as ε → 0: `singular` is reported, never NaN. */
export function LuSpecimen() {
  const [epsilon, setEpsilon] = useState(0.5)
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
    <MatrixDecompositionView
      controls={<Slider label="ε (A₃₃ = 9 + ε)" value={epsilon} onChange={setEpsilon} min={0} max={2} />}
      input={{ name: 'A', tensor: A }}
      factors={[
        { name: 'P', tensor: f.P },
        { name: 'L', tensor: f.L },
        { name: 'U', tensor: f.U },
      ]}
      equation="P A = L U"
      notes={
        <>
          <Readout label="singular" value={String(f.singular)} />
          <Readout label="U₃₃" value={formatValue(f.U.data[8])} />
        </>
      }
    />
  )
}

/** Symmetric eigendecomposition: eigenvalues descending, eigenvectors as columns. */
export function EighSpecimen() {
  const [lengthscale, setLengthscale] = useState(0.2)
  const K = useMemo(() => gram(24, lengthscale), [lengthscale])
  const { values, vectors, sweeps } = useMemo(() => eigh(K), [K])
  const leading = useMemo(() => slice(vectors, null, [0, 4]) as Tensor, [vectors])
  const bars = useMemo(() => {
    const v = toFlat(values)
    return { x: v.map((_, i) => i), y: v.map((s) => Math.log10(Math.max(s, 1e-18))) }
  }, [values])
  return (
    <Figure
      title="Spectrum and leading eigenvectors of K"
      defaultSize="L"
      controls={<Slider label="lengthscale ℓ" value={lengthscale} onChange={setLengthscale} min={0.03} max={0.6} />}
      readouts={<Readout label="Jacobi sweeps" value={sweeps} />}
    >
      <ChartSize scale={0.5}>
        <div className="flex flex-col gap-4">
          <XYChart
            xLabel="index"
            yLabel="log₁₀ eigenvalue"
            integerX
            series={[{ name: 'eigenvalues', type: 'bar', x: bars.x, y: bars.y }]}
          />
          <TensorPanel name="the four leading eigenvectors (columns)" tensor={leading} />
        </div>
      </ChartSize>
    </Figure>
  )
}

/** Truncated SVD of a small pattern: the rank-k reconstruction and the singular values. */
export function SvdSpecimen() {
  const [rank, setRank] = useState(3)
  const image = useMemo(() => {
    const y = reshape(linspace(-1, 1, 24), [24, 1])
    const x = reshape(linspace(-1, 1, 32), [1, 32])
    const r = add(square(x), square(y))
    return add(mul(cos(mul(6, r)), exp(mul(-1.5, r))), mul(0.3, sin(mul(4, add(x, y))))) as Tensor
  }, [])
  const { U, S, V } = useMemo(() => svd(image), [image])
  const approx = useMemo(() => {
    const Uk = slice(U, null, [0, rank])
    const Vk = slice(V, null, [0, rank])
    return matmul(matmul(Uk, diag(slice(S, [0, rank]))), transpose(Vk)) as Tensor
  }, [U, S, V, rank])
  // Eckart–Young: the Frobenius error of the best rank-k approximation is the norm of the discarded singular values.
  const error = useMemo(() => (rank >= S.shape[0] ? 0 : (norm(slice(S, [rank, null])) as number)), [S, rank])
  const bars = useMemo(() => ({ x: toFlat(arange(S.shape[0])), y: toFlat(S) }), [S])
  const last = S.shape[0]
  return (
    <Figure
      title="A pattern and its rank-k reconstruction"
      defaultSize="L"
      controls={<Slider label="rank k" value={rank} onChange={setRank} min={1} max={last} step={1} />}
      readouts={<Readout label="‖A − A_k‖_F = √(Σ_{j>k} σⱼ²)" value={formatValue(error)} />}
      caption="Drag the line on the singular values to choose k: the values left of it are kept."
    >
      <div className="flex flex-col gap-4">
        <ChartSize scale={0.7}>
          <div className="grid grid-cols-[repeat(auto-fill,minmax(16rem,1fr))] gap-4">
            <TensorPanel name="A" tensor={image} />
            <TensorPanel name={`A_${rank}`} tensor={approx} />
          </div>
        </ChartSize>
        <ChartSize scale={0.5}>
          <XYChart
            xLabel="j"
            yLabel="σⱼ"
            integerX
            yLog
            series={[{ name: 'singular values', type: 'bar', x: bars.x, y: bars.y }]}
            handles={[
              {
                kind: 'x',
                at: rank - 0.5,
                label: 'k',
                onDrag: (x) => setRank(Math.min(Math.max(Math.round(x + 0.5), 1), last)),
              },
            ]}
          />
        </ChartSize>
      </div>
    </Figure>
  )
}

/** Householder QR of a tall matrix: orthonormal Q and upper-triangular R. */
export function QrSpecimen() {
  const A = useMemo(() => reshape(sin(mul(0.7, arange(8 * 4))), [8, 4]) as Tensor, [])
  const { Q, R } = useMemo(() => qr(A), [A])
  const gram = useMemo(() => matmul(transpose(Q), Q) as Tensor, [Q])
  return (
    <MatrixDecompositionView
      input={{ name: 'A', tensor: A }}
      factors={[
        { name: 'Q', tensor: Q },
        { name: 'R', tensor: R },
        { name: 'QᵀQ', tensor: gram },
      ]}
      equation="A = Q R (reduced), QᵀQ = I"
    />
  )
}
