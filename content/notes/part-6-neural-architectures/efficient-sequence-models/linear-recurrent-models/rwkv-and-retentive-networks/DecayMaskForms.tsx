import { useMemo } from 'react'
import {
  Curve,
  Figure,
  formatNumber,
  int,
  Plot,
  Plots,
  Points,
  Raster,
  Readout,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { normals, stream } from 'aifn-compute/foundation/random'
import {
  add,
  arange,
  cumsum,
  einsum,
  exp,
  expandDims,
  greaterEqual,
  matmul,
  mul,
  neg,
  sub,
  type Tensor,
  toFlat,
  toRows,
  transpose,
  where,
} from 'aifn-compute/foundation/tensor'
import { linearRecurrence } from 'aifn-compute/nn/sequence'

const DK = 4
const DV = 2

/**
 * Retention with a decay per step a_t = exp(−exp(d_t)), d_t = log(−log γ) + s z_t (Finch's parameterisation, with a
 * scalar decay per head). s = 0 is RetNet's fixed decay γ. Both forms are computed: the parallel form
 * ((Q Kᵀ) ∘ D) V with D_nm = a_{m+1} ⋯ a_n, and the recurrence S_t = a_t S_{t−1} + k_t v_tᵀ, o_t = S_tᵀ q_t.
 */
function forms(n: number, gamma: number, spread: number) {
  const s = stream(3)
  const q = normals(s, [n, DK], 0, 0.5)
  const k = normals(s, [n, DK], 0, 0.5)
  const v = normals(s, [n, DV])
  const z = normals(s, [n])
  const logDecay = neg(exp(add(Math.log(-Math.log(gamma)), mul(spread, z))))
  // Cumulative log decays L_t; the mask is exp(L_n − L_m) on and below the diagonal.
  const L = cumsum(logDecay)
  const steps = arange(n)
  const causal = greaterEqual(expandDims(steps, -1), expandDims(steps, 0))
  const mask = where(causal, exp(sub(expandDims(L, -1), expandDims(L, 0))), 0) as Tensor
  const parallel = matmul(mul(matmul(q, transpose(k)), mask), v) as Tensor
  const states = linearRecurrence(expandDims(expandDims(exp(logDecay), -1), -1), einsum('tk,tv->tkv', k, v))
  const recurrent = einsum('tk,tkv->tv', q, states) as Tensor
  const a = toFlat(exp(logDecay) as Tensor)
  return { mask: toRows(mask), parallel: toFlat(parallel), recurrent: toFlat(recurrent), a }
}

/** The decay mask of retention and its parallel and recurrent forms, with a fixed or an input-dependent decay. */
export function DecayMaskForms() {
  const state = useFigureState({
    gamma: slider(0.5, 0.99, 0.8, { step: 0.01, label: 'decay γ' }),
    spread: slider(0, 1.5, 0, { step: 0.05, label: 'input dependence s (0 = RetNet)' }),
    n: int(24, { min: 4, max: 64, suggestions: [8, 16, 24, 48, 64], label: 'sequence length n' }),
  })
  const { gamma, spread, n } = state
  const r = useMemo(() => forms(n, gamma, spread), [n, gamma, spread])
  const grid = useMemo(() => r.mask.map((row, i) => row.map((x, j) => (j > i ? Number.NaN : x))), [r.mask])
  const pos = useMemo(() => Array.from({ length: n }, (_, i) => i + 1), [n])
  const out1 = (f: number[]) => pos.map((_, t) => f[t * DV])
  const gap = Math.max(...r.parallel.map((x, i) => Math.abs(x - r.recurrent[i])))
  const minA = Math.min(...r.a)
  const maxA = Math.max(...r.a)

  const keyAxis = useAxis({ label: 'key position m', integer: true })
  const queryAxis = useAxis({ label: 'query position n', integer: true, inverse: true })
  const tAxis = useAxis({ label: 'position t', integer: true })
  const yAxis = useAxis({ label: 'output, channel 1' })

  return (
    <Figure
      title="One retention layer, two ways"
      state={state}
      caption="Left: the decay mask D. Row n is a query, column m a key, and the cell is the weight a_{m+1}⋯a_n that the key's value still has when query n reads the state; the upper triangle is empty because the layer is causal. With s = 0 every step decays by γ, the mask is Toeplitz (constant along diagonals) and its weights fall as γ^(n−m): this is RetNet. Raising s makes the per-step decay depend on a token-level signal, as in RWKV-6 (Finch), Mamba-2 and gated linear attention, and the bands break up where a strong decay erases the past. Right: the first output channel computed by the parallel form ((QKᵀ) ∘ D)V (line) and by the recurrence Sₜ = aₜ Sₜ₋₁ + kₜvₜᵀ (points). They agree to rounding for every setting."
      readouts={
        <>
          <Readout label="per-step decay aₜ" value={`${formatNumber(minA)} to ${formatNumber(maxA)}`} />
          <Readout label="memory 1 / (1 − γ)" value={`${formatNumber(1 / (1 - gamma))} tokens`} />
          <Readout label="largest |parallel − recurrent|" value={formatNumber(gap)} />
          <Readout label="state per head" value={`${DK} × ${DV} numbers`} />
        </>
      }
    >
      <Plots cols={2}>
        <Plot x={keyAxis} y={queryAxis}>
          <Raster x={pos} y={pos} z={grid} range={[0, 1]} valueLabel="weight Dₙₘ" />
        </Plot>
        <Plot x={tAxis} y={yAxis}>
          <Curve name="parallel form" x={pos} y={out1(r.parallel)} slot={0} />
          <Points name="recurrent form" x={pos} y={out1(r.recurrent)} slot={1} />
        </Plot>
      </Plots>
    </Figure>
  )
}
