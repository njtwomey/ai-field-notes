import { useMemo } from 'react'
import {
  Annotation,
  Curve,
  Figure,
  float,
  formatNumber,
  Handle,
  int,
  Plot,
  Plots,
  Readout,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { normals, stream } from 'aifn-compute/foundation/random'
import { div, einsum, expandDims, norm, square, sub, sum, type Tensor, toRows } from 'aifn-compute/foundation/tensor'
import { deltaRule, linearRecurrence } from 'aifn-compute/nn/sequence'

/**
 * Write T = 3d random pairs (unit keys k_s in R^d, values v_s in R^d) into a d × d memory, one per step, and after each
 * write read back every pair stored so far. err[t][s] = ‖S_t k_s − v_s‖² / ‖v_s‖² for s ≤ t.
 */
function recall(d: number, beta: number, alpha: number) {
  const T = 3 * d
  const s = stream(11)
  const raw = normals(s, [T, d])
  const k = div(raw, norm(raw, -1, true))
  const v = normals(s, [T, d])
  const delta = deltaRule(k, k, v, beta, { decay: alpha }).states
  const additive = linearRecurrence(alpha, einsum('tv,tk->tvk', v, k))
  const vv = sum(square(v), -1)
  const errors = (states: unknown) => {
    const reads = einsum('tvk,sk->tsv', states as Tensor, k)
    return toRows(div(sum(square(sub(reads, expandDims(v, 0))), -1), expandDims(vv, 0)) as Tensor)
  }
  return { T, delta: errors(delta), additive: errors(additive) }
}

const meanStored = (err: number[][]) => err.map((row, t) => row.slice(0, t + 1).reduce((a, b) => a + b, 0) / (t + 1))

/** Retrieval error of the additive (linear attention) and delta-rule memories as pairs are written. */
export function AssociativeRecall() {
  const state = useFigureState({
    d: int(16, { min: 2, max: 32, suggestions: [4, 8, 16, 32], label: 'key and value dimension d' }),
    beta: float(1, { gt: 0, lt: 2, suggestions: [0.25, 0.5, 1, 1.5], label: 'writing strength β' }),
    alpha: float(1, { gt: 0, le: 1, suggestions: [1, 0.99, 0.95, 0.9], label: 'decay α (1 = no decay)' }),
    N: slider(1, 96, 24, { step: 1, onChart: true }),
  })
  const { d, beta, alpha } = state
  const r = useMemo(() => recall(d, beta, alpha), [d, beta, alpha])
  const N = Math.max(1, Math.min(Math.round(state.N), r.T))

  const written = useMemo(() => Array.from({ length: r.T }, (_, i) => i + 1), [r.T])
  const meanDelta = useMemo(() => meanStored(r.delta), [r.delta])
  const meanAdditive = useMemo(() => meanStored(r.additive), [r.additive])
  const pairs = written.slice(0, N)
  const xTop = useAxis({ label: 'pairs written N', integer: true })
  const yTop = useAxis({ label: 'mean relative error, stored pairs' })
  const xBottom = useAxis({ label: 'pair s (1 = oldest), after N writes', integer: true, range: [0.5, r.T + 0.5] })
  const yBottom = useAxis({ label: 'relative error of pair s' })

  return (
    <Figure
      title="Associative recall: additive and delta-rule memories"
      state={state}
      caption="A d × d memory receives one random pair (unit key, value) per step. After each write, every pair stored so far is read back by its key. Top: the mean relative error ‖S kₛ − vₛ‖² / ‖vₛ‖² over the stored pairs against the number written. The additive memory of linear attention adds every pair, so each read also returns the other values weighted by their keys' overlaps, and the error grows with N from the second write. The delta rule first subtracts what the memory already returns for the key, so with β = 1 the newest pair is stored exactly and older pairs change only along the new key. Past N = d no d × d memory can hold every pair; the delta rule then keeps the recent ones and gives up the oldest. Bottom: the error of each pair after N writes, oldest first. Drag N on the top panel. A β below 1 stores each pair only partly and disturbs the old ones less; α below 1 adds Gated DeltaNet's decay."
      readouts={
        <>
          <Readout label="pairs written N" value={String(N)} />
          <Readout label="mean error, additive" value={formatNumber(meanAdditive[N - 1])} />
          <Readout label="mean error, delta rule" value={formatNumber(meanDelta[N - 1])} />
          <Readout label="newest pair, delta rule" value={formatNumber(r.delta[N - 1][N - 1])} />
          <Readout label="memory size" value={`${d} × ${d} = ${d * d} numbers`} />
        </>
      }
    >
      <Plots rows={2}>
        <Plot x={xTop} y={yTop}>
          <Curve name="additive (linear attention)" x={written} y={meanAdditive} slot={0} />
          <Curve name="delta rule" x={written} y={meanDelta} slot={1} />
          <Annotation x={d} text="N = d" dashed />
          <Handle {...state.handle('N', { label: 'N' })} />
        </Plot>
        <Plot x={xBottom} y={yBottom}>
          <Curve name="additive (linear attention)" x={pairs} y={r.additive[N - 1].slice(0, N)} slot={0} showPoints />
          <Curve name="delta rule" x={pairs} y={r.delta[N - 1].slice(0, N)} slot={1} showPoints />
        </Plot>
      </Plots>
    </Figure>
  )
}
