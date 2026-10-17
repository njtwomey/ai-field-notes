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
  Readout,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { stream } from 'aifn-compute/foundation/random'
import { associativeRecallByDimension } from 'aifn-methods/neural/sequence'

const DIMS = [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]

/** MQAR accuracy of softmax attention, linear attention and the delta rule against the key dimension d. */
export function RecallByStateSize() {
  const state = useFigureState({
    pairs: int(16, { min: 2, max: 48, suggestions: [8, 16, 32], label: 'key-value pairs N' }),
    beta: float(1, { gt: 0, lt: 2, suggestions: [0.5, 1, 1.5], label: 'delta-rule strength β' }),
    d: slider(2, 64, 16, { step: 1, onChart: true }),
  })
  const { pairs, beta } = state
  const r = useMemo(
    () => associativeRecallByDimension(stream(5), { pairs, dims: DIMS, trials: 8, valueVocab: 32, scale: 30, beta }),
    [pairs, beta],
  )
  const i = DIMS.reduce((best, d, k) => (Math.abs(d - state.d) < Math.abs(DIMS[best] - state.d) ? k : best), 0)
  const d = DIMS[i]

  const x = useAxis({ label: 'key and value dimension d', log: true, range: [2, 64] })
  const y = useAxis({ label: 'MQAR accuracy', range: [0, 1.02] })

  return (
    <Figure
      title="Associative recall against state size"
      state={state}
      caption="Multi-query associative recall with N pairs, then every key as a query. Keys and values are random unit vectors in d dimensions, the queries are the keys themselves, and a read counts as correct when its nearest value embedding (of 32) is the right one. Nothing is trained: each memory has ideal projections, so the curves show what the memory can hold. Softmax attention keeps all N keys and values and finds the matching key once d is large enough to tell N random directions apart. Linear attention (an unnormalised sum of value-key outer products) and the delta rule keep a d × d state, and come close to full accuracy only once d is about twice N, where their state of d² numbers is as large as the attention cache of 2Nd numbers. Drag the line to read the accuracies and memory sizes at one d. The delta rule with β = 1 stores each new pair exactly at the cost of the earlier ones, which with random keys that are never written twice does worse than adding; β = 0.5 spreads the error and roughly matches the additive memory."
      readouts={
        <>
          <Readout label="d" value={String(d)} />
          <Readout label="attention" value={formatNumber(r.attention[i])} />
          <Readout label="linear attention" value={formatNumber(r.linear[i])} />
          <Readout label="delta rule" value={formatNumber(r.delta[i])} />
          <Readout label="attention cache, numbers" value={`2Nd = ${2 * pairs * d}`} />
          <Readout label="recurrent state, numbers" value={`d² = ${d * d}`} />
        </>
      }
    >
      <Plot x={x} y={y}>
        <Curve name="softmax attention" x={DIMS} y={r.attention} slot={1} showPoints />
        <Curve name="linear attention" x={DIMS} y={r.linear} slot={0} showPoints />
        <Curve name="delta rule" x={DIMS} y={r.delta} slot={2} showPoints />
        <Annotation x={pairs} text="d = N" dashed />
        <Handle {...state.handle('d', { label: 'd' })} />
      </Plot>
    </Figure>
  )
}
