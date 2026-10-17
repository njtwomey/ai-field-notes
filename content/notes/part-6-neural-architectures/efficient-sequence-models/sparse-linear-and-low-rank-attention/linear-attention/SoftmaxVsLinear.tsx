import { useMemo, useState } from 'react'
import {
  choice,
  Curve,
  Figure,
  formatNumber,
  int,
  Player,
  Plot,
  Plots,
  Points,
  Raster,
  Readout,
  Segments,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { normals, stream } from 'aifn-compute/foundation/random'
import { mul, type Tensor, toFlat, toRows, type Value } from 'aifn-compute/foundation/tensor'
import { scaledDotProductAttention } from 'aifn-compute/nn/attention'
import {
  eluFeatureMap,
  linearAttention,
  linearAttentionRecurrent,
  linearAttentionWeights,
  positiveRandomFeatures,
} from 'aifn-compute/nn/sequence'

const N = 16
const D = 4
const DV = 2

type MapKind = 'elu' | 'favor'

/** Fixed queries, keys and values for the figure: N tokens, width D, value width DV. */
function tokens() {
  const s = stream(21)
  return { q: normals(s, [N, D]), k: normals(s, [N, D]), v: normals(s, [N, DV]) }
}

/** A box around query row t of an N × N raster whose rows run 0 … N − 1. */
function rowBox(t: number) {
  const [x0, x1, y0, y1] = [-0.5, N - 0.5, t - 0.5, t + 0.5]
  return [
    { from: [x0, y0], to: [x1, y0] },
    { from: [x1, y0], to: [x1, y1] },
    { from: [x1, y1], to: [x0, y1] },
    { from: [x0, y1], to: [x0, y0] },
  ] as const
}

/** Causal softmax attention beside causal linear attention, with linear attention's recurrent state step by step. */
export function SoftmaxVsLinear() {
  const state = useFigureState({
    map: choice<MapKind>(
      [
        { value: 'elu', label: 'elu(x) + 1' },
        { value: 'favor', label: 'positive random features (Performer)' },
      ],
      'elu',
      { label: 'feature map φ' },
    ),
    m: int(64, {
      min: 1,
      max: 1024,
      suggestions: [4, 16, 64, 256, 1024],
      label: 'random features m',
      when: (v) => v.map === 'favor',
    }),
  })
  const { map, m } = state

  const r = useMemo(() => {
    const { q, k, v } = tokens()
    const exact = scaledDotProductAttention(q, k, v, { causal: true })
    const soft = toRows(exact.weights as Tensor)
    const softOut = toRows(exact.output as Tensor)
    // The softmax kernel exp(qᵀk/√d) is exp(q'ᵀk') with q' = q d^{−1/4}, k' = k d^{−1/4}.
    const qs = mul(q, D ** -0.25)
    const ks = mul(k, D ** -0.25)
    const featureMap: (x: Value) => Value =
      map === 'elu' ? eluFeatureMap : positiveRandomFeatures(stream(5), D, m).featureMap
    const features = map === 'elu' ? D : m
    const lin = toRows(linearAttentionWeights(qs, ks, featureMap) as Tensor)
    const parallel = toRows(linearAttention(qs, ks, v, { featureMap }) as Tensor)
    const rec = linearAttentionRecurrent(qs, ks, v, { featureMap })
    const recurrent = toRows(rec.output as Tensor)
    const flatStates = toFlat(rec.states as Tensor)
    // states[t] is features × DV; draw it as DV rows (value channels) by `features` columns.
    const states = Array.from({ length: N }, (_, t) =>
      Array.from({ length: DV }, (_, c) =>
        Array.from({ length: features }, (_, f) => flatStates[t * features * DV + f * DV + c]),
      ),
    )
    const tv = soft.map((row, i) => row.reduce((a, p, j) => a + Math.abs(p - lin[i][j]), 0) / 2)
    const gap = Math.max(...parallel.flatMap((row, i) => row.map((x, c) => Math.abs(x - recurrent[i][c]))))
    return { soft, softOut, lin, parallel, recurrent, states, tv, gap, features }
  }, [map, m])

  const [step, setStep] = useState(0)
  const t = Math.min(step, N - 1)
  const pos = useMemo(() => Array.from({ length: N }, (_, i) => i), [])
  const featureIdx = useMemo(() => Array.from({ length: r.features }, (_, f) => f + 1), [r.features])
  const channels = useMemo(() => Array.from({ length: DV }, (_, c) => c + 1), [])
  const upper = (rows: number[][]) => rows.map((row, i) => row.map((x, j) => (j > i ? Number.NaN : x)))
  const box = useMemo(() => rowBox(t), [t])
  const meanTv = r.tv.reduce((a, b) => a + b, 0) / N
  const shown = pos.slice(0, t + 1)

  const keyAxis = useAxis({ label: 'key position j', integer: true })
  const queryAxis = useAxis({ label: 'query position i', integer: true, inverse: true })
  const featureAxis = useAxis({ label: 'feature index', integer: true })
  const channelAxis = useAxis({ label: 'value channel', integer: true })
  const tAxis = useAxis({ label: 'position t', integer: true })
  const outAxis = useAxis({ label: 'output, channel 1' })

  return (
    <Figure
      title="Softmax and linear attention, and the state that replaces the cache"
      state={state}
      caption="Sixteen tokens with random queries, keys and values (width 4, causal). Top: the attention weights of softmax attention and of linear attention with the chosen feature map; row i is a query, the box marks the current position t. Bottom left: linear attention's state Sₜ = Σ_{s≤t} φ(kₛ)vₛᵀ after position t, one row per value channel and one column per feature; it has the same size at every t, while a softmax layer's cache holds t keys and values. Bottom right: the first output channel of softmax attention, of linear attention in its parallel form (lines) and of the recurrence run up to t (points), which agree. With elu + 1 the weights are flatter than softmax's. With positive random features the weights estimate softmax's, closer as m grows. Step through t with the player; set the feature map and the number of random features."
      controls={
        <Player value={t} onChange={setStep} count={N} label="position t" format={(i) => `t = ${i + 1} of ${N}`} />
      }
      readouts={
        <>
          <Readout label="row t: weight misplaced vs softmax" value={formatNumber(r.tv[t])} />
          <Readout label="mean over rows" value={formatNumber(meanTv)} />
          <Readout label="linear state" value={`${r.features} × ${DV} + ${r.features} numbers`} />
          <Readout label="softmax cache at t" value={`${t + 1} × (${D} + ${DV}) numbers`} />
          <Readout label="largest |parallel − recurrent|" value={formatNumber(r.gap)} />
        </>
      }
    >
      <Plots cols={2} rows={2} heights={[3, 2]}>
        <Plot x={keyAxis} y={queryAxis} title="softmax attention">
          <Raster x={pos} y={pos} z={upper(r.soft)} range={[0, 1]} valueLabel="weight" />
          <Segments segments={box} emphasis width={2} />
        </Plot>
        <Plot x={keyAxis} y={queryAxis} title="linear attention">
          <Raster x={pos} y={pos} z={upper(r.lin)} range={[0, 1]} valueLabel="weight" />
          <Segments segments={box} emphasis width={2} />
        </Plot>
        <Plot x={featureAxis} y={channelAxis} title={`state S after t = ${t + 1}`}>
          <Raster x={featureIdx} y={channels} z={r.states[t]} scale="diverging" valueLabel="state entry" />
        </Plot>
        <Plot x={tAxis} y={outAxis} title="outputs">
          <Curve name="softmax" x={pos.map((i) => i + 1)} y={r.softOut.map((o) => o[0])} slot={0} />
          <Curve name="linear, parallel" x={pos.map((i) => i + 1)} y={r.parallel.map((o) => o[0])} slot={1} />
          <Points
            name="linear, recurrent"
            x={shown.map((i) => i + 1)}
            y={shown.map((i) => r.recurrent[i][0])}
            slot={1}
          />
        </Plot>
      </Plots>
    </Figure>
  )
}
