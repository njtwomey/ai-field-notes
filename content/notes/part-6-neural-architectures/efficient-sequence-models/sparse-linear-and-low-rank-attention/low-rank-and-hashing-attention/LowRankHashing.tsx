import { useMemo } from 'react'
import {
  choice,
  Curve,
  Figure,
  formatNumber,
  Handle,
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
import { stream } from 'aifn-compute/foundation/random'
import { type Tensor, tensor, toRows } from 'aifn-compute/foundation/tensor'
import { scaledDotProductAttention } from 'aifn-compute/nn/attention'
import {
  angularRotations,
  linformerMatrix,
  lowRankAt,
  lowRankSeries,
  lshAttentionMatrix,
  nystromMatrix,
  rowTotalVariation,
  segmentMeans,
  topicTokens,
} from 'aifn-methods/neural/sequence'

const N = 128
const D = 16
const RANKS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]
const BUCKETS = [2, 4, 8, 16, 32, 64]

type Structure = 'four' | 'eight' | 'none'
type Method = 'svd' | 'linformer' | 'nystrom' | 'lsh'

const TOPICS: Record<Structure, number> = { four: 4, eight: 8, none: 0 }

/** The attention matrix of 128 tokens and its low-rank and hashing approximations, scored by misplaced weight. */
export function LowRankHashing() {
  const state = useFigureState({
    structure: choice<Structure>(
      [
        { value: 'four', label: 'four topics in runs' },
        { value: 'eight', label: 'eight topics in runs' },
        { value: 'none', label: 'no structure' },
      ],
      'four',
      { label: 'tokens' },
    ),
    beta: slider(1, 20, 6, { step: 0.5, label: 'sharpness β' }),
    budget: int(8, { min: 1, max: 64, suggestions: [2, 4, 8, 16, 32], label: 'rank or keys per query k' }),
    rounds: int(2, { min: 1, max: 8, label: 'hashing rounds r' }),
    method: choice<Method>(
      [
        { value: 'svd', label: 'best rank k (SVD)' },
        { value: 'linformer', label: 'Linformer, pooled' },
        { value: 'nystrom', label: 'Nyström' },
        { value: 'lsh', label: 'LSH buckets' },
      ],
      'linformer',
      { label: 'approximation shown' },
    ),
  })
  const { structure, beta, budget, rounds, method } = state

  const base = useMemo(() => {
    const { q, k } = topicTokens(stream(5), N, D, TOPICS[structure], 0.6)
    const P = toRows(scaledDotProductAttention(tensor(q), tensor(k), tensor(k), { scale: beta }).weights as Tensor)
    const series = lowRankSeries(P)
    const curves = RANKS.map((r) => ({
      svd: rowTotalVariation(lowRankAt(series, r), P),
      linformer: rowTotalVariation(linformerMatrix(q, k, beta, segmentMeans(N, r)), P),
      nystrom: rowTotalVariation(nystromMatrix(q, k, beta, r), P),
    }))
    return { q, k, P, series, curves }
  }, [structure, beta])

  const lsh = useMemo(
    () =>
      [...BUCKETS].reverse().map((b) => {
        const res = lshAttentionMatrix(base.q, base.k, beta, angularRotations(stream(9), D, b, rounds))
        return { buckets: b, keys: res.keysPerQuery, tv: rowTotalVariation(res.weights, base.P), weights: res.weights }
      }),
    [base, beta, rounds],
  )

  const k = Math.min(64, Math.max(1, Math.round(budget)))
  const nearestLsh = lsh.reduce((best, x) =>
    Math.abs(Math.log(x.keys / k)) < Math.abs(Math.log(best.keys / k)) ? x : best,
  )
  const approx = useMemo(() => {
    switch (method) {
      case 'svd':
        return lowRankAt(base.series, k)
      case 'linformer':
        return linformerMatrix(base.q, base.k, beta, segmentMeans(N, k))
      case 'nystrom':
        return nystromMatrix(base.q, base.k, beta, k)
      case 'lsh':
        return nearestLsh.weights
    }
  }, [method, base, beta, k, nearestLsh])
  const tvAt = rowTotalVariation(approx, base.P)

  const pos = useMemo(() => Array.from({ length: N }, (_, i) => i), [])
  const top = useMemo(() => Math.max(...base.P.flat()), [base])
  const sv = base.series.singularValues.slice(0, 64).map((v) => Math.max(v, 1e-12))
  const idx = sv.map((_, i) => i + 1)

  const keyAxis = useAxis({ label: 'key position j', integer: true })
  const queryAxis = useAxis({ label: 'query position i', integer: true, inverse: true })
  const indexAxis = useAxis({ label: 'index t (rank)', log: true })
  const svAxis = useAxis({ label: 'singular value σₜ', log: true })
  const budgetAxis = useAxis({ label: 'rank k, or keys scored per query', log: true })
  const tvAxis = useAxis({ label: 'share of weight misplaced', range: [0, 1] })
  const drag = (x: number) => state.set('budget', Math.round(Math.min(64, Math.max(1, x))))
  const label = { svd: 'best rank k', linformer: 'Linformer, pooled', nystrom: 'Nyström', lsh: 'LSH' }[method]

  return (
    <Figure
      title="Low-rank and hashed approximations of an attention matrix"
      state={state}
      caption="128 tokens of width 16 whose queries and keys follow topics in runs, attention P = softmax(β Q Kᵀ). Top left: P. Top right: the chosen approximation at budget k. Bottom left: the singular values of P; with c topics, about c are large and the rest small, so P is close to rank c, until a larger β sharpens the attention and the tail rises. Bottom right: the share of each query's attention weight that each method puts in the wrong place (½ Σⱼ |P̂ᵢⱼ − Pᵢⱼ|, averaged over queries) against its budget: the rank k for the best rank-k matrix (the floor for any rank-k method), for Linformer with mean-pooling projections and for Nyström with segment-mean landmarks; the mean number of keys a query scores for LSH attention with 2 to 64 buckets and r rounds. Nyström's matrix can have negative entries and leaves the chart when its landmark matrix is ill-conditioned. Drag the vertical line in either bottom panel to set k; set the token structure, the sharpness, the hashing rounds and the approximation shown."
      readouts={
        <>
          <Readout label={`${label}, k = ${k}: misplaced`} value={formatNumber(Math.round(tvAt * 1000) / 1000)} />
          <Readout label="σₖ₊₁ / σ₁" value={formatNumber(Math.round((1000 * (sv[k] ?? 0)) / sv[0]) / 1000)} />
          <Readout
            label="LSH nearest budget"
            value={`${nearestLsh.buckets} buckets, ${formatNumber(Math.round(nearestLsh.keys))} keys, ${formatNumber(Math.round(nearestLsh.tv * 1000) / 1000)}`}
          />
          <Readout label="full attention" value={`${N} keys per query`} />
        </>
      }
    >
      <Plots cols={2} rows={2}>
        <Plot x={keyAxis} y={queryAxis} title="exact P">
          <Raster x={pos} y={pos} z={base.P} range={[0, top]} valueLabel="weight" />
        </Plot>
        <Plot x={keyAxis} y={queryAxis} title={`${label}, k = ${k}`}>
          <Raster x={pos} y={pos} z={approx} range={[0, top]} valueLabel="weight" />
        </Plot>
        <Plot x={indexAxis} y={svAxis} title="spectrum of P">
          <Points name="σₜ" x={idx} y={sv} slot={0} />
          <Handle kind="x" at={k} label="k" onDrag={drag} />
        </Plot>
        <Plot x={budgetAxis} y={tvAxis} title="error against budget">
          <Curve name="best rank k" x={RANKS} y={base.curves.map((c) => c.svd)} emphasis showPoints />
          <Curve name="Linformer, pooled" x={RANKS} y={base.curves.map((c) => c.linformer)} slot={0} showPoints />
          <Curve name="Nyström" x={RANKS} y={base.curves.map((c) => c.nystrom)} slot={1} showPoints />
          <Curve
            name={`LSH, ${rounds} round${rounds > 1 ? 's' : ''}`}
            x={lsh.map((l) => l.keys)}
            y={lsh.map((l) => l.tv)}
            slot={2}
            showPoints
          />
          <Handle kind="x" at={k} label="k" onDrag={drag} />
        </Plot>
      </Plots>
    </Figure>
  )
}
