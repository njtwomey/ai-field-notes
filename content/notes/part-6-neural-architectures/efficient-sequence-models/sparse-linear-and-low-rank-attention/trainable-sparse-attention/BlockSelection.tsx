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
  Raster,
  Readout,
  Segments,
  setting,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { integers, normals, stream } from 'aifn-compute/foundation/random'
import { tensor, type Tensor, toFlat, toRows } from 'aifn-compute/foundation/tensor'
import {
  attentionCost,
  blockScores,
  blockSparseMask,
  capturedWeight,
  scaledDotProductAttention,
  selectBlocks,
} from 'aifn-compute/nn/attention'

const N = 64
const D = 8
const RUN = 8
const TOPICS = 4
const LENGTHS = Array.from({ length: 41 }, (_, i) => Math.round(2 ** (10 + i / 4)))

/**
 * A sequence of N tokens in runs of RUN tokens, each run on one of TOPICS topics drawn at random. A token's query and
 * key are its topic's centre plus noise, scaled to unit length, so a query attends mostly to earlier runs on its topic.
 */
function tokens() {
  const s = stream(31)
  const centres = toRows(normals(s, [TOPICS, D]) as Tensor)
  const topicOfRun = Array.from({ length: N / RUN }, () => integers(s, TOPICS) as number)
  const noiseQ = toRows(normals(s, [N, D]) as Tensor)
  const noiseK = toRows(normals(s, [N, D]) as Tensor)
  const make = (z: number[], i: number) => {
    const v = z.map((x, t) => centres[topicOfRun[Math.floor(i / RUN)]][t] + 0.7 * x)
    const r = Math.hypot(...v)
    return v.map((x) => x / r)
  }
  return { q: noiseQ.map(make), k: noiseK.map(make) }
}

/** Grid lines between blocks of B positions over an N × N raster. */
function blockLines(B: number) {
  const lines: { from: [number, number]; to: [number, number] }[] = []
  for (let b = B; b < N; b += B) {
    lines.push({ from: [b - 0.5, -0.5], to: [b - 0.5, N - 0.5] })
    lines.push({ from: [-0.5, b - 0.5], to: [N - 0.5, b - 0.5] })
  }
  return lines
}

/** Each query picks its blocks of keys from block scores; the kept blocks against the full attention weights. */
export function BlockSelection() {
  const state = useFigureState({
    blockSize: choice<number>([4, 8, 16], 8, { label: 'block size B' }),
    topK: int(2, { min: 0, max: 8, suggestions: [0, 1, 2, 4], label: 'blocks chosen k' }),
    current: setting(true, 'always keep own block'),
    first: setting(false, 'always keep first block'),
    beta: slider(1, 12, 6, { step: 0.5, label: 'sharpness β' }),
    query: int(60, { min: 0, max: N - 1, label: 'query i' }),
  })
  const { blockSize, topK, current, first, beta, query } = state
  const { q, k } = useMemo(() => tokens(), [])

  const full = useMemo(
    () =>
      toRows(
        scaledDotProductAttention(tensor(q), tensor(k), tensor(k), { causal: true, scale: beta }).weights as Tensor,
      ),
    [q, k, beta],
  )
  const sel = useMemo(() => {
    const scoreTensor = blockScores(q, k, blockSize, { scale: beta })
    const scores = toRows(scoreTensor)
    const kept = selectBlocks(scoreTensor, blockSize, { topK, current, first })
    const mask = toRows(blockSparseMask(N, blockSize, kept))
    // Rows with no kept key (own block off and nothing chosen) keep their own key, so the softmax is defined.
    for (let i = 0; i < N; i++) if (!mask[i].some((v) => v)) mask[i][i] = 1
    const sparse = toRows(
      scaledDotProductAttention(tensor(q), tensor(k), tensor(k), { mask: tensor(mask), scale: beta }).weights as Tensor,
    )
    const captured = toFlat(capturedWeight(full, mask))
    const pairs = mask.reduce((a, row) => a + row.reduce((s, v) => s + v, 0), 0)
    return { scores, kept, mask, sparse, captured, pairs }
  }, [q, k, blockSize, topK, current, first, beta, full])

  const pos = useMemo(() => Array.from({ length: N }, (_, i) => i), [])
  const lines = useMemo(() => blockLines(blockSize), [blockSize])
  const fullGrid = useMemo(() => full.map((row, i) => row.map((x, j) => (j > i ? Number.NaN : x))), [full])
  const sparseGrid = useMemo(
    () => sel.sparse.map((row, i) => row.map((x, j) => (sel.mask[i][j] ? x : Number.NaN))),
    [sel],
  )
  const meanCaptured = sel.captured.reduce((a, b) => a + b, 0) / N
  const costs = useMemo(() => {
    const options = { headWidth: D, blockSize: 64, topK, window: (topK + 1) * 64 }
    return {
      full: LENGTHS.map((L) => attentionCost('full', L, options).multiplyAdds),
      window: LENGTHS.map((L) => attentionCost('window', L, options).multiplyAdds),
      block: LENGTHS.map((L) => attentionCost('block', L, options).multiplyAdds),
    }
  }, [topK])
  const i = Math.min(N - 1, Math.max(0, query))
  const visibleScores = sel.scores[i].map((s, b) =>
    Number.isFinite(s) ? `${b}: ${formatNumber(Math.round(s * 100) / 100)}` : null,
  )

  const keyAxis = useAxis({ label: 'key position j', integer: true })
  const queryAxis = useAxis({ label: 'query position i', integer: true, inverse: true })
  const posAxis = useAxis({ label: 'query position i', integer: true })
  const shareAxis = useAxis({ label: 'share of full weight kept', range: [0, 1.05] })
  const nAxis = useAxis({ label: 'sequence length n', log: true })
  const costAxis = useAxis({ label: 'multiply–adds per head (d = 8)', log: true })
  const pick = (y: number) => state.set('query', Math.round(Math.min(N - 1, Math.max(0, y))))

  return (
    <Figure
      title="Choosing blocks of keys per query"
      state={state}
      caption="Sixty-four tokens in runs of eight, each run on one of four topics; a query attends mostly to earlier runs on its own topic. Top left: the full causal softmax weights, with the block grid. Top right: the weights after each query keeps its own block and the k blocks whose mean key scores highest against it, renormalised over the kept keys; empty cells are never scored. Bottom left: the share of each query's full softmax weight that falls in its kept blocks. Bottom right: the cost of one head against length for full attention, a window of the same number of keys, and block selection with blocks of 64 (the scoring of block summaries keeps a small n² term). Drag the horizontal line to pick a query; set the block size, the number of chosen blocks, the fixed blocks and the sharpness."
      readouts={
        <>
          <Readout label={`query ${i}: kept blocks`} value={sel.kept[i].join(', ')} />
          <Readout label={`query ${i}: block scores`} value={visibleScores.filter((x) => x !== null).join(' · ')} />
          <Readout label={`query ${i}: weight kept`} value={formatNumber(Math.round(sel.captured[i] * 1000) / 1000)} />
          <Readout label="mean weight kept" value={formatNumber(Math.round(meanCaptured * 1000) / 1000)} />
          <Readout label="scores computed" value={`${sel.pairs} of ${(N * (N + 1)) / 2} causal pairs`} />
        </>
      }
    >
      <Plots cols={2} rows={2} heights={[3, 2]}>
        <Plot x={keyAxis} y={queryAxis} title="full attention">
          <Raster x={pos} y={pos} z={fullGrid} range={[0, 1]} valueLabel="weight" />
          <Segments segments={lines} muted />
          <Handle kind="y" at={i} label="query" onDrag={pick} />
        </Plot>
        <Plot x={keyAxis} y={queryAxis} title="kept blocks">
          <Raster x={pos} y={pos} z={sparseGrid} range={[0, 1]} valueLabel="weight" />
          <Segments segments={lines} muted />
          <Handle kind="y" at={i} label="query" onDrag={pick} />
        </Plot>
        <Plot x={posAxis} y={shareAxis} title="weight kept per query">
          <Curve name="share kept" x={pos} y={sel.captured} slot={0} showPoints />
        </Plot>
        <Plot x={nAxis} y={costAxis} title="cost">
          <Curve name="full" x={LENGTHS} y={costs.full} slot={1} />
          <Curve name="window, same keys" x={LENGTHS} y={costs.window} slot={2} />
          <Curve name="block selection" x={LENGTHS} y={costs.block} slot={0} />
        </Plot>
      </Plots>
    </Figure>
  )
}
