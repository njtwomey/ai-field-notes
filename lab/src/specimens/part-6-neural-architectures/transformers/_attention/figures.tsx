import { useMemo, useState } from 'react'
import { normal, stream } from 'aifn-compute/foundation/random'
import { add, fromData, mul, reshape, slice, tensor, toFlat, unwrap, type Tensor } from 'aifn-compute/foundation/tensor'
import { trace } from 'aifn-compute/foundation/trace'
import {
  alibiBias,
  alibiSlopes,
  applyRope,
  flashAttentionSteps,
  kvCacheMemory,
  multiHeadAttention,
  MultiHeadAttention,
  paddingMask,
  positionMask,
  positionRange,
  ropeFrequencies,
  scaledDotProductAttention,
  sinusoidalPositions,
  t5RelativeBias,
  t5RelativeBucket,
  type RopeScaling,
} from 'aifn-compute/nn/attention'
import { Figure } from 'aifn-render/layout'
import { Player } from 'aifn-render/controls'
import { Input } from 'aifn-render/ui/input'
import { choice, row, setting, slider, useFigureState, when } from 'aifn-render/state'
import { Annotation, Bars, Curve, Plot, Plots, Raster, Readout, Vectors, formatNumber, useAxis } from 'aifn-render/viz'
import { AttentionPanel, attentionPattern } from '@lab/views'

const fmt = (v: number) => formatNumber(v)
const raw = (v: unknown) => unwrap(v as Tensor) as Tensor
const range = (n: number) => Array.from({ length: n }, (_, i) => i)
const dotOf = (a: readonly number[], b: readonly number[]) => a.reduce((acc, v, j) => acc + v * b[j], 0)

// ---------------------------------------------------------------------------------------------------------------------
// 1. Attention over an editable sequence: positional schemes and masks.

const D = 16
const HEADS = 2
const SCHEMES = [
  { value: 'none', label: 'none (NoPE)' },
  { value: 'sinusoidal', label: 'sinusoidal' },
  { value: 'learned', label: 'learned' },
  { value: 'rope', label: 'RoPE' },
  { value: 'alibi', label: 'ALiBi' },
  { value: 't5', label: 'T5 bias' },
] as const
const MASKS = [
  { value: 'none', label: 'none (bidirectional)' },
  { value: 'causal', label: 'causal' },
  { value: 'window', label: 'sliding window' },
  { value: 'padding', label: 'padding (last 2)' },
] as const

/** A fixed vector per word, so a repeated word has the same embedding wherever it sits. */
const embeddingOf = (word: string): number[] => toFlat(normal(stream(`word:${word}`), 0, 1, { shape: [D] }) as Tensor)

export function AttentionPositionsSpecimen() {
  const [text, setText] = useState('the cat sat on the mat and the cat slept')
  const state = useFigureState({
    position: row('1 · positions', {
      scheme: choice(SCHEMES, 'none', { label: 'scheme' }),
      base: slider(10, 10000, 100, { label: 'RoPE base', step: 10, when: when('scheme', 'rope') }),
    }),
    mask: row('2 · mask', {
      mask: choice(MASKS, 'none', { label: 'mask' }),
      window: slider(1, 6, 3, { label: 'window', step: 1, when: when('mask', 'window') }),
    }),
    reveal: row('3 · reveal', { sharpness: slider(0.5, 6, 3, { label: 'score scale (× 1/√d)', step: 0.1 }) }),
  })
  const { scheme, base } = state.position
  const { mask: maskKind, window } = state.mask
  const { sharpness } = state.reveal
  const tokens = useMemo(() => text.trim().split(/\s+/).filter(Boolean).slice(0, 12), [text])
  const T = Math.max(1, tokens.length)
  const words = useMemo(() => (tokens.length ? tokens : ['∅']), [tokens])
  const layer = useMemo(() => MultiHeadAttention(D, { heads: HEADS }), [])
  const params = useMemo(() => layer.init(stream('attention-page')), [layer])
  const result = useMemo(() => {
    let x: Tensor = tensor(words.map(embeddingOf))
    if (scheme === 'sinusoidal') x = raw(add(x, sinusoidalPositions(T, D)))
    if (scheme === 'learned')
      x = raw(add(x, slice(normal(stream('learned-positions'), 0, 1, { shape: [16, D] }) as Tensor, [0, T])))
    const pos = positionRange(T)
    const bias =
      scheme === 'alibi'
        ? alibiBias(HEADS, pos, pos)
        : scheme === 't5'
          ? t5RelativeBias(normal(stream('t5-table'), 0, 1.5, { shape: [32, HEADS] }) as Tensor, pos, pos, {
              bidirectional: maskKind !== 'causal' && maskKind !== 'window',
            })
          : undefined
    const padding = maskKind === 'padding' ? reshape(paddingMask([Math.max(1, T - 2)], T), [1, T]) : undefined
    const r = multiHeadAttention(params, x, x, {
      heads: HEADS,
      causal: maskKind === 'causal' || maskKind === 'window',
      window: maskKind === 'window' ? window : undefined,
      mask: padding,
      bias,
      rope: scheme === 'rope' ? { base } : undefined,
      scale: sharpness / Math.sqrt(D / HEADS),
    })
    const shown = r.mask ?? (padding ? raw(mul(padding, fromData(new Float64Array(T * T).fill(1), [T, T]))) : null)
    return { weights: raw(r.weights), mask: shown }
  }, [words, T, scheme, base, maskKind, window, sharpness, params])
  // Repeated words: with no positions and no mask, attention cannot tell their rows apart.
  const repeat = useMemo(() => {
    const w = toFlat(result.weights)
    let worst = -1
    words.forEach((a, i) =>
      words.forEach((b, j) => {
        if (j <= i || a !== b) return
        for (let h = 0; h < HEADS; h++)
          for (let k = 0; k < T; k++) worst = Math.max(worst, Math.abs(w[(h * T + i) * T + k] - w[(h * T + j) * T + k]))
      }),
    )
    return worst
  }, [result, words, T])
  const pattern = useMemo(
    () => attentionPattern({ keys: words, weights: result.weights, mask: result.mask }),
    [words, result],
  )
  return (
    <Figure
      title="Attention, masks and positions"
      purpose="Self-attention sees a set, not a sequence: without positions a repeated word gets the same row of weights wherever it sits; a positional scheme breaks the tie, and a mask decides which keys each query may see."
      defaultSize="L"
      state={state}
      controls={
        <div className="flex items-center gap-2">
          <span className="text-xs text-muted-foreground">sequence (edit)</span>
          <Input
            value={text}
            onChange={(e) => setText(e.target.value)}
            className="max-w-md font-mono"
            aria-label="sequence"
          />
        </div>
      }
      readouts={{
        sequence: (
          <>
            <Readout label="tokens" value={String(T)} />
            <Readout
              label="largest difference between rows of a repeated word"
              value={repeat < 0 ? 'no repeated word' : fmt(repeat)}
            />
          </>
        ),
      }}
      caption="aifn-compute/nn/attention multiHeadAttention (2 heads, d = 16, fixed random projections) over fixed random word embeddings. Sinusoidal and learned positions are added to the embeddings; RoPE rotates queries and keys; ALiBi and T5 add a bias per head to the scores. Crossed cells are hidden by the mask. Hover a query token to see its row. With the scheme at none and no mask, the two rows of “the” and of “cat” coincide."
    >
      <AttentionPanel pattern={pattern} head="all" />
    </Figure>
  )
}

// ---------------------------------------------------------------------------------------------------------------------
// 2. RoPE: rotation of one coordinate pair, the score against the offset, and the frequencies under rescaling.

const ROPE_D = 32
const SCALINGS = [
  { value: 'none', label: 'none' },
  { value: 'linear', label: 'position interpolation' },
  { value: 'ntk', label: 'NTK-aware' },
  { value: 'yarn', label: 'YaRN' },
] as const

export function RopeSpecimen() {
  const state = useFigureState({
    where: row('1 · positions', {
      p: slider(0, 64, 12, { label: 'query position p', step: 1 }),
      s: slider(0, 64, 4, { label: 'key position s', step: 1 }),
    }),
    pair: row('2 · coordinate pair', { i: slider(0, ROPE_D / 2 - 1, 1, { label: 'pair i', step: 1 }) }),
    scaling: row('3 · rescaling', {
      base: slider(100, 100000, 10000, { label: 'base', step: 100 }),
      kind: choice(SCALINGS, 'none', { label: 'rescaling' }),
      factor: slider(1, 8, 4, { label: 'factor s', step: 0.5 }),
    }),
  })
  const { p, s } = state.where
  const { i } = state.pair
  const { base, kind, factor } = state.scaling
  const scaling: RopeScaling | undefined = useMemo(
    () =>
      kind === 'none'
        ? undefined
        : kind === 'yarn'
          ? { kind: 'yarn', factor, originalLength: 64 }
          : { kind: kind as 'linear' | 'ntk', factor },
    [kind, factor],
  )
  const q = useMemo(() => normal(stream('rope-q'), 0, 1, { shape: [1, ROPE_D] }) as Tensor, [])
  const k = useMemo(() => normal(stream('rope-k'), 0, 1, { shape: [1, ROPE_D] }) as Tensor, [])
  const opts = useMemo(() => ({ base, scaling }), [base, scaling])
  const rotated = useMemo(() => {
    const qp = toFlat(raw(applyRope(q, [p], opts)))
    const ks = toFlat(raw(applyRope(k, [s], opts)))
    const h = ROPE_D / 2
    return {
      q: [qp[i], qp[i + h]] as [number, number],
      k: [ks[i], ks[i + h]] as [number, number],
      score: dotOf(qp, ks),
    }
  }, [q, k, p, s, i, opts])
  const shifted = useMemo(
    () => dotOf(toFlat(raw(applyRope(q, [p + 10], opts))), toFlat(raw(applyRope(k, [s + 10], opts)))),
    [q, k, p, s, opts],
  )
  const offsets = useMemo(() => range(161).map((o) => o - 80), [])
  // q at position 80 + o against k at 80, for offsets o = −80 … 80.
  const scoreCurve = useMemo(() => {
    const qv = toFlat(q)
    const rows = fromData(Float64Array.from(offsets.flatMap(() => qv)), [offsets.length, ROPE_D])
    const qs = toFlat(
      raw(
        applyRope(
          rows,
          offsets.map((o) => 80 + o),
          opts,
        ),
      ),
    )
    const k80 = toFlat(raw(applyRope(k, [80], opts)))
    return offsets.map((_, r) => dotOf(qs.slice(r * ROPE_D, (r + 1) * ROPE_D), k80))
  }, [q, k, offsets, opts])
  const freqs = useMemo(() => {
    const plain = ropeFrequencies(ROPE_D, { base }).frequencies
    const scaled = ropeFrequencies(ROPE_D, opts).frequencies
    return { plain, scaled }
  }, [base, opts])
  const theta = freqs.scaled[i]
  const vectors = useMemo(
    () => [
      { from: [0, 0] as [number, number], to: rotated.q, slot: 0, label: `q at ${p}` },
      { from: [0, 0] as [number, number], to: rotated.k, slot: 1, label: `k at ${s}` },
    ],
    [rotated, p, s],
  )
  const ax = useAxis({ label: `coordinate ${i}`, range: [-3, 3] })
  const ay = useAxis({ label: `coordinate ${i + ROPE_D / 2}`, range: [-3, 3], equal: ax })
  const ox = useAxis({ label: 'offset p − s', range: [-80, 80] })
  const oy = useAxis({ label: 'score q·k', hold: 'union', key: `${base}${kind}${factor}` })
  const fx = useAxis({ label: 'pair i' })
  const fy = useAxis({ label: 'θ_i (radians per position)', log: true })
  return (
    <Figure
      title="RoPE rotates queries and keys"
      purpose="Rotary embedding turns each coordinate pair of a query or key by its position times a frequency θ_i, so the score q·k depends only on the offset p − s; rescaling the frequencies stretches the positions a model was trained on."
      defaultSize="L"
      state={state}
      readouts={{
        score: (
          <>
            <Readout label="q(p) · k(s)" value={fmt(rotated.score)} />
            <Readout label="q(p + 10) · k(s + 10)" value={fmt(shifted)} />
            <Readout label={`θ_${i}`} value={fmt(theta)} />
            <Readout label="angle of the pair, (p − s)·θ_i" value={`${fmt(((p - s) * theta * 180) / Math.PI)}°`} />
          </>
        ),
      }}
      caption="aifn-compute/nn/attention applyRope and ropeFrequencies on fixed random q and k of width 32 (half layout: pair i is coordinates i and i + 16). Left: pair i of q at position p and k at position s; both turn as the positions move, but their angle depends on p − s only, so shifting both by 10 leaves the score unchanged. Middle: the score against the offset. Right: the frequencies, plain (grey) and rescaled; YaRN interpolates only the slow pairs."
    >
      <Plots cols={3} widths={[1, 1.3, 1]}>
        <Plot x={ax} y={ay}>
          <Vectors vectors={vectors} live />
        </Plot>
        <Plot x={ox} y={oy}>
          <Curve name="q·k by offset" x={offsets} y={scoreCurve} slot={0} />
          <Annotation x={p - s} text="p − s" dashed />
        </Plot>
        <Plot x={fx} y={fy}>
          <Curve name="plain" x={range(ROPE_D / 2)} y={freqs.plain} muted showPoints />
          <Curve
            name={kind === 'none' ? 'frequencies' : 'rescaled'}
            x={range(ROPE_D / 2)}
            y={freqs.scaled}
            slot={0}
            showPoints
          />
          <Annotation x={i} text={`pair ${i}`} dashed />
        </Plot>
      </Plots>
    </Figure>
  )
}

// ---------------------------------------------------------------------------------------------------------------------
// 3. Additive biases: ALiBi's slopes and T5's buckets.

const BIAS_T = 24

export function BiasSpecimen() {
  const state = useFigureState({
    scheme: row('1 · scheme', {
      kind: choice(
        [
          { value: 'alibi', label: 'ALiBi' },
          { value: 't5', label: 'T5 buckets' },
        ],
        'alibi',
        { label: 'bias' },
      ),
      heads: slider(1, 12, 8, { label: 'heads', step: 1 }),
      head: slider(1, 12, 1, { label: 'head shown', step: 1 }),
    }),
    t5: row('2 · T5 buckets', {
      buckets: slider(4, 32, 16, { label: 'buckets', step: 2 }),
      maxDistance: slider(4, 64, 16, { label: 'max distance', step: 1 }),
      bidirectional: setting(false, 'bidirectional'),
    }),
  })
  const { kind, heads, head } = state.scheme
  const { buckets, maxDistance, bidirectional } = state.t5
  const h = Math.min(head, heads) - 1
  const pos = useMemo(() => positionRange(BIAS_T), [])
  const grid = useMemo(() => {
    let b: number[]
    if (kind === 'alibi') b = toFlat(alibiBias(heads, pos, pos)).slice(h * BIAS_T * BIAS_T, (h + 1) * BIAS_T * BIAS_T)
    else b = pos.flatMap((p) => pos.map((q) => t5RelativeBucket(q - p, { buckets, maxDistance, bidirectional })))
    const m = toFlat(positionMask(pos, pos, { causal: !bidirectional || kind === 'alibi' }))
    return range(BIAS_T)
      .map((i) => range(BIAS_T).map((j) => (m[i * BIAS_T + j] ? b[i * BIAS_T + j] : NaN)))
      .reverse()
  }, [kind, heads, h, pos, buckets, maxDistance, bidirectional])
  const rel = useMemo(() => range(81).map((r) => r - 40), [])
  const curves = useMemo(() => {
    if (kind === 'alibi') return alibiSlopes(heads).map((m) => rel.map((r) => (r <= 0 ? m * r : NaN)))
    return [rel.map((r) => t5RelativeBucket(r, { buckets, maxDistance, bidirectional }))]
  }, [kind, heads, rel, buckets, maxDistance, bidirectional])
  const kx = useAxis({ label: 'key position' })
  const qy = useAxis({ label: 'query position (top: 0)', equal: kx })
  const rx = useAxis({ label: 'relative position (key − query)', range: [-40, 40] })
  const ry = useAxis({ label: kind === 'alibi' ? 'bias −m_h·distance' : 'bucket', hold: 'union', key: kind })
  return (
    <Figure
      title="ALiBi and T5 position biases"
      purpose="Relative position can enter as a bias added to the scores: ALiBi penalises distance linearly at a fixed geometric slope per head; T5 learns one scalar per bucket of distance, with exact buckets nearby and logarithmic ones far away."
      state={state}
      readouts={{
        [kind === 'alibi' ? 'slopes' : 'buckets']:
          kind === 'alibi' ? (
            <Readout label={`m_${h + 1}`} value={fmt(alibiSlopes(heads)[h])} />
          ) : (
            <Readout
              label="bucket at distance 100"
              value={String(t5RelativeBucket(-100, { buckets, maxDistance, bidirectional }))}
            />
          ),
      }}
      caption="aifn-compute/nn/attention alibiBias and t5RelativeBucket. Left: the bias (ALiBi, for the head shown) or the bucket index (T5) for every query and key in a causal model, or both directions with T5 bidirectional. Right: ALiBi's lines, one per head, steepest for head 1; or T5's bucket against the relative position, flat beyond the max distance."
    >
      <Plots cols={2}>
        <Plot x={kx} y={qy}>
          <Raster x={pos} y={pos} z={grid} valueLabel={kind === 'alibi' ? 'bias' : 'bucket'} />
        </Plot>
        <Plot x={rx} y={ry}>
          {curves.map((c, k) => (
            <Curve
              key={k}
              name={kind === 'alibi' ? `head ${k + 1}` : 'bucket'}
              x={rel}
              y={c}
              slot={kind === 'alibi' ? k % 8 : 0}
              thin={kind === 'alibi' && k !== h}
            />
          ))}
        </Plot>
      </Plots>
    </Figure>
  )
}

// ---------------------------------------------------------------------------------------------------------------------
// 4. Key–value cache memory.

export function KvCacheSpecimen() {
  const state = useFigureState({
    model: row('1 · model', {
      layers: slider(1, 96, 32, { label: 'layers', step: 1 }),
      heads: slider(1, 128, 32, { label: 'query heads h', step: 1 }),
      headDim: slider(32, 256, 128, { label: 'head width d_h', step: 32 }),
    }),
    sharing: row('2 · sharing', {
      kvHeads: slider(1, 128, 8, { label: 'GQA key–value heads g', step: 1 }),
      latent: slider(64, 1024, 512, { label: 'MLA latent d_c', step: 64 }),
      rope: slider(0, 128, 64, { label: 'MLA rotary d_R', step: 16 }),
    }),
    use: row('3 · use', {
      context: slider(512, 131072, 8192, { label: 'context tokens', step: 512 }),
      batch: slider(1, 64, 1, { label: 'batch', step: 1 }),
      bytes: choice(
        [
          { value: 2, label: '16-bit' },
          { value: 1, label: '8-bit' },
          { value: 4, label: '32-bit' },
        ],
        2,
        { label: 'precision' },
      ),
    }),
  })
  const { layers, heads, headDim } = state.model
  const { latent, rope } = state.sharing
  const kvHeads = Math.min(state.sharing.kvHeads, heads)
  const { context, batch, bytes } = state.use
  const layouts = useMemo(() => {
    const base = { layers, heads, headDim, bytesPerValue: bytes }
    return [
      { name: 'MHA', m: kvCacheMemory(base, context, batch) },
      { name: `GQA (g = ${kvHeads})`, m: kvCacheMemory({ ...base, kvHeads }, context, batch) },
      { name: 'MQA', m: kvCacheMemory({ ...base, kvHeads: 1 }, context, batch) },
      { name: 'MLA', m: kvCacheMemory({ ...base, latentDim: latent, ropeDim: rope }, context, batch) },
    ]
  }, [layers, heads, headDim, bytes, kvHeads, latent, rope, context, batch])
  const gib = (b: number) => b / 2 ** 30
  const cx = useAxis({ label: 'attention', categories: layouts.map((l) => l.name) })
  const cy = useAxis({ label: 'cache (GiB)', hold: 'union', key: `${layers}${heads}${headDim}` })
  return (
    <Figure
      title="Key–value cache memory"
      purpose="A decoder caches a key and a value per layer, head and token, 2·layers·g·d_h numbers per token; sharing key–value heads (GQA, MQA) or caching one latent (MLA) shrinks the cache, which bounds the batch and context that fit."
      state={state}
      readouts={{
        'per token': (
          <>
            {layouts.map((l) => (
              <Readout
                key={l.name}
                label={l.name}
                value={`${fmt(l.m.bytesPerToken / 1024)} KiB (${fmt(l.m.relativeToMultiHead * 100)}%)`}
              />
            ))}
          </>
        ),
      }}
      caption="aifn-compute/nn/attention kvCacheMemory: total cache for the batch at the chosen context length, by attention layout. Percentages are relative to multi-head attention with the same h and d_h."
    >
      <Plot x={cx} y={cy}>
        <Bars name="cache" x={range(4)} y={layouts.map((l) => gib(l.m.bytes))} slot={0} />
      </Plot>
    </Figure>
  )
}

// ---------------------------------------------------------------------------------------------------------------------
// 5. FlashAttention's tiled online softmax.

const FT = 12

export function FlashAttentionSpecimen() {
  const state = useFigureState({
    tiles: row('1 · tiles', {
      queryBlock: slider(1, 6, 4, { label: 'queries per tile', step: 1 }),
      keyBlock: slider(1, 6, 3, { label: 'keys per tile', step: 1 }),
      causal: setting(true, 'causal'),
    }),
  })
  const { queryBlock, keyBlock, causal } = state.tiles
  const qkv = useMemo(() => {
    const draw = (name: string, d: number) => normal(stream(`flash-${name}`), 0, 1.2, { shape: [FT, d] }) as Tensor
    return { q: draw('q', 4), k: draw('k', 4), v: draw('v', 2) }
  }, [])
  const exact = useMemo(
    () => toFlat(raw(scaledDotProductAttention(qkv.q, qkv.k, qkv.v, { causal }).output)),
    [qkv, causal],
  )
  const tr = useMemo(
    () =>
      trace(flashAttentionSteps(qkv.q, qkv.k, qkv.v, { queryBlock, keyBlock, causal }), undefined, 1000, {
        keep: 'all',
      }),
    [qkv, queryBlock, keyBlock, causal],
  )
  const [position, setPosition] = useState(0)
  const pos = Math.min(position, tr.steps.length - 1)
  const at = tr.steps[pos]
  const order = useMemo(() => {
    const z = range(FT).map(() => range(FT).map(() => NaN))
    at.tiles.forEach((tile, n) => {
      if (n >= at.t) return
      for (let i = tile.queries[0]; i < tile.queries[1]; i++)
        for (let j = tile.keys[0]; j < tile.keys[1]; j++) if (!causal || j <= i) z[i][j] = n + 1
    })
    return z.reverse()
  }, [at, causal])
  const box = useMemo(() => {
    if (!at.tile) return { x: [], y: [] }
    const [q0, q1] = at.tile.queries
    const [k0, k1] = at.tile.keys
    const top = FT - 1 - q0 + 0.5
    const bottom = FT - 1 - (q1 - 1) - 0.5
    return { x: [k0 - 0.5, k1 - 0.5, k1 - 0.5, k0 - 0.5, k0 - 0.5], y: [bottom, bottom, top, top, bottom] }
  }, [at])
  const errors = useMemo(() => {
    const a = toFlat(at.accumulator)
    const l = toFlat(at.normaliser)
    return range(FT).map((i) =>
      l[i] > 0 ? Math.max(...[0, 1].map((c) => Math.abs(a[i * 2 + c] / l[i] - exact[i * 2 + c]))) : NaN,
    )
  }, [at, exact])
  const worst = Math.max(...errors.map((e) => (Number.isFinite(e) ? e : 0)))
  const kx = useAxis({ label: 'key' })
  const qy = useAxis({ label: 'query (top: 0)', equal: kx })
  const ex = useAxis({ label: 'query' })
  const ey = useAxis({ label: '|a/ℓ − exact output|', log: true, range: [1e-17, 10] })
  return (
    <Figure
      title="Tiled attention with the online softmax"
      purpose="FlashAttention never stores the whole score matrix: it folds one tile of scores at a time into a running maximum, normaliser and output per query, rescaling what it has when the maximum grows, and ends on exactly softmax(QKᵀ)V."
      defaultSize="L"
      state={state}
      controls={
        <Player
          value={pos}
          onChange={setPosition}
          count={tr.steps.length}
          label="2 · tile"
          format={(k) => `${k} of ${tr.steps.length - 1}`}
        />
      }
      readouts={{
        tiles: (
          <>
            <Readout label="tiles done" value={`${at.t} of ${at.tiles.length}`} />
            <Readout label="skipped by the causal mask" value={String(at.skipped)} />
            <Readout label="largest error of a/ℓ so far" value={worst.toExponential(1)} />
          </>
        ),
      }}
      caption="aifn-compute/nn/attention flashAttentionSteps on random q, k (12 × 4) and v (12 × 2). Left: the score matrix coloured by the step at which each cell's tile was folded in (empty: not yet, or hidden); the outline is the current tile. Right: for each query, how far the running output a/ℓ is from the exact attention output; every row reaches rounding error once its last tile is in."
    >
      <Plots cols={2}>
        <Plot x={kx} y={qy}>
          <Raster x={range(FT)} y={range(FT)} z={order} valueLabel="tile step" />
          <Curve name="current tile" x={box.x} y={box.y} emphasis live />
        </Plot>
        <Plot x={ex} y={ey}>
          <Bars
            name="error"
            x={range(FT)}
            y={errors.map((e) => (Number.isFinite(e) ? Math.max(e, 1e-17) : NaN))}
            slot={0}
          />
        </Plot>
      </Plots>
    </Figure>
  )
}
