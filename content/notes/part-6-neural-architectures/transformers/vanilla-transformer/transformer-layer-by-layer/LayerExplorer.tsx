import { useMemo, useState } from 'react'
import {
  Heatmap,
  Interactive,
  ParamButton,
  ParamChoice,
  ParamSlider,
  ParamSwitch,
  Readout,
  formatNumber,
  useParam,
} from 'aifn-render'

type Matrix = number[][]

const D = 4
const TOKENS = ['the', 'cat', 'sat', 'down'] as const
type Token = (typeof TOKENS)[number]
const TOKEN_OPTIONS = TOKENS.map((t) => ({ value: t, label: t }))

/** The note's embeddings: one row per token, before the √d scaling. */
const E0: Matrix = [
  [0.5, 0, 0.5, 0],
  [0, 0.5, 0, 0.5],
  [0.5, 0.5, 0, 0],
  [0, 0, 0.5, 0.5],
]

/** W_K swaps the two features of each pair; W_Q, W_V and W_O are identities. */
const WK: Matrix = [
  [0, 1, 0, 0],
  [1, 0, 0, 0],
  [0, 0, 0, 1],
  [0, 0, 1, 0],
]

/** W1 = ½ [I | M] with M mixing the pairs; W2 = W1ᵀ; biases are 0. */
const W1: Matrix = [
  [1, 0, 0, 0, 1, 0, -1, 0],
  [0, 1, 0, 0, 0, 1, 0, -1],
  [0, 0, 1, 0, -1, 0, 1, 0],
  [0, 0, 0, 1, 0, -1, 0, 1],
].map((row) => row.map((v) => v / 2))
const W2: Matrix = W1[0].map((_, j) => W1.map((row) => row[j]))

const matmul = (a: Matrix, b: Matrix): Matrix =>
  a.map((row) => b[0].map((_, j) => row.reduce((s, v, k) => s + v * b[k][j], 0)))
const add = (a: Matrix, b: Matrix): Matrix => a.map((row, i) => row.map((v, j) => v + b[i][j]))
const transpose = (a: Matrix): Matrix => a[0].map((_, j) => a.map((row) => row[j]))

/** Row-wise layer normalisation with γ = 1, β = 0, population variance and ε = 1e-6. */
const layerNorm = (a: Matrix): Matrix =>
  a.map((row) => {
    const mean = row.reduce((s, v) => s + v, 0) / row.length
    const variance = row.reduce((s, v) => s + (v - mean) ** 2, 0) / row.length
    return row.map((v) => (v - mean) / Math.sqrt(variance + 1e-6))
  })

const softmaxRows = (s: Matrix, causal: boolean): Matrix =>
  s.map((row, i) => {
    const kept = row.map((v, j) => (causal && j > i ? -Infinity : v))
    const m = Math.max(...kept)
    const e = kept.map((v) => Math.exp(v - m))
    const z = e.reduce((a, b) => a + b, 0)
    return e.map((v) => v / z)
  })

/** Sinusoidal encodings for positions 0..3 at width 4. */
const PE: Matrix = Array.from({ length: TOKENS.length }, (_, t) =>
  Array.from({ length: D }, (_, j) => {
    const angle = t * 10000 ** (-(2 * Math.floor(j / 2)) / D)
    return j % 2 === 0 ? Math.sin(angle) : Math.cos(angle)
  }),
)
const ZERO: Matrix = PE.map((row) => row.map(() => 0))

type Stage = { key: string; label: string; m: Matrix; kind: 'signed' | 'weights'; xLabel: string }

/** One encoder block, post-norm, one head, no dropout: every intermediate matrix in order. */
function forward(emb: Matrix, qkScale: number, causal: boolean, positions: boolean): Stage[] {
  const P = positions ? PE : ZERO
  const X = add(
    emb.map((row) => row.map((v) => Math.sqrt(D) * v)),
    P,
  )
  const Q = X.map((row) => row.map((v) => qkScale * v))
  const K = matmul(X, WK)
  const V = X
  const S = matmul(Q, transpose(K)).map((row) => row.map((v) => v / Math.sqrt(D)))
  const A = softmaxRows(S, causal)
  const O = matmul(A, V)
  const R1 = add(X, O)
  const Z = layerNorm(R1)
  const H = matmul(Z, W1).map((row) => row.map((v) => Math.max(0, v)))
  const F = matmul(H, W2)
  const R2 = add(Z, F)
  const Y = layerNorm(R2)
  const feature = 'feature'
  return [
    { key: 'P', label: 'P: positional encodings', m: P, kind: 'signed', xLabel: feature },
    { key: 'X', label: 'X = √d·E + P: block input', m: X, kind: 'signed', xLabel: feature },
    { key: 'Q', label: 'Q = X W_Q: queries', m: Q, kind: 'signed', xLabel: feature },
    { key: 'K', label: 'K = X W_K: keys', m: K, kind: 'signed', xLabel: feature },
    { key: 'V', label: 'V = X W_V: values', m: V, kind: 'signed', xLabel: feature },
    { key: 'S', label: 'S = QKᵀ/√d: scores (before any mask)', m: S, kind: 'signed', xLabel: 'key position' },
    { key: 'A', label: 'A = softmax(S): attention weights', m: A, kind: 'weights', xLabel: 'key position' },
    { key: 'O', label: 'O = A V W_O: attention output', m: O, kind: 'signed', xLabel: feature },
    { key: 'R1', label: 'X + O: first residual sum', m: R1, kind: 'signed', xLabel: feature },
    { key: 'Z', label: 'Z = LN(X + O)', m: Z, kind: 'signed', xLabel: feature },
    { key: 'H', label: 'ReLU(Z W₁): hidden layer, width 8', m: H, kind: 'signed', xLabel: 'hidden unit' },
    { key: 'F', label: 'FFN(Z) = ReLU(Z W₁) W₂', m: F, kind: 'signed', xLabel: feature },
    { key: 'R2', label: 'Z + FFN(Z): second residual sum', m: R2, kind: 'signed', xLabel: feature },
    { key: 'Y', label: 'Y = LN(Z + FFN(Z)): block output', m: Y, kind: 'signed', xLabel: feature },
  ]
}

const COLS4 = [0, 1, 2, 3]
const COLS8 = [0, 1, 2, 3, 4, 5, 6, 7]
const ROWS = [0, 1, 2, 3]
const STAGES = 14

/** A worked encoder block on four tokens: pick a stage to see its matrix; edit a token's embedding to follow it through. */
export function LayerExplorer() {
  const stage = useParam(6, { min: 0, max: STAGES - 1, step: 1 })
  const [emb, setEmb] = useState<Matrix>(E0)
  const [token, setToken] = useState<Token>('cat')
  const qkScale = useParam(1, { min: 0, max: 4, step: 0.1 })
  const [causal, setCausal] = useState(false)
  const [positions, setPositions] = useState(true)
  const row = TOKENS.indexOf(token)

  const stages = useMemo(() => forward(emb, qkScale.value, causal, positions), [emb, qkScale.value, causal, positions])
  const current = stages[stage.value]
  const weights = current.kind === 'weights'
  const bound = useMemo(() => {
    const max = Math.max(...current.m.flat().map(Math.abs), 1e-9)
    return Math.ceil(max * 2) / 2
  }, [current])
  const lo = weights ? 0 : -bound
  const hi = weights ? 1 : bound
  const range = useMemo<[number, number]>(() => [lo, hi], [lo, hi])
  const setComponent = (j: number) => (v: number) =>
    setEmb((e) => e.map((r, i) => (i === row ? r.map((x, k) => (k === j ? v : x)) : r)))

  return (
    <Interactive
      title="One encoder block, matrix by matrix"
      caption="Four tokens at positions 0 to 3 pass through one post-norm encoder block with one head, d = 4 and the weights given in the text. Step through the stages to see each intermediate matrix; row t is the token at position t. Edit a token's embedding and watch the change spread: it alters that token's row until attention, then every row. The query–key scale multiplies W_Q and so sharpens or flattens the attention weights."
      controls={
        <>
          <div className="sm:col-span-2 lg:col-span-3">
            <ParamSlider label="stage" param={stage} format={(v) => stages[v].label} withArrows />
          </div>
          <ParamChoice label="token to edit and read" value={token} onChange={setToken} options={TOKEN_OPTIONS} />
          {[0, 1, 2, 3].map((j) => (
            <ParamSlider
              key={j}
              label={`${token} embedding, feature ${j}`}
              value={emb[row][j]}
              onChange={setComponent(j)}
              min={-1}
              max={1}
              step={0.05}
            />
          ))}
          <ParamSlider label="query–key scale" param={qkScale} />
          <ParamSwitch label="causal mask" checked={causal} onChange={setCausal} />
          <ParamSwitch label="positional encoding" checked={positions} onChange={setPositions} />
          <ParamButton onClick={() => setEmb(E0)}>Reset embeddings</ParamButton>
        </>
      }
      readout={
        <>
          <Readout
            label={`row for "${token}"`}
            value={`(${current.m[row].map((v) => formatNumber(Math.round(v * 1000) / 1000)).join(', ')})`}
          />
          {weights && <Readout label="row sum" value={formatNumber(current.m[row].reduce((s, v) => s + v, 0))} />}
        </>
      }
    >
      <Heatmap
        x={current.m[0].length === 8 ? COLS8 : COLS4}
        y={ROWS}
        z={current.m}
        scale={weights ? 'sequential' : 'diverging'}
        range={range}
        xLabel={current.xLabel}
        yLabel="position t"
        valueLabel={current.key}
        height={300}
      />
    </Interactive>
  )
}
