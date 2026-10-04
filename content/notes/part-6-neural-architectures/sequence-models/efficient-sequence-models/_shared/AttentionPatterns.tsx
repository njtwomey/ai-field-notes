import { useMemo } from 'react'
import { choice, Figure, formatNumber, int, Plot, Raster, Readout, setting, useAxis, useFigureState } from 'aifn-render'
import { stream, uniform } from 'aifn/foundation/random'

export type Pattern = 'full' | 'sliding' | 'dilated' | 'strided' | 'longformer' | 'bigbird'

const PATTERNS: { value: Pattern; label: string }[] = [
  { value: 'full', label: 'full' },
  { value: 'sliding', label: 'sliding' },
  { value: 'dilated', label: 'dilated' },
  { value: 'strided', label: 'strided' },
  { value: 'longformer', label: 'local + global' },
  { value: 'bigbird', label: 'BigBird' },
]

const GROWTH: Record<Pattern, string> = {
  full: 'n²',
  sliding: 'n·w',
  dilated: 'n·w',
  strided: 'n·√n',
  longformer: 'n·(w + g)',
  bigbird: 'n·(w + g + r)',
}

type Options = { n: number; w: number; dilation: number; stride: number; g: number; r: number; causal: boolean }

/** Binary mask: row i is query position i, column j is key position j, 1 where query i attends to key j. */
function mask(pattern: Pattern, o: Options): number[][] {
  const { n, w, dilation, stride, g, r, causal } = o
  const m = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => {
      const d = Math.abs(i - j)
      const local = d <= w
      const global = i < g || j < g
      switch (pattern) {
        case 'full':
          return 1
        case 'sliding':
          return local ? 1 : 0
        case 'dilated':
          return d <= w * dilation && d % dilation === 0 ? 1 : 0
        case 'strided':
          return d < stride || d % stride === 0 ? 1 : 0
        case 'longformer':
        case 'bigbird':
          return local || global ? 1 : 0
      }
    }),
  )
  if (pattern === 'bigbird') {
    // r random keys per query, drawn from the keys the query may see. Fixed seed, so the pattern is stable.
    const u = stream(11)
    for (let i = 0; i < n; i++) {
      const hi = causal ? i + 1 : n
      for (let k = 0; k < r; k++) m[i][Math.floor(uniform(u) * hi)] = 1
    }
  }
  if (causal) for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) m[i][j] = 0
  return m
}

/** Patterns with a local window of half-width w. */
const WINDOWED: Pattern[] = ['sliding', 'dilated', 'longformer', 'bigbird']

/** Sparse attention patterns as query-by-key masks, with the fraction of the full n × n score matrix each one keeps. */
export function AttentionPatterns({ initial = 'sliding' }: { initial?: Pattern }) {
  const state = useFigureState({
    pattern: choice<Pattern>(PATTERNS, initial, { label: 'pattern' }),
    causal: setting(false, 'causal'),
    n: int(64, { min: 16, max: 128, step: 8, suggestions: [16, 32, 64, 128], label: 'sequence length n' }),
    w: int(4, { min: 1, max: 16, label: 'half-width w', when: (v) => WINDOWED.includes(v.pattern as Pattern) }),
    dilation: int(2, { min: 2, max: 4, label: 'dilation', when: (v) => v.pattern === 'dilated' }),
    stride: int(8, { min: 2, max: 16, label: 'stride', when: (v) => v.pattern === 'strided' }),
    g: int(2, {
      min: 0,
      max: 8,
      label: 'global tokens g',
      when: (v) => v.pattern === 'longformer' || v.pattern === 'bigbird',
    }),
    r: int(2, { min: 0, max: 6, label: 'random keys r', when: (v) => v.pattern === 'bigbird' }),
  })
  const { pattern, n, w, dilation, stride, g, r, causal } = state

  const z = useMemo(
    () => mask(pattern, { n, w, dilation, stride, g, r, causal }),
    [pattern, n, w, dilation, stride, g, r, causal],
  )
  const pos = useMemo(() => Array.from({ length: n }, (_, i) => i), [n])
  const pairs = z.reduce((a, row) => a + row.reduce((s, v) => s + v, 0), 0)

  const xAxis = useAxis({ label: 'key position j', hold: 'union' })
  const yAxis = useAxis({ label: 'query position i', hold: 'union' })
  return (
    <Figure
      title="Attention patterns"
      caption="Each row is one query position and each column one key position; a filled cell is a score that is computed. The fraction of cells filled is the fraction of the n × n score matrix, and of the score and weighted-sum arithmetic, that the pattern keeps. Sliding windows keep a band of half-width w. Dilated windows skip keys. Strided attention (Sparse Transformer) adds every stride-th key. Global tokens fill whole rows and columns. BigBird adds r random keys per query. The causal switch removes every key after its query."
      state={state}
      readouts={
        <>
          <Readout label="scores computed" value={`${pairs} of ${n * n}`} />
          <Readout label="FLOPs and memory vs full" value={`${formatNumber((100 * pairs) / (n * n))}%`} />
          <Readout label="keys per query" value={formatNumber(pairs / n)} />
          <Readout label="grows as" value={GROWTH[pattern]} />
        </>
      }
    >
      <Plot x={xAxis} y={yAxis} height={400}>
        <Raster x={pos} y={pos} z={z} range={[0, 1]} valueLabel={'computed'} />
      </Plot>
    </Figure>
  )
}
