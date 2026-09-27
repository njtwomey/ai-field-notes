import { useMemo, useState } from 'react'
import { Heatmap, Interactive, ParamChoice, ParamSlider, ParamSwitch, Readout, formatNumber } from '@/components/viz'
import { rng } from '@/lib/math'

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
    const u = rng(11)
    for (let i = 0; i < n; i++) {
      const hi = causal ? i + 1 : n
      for (let k = 0; k < r; k++) m[i][Math.floor(u.uniform() * hi)] = 1
    }
  }
  if (causal) for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) m[i][j] = 0
  return m
}

/** Sparse attention patterns as query-by-key masks, with the fraction of the full n × n score matrix each one keeps. */
export function AttentionPatterns({ initial = 'sliding' }: { initial?: Pattern }) {
  const [pattern, setPattern] = useState<Pattern>(initial)
  const [n, setN] = useState(64)
  const [w, setW] = useState(4)
  const [dilation, setDilation] = useState(2)
  const [stride, setStride] = useState(8)
  const [g, setG] = useState(2)
  const [r, setR] = useState(2)
  const [causal, setCausal] = useState(false)

  const z = useMemo(
    () => mask(pattern, { n, w, dilation, stride, g, r, causal }),
    [pattern, n, w, dilation, stride, g, r, causal],
  )
  const pos = useMemo(() => Array.from({ length: n }, (_, i) => i), [n])
  const pairs = z.reduce((a, row) => a + row.reduce((s, v) => s + v, 0), 0)
  const usesWindow = pattern === 'sliding' || pattern === 'dilated' || pattern === 'longformer' || pattern === 'bigbird'

  return (
    <Interactive
      title="Attention patterns"
      caption="Each row is one query position and each column one key position; a filled cell is a score that is computed. The fraction of cells filled is the fraction of the n × n score matrix, and of the score and weighted-sum arithmetic, that the pattern keeps. Sliding windows keep a band of half-width w. Dilated windows skip keys. Strided attention (Sparse Transformer) adds every stride-th key. Global tokens fill whole rows and columns. BigBird adds r random keys per query. The causal switch removes every key after its query."
      controls={
        <>
          <ParamChoice label="pattern" value={pattern} onChange={setPattern} options={PATTERNS} />
          <ParamSwitch label="causal" checked={causal} onChange={setCausal} />
          <ParamSlider label="sequence length n" value={n} onChange={setN} min={16} max={128} step={8} />
          {usesWindow && <ParamSlider label="half-width w" value={w} onChange={setW} min={1} max={16} step={1} />}
          {pattern === 'dilated' && (
            <ParamSlider label="dilation" value={dilation} onChange={setDilation} min={2} max={4} step={1} />
          )}
          {pattern === 'strided' && (
            <ParamSlider label="stride" value={stride} onChange={setStride} min={2} max={16} step={1} />
          )}
          {(pattern === 'longformer' || pattern === 'bigbird') && (
            <ParamSlider label="global tokens g" value={g} onChange={setG} min={0} max={8} step={1} />
          )}
          {pattern === 'bigbird' && (
            <ParamSlider label="random keys r" value={r} onChange={setR} min={0} max={6} step={1} />
          )}
        </>
      }
      readout={
        <>
          <Readout label="scores computed" value={`${pairs} of ${n * n}`} />
          <Readout label="FLOPs and memory vs full" value={`${formatNumber((100 * pairs) / (n * n))}%`} />
          <Readout label="keys per query" value={formatNumber(pairs / n)} />
          <Readout label="grows as" value={GROWTH[pattern]} />
        </>
      }
    >
      <Heatmap
        x={pos}
        y={pos}
        z={z}
        range={[0, 1]}
        xLabel="key position j"
        yLabel="query position i"
        valueLabel="computed"
        height={400}
      />
    </Interactive>
  )
}
