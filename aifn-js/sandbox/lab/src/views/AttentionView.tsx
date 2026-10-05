import { useMemo, useState } from 'react'
import { toFlat } from 'aifn/foundation/tensor'
import { PanelSlot } from '@lab/layout'
import { choice } from '@lab/state'
import { Bars, Curve, Plot, Plots, Raster, Readout, Segments, useAxis } from '@lab/viz'
import { cn } from '@lab/lib/utils'
import type { AttentionPattern } from './attention-pattern'
import { formatValue } from './format'
import { registerView } from './registry'

export type AttentionPanelProps = {
  pattern: AttentionPattern
  /** The head to draw, or `all` for one heatmap per head (default `all` up to four heads, else 0). */
  head?: number | 'all'
  /** The query whose row is highlighted before any hover (default the last). */
  focus?: number
}

/** Unique axis labels: a repeated token gets its position as a suffix. */
function labelsOf(tokens: readonly string[]): string[] {
  const seen = new Map<string, number>()
  tokens.forEach((t) => seen.set(t, (seen.get(t) ?? 0) + 1))
  return tokens.map((t, i) => ((seen.get(t) ?? 0) > 1 ? `${t}·${i}` : t))
}

const f3 = (v: number) => formatValue(Number(v.toPrecision(3)))

/**
 * Attention as a matrix of tokens: queries down, keys across, one heatmap per head; hidden (masked) cells are empty
 * and crossed. Hover a query token in the strip to outline its row in every head and see that row's weights as bars
 * below, with its entropy (nats) and the key it weighs most. Keys held in a cache are marked in the strip.
 */
export function AttentionPanel({ pattern, head, focus }: AttentionPanelProps) {
  const { keys, weights, mask, cached = 0 } = pattern
  const queries = pattern.queries ?? keys
  const [H, Tq, Tk] = weights.shape.length === 3 ? weights.shape : [1, ...weights.shape]
  const shown = head === undefined ? (H <= 4 ? 'all' : 0) : head
  const heads = useMemo(
    () => (shown === 'all' ? Array.from({ length: H }, (_, h) => h) : [Math.min(shown, H - 1)]),
    [shown, H],
  )
  const [hover, setHover] = useState<number | null>(null)
  const row = Math.min(hover ?? focus ?? Tq - 1, Tq - 1)

  const qLabels = useMemo(() => labelsOf(queries), [queries])
  const kLabels = useMemo(() => labelsOf(keys), [keys])
  const qUp = useMemo(() => [...qLabels].reverse(), [qLabels])
  const kx = useMemo(() => Array.from({ length: Tk }, (_, j) => j), [Tk])
  const qy = useMemo(() => Array.from({ length: Tq }, (_, i) => i), [Tq])

  // Per head: rows drawn top-down (query 0 at the top), hidden cells NaN.
  const grids = useMemo(() => {
    const w = toFlat(weights)
    const m = mask ? toFlat(mask) : null
    return Array.from({ length: H }, (_, h) =>
      Array.from({ length: Tq }, (_, i) =>
        Array.from({ length: Tk }, (_, j) => (m && m[i * Tk + j] === 0 ? NaN : w[(h * Tq + i) * Tk + j])),
      ).reverse(),
    )
  }, [weights, mask, H, Tq, Tk])
  // Each hidden cell crossed by its two diagonals.
  const crosses = useMemo(() => {
    if (!mask) return []
    const m = toFlat(mask)
    const out: { from: [number, number]; to: [number, number] }[] = []
    const r = 0.32
    for (let i = 0; i < Tq; i++)
      for (let j = 0; j < Tk; j++)
        if (m[i * Tk + j] === 0) {
          const y = Tq - 1 - i
          out.push({ from: [j - r, y - r], to: [j + r, y + r] }, { from: [j - r, y + r], to: [j + r, y - r] })
        }
    return out
  }, [mask, Tq, Tk])
  const outline = useMemo(() => {
    const y = Tq - 1 - row
    return { x: [-0.5, Tk - 0.5, Tk - 0.5, -0.5, -0.5], y: [y - 0.5, y - 0.5, y + 0.5, y + 0.5, y - 0.5] }
  }, [row, Tq, Tk])
  const rowWeights = useMemo(
    () => heads.map((h) => grids[h][Tq - 1 - row].map((v) => (Number.isFinite(v) ? v : 0))),
    [heads, grids, row, Tq],
  )
  const stats = rowWeights.map((r) => {
    const entropy = -r.reduce((a, p) => a + (p > 0 ? p * Math.log(p) : 0), 0)
    const top = r.indexOf(Math.max(...r))
    return { entropy, top }
  })

  const xAxis = useAxis({ label: 'key', categories: kLabels })
  const yAxis = useAxis({ label: 'query', categories: qUp, equal: xAxis })
  const barX = useAxis({ label: 'key', categories: kLabels })
  const barY = useAxis({ label: `weights of “${queries[row]}”`, range: [0, 1] })
  const width = 0.8 / heads.length
  const cols = Math.min(heads.length, 4)

  return (
    <>
      <PanelSlot slot="controls">
        <div className="flex flex-wrap items-center gap-1 text-sm" onMouseLeave={() => setHover(null)}>
          <span className="mr-1 text-xs text-muted-foreground">queries</span>
          {queries.map((t, i) => (
            <button
              key={i}
              type="button"
              onMouseEnter={() => setHover(i)}
              onFocus={() => setHover(i)}
              className={cn(
                'rounded border px-1.5 py-0.5 font-mono text-xs',
                i === row ? 'border-foreground bg-muted' : 'border-border',
                i < cached && 'border-dashed text-muted-foreground',
              )}
            >
              {t === ' ' ? '␣' : t}
            </button>
          ))}
          {cached > 0 && <span className="ml-2 text-xs text-muted-foreground">dashed: from the cache</span>}
        </div>
      </PanelSlot>
      <PanelSlot slot="readouts">
        <>
          <Readout label="query" value={`“${queries[row]}” (position ${row})`} />
          {heads.map((h, k) => (
            <Readout
              key={h}
              label={heads.length > 1 ? `head ${h + 1}` : 'row'}
              value={`entropy ${f3(stats[k].entropy)} nats, most on “${keys[stats[k].top]}” (${f3(rowWeights[k][stats[k].top])})`}
            />
          ))}
        </>
      </PanelSlot>
      <Plots cols={cols} scale={0.7}>
        {heads.map((h) => (
          <Plot key={h} x={xAxis} y={yAxis} title={heads.length > 1 ? `head ${h + 1}` : undefined}>
            <Raster x={kx} y={qy} z={grids[h]} range={[0, 1]} valueLabel="weight" colorBar={h === heads.at(-1)} />
            {crosses.length > 0 && <Segments segments={crosses} width={1} />}
            <Curve name="hovered query" x={outline.x} y={outline.y} emphasis live />
          </Plot>
        ))}
      </Plots>
      <Plot x={barX} y={barY} scale={0.3}>
        {heads.map((h, k) => (
          <Bars
            key={h}
            name={heads.length > 1 ? `head ${h + 1}` : 'weight'}
            x={kx.map((j) => j + (k - (heads.length - 1) / 2) * width)}
            y={rowWeights[k]}
            width={width}
            slot={k}
          />
        ))}
      </Plot>
    </>
  )
}

const HEAD_OPTIONS = choice<string | number>(
  [
    { value: 'all', label: 'all heads' },
    { value: 0, label: 'head 1' },
    { value: 1, label: 'head 2' },
  ],
  'all',
  { label: 'heads' },
)

registerView<AttentionPattern, { head: typeof HEAD_OPTIONS }>({
  key: 'attention/matrix',
  kind: 'attention',
  description:
    'Attention weights as a token-by-token heatmap per head, masked cells crossed out; hover a query token to outline its row and see its weights as bars.',
  title: () => 'Attention weights',
  options: { head: HEAD_OPTIONS },
  render: (p, { head }) => <AttentionPanel pattern={p} head={head as number | 'all'} />,
})
