import { useContext, type ReactNode } from 'react'
import type { Encoding } from 'aifn/text/pipeline'
import { categorical } from '@lab/design/palette'
import { useTheme } from '@lab/design/theme'
import { DEFAULT_HEIGHT, FrameContext } from '@lab/viz'
import { visible } from './visible'

/** A scrolling body that fills the figure's frame height. */
export function Body({ children }: { children: ReactNode }) {
  const { height } = useContext(FrameContext)
  return (
    <div className="flex flex-col gap-3 overflow-auto pr-1" style={{ height: height ?? DEFAULT_HEIGHT }}>
      {children}
    </div>
  )
}

/** A [start, end) range of the source text. */
export type Span = readonly [number, number]

const overlaps = (a: Span, b: Span) => (a[0] === a[1] ? a[0] >= b[0] && a[0] < b[1] : a[0] < b[1] && b[0] < a[1])

/**
 * An encoding as chips: tokens tinted in two alternating colours; special tokens dashed, unknown tokens in the warning
 * slot (red), byte-fallback `<0xNN>` tokens in green. Hovering a chip reports its source span; chips overlapping the
 * hovered span are outlined, so one span can be followed across tokenisers.
 */
export function TokenChips({
  e,
  unknownId,
  hot,
  onHover,
  pinned = null,
  onPin,
}: {
  e: Encoding
  unknownId: number
  hot: Span | null
  onHover: (span: Span | null) => void
  /** The pinned span: its chips keep an outline. */
  pinned?: Span | null
  /** Click a chip to pin its span. */
  onPin?: (span: Span) => void
}) {
  const { resolved: mode } = useTheme()
  const c = categorical(mode)
  const ids = e.ids.data
  const o = e.offsets.data
  const special = e.specialTokensMask.data
  return (
    <div className="flex flex-wrap gap-0.5 font-mono text-xs leading-5">
      {e.tokens.map((tok, k) => {
        const span: Span = [o[2 * k], o[2 * k + 1]]
        const isSpecial = special[k] === 1
        const isUnknown = ids[k] === unknownId
        const isByte = /^<0x[0-9A-F]{2}>$/u.test(tok)
        const colour = isUnknown ? c[7] : isByte ? c[2] : c[k % 2 ? 6 : 0]
        const lit = hot !== null && !isSpecial && overlaps(span, hot)
        const dim = hot !== null && !lit
        const isPinned = pinned !== null && !isSpecial && overlaps(span, pinned)
        return (
          <span
            key={k}
            title={`${JSON.stringify(tok)}  id ${ids[k]}  [${span[0]}, ${span[1]})`}
            onMouseEnter={() => onHover(isSpecial ? null : span)}
            onMouseLeave={() => onHover(null)}
            onClick={(ev) => {
              ev.stopPropagation()
              if (!isSpecial) onPin?.(span)
            }}
            className={
              'cursor-default rounded-sm px-1 ' + (isSpecial ? 'border border-dashed text-muted-foreground' : '')
            }
            style={{
              background: isSpecial
                ? undefined
                : `color-mix(in srgb, ${colour} ${lit ? 60 : isUnknown ? 45 : 22}%, transparent)`,
              fontWeight: isUnknown ? 600 : undefined,
              opacity: dim ? 0.3 : undefined,
              boxShadow: isPinned ? 'inset 0 0 0 1.5px currentColor' : undefined,
              outline: lit ? `1.5px solid ${colour}` : undefined,
            }}
          >
            {visible(tok)}
          </span>
        )
      })}
    </div>
  )
}

/**
 * The source text with one span highlighted (the offsets of the hovered token). Each run of non-space characters is
 * itself hoverable: hovering a word reports its span, so every tokeniser's chips for that word light up.
 */
export function SourceSpan({
  text,
  span,
  onHover,
  pinned = null,
  onPin,
}: {
  text: string
  span: Span | null
  onHover?: (span: Span | null) => void
  /** The pinned span, outlined (dashed). */
  pinned?: Span | null
  /** Click a word to pin its span. */
  onPin?: (span: Span) => void
}) {
  const { resolved: mode } = useTheme()
  const c = categorical(mode)
  // Words and the gaps between them, each [start, end, isWord].
  const parts: [number, number, boolean][] = []
  let at = 0
  for (const m of text.matchAll(/\S+/gu)) {
    if (m.index > at) parts.push([at, m.index, false])
    parts.push([m.index, m.index + m[0].length, true])
    at = m.index + m[0].length
  }
  if (at < text.length) parts.push([at, text.length, false])
  const mark = { background: `color-mix(in srgb, ${c[0]} 45%, transparent)`, outline: `1.5px solid ${c[0]}` }
  // Cut each part at the span's ends so the highlight covers exactly the span.
  const cuts = [...(span ? span : []), ...(pinned ? pinned : [])]
  const pieces: { s: number; e: number; word: [number, number] | null }[] = []
  for (const [s, e, isWord] of parts) {
    const points = [s, ...cuts.filter((x) => x > s && x < e), e]
    for (let k = 0; k + 1 < points.length; k++)
      pieces.push({ s: points[k], e: points[k + 1], word: isWord ? [s, e] : null })
  }
  return (
    <div className="font-mono text-sm leading-7 whitespace-pre-wrap" onMouseLeave={() => onHover?.(null)}>
      {pieces.map((p, k) => {
        const inside = span !== null && p.s >= span[0] && p.e <= span[1] && span[1] > span[0]
        const inPin = pinned !== null && p.s >= pinned[0] && p.e <= pinned[1] && pinned[1] > pinned[0]
        const word = p.word
        return (
          <span
            key={k}
            className={word && onHover ? 'cursor-pointer' : undefined}
            onMouseEnter={word && onHover ? () => onHover(word) : undefined}
            onClick={
              word && onPin
                ? (ev) => {
                    ev.stopPropagation()
                    onPin(word)
                  }
                : undefined
            }
            style={{
              ...(inside ? mark : {}),
              ...(inPin ? { textDecoration: 'underline dashed', textUnderlineOffset: 4 } : {}),
            }}
          >
            {text.slice(p.s, p.e)}
          </span>
        )
      })}
      {span && span[0] === span[1] && <span style={mark}>▏</span>}
    </div>
  )
}
