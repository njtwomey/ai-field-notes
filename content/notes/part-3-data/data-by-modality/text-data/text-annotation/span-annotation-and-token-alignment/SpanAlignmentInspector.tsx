import { useMemo, useState } from 'react'

interface EntitySpan {
  start: number
  end: number
  label: string
}

interface Preset {
  id: string
  label: string
  text: string
  spans: EntitySpan[]
  words: { text: string; start: number; end: number; subwords: string[] }[]
}

const PRESETS: Preset[] = [
  {
    id: 'ner',
    label: 'Diplomatic report (named entities)',
    text: 'Dr. Eleanor Vance met with the United Nations delegation in Geneva.',
    spans: [
      { start: 4, end: 17, label: 'PER' },
      { start: 31, end: 45, label: 'ORG' },
      { start: 60, end: 66, label: 'LOC' },
    ],
    words: [
      { text: 'Dr.', start: 0, end: 3, subwords: ['Dr', '.'] },
      { text: 'Eleanor', start: 4, end: 11, subwords: ['Ele', '##anor'] },
      { text: 'Vance', start: 12, end: 17, subwords: ['Van', '##ce'] },
      { text: 'met', start: 18, end: 21, subwords: ['met'] },
      { text: 'with', start: 22, end: 26, subwords: ['with'] },
      { text: 'the', start: 27, end: 30, subwords: ['the'] },
      { text: 'United', start: 31, end: 37, subwords: ['United'] },
      { text: 'Nations', start: 38, end: 45, subwords: ['Nations'] },
      { text: 'delegation', start: 46, end: 56, subwords: ['delega', '##tion'] },
      { text: 'in', start: 57, end: 59, subwords: ['in'] },
      { text: 'Geneva', start: 60, end: 66, subwords: ['Gene', '##va'] },
      { text: '.', start: 66, end: 67, subwords: ['.'] },
    ],
  },
  {
    id: 'tech',
    label: 'Tech news (single vs multi-token)',
    text: 'Alan Turing founded foundations before Apple Inc. expanded to Cupertino.',
    spans: [
      { start: 0, end: 11, label: 'PER' },
      { start: 39, end: 49, label: 'ORG' },
      { start: 62, end: 71, label: 'LOC' },
    ],
    words: [
      { text: 'Alan', start: 0, end: 4, subwords: ['Alan'] },
      { text: 'Turing', start: 5, end: 11, subwords: ['Tur', '##ing'] },
      { text: 'founded', start: 12, end: 19, subwords: ['founded'] },
      { text: 'foundations', start: 20, end: 31, subwords: ['foundation', '##s'] },
      { text: 'before', start: 32, end: 38, subwords: ['before'] },
      { text: 'Apple', start: 39, end: 44, subwords: ['Apple'] },
      { text: 'Inc.', start: 45, end: 49, subwords: ['Inc', '.'] },
      { text: 'expanded', start: 50, end: 58, subwords: ['expanded'] },
      { text: 'to', start: 59, end: 61, subwords: ['to'] },
      { text: 'Cupertino', start: 62, end: 71, subwords: ['Cuper', '##tino'] },
      { text: '.', start: 71, end: 72, subwords: ['.'] },
    ],
  },
]

type TaggingScheme = 'bio' | 'bilou' | 'offsets'
type SubwordStrategy = 'first' | 'all-i' | 'all-repeat'

interface TokenDisplay {
  text: string
  start: number
  end: number
  wordIndex: number
  isSubwordContinuation: boolean
  tag: string
  isIgnored: boolean
  spanLabel?: string
}

export function SpanAlignmentInspector() {
  const [selectedPresetId, setSelectedPresetId] = useState<string>('ner')
  const [useSubwords, setUseSubwords] = useState<boolean>(true)
  const [scheme, setScheme] = useState<TaggingScheme>('bio')
  const [subwordStrategy, setSubwordStrategy] = useState<SubwordStrategy>('first')
  const [hoveredTokenIndex, setHoveredTokenIndex] = useState<number | null>(null)

  const preset = useMemo(() => {
    return PRESETS.find((p) => p.id === selectedPresetId) ?? PRESETS[0]
  }, [selectedPresetId])

  // Build flattened token list with character offsets
  const tokens = useMemo<TokenDisplay[]>(() => {
    const list: TokenDisplay[] = []

    preset.words.forEach((w, wIdx) => {
      if (!useSubwords) {
        list.push({
          text: w.text,
          start: w.start,
          end: w.end,
          wordIndex: wIdx,
          isSubwordContinuation: false,
          tag: 'O',
          isIgnored: false,
        })
      } else {
        // Calculate subword character offsets
        let currentOffset = w.start
        w.subwords.forEach((sw, swIdx) => {
          const rawSw = sw.replace(/^##/, '')
          const swLen = rawSw.length
          const swEnd = Math.min(w.end, currentOffset + swLen)
          list.push({
            text: sw,
            start: currentOffset,
            end: swEnd,
            wordIndex: wIdx,
            isSubwordContinuation: swIdx > 0,
            tag: 'O',
            isIgnored: false,
          })
          currentOffset = swEnd
        })
      }
    })

    // Assign tags according to spans and selected scheme
    // First identify word-level span membership
    const wordEntityMap = new Map<number, { label: string; isStart: boolean; isEnd: boolean; isSingle: boolean }>()

    preset.words.forEach((w, wIdx) => {
      const span = preset.spans.find((s) => w.start >= s.start && w.end <= s.end)
      if (span) {
        // Check if this word is start, end, or inside
        const isStart =
          w.start === span.start || !preset.words.some((pw) => pw.start >= span.start && pw.start < w.start)
        const isEnd = w.end === span.end || !preset.words.some((pw) => pw.end <= span.end && pw.end > w.end)
        const isSingle = isStart && isEnd
        wordEntityMap.set(wIdx, { label: span.label, isStart, isEnd, isSingle })
      }
    })

    // Assign token tags
    return list.map((tok) => {
      const wInfo = wordEntityMap.get(tok.wordIndex)
      if (!wInfo) {
        return { ...tok, tag: scheme === 'offsets' ? `[${tok.start}, ${tok.end})` : 'O' }
      }

      if (scheme === 'offsets') {
        return {
          ...tok,
          tag: `${wInfo.label} [${tok.start}, ${tok.end})`,
          spanLabel: wInfo.label,
        }
      }

      if (!tok.isSubwordContinuation) {
        // First token of the word
        if (scheme === 'bio') {
          const tag = wInfo.isStart ? `B-${wInfo.label}` : `I-${wInfo.label}`
          return { ...tok, tag, spanLabel: wInfo.label }
        } else {
          // BILOU
          let prefix = 'I'
          if (wInfo.isSingle) prefix = 'U'
          else if (wInfo.isStart) prefix = 'B'
          else if (wInfo.isEnd) prefix = 'L'
          return { ...tok, tag: `${prefix}-${wInfo.label}`, spanLabel: wInfo.label }
        }
      } else {
        // Subsequent subword within the word
        if (subwordStrategy === 'first') {
          return { ...tok, tag: '-100 (mask)', isIgnored: true, spanLabel: wInfo.label }
        } else if (subwordStrategy === 'all-i') {
          const tag = `I-${wInfo.label}`
          return { ...tok, tag, spanLabel: wInfo.label }
        } else {
          // repeat parent tag
          if (scheme === 'bio') {
            const tag = wInfo.isStart ? `B-${wInfo.label}` : `I-${wInfo.label}`
            return { ...tok, tag, spanLabel: wInfo.label }
          } else {
            let prefix = 'I'
            if (wInfo.isSingle) prefix = 'U'
            else if (wInfo.isStart) prefix = 'B'
            else if (wInfo.isEnd) prefix = 'L'
            return { ...tok, tag: `${prefix}-${wInfo.label}`, spanLabel: wInfo.label }
          }
        }
      }
    })
  }, [preset, useSubwords, scheme, subwordStrategy])

  const stats = useMemo(() => {
    const totalTokens = tokens.length
    const ignoredTokens = tokens.filter((t) => t.isIgnored).length
    const activeTokens = totalTokens - ignoredTokens
    const entityTokens = tokens.filter((t) => t.spanLabel && !t.isIgnored).length
    return { totalTokens, ignoredTokens, activeTokens, entityTokens }
  }, [tokens])

  const activeToken = hoveredTokenIndex !== null ? tokens[hoveredTokenIndex] : null

  return (
    <div className="my-6 rounded-lg border border-border bg-card p-4 text-card-foreground shadow-sm">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3 border-b border-border pb-3">
        <div>
          <h4 className="text-text font-semibold">Span Annotation & Token Alignment Inspector</h4>
          <p className="text-text-muted text-xs">
            Inspect character spans, subword tokenisation, and BIO / BILOU tag alignment.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <label className="text-text-muted text-xs font-medium">Preset:</label>
          <select
            value={selectedPresetId}
            onChange={(e) => setSelectedPresetId(e.target.value)}
            className="bg-surface text-text rounded border border-border px-2 py-1 text-xs shadow-sm"
          >
            {PRESETS.map((p) => (
              <option key={p.id} value={p.id}>
                {p.label}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Control bar */}
      <div className="mb-4 grid grid-cols-1 gap-3 sm:grid-cols-3">
        <div className="bg-surface/50 rounded border border-border p-2.5">
          <label className="text-text-muted mb-1 block text-xs font-semibold">Tokenisation Unit</label>
          <div className="flex gap-2 text-xs">
            <button
              onClick={() => setUseSubwords(false)}
              className={`flex-1 rounded px-2 py-1 font-medium transition-colors ${
                !useSubwords ? 'bg-primary text-primary-foreground' : 'bg-surface hover:bg-surface-elevated text-text'
              }`}
            >
              Words
            </button>
            <button
              onClick={() => setUseSubwords(true)}
              className={`flex-1 rounded px-2 py-1 font-medium transition-colors ${
                useSubwords ? 'bg-primary text-primary-foreground' : 'bg-surface hover:bg-surface-elevated text-text'
              }`}
            >
              BPE Subwords
            </button>
          </div>
        </div>

        <div className="bg-surface/50 rounded border border-border p-2.5">
          <label className="text-text-muted mb-1 block text-xs font-semibold">Labelling Scheme</label>
          <div className="flex gap-1.5 text-xs">
            {(['bio', 'bilou', 'offsets'] as TaggingScheme[]).map((s) => (
              <button
                key={s}
                onClick={() => setScheme(s)}
                className={`flex-1 rounded px-1.5 py-1 font-medium uppercase transition-colors ${
                  scheme === s ? 'bg-primary text-primary-foreground' : 'bg-surface hover:bg-surface-elevated text-text'
                }`}
              >
                {s}
              </button>
            ))}
          </div>
        </div>

        <div className="bg-surface/50 rounded border border-border p-2.5">
          <label className="text-text-muted mb-1 block text-xs font-semibold">
            Subword Alignment {useSubwords ? '' : '(N/A in words)'}
          </label>
          <select
            disabled={!useSubwords || scheme === 'offsets'}
            value={subwordStrategy}
            onChange={(e) => setSubwordStrategy(e.target.value as SubwordStrategy)}
            className="bg-surface text-text w-full rounded border border-border px-2 py-1 text-xs disabled:opacity-50"
          >
            <option value="first">First subword (mask rest: -100)</option>
            <option value="all-i">All subwords (propagate I-tag)</option>
            <option value="all-repeat">Repeat full tag on all pieces</option>
          </select>
        </div>
      </div>

      {/* Raw text with highlighted character spans */}
      <div className="bg-surface mb-4 rounded border border-border p-3">
        <div className="text-text-muted mb-1.5 flex items-center justify-between text-xs">
          <span className="font-semibold tracking-wider uppercase">Raw Character Stream with Offset Spans</span>
          <span>{preset.text.length} characters</span>
        </div>
        <div className="text-text font-mono text-sm leading-relaxed">
          {preset.text.split('').map((char, idx) => {
            const span = preset.spans.find((s) => idx >= s.start && idx < s.end)
            const isHovered = activeToken && idx >= activeToken.start && idx < activeToken.end
            let bgClass = 'hover:bg-accent/15'
            if (isHovered) {
              bgClass = 'bg-accent/30 font-bold'
            } else if (span) {
              bgClass = 'bg-primary/15 border-b-2 border-primary'
            }
            return (
              <span key={idx} className={`inline-block transition-colors ${bgClass}`} title={`char ${idx}: "${char}"`}>
                {char}
              </span>
            )
          })}
        </div>
        <div className="mt-2 flex flex-wrap gap-2 text-xs">
          {preset.spans.map((s, idx) => (
            <span
              key={idx}
              className="bg-surface-elevated text-text inline-flex items-center gap-1 rounded px-2 py-0.5 font-mono"
            >
              <span className="font-semibold text-primary">{s.label}:</span>
              <span>
                [{s.start}, {s.end}) &ldquo;{preset.text.slice(s.start, s.end)}&rdquo;
              </span>
            </span>
          ))}
        </div>
      </div>

      {/* Tokenised sequence with tags */}
      <div className="mb-4">
        <div className="text-text-muted mb-1.5 flex items-center justify-between text-xs">
          <span className="font-semibold tracking-wider uppercase">Token Sequence & Supervised Labels</span>
          <span>Hover a token to locate its exact character span</span>
        </div>
        <div className="bg-surface flex flex-wrap gap-2 rounded border border-border p-3">
          {tokens.map((tok, idx) => {
            const isHovered = hoveredTokenIndex === idx
            let tagBg = 'bg-surface-elevated text-text-muted'
            if (tok.isIgnored) {
              tagBg = 'bg-amber-500/15 text-amber-700 dark:text-amber-300 font-mono'
            } else if (tok.tag.startsWith('B-') || tok.tag.startsWith('U-')) {
              tagBg = 'bg-primary/20 text-primary font-bold'
            } else if (tok.tag.startsWith('I-') || tok.tag.startsWith('L-')) {
              tagBg = 'bg-sky-500/15 text-sky-700 dark:text-sky-300'
            }

            return (
              <div
                key={idx}
                onMouseEnter={() => setHoveredTokenIndex(idx)}
                onMouseLeave={() => setHoveredTokenIndex(null)}
                className={`group flex cursor-pointer flex-col items-center rounded border px-2 py-1 transition-all ${
                  isHovered
                    ? 'scale-105 border-accent bg-accent/10 shadow-sm'
                    : 'hover:border-text-muted border-border bg-card'
                }`}
              >
                <div className="flex items-center gap-1">
                  <span
                    className={`font-mono text-xs font-semibold ${
                      tok.isSubwordContinuation ? 'text-text-muted italic' : 'text-text'
                    }`}
                  >
                    {tok.text}
                  </span>
                </div>
                <span className={`mt-1 rounded px-1.5 py-0.5 text-[10px] tracking-wide ${tagBg}`}>{tok.tag}</span>
                <span className="text-text-muted mt-0.5 font-mono text-[9px]">
                  [{tok.start}:{tok.end}]
                </span>
              </div>
            )
          })}
        </div>
      </div>

      {/* Stats summary */}
      <div className="bg-surface/50 text-text-muted grid grid-cols-2 gap-2 rounded p-2.5 text-xs sm:grid-cols-4">
        <div>
          <span className="text-text block font-semibold">Total Tokens:</span>
          <span>{stats.totalTokens} tokens</span>
        </div>
        <div>
          <span className="text-text block font-semibold">Supervised Tokens:</span>
          <span>
            {stats.activeTokens} ({stats.entityTokens} entity)
          </span>
        </div>
        <div>
          <span className="text-text block font-semibold">Masked (-100):</span>
          <span>{stats.ignoredTokens} subwords</span>
        </div>
        <div>
          <span className="text-text block font-semibold">Subword Expansion:</span>
          <span>{(stats.totalTokens / preset.words.length).toFixed(2)}x word count</span>
        </div>
      </div>
    </div>
  )
}
