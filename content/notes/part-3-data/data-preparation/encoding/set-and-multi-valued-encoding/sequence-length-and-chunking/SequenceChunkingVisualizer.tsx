import React, { useState, useMemo } from 'react'

type ChunkingMode = 'sliding' | 'head' | 'tail' | 'head_tail'

export const SequenceChunkingVisualizer: React.FC = () => {
  const [mode, setMode] = useState<ChunkingMode>('sliding')
  const [windowSize, setWindowSize] = useState<number>(4)
  const [overlap, setOverlap] = useState<number>(2)

  // 10 mock paragraphs / sections in a long document
  const paragraphs = useMemo(
    () => [
      { id: 1, title: 'Introduction', role: 'Context' },
      { id: 2, title: 'Background', role: 'Context' },
      { id: 3, title: 'Prior Literature', role: 'Context' },
      { id: 4, title: 'Methodology Part 1', role: 'Core' },
      { id: 5, title: 'Methodology Part 2', role: 'Core' },
      { id: 6, title: 'Experiment Setup', role: 'Evaluation' },
      { id: 7, title: 'Empirical Results', role: 'Evaluation' },
      { id: 8, title: 'Error Analysis', role: 'Evaluation' },
      { id: 9, title: 'Discussion', role: 'Conclusion' },
      { id: 10, title: 'Conclusion', role: 'Conclusion' },
    ],
    [],
  )

  // Compute chunks based on active mode
  const chunks = useMemo(() => {
    if (mode === 'head') {
      return [
        {
          chunkId: 1,
          items: paragraphs.slice(0, windowSize),
          name: `Head Chunk (First ${windowSize} Sections)`,
        },
      ]
    }

    if (mode === 'tail') {
      return [
        {
          chunkId: 1,
          items: paragraphs.slice(paragraphs.length - windowSize),
          name: `Tail Chunk (Last ${windowSize} Sections)`,
        },
      ]
    }

    if (mode === 'head_tail') {
      const half = Math.floor(windowSize / 2)
      const headItems = paragraphs.slice(0, half)
      const tailItems = paragraphs.slice(paragraphs.length - half)
      return [
        {
          chunkId: 1,
          items: [...headItems, ...tailItems],
          name: `Head + Tail Chunk (${half} head, ${half} tail)`,
        },
      ]
    }

    // Sliding window with overlap
    const stride = Math.max(1, windowSize - overlap)
    const result: { chunkId: number; items: typeof paragraphs; name: string }[] = []
    let start = 0
    let chunkNum = 1

    while (start < paragraphs.length) {
      const chunkItems = paragraphs.slice(start, start + windowSize)
      result.push({
        chunkId: chunkNum,
        items: chunkItems,
        name: `Chunk ${chunkNum} [Sec ${chunkItems[0].id}-${chunkItems[chunkItems.length - 1].id}]`,
      })
      if (start + windowSize >= paragraphs.length) break
      start += stride
      chunkNum++
    }

    return result
  }, [paragraphs, mode, windowSize, overlap])

  // Length bucketing comparison simulation
  const sequences = [8, 12, 42, 48]
  const maxSeq = 48
  const naivePadTokens = sequences.reduce((s, len) => s + (maxSeq - len), 0)
  const naiveTotalSlots = sequences.length * maxSeq
  const naiveWastePct = Math.round((naivePadTokens / naiveTotalSlots) * 100)

  // Bucketed: Batch 1: [8, 12] padded to 12. Batch 2: [42, 48] padded to 48.
  const bucketedPadTokens = 12 - 8 + (12 - 12) + (48 - 42) + (48 - 48)
  const bucketedTotalSlots = 2 * 12 + 2 * 48
  const bucketedWastePct = Math.round((bucketedPadTokens / bucketedTotalSlots) * 100)

  return (
    <div className="not-prose my-6 rounded-xl border border-border bg-card p-5 text-card-foreground shadow-sm">
      <div className="mb-4 flex flex-col gap-1 border-b border-border pb-4">
        <h3 className="text-base font-semibold text-foreground">
          Sequence Length, Overlapping Chunking &amp; Length Bucketing
        </h3>
        <p className="text-xs text-muted-foreground">
          Models possess finite context windows <em>L</em>
          <sub>max</sub>. Contrast naive truncation with overlapping sliding windows, aggregation strategies, and
          dynamic length bucketing to eliminate padding waste.
        </p>
      </div>

      {/* Control Grid */}
      <div className="mb-5 grid grid-cols-1 gap-4 text-xs md:grid-cols-3">
        <div>
          <label className="mb-1 block font-medium text-foreground">Context Window Strategy:</label>
          <div className="flex flex-col gap-1.5">
            {[
              { id: 'sliding', label: '1. Overlapping Sliding Window' },
              { id: 'head', label: '2. Head Truncation (Prefix Only)' },
              { id: 'tail', label: '3. Tail Truncation (Suffix Only)' },
              { id: 'head_tail', label: '4. Head + Tail Split' },
            ].map((opt) => (
              <button
                key={opt.id}
                onClick={() => setMode(opt.id as ChunkingMode)}
                className={`rounded border px-2.5 py-1.5 text-left text-xs transition-colors ${
                  mode === opt.id
                    ? 'border-primary bg-primary font-medium text-primary-foreground'
                    : 'border-transparent bg-muted/40 text-muted-foreground hover:bg-muted'
                }`}
              >
                {opt.label}
              </button>
            ))}
          </div>
        </div>

        <div>
          <label className="mb-1 block font-medium text-foreground">
            Window Capacity ($W$): <span className="font-mono text-primary">{windowSize} sections</span>
          </label>
          <input
            type="range"
            min={3}
            max={6}
            step={1}
            value={windowSize}
            onChange={(e) => setWindowSize(Number(e.target.value))}
            className="w-full accent-primary"
          />
          <span className="mt-1 block text-[10px] text-muted-foreground">
            Maximum tokens/sections that fit into one transformer forward pass.
          </span>
        </div>

        {mode === 'sliding' && (
          <div>
            <label className="mb-1 block font-medium text-foreground">
              Boundary Overlap ($O$): <span className="font-mono text-primary">{overlap} sections</span>
            </label>
            <input
              type="range"
              min={0}
              max={windowSize - 1}
              step={1}
              value={overlap}
              onChange={(e) => setOverlap(Number(e.target.value))}
              className="w-full accent-primary"
            />
            <span className="mt-1 block text-[10px] text-muted-foreground">
              Overlap stride $S = W - O = {windowSize - overlap}$. Preserves cross-boundary semantics.
            </span>
          </div>
        )}
      </div>

      {/* Visual Chunk Breakdown */}
      <div className="mb-5 rounded-lg border border-border bg-card/60 p-4">
        <div className="mb-3 flex items-center justify-between text-xs">
          <span className="font-semibold text-foreground">
            Document Section Coverage ({chunks.length} Total Model Forwards)
          </span>
          <span className="text-[11px] text-muted-foreground">Document: 10 Total Sections</span>
        </div>

        {/* Global Sequence Track */}
        <div className="mb-4">
          <span className="mb-1 block text-[10px] text-muted-foreground">Full Document Timeline:</span>
          <div className="grid grid-cols-10 gap-1 text-center font-mono text-[10px]">
            {paragraphs.map((p) => (
              <div key={p.id} className="rounded border border-border/60 bg-muted/40 p-1.5">
                <div className="font-bold text-foreground">S{p.id}</div>
                <div className="truncate text-[8px] text-muted-foreground">{p.title.split(' ')[0]}</div>
              </div>
            ))}
          </div>
        </div>

        {/* Chunks Stack */}
        <div className="space-y-2">
          {chunks.map((ch) => (
            <div
              key={ch.chunkId}
              className="flex items-center justify-between rounded-lg border border-border bg-muted/20 p-2.5 text-xs"
            >
              <span className="font-mono text-xs font-bold text-primary">{ch.name}</span>
              <div className="flex gap-1.5">
                {ch.items.map((it) => (
                  <span
                    key={it.id}
                    className="rounded border border-primary/30 bg-primary/10 px-2 py-0.5 font-mono text-[10px] text-primary"
                  >
                    S{it.id}: {it.title}
                  </span>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Dynamic Length Bucketing vs Naive Padding */}
      <div className="rounded-lg border border-border bg-card/60 p-4 text-xs">
        <div className="mb-2 text-xs font-bold tracking-wider text-foreground uppercase">
          Padding Efficiency: Naive Padding vs Dynamic Length Bucketing
        </div>
        <p className="mb-3 text-[11px] leading-relaxed text-muted-foreground">
          In variable-length batches (e.g. sequences of length [8, 12, 42, 48]), padding all inputs to the dataset
          maximum wastes computational attention on meaningless <code>&lt;PAD&gt;</code> tokens:
        </p>

        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          <div className="rounded-lg border border-rose-500/20 bg-rose-500/10 p-3">
            <div className="mb-1 flex items-center justify-between">
              <span className="font-semibold text-rose-600 dark:text-rose-400">1. Global Max Padding</span>
              <span className="font-mono font-bold text-rose-600 dark:text-rose-400">{naiveWastePct}% Waste</span>
            </div>
            <p className="text-[10px] leading-normal text-muted-foreground">
              Pads all sequences to <em>L</em>
              <sub>max</sub> = 48. Total padding tokens inserted: {naivePadTokens} across {naiveTotalSlots} slots.
              Attention quadratic complexity <em>O</em>(<em>L</em>
              <sup>2</sup>) wastes memory computing self-attention over zeros.
            </p>
          </div>

          <div className="rounded-lg border border-emerald-500/20 bg-emerald-500/10 p-3">
            <div className="mb-1 flex items-center justify-between">
              <span className="font-semibold text-emerald-600 dark:text-emerald-400">2. Dynamic Length Bucketing</span>
              <span className="font-mono font-bold text-emerald-600 dark:text-emerald-400">
                {bucketedWastePct}% Waste
              </span>
            </div>
            <p className="text-[10px] leading-normal text-muted-foreground">
              Sorts dataset by length and batches similar sequences together. Batch A: [8, 12] padded to 12. Batch B:
              [42, 48] padded to 48. Padding waste drops from {naiveWastePct}% to {bucketedWastePct}%, yielding a ~2.5x
              training speedup.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
