import { useMemo, useState } from 'react'
import { Interactive, Readout, StepControls } from 'aifn-render'

const END = '</w>'
const CORPUS: [string, number][] = [
  ['low', 5],
  ['lower', 2],
  ['newest', 6],
  ['widest', 3],
]
const TESTS = ['lowest', 'newer', 'wider']
const MAX_MERGES = 14

type Pair = [string, string]

const split = (word: string) => [...word.split(''), END]

/** Replace every adjacent occurrence of the pair in a symbol sequence by the merged symbol, left to right. */
function applyMerge(symbols: string[], [a, b]: Pair): string[] {
  const out: string[] = []
  for (let i = 0; i < symbols.length; i++) {
    if (i < symbols.length - 1 && symbols[i] === a && symbols[i + 1] === b) {
      out.push(a + b)
      i++
    } else out.push(symbols[i])
  }
  return out
}

/** Learn merges on the toy corpus. Ties go to the pair seen first when the corpus is read left to right. */
function learn(): { merges: Pair[]; counts: number[] } {
  let words = CORPUS.map(([w, f]) => ({ symbols: split(w), f }))
  const merges: Pair[] = []
  const counts: number[] = []
  for (let m = 0; m < MAX_MERGES; m++) {
    const tally = new Map<string, { pair: Pair; n: number }>()
    for (const { symbols, f } of words) {
      for (let i = 0; i < symbols.length - 1; i++) {
        const key = symbols[i] + '\u0000' + symbols[i + 1]
        const entry = tally.get(key) ?? { pair: [symbols[i], symbols[i + 1]] as Pair, n: 0 }
        entry.n += f
        tally.set(key, entry)
      }
    }
    let best: { pair: Pair; n: number } | undefined
    for (const entry of tally.values()) if (!best || entry.n > best.n) best = entry
    if (!best) break
    merges.push(best.pair)
    counts.push(best.n)
    words = words.map(({ symbols, f }) => ({ symbols: applyMerge(symbols, best.pair), f }))
  }
  return { merges, counts }
}

const encode = (word: string, merges: Pair[]) => merges.reduce(applyMerge, split(word))

function Tokens({ symbols }: { symbols: string[] }) {
  return (
    <span className="inline-flex flex-wrap gap-1">
      {symbols.map((s, i) => (
        <span key={i} className="rounded border bg-muted px-1.5 py-0.5 font-mono text-xs">
          {s}
        </span>
      ))}
    </span>
  )
}

/** Byte-pair encoding learned step by step on Sennrich et al.'s toy corpus. */
export function BpeSteps() {
  const { merges, counts } = useMemo(() => learn(), [])
  const [step, setStep] = useState(0)
  const active = merges.slice(0, step)
  const symbolTypes = new Set(CORPUS.flatMap(([w]) => encode(w, active))).size
  const corpusTokens = CORPUS.reduce((n, [w, f]) => n + f * encode(w, active).length, 0)

  return (
    <Interactive
      title="Byte-pair encoding, one merge at a time"
      caption="The corpus holds 'low' five times, 'lower' twice, 'newest' six times and 'widest' three times; </w> marks the end of a word. Each step merges the most frequent adjacent pair of symbols, weighting each word by its count. The unseen words at the bottom are encoded by replaying the learned merges in order."
      controls={
        <StepControls
          onStep={() => setStep((s) => Math.min(s + 1, merges.length))}
          onRun={() => setStep(merges.length)}
          onReset={() => setStep(0)}
          done={step >= merges.length}
        />
      }
      readout={
        <>
          <Readout label="merges" value={step} />
          <Readout label="corpus length (tokens)" value={corpusTokens} />
          <Readout label="distinct symbols in corpus" value={symbolTypes} />
        </>
      }
    >
      <div className="grid gap-4 text-sm md:grid-cols-2">
        <div className="space-y-2">
          <div className="text-xs font-medium text-muted-foreground">Corpus</div>
          {CORPUS.map(([w, f]) => (
            <div key={w} className="flex items-center gap-2">
              <span className="w-8 text-right text-xs text-muted-foreground tabular-nums">{f}×</span>
              <Tokens symbols={encode(w, active)} />
            </div>
          ))}
          <div className="pt-2 text-xs font-medium text-muted-foreground">Unseen words</div>
          {TESTS.map((w) => (
            <div key={w} className="flex items-center gap-2">
              <span className="w-14 text-xs text-muted-foreground">{w}</span>
              <Tokens symbols={encode(w, active)} />
            </div>
          ))}
        </div>
        <div className="space-y-1">
          <div className="text-xs font-medium text-muted-foreground">Merge table</div>
          {active.length === 0 && <div className="text-xs text-muted-foreground">No merges yet.</div>}
          <ol className="space-y-1">
            {active.map(([a, b], i) => (
              <li key={i} className="flex items-center gap-2 font-mono text-xs">
                <span className="w-5 text-right text-muted-foreground tabular-nums">{i + 1}.</span>
                <span>
                  {a} + {b} → {a + b}
                </span>
                <span className="text-muted-foreground">(count {counts[i]})</span>
              </li>
            ))}
          </ol>
        </div>
      </div>
    </Interactive>
  )
}
