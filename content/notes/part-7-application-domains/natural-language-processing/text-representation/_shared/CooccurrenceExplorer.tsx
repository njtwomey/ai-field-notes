import { useMemo, useState } from 'react'
import { Interactive, ParamChoice, ParamSlider, Readout, formatNumber, sequential } from 'aifn-render'
import { cn } from '@/lib/utils'

const TOY_CORPUS = [
  'the cat is a pet',
  'the dog is a pet',
  'we fed the cat',
  'we fed the dog',
  'the vet saw the cat',
  'the vet saw the dog',
  'the cat chased the mouse',
  'the dog chased the cat',
  'the mouse ate the cheese',
  'the mouse ate the grain',
  'we ate the cheese',
  'the cat slept',
  'the dog slept',
]

const SENTENCES = TOY_CORPUS.map((s) => s.split(' '))
const VOCAB = [...new Set(SENTENCES.flat())].sort()
const INDEX = new Map(VOCAB.map((w, i) => [w, i]))
const ROWS = ['cat', 'dog', 'mouse', 'cheese', 'grain', 'vet']

/** Symmetric window counts within each sentence: M[w][c] = number of times c is within `window` tokens of w. */
function cooccurrence(window: number): number[][] {
  const M = VOCAB.map(() => VOCAB.map(() => 0))
  for (const s of SENTENCES)
    s.forEach((w, i) => {
      for (let j = Math.max(0, i - window); j <= Math.min(s.length - 1, i + window); j++)
        if (j !== i) M[INDEX.get(w)!][INDEX.get(s[j])!]++
    })
  return M
}

/** Positive PMI in bits, with the context distribution raised to the power alpha and renormalised. */
function ppmi(M: number[][], alpha: number): number[][] {
  const total = M.flat().reduce((a, b) => a + b, 0)
  const rowSum = M.map((r) => r.reduce((a, b) => a + b, 0))
  const colSum = VOCAB.map((_, j) => M.reduce((a, r) => a + r[j], 0))
  const smoothed = colSum.map((c) => c ** alpha)
  const z = smoothed.reduce((a, b) => a + b, 0)
  return M.map((r, i) =>
    r.map((n, j) => (n === 0 ? 0 : Math.max(0, Math.log2(n / total / ((rowSum[i] / total) * (smoothed[j] / z)))))),
  )
}

function cosine(a: number[], b: number[]) {
  let ab = 0
  let aa = 0
  let bb = 0
  a.forEach((x, i) => {
    ab += x * b[i]
    aa += x * x
    bb += b[i] * b[i]
  })
  return aa && bb ? ab / Math.sqrt(aa * bb) : 0
}

/** Word–context matrix of a thirteen-sentence corpus, as raw counts or PPMI, with the cosines they imply. */
export function CooccurrenceExplorer() {
  const [span, setSpan] = useState(2)
  const [measure, setMeasure] = useState<'count' | 'ppmi'>('ppmi')
  const [alpha, setAlpha] = useState<'1' | '0.75'>('1')

  const { M, W, max } = useMemo(() => {
    const counts = cooccurrence(span)
    const weights = measure === 'count' ? counts : ppmi(counts, Number(alpha))
    const peak = Math.max(...ROWS.flatMap((r) => weights[INDEX.get(r)!]))
    return { M: counts, W: weights, max: peak || 1 }
  }, [span, measure, alpha])

  const row = (w: string) => W[INDEX.get(w)!]
  const total = M.flat().reduce((a, b) => a + b, 0)
  const shade = (v: number) =>
    sequential[Math.min(sequential.length - 1, Math.floor((v / max) * (sequential.length - 1)))]

  return (
    <Interactive
      title="Word–context matrix: counts against PPMI"
      caption="Each row is a word's vector over contexts: how often each context word falls within the window around it, in the thirteen sentences listed in the note. Raw counts are dominated by 'the', so every noun looks like every other. PPMI keeps only contexts that co-occur more than chance. Context smoothing (α = 0.75) lowers the PMI of rare contexts."
      controls={
        <>
          <ParamSlider
            label="window (tokens each side)"
            value={span}
            onChange={setSpan}
            min={1}
            max={4}
            step={1}
            debounceMs={0}
          />
          <ParamChoice
            label="cell value"
            value={measure}
            onChange={setMeasure}
            options={[
              { value: 'count', label: 'count' },
              { value: 'ppmi', label: 'PPMI (bits)' },
            ]}
          />
          <ParamChoice
            label="context smoothing α"
            value={alpha}
            onChange={setAlpha}
            options={[
              { value: '1', label: '1 (none)' },
              { value: '0.75', label: '0.75' },
            ]}
          />
        </>
      }
      readout={
        <>
          <Readout label="pairs |D|" value={total} />
          <Readout label="cos(cat, dog)" value={formatNumber(cosine(row('cat'), row('dog')))} />
          <Readout label="cos(cat, mouse)" value={formatNumber(cosine(row('cat'), row('mouse')))} />
          <Readout label="cos(cat, cheese)" value={formatNumber(cosine(row('cat'), row('cheese')))} />
          <Readout label="cos(mouse, cheese)" value={formatNumber(cosine(row('mouse'), row('cheese')))} />
        </>
      }
    >
      <div className="overflow-x-auto">
        <table className="border-separate border-spacing-0.5 font-mono text-[11px] tabular-nums">
          <thead>
            <tr>
              <th />
              {VOCAB.map((c) => (
                <th key={c} className="h-16 w-9 align-bottom font-normal text-muted-foreground">
                  <span className="inline-block origin-bottom-left translate-x-3 -rotate-60 whitespace-nowrap">
                    {c}
                  </span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {ROWS.map((r) => (
              <tr key={r}>
                <th className="pr-2 text-right font-normal text-muted-foreground">{r}</th>
                {row(r).map((v, j) => (
                  <td
                    key={j}
                    className={cn(
                      'h-7 w-9 rounded-sm text-center',
                      v === 0 ? 'bg-muted/40 text-muted-foreground/50' : v / max > 0.55 ? 'text-white' : 'text-black',
                    )}
                    style={v === 0 ? undefined : { backgroundColor: shade(v) }}
                  >
                    {v === 0 ? '·' : measure === 'count' ? v : v.toFixed(1)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Interactive>
  )
}
