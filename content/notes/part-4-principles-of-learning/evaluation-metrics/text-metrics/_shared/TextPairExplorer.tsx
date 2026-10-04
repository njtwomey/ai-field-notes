import { useMemo, useState } from 'react'
import {
  Bars,
  choice,
  Curve,
  Figure,
  formatNumber,
  Plot,
  Readout,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
  Textarea,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { cn } from '@/lib/utils'
import { bleu, chrf, errorRates, rougeL, rougeN, tokens } from './text'

type Focus = 'bleu' | 'chrf' | 'rouge' | 'wer'

const PRESETS = {
  paraphrase: { label: 'one word changed', ref: 'the cat is on the mat', hyp: 'the cat sat on the mat' },
  order: { label: 'word order', ref: 'police killed the gunman', hyp: 'the gunman killed police' },
  short: { label: 'too short', ref: 'the quick brown fox jumps over the lazy dog', hyp: 'the quick fox' },
  asr: {
    label: 'speech transcript',
    ref: 'the quick brown fox jumps over the lazy dog',
    hyp: 'the quick brown fox jumped over a lazy dog today',
  },
} as const
type Preset = keyof typeof PRESETS | 'custom'

const TITLES: Record<Focus, string> = {
  bleu: 'BLEU on one sentence pair',
  chrf: 'chrF on one sentence pair',
  rouge: 'ROUGE on one sentence pair',
  wer: 'Word error rate and its alignment',
}

const CAPTIONS: Record<Focus, string> = {
  bleu: 'Edit the reference and the hypothesis, or pick an example. The bars are the clipped n-gram precisions p₁ to p₄; BLEU is their geometric mean times the brevity penalty. A single missing 4-gram match sends sentence-level BLEU to zero, which is why it is reported over a corpus or smoothed.',
  chrf: 'Edit the reference and the hypothesis, or pick an example. The lines are the character n-gram precision and recall for n = 1 to 6, with spaces removed; chrF is the F_β score of their averages. Character n-grams give partial credit to a word with the right stem and the wrong ending.',
  rouge:
    'Edit the reference and the hypothesis, or pick an example. ROUGE-1 and ROUGE-2 count shared unigrams and bigrams; ROUGE-L uses the longest common subsequence, which rewards words in the right order without requiring them to be adjacent.',
  wer: 'Edit the reference and the hypothesis, or pick an example. The alignment is the minimum-edit one: each reference word is matched, substituted or deleted, and extra hypothesis words are insertions. WER divides the edits by the reference length, so it exceeds 1 when the hypothesis is much longer.',
}

/** A sentence pair with every text metric computed live; `focus` chooses the detail shown. */
export function TextPairExplorer({ focus }: { focus: Focus }) {
  const first = focus === 'wer' ? 'asr' : 'paraphrase'
  // The text typed by the reader; shown while the example is 'custom'.
  const [custom, setCustom] = useState<{ ref: string; hyp: string }>({
    ref: PRESETS[first].ref,
    hyp: PRESETS[first].hyp,
  })
  const state = useFigureState({
    preset: choice<Preset>(
      [
        ...(Object.keys(PRESETS) as (keyof typeof PRESETS)[]).map((k) => ({ value: k, label: PRESETS[k].label })),
        { value: 'custom', label: 'custom' },
      ],
      first,
      { label: 'example' },
    ),
    beta: choice<'1' | '2' | '3'>(
      [
        { value: '1', label: '1' },
        { value: '2', label: '2' },
        { value: '3', label: '3' },
      ],
      '2',
      { label: 'β (weight of recall)', when: () => focus === 'chrf' },
    ),
  })

  const { ref, hyp } = state.preset === 'custom' ? custom : PRESETS[state.preset]
  const edit = (next: { ref: string; hyp: string }) => {
    setCustom(next)
    state.set('preset', 'custom')
  }

  const m = useMemo(() => {
    const r = tokens(ref)
    const h = tokens(hyp)
    const b = bleu([{ hyp: h, refs: [r] }])
    const bs = bleu([{ hyp: h, refs: [r] }], 4, true)
    const c = chrf(hyp, ref, 6, Number(state.beta))
    const r1 = rougeN(h, r, 1)
    const r2 = rougeN(h, r, 2)
    const rl = rougeL(h, r)
    const e = errorRates(r, h)
    const cer = errorRates([...ref.replace(/\s+/g, ' ').trim()], [...hyp.replace(/\s+/g, ' ').trim()]).wer
    return { b, bs, c, r1, r2, rl, e, cer }
  }, [ref, hyp, state.beta])

  const bleuBars = [{ name: 'clipped precision pₙ', x: [1, 2, 3, 4], y: m.b.precisions, slot: 0 }] as const
  const chrfLines = [
    { name: 'precision', x: [1, 2, 3, 4, 5, 6], y: m.c.precisions, slot: 0 },
    { name: 'recall', x: [1, 2, 3, 4, 5, 6], y: m.c.recalls, slot: 1, dashed: true },
  ] as const

  const xAxis = useAxis({ label: 'n-gram order n', hold: 'union' })
  const yAxis = useAxis({ label: 'pₙ', range: [0, 1] })
  const xAxis2 = useAxis({ label: 'character n-gram order n', hold: 'union' })
  const yAxis2 = useAxis({ label: 'score', range: [0, 1] })
  return (
    <Figure
      title={TITLES[focus]}
      state={state}
      caption={CAPTIONS[focus]}
      readouts={
        <>
          <Readout label="BLEU" value={formatNumber(m.b.score)} />
          <Readout label="smoothed BLEU" value={formatNumber(m.bs.score)} />
          <Readout label={`chrF${state.beta}`} value={formatNumber(m.c.F)} />
          <Readout label="ROUGE-L F" value={formatNumber(m.rl.F)} />
          <Readout label="WER" value={formatNumber(m.e.wer)} />
          <Readout label="CER" value={formatNumber(m.cer)} />
        </>
      }
    >
      <div className="grid gap-3 sm:grid-cols-2">
        <label className="space-y-1 text-xs text-muted-foreground">
          reference
          <Textarea
            value={ref}
            onChange={(e) => edit({ ref: e.target.value, hyp })}
            className="min-h-10 font-mono text-sm text-foreground"
          />
        </label>
        <label className="space-y-1 text-xs text-muted-foreground">
          hypothesis
          <Textarea
            value={hyp}
            onChange={(e) => edit({ ref, hyp: e.target.value })}
            className="min-h-10 font-mono text-sm text-foreground"
          />
        </label>
      </div>

      {focus === 'bleu' && (
        <div className="space-y-2">
          <Plot x={xAxis} y={yAxis} height={220}>
            <Bars {...bleuBars[0]} />
          </Plot>
          <div className="flex flex-wrap gap-x-6 gap-y-1 text-xs text-muted-foreground">
            <Readout
              label="matched / total n-grams"
              value={m.b.matched.map((v, i) => `${v}/${m.b.total[i]}`).join(', ')}
            />
            <Readout label="hypothesis length c" value={m.b.hypLength} />
            <Readout label="reference length r" value={m.b.refLength} />
            <Readout label="brevity penalty" value={formatNumber(m.b.bp)} />
          </div>
        </div>
      )}

      {focus === 'chrf' && (
        <div className="space-y-2">
          <Plot x={xAxis2} y={yAxis2} height={220}>
            <Curve {...chrfLines[0]} />
            <Curve {...chrfLines[1]} />
          </Plot>
          <div className="flex flex-wrap gap-x-6 gap-y-1 text-xs text-muted-foreground">
            <Readout label="average precision" value={formatNumber(m.c.P)} />
            <Readout label="average recall" value={formatNumber(m.c.R)} />
          </div>
        </div>
      )}

      {focus === 'rouge' && (
        <div className="overflow-x-auto rounded-lg border">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>score</TableHead>
                <TableHead>matches</TableHead>
                <TableHead>precision</TableHead>
                <TableHead>recall</TableHead>
                <TableHead>F1</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {[
                ['ROUGE-1', m.r1.matched, m.r1],
                ['ROUGE-2', m.r2.matched, m.r2],
                ['ROUGE-L', m.rl.lcs, m.rl],
              ].map(([name, k, s]) => {
                const v = s as { P: number; R: number; F: number }
                return (
                  <TableRow key={name as string}>
                    <TableCell className="font-mono">{name as string}</TableCell>
                    <TableCell className="font-mono tabular-nums">{k as number}</TableCell>
                    <TableCell className="font-mono tabular-nums">{formatNumber(v.P)}</TableCell>
                    <TableCell className="font-mono tabular-nums">{formatNumber(v.R)}</TableCell>
                    <TableCell className="font-mono tabular-nums">{formatNumber(v.F)}</TableCell>
                  </TableRow>
                )
              })}
            </TableBody>
          </Table>
        </div>
      )}

      {focus === 'wer' && (
        <div className="space-y-2">
          <div className="flex flex-wrap gap-1.5">
            {m.e.ops.map((o, i) => (
              <span
                key={i}
                className={cn(
                  'inline-flex flex-col items-center rounded-md border px-2 py-1 font-mono text-xs',
                  o.op === 'hit' ? 'border-transparent bg-muted' : 'border-foreground',
                )}
              >
                <span className={cn(o.op === 'ins' && 'text-muted-foreground')}>{o.ref ?? '∅'}</span>
                <span className={cn(o.op === 'del' && 'text-muted-foreground')}>{o.hyp ?? '∅'}</span>
                <span className="text-[10px] text-muted-foreground uppercase">
                  {o.op === 'hit' ? 'match' : o.op === 'sub' ? 'S' : o.op === 'del' ? 'D' : 'I'}
                </span>
              </span>
            ))}
          </div>
          <div className="flex flex-wrap gap-x-6 gap-y-1 text-xs text-muted-foreground">
            <Readout label="H, S, D, I" value={`${m.e.H}, ${m.e.S}, ${m.e.D}, ${m.e.I}`} />
            <Readout label="MER" value={formatNumber(m.e.mer)} />
            <Readout label="WIL" value={formatNumber(m.e.wil)} />
            <Readout label="WIP" value={formatNumber(m.e.wip)} />
          </div>
        </div>
      )}
    </Figure>
  )
}
