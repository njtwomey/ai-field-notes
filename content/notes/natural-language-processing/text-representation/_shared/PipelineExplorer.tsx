import { useMemo, useState } from 'react'
import { Interactive, ParamChoice, ParamSwitch, Readout, formatNumber } from 'aifn-render'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from 'aifn-render'
import { Textarea } from 'aifn-render'
import { cn } from '@/lib/utils'
import { NEGATIONS, STOP_WORDS, counts, lemmatise, ngrams, normalise, porterStem, tokenise } from './text'

const CORPUS = [
  'The cats are chasing the mice in the garden.',
  'A cat chased a mouse, and the mouse ran away.',
  "Dogs don't like cats, but this dog likes the cat.",
  'The children were running while the dog was barking.',
  "It isn't a bad film; it's not boring at all.",
  "The Café's résumé-writing course starts on 3 May.",
]

type Morph = 'none' | 'porter' | 'lemma'
type Grams = '1' | '1-2' | '2'
type Weight = 'count' | 'binary' | 'tfidf'
type Settings = {
  casefold: boolean
  accents: boolean
  contractions: boolean
  stop: boolean
  keepNegation: boolean
  morph: Morph
  grams: Grams
}

/** One token after the pipeline, with whether a stop list removed it. */
type Tok = { raw: string; term: string; removed: boolean }

function process(text: string, s: Settings): Tok[] {
  const norm = normalise(text, {
    unicode: 'NFKC',
    casefold: s.casefold,
    stripAccents: s.accents,
    expandContractions: s.contractions,
    numbers: true,
  })
  return tokenise(norm).map((raw) => {
    const lower = raw.toLowerCase()
    // The list is lower case, so without case folding it misses "The": the interaction the notes describe.
    const removed = s.stop && STOP_WORDS.has(raw) && !(s.keepNegation && NEGATIONS.has(raw))
    const term = s.morph === 'porter' ? porterStem(raw) : s.morph === 'lemma' ? lemmatise(raw) : raw
    // Stemmers and the lemmatiser work on lower case; keep the surface case when folding is off and nothing changed.
    return { raw, term: s.morph !== 'none' && term === lower ? (s.casefold ? lower : raw) : term, removed }
  })
}

function features(toks: Tok[], grams: Grams): string[] {
  const kept = toks.filter((t) => !t.removed).map((t) => t.term)
  const orders = grams === '1' ? [1] : grams === '2' ? [2] : [1, 2]
  return orders.flatMap((n) => ngrams(kept, n))
}

/** Raw text → normalised tokens → stop words → stems or lemmas → n-grams → weighted bag, over a six-sentence corpus. */
export function PipelineExplorer() {
  const [choice, setChoice] = useState<string>('0')
  const [text, setText] = useState(CORPUS[0])
  const [casefold, setCasefold] = useState(true)
  const [accents, setAccents] = useState(false)
  const [contractions, setContractions] = useState(false)
  const [stop, setStop] = useState(false)
  const [keepNegation, setKeepNegation] = useState(true)
  const [morph, setMorph] = useState<Morph>('none')
  const [grams, setGrams] = useState<Grams>('1')
  const [weight, setWeight] = useState<Weight>('count')

  const pick = (v: string) => {
    setChoice(v)
    if (v !== 'custom') setText(CORPUS[Number(v)])
  }

  const r = useMemo(() => {
    const s: Settings = { casefold, accents, contractions, stop, keepNegation, morph, grams }
    const docs = choice === 'custom' ? [...CORPUS, text] : CORPUS
    const docFeatures = docs.map((d) => features(process(d, s), grams))
    const df = counts(docFeatures.flatMap((f) => [...new Set(f)]))
    const toks = process(text, s)
    const tf = counts(features(toks, grams))
    const N = docs.length
    const rows = [...tf.entries()]
      .map(([term, n]) => {
        const d = df.get(term) ?? 1
        const idf = Math.log(N / d)
        const w = weight === 'count' ? n : weight === 'binary' ? 1 : n * idf
        return { term, n, d, idf, w }
      })
      .sort((a, b) => b.w - a.w || a.term.localeCompare(b.term))
    return { toks, rows, N, vocab: df.size }
  }, [text, choice, casefold, accents, contractions, stop, keepNegation, morph, grams, weight])

  return (
    <Interactive
      title="From raw text to a bag of words"
      caption="Pick a sentence or type one. Each switch is one pipeline step; the chips show the tokens after it (struck through when a stop list removes them), and the table shows the resulting features with their weight. Corpus vocabulary is the number of distinct features across all six sentences: watch it shrink with case folding and stemming, and grow with bigrams. Stemming uses Porter's 1980 rules; lemmatisation uses a small built-in dictionary."
      controls={
        <>
          <ParamChoice
            label="sentence"
            value={choice}
            onChange={pick}
            options={[
              ...CORPUS.map((_, i) => ({ value: String(i), label: String(i + 1) })),
              { value: 'custom', label: 'custom' },
            ]}
          />
          <ParamChoice
            label="stemming or lemmatisation"
            value={morph}
            onChange={setMorph}
            options={[
              { value: 'none', label: 'none' },
              { value: 'porter', label: 'Porter stem' },
              { value: 'lemma', label: 'lemma' },
            ]}
          />
          <ParamChoice
            label="n-grams"
            value={grams}
            onChange={setGrams}
            options={[
              { value: '1', label: 'unigrams' },
              { value: '1-2', label: '1 + 2' },
              { value: '2', label: 'bigrams' },
            ]}
          />
          <ParamChoice
            label="weighting"
            value={weight}
            onChange={setWeight}
            options={[
              { value: 'count', label: 'count' },
              { value: 'binary', label: 'binary' },
              { value: 'tfidf', label: 'TF-IDF' },
            ]}
          />
          <ParamSwitch label="case folding" checked={casefold} onChange={setCasefold} />
          <ParamSwitch label="strip accents" checked={accents} onChange={setAccents} />
          <ParamSwitch label="expand contractions" checked={contractions} onChange={setContractions} />
          <ParamSwitch label="remove stop words" checked={stop} onChange={setStop} />
          <ParamSwitch label="…but keep negations" checked={keepNegation} onChange={setKeepNegation} />
        </>
      }
      readout={
        <>
          <Readout label="tokens kept" value={`${r.toks.filter((t) => !t.removed).length} / ${r.toks.length}`} />
          <Readout label="distinct features here" value={r.rows.length} />
          <Readout label="corpus vocabulary" value={r.vocab} />
          <Readout label="documents N" value={r.N} />
        </>
      }
    >
      <Textarea
        value={text}
        onChange={(e) => {
          setText(e.target.value)
          setChoice('custom')
        }}
        className="min-h-10 font-mono text-sm"
        aria-label="text to process"
      />
      <div className="flex flex-wrap gap-1.5">
        {r.toks.map((t, i) => (
          <span
            key={i}
            className={cn(
              'inline-flex flex-col items-center rounded-md border px-2 py-1 font-mono text-xs',
              t.removed && 'border-dashed text-muted-foreground line-through',
            )}
          >
            <span>{t.term}</span>
            {t.term !== t.raw && <span className="text-[10px] text-muted-foreground">{t.raw}</span>}
          </span>
        ))}
      </div>
      <div className="max-h-72 overflow-auto rounded-lg border">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>feature</TableHead>
              <TableHead>tf</TableHead>
              <TableHead>df</TableHead>
              <TableHead>idf = ln(N / df)</TableHead>
              <TableHead>weight</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {r.rows.map((row) => (
              <TableRow key={row.term}>
                <TableCell className="font-mono text-xs">{row.term}</TableCell>
                <TableCell className="font-mono text-xs tabular-nums">{row.n}</TableCell>
                <TableCell className="font-mono text-xs tabular-nums">{row.d}</TableCell>
                <TableCell className="font-mono text-xs tabular-nums">{formatNumber(row.idf)}</TableCell>
                <TableCell className="font-mono text-xs tabular-nums">{formatNumber(row.w)}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
    </Interactive>
  )
}
