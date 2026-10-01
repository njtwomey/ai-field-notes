import { useContext, useMemo, useState, type ReactNode } from 'react'
import { stream } from 'aifn/foundation/random'
import { toFlat, toRows } from 'aifn/foundation/tensor'
import { trace } from 'aifn/foundation/trace'
import { normalise } from 'aifn/text/normalise'
import { characterTokenise, tokenise, type Tokenisation, type TokenPattern } from 'aifn/text/tokenise'
import { porterStem, removeStopWords } from 'aifn/text/stem'
import { bpeEncode, bpeModel, bpeSteps, type BpeState } from 'aifn/text/subword'
import { bagOfWords, bm25, bm25Weights, characterNgrams, featureHash, tfidf, wordNgrams } from 'aifn/text/features'
import type { IdfScheme, NormScheme, TfScheme } from 'aifn/text/features'
import { tokenId } from 'aifn/text/vocabulary'
import { NAMED_CORPORA, namedCorpus, toyCorpus, type CorpusName } from 'aifn-applied/text/corpora'
import { Player } from '@lab/controls'
import { categorical } from '@lab/design/palette'
import { useTheme } from '@lab/design/theme'
import { Columns, Figure } from '@lab/layout'
import { choice, setting, slider, useFigureState, when } from '@lab/state'
import { Textarea } from '@lab/ui/textarea'
import { Input } from '@lab/ui/input'
import { Bars, Curve, Plot, Plots, Points, Raster, Readout, useAxis, DEFAULT_HEIGHT, FrameContext } from '@lab/viz'
import { formatValue } from '@lab/views'

// ── The corpus, shared by every figure on the page ───────────────────────────────────────────────────────────────────

type Source = CorpusName | 'toy corpus' | 'your text'
const SOURCES: Source[] = [...(Object.keys(NAMED_CORPORA) as CorpusName[]), 'toy corpus', 'your text']
type Unit = 'words' | 'wordsAndPunctuation' | 'whitespace' | 'alphanumeric' | 'gpt2' | 'characters'
const UNITS: Unit[] = ['words', 'wordsAndPunctuation', 'whitespace', 'alphanumeric', 'gpt2', 'characters']

const DEFAULT_TEXT = `Tokenisers split text into tokens.
The café's naïve résumé — ﬁne print, ２０２４.
Stemming maps connected, connecting and connection to one stem.`

/** A processed corpus: raw and normalised documents, their tokenisations, and the tokens features are built from. */
interface Processed {
  raw: string[]
  docs: string[]
  tokenised: Tokenisation[]
  /** Word tokens after stop-word removal and stemming, per document: what the features count. */
  terms: string[][]
  /** Word tokens of the normalised text (no stemming), what BPE trains on. */
  words: string[]
}

/** Text with each token's span tinted, alternating two colours; hovering a token shows its offsets. */
function TokenText({ t, highlight }: { t: Tokenisation; highlight?: (token: string, k: number) => boolean }) {
  const { resolved: mode } = useTheme()
  const colours = categorical(mode)
  const o = t.offsets.data
  const parts: ReactNode[] = []
  let at = 0
  t.tokens.forEach((tok, k) => {
    const [s, e] = [o[2 * k], o[2 * k + 1]]
    if (s > at) parts.push(<span key={`g${k}`}>{t.source.slice(at, s)}</span>)
    const hot = highlight?.(tok, k) ?? false
    const c = colours[hot ? 1 : 0]
    parts.push(
      <span
        key={k}
        title={`${JSON.stringify(tok)}  [${s}, ${e})`}
        className="rounded-sm px-px"
        style={{
          background: `color-mix(in srgb, ${c} ${hot ? 45 : k % 2 ? 14 : 26}%, transparent)`,
          outline: hot ? `1px solid ${c}` : undefined,
        }}
      >
        {s === e ? <span className="text-muted-foreground">·</span> : t.source.slice(s, e)}
      </span>,
    )
    at = Math.max(at, e)
  })
  if (at < t.source.length) parts.push(<span key="end">{t.source.slice(at)}</span>)
  return <div className="font-mono text-sm leading-7 whitespace-pre-wrap">{parts}</div>
}

/** A scrolling body that fills the figure's frame height. */
function Body({ children }: { children: ReactNode }) {
  const { height } = useContext(FrameContext)
  return (
    <div className="flex flex-col gap-3 overflow-auto pr-1" style={{ height: height ?? DEFAULT_HEIGHT }}>
      {children}
    </div>
  )
}

function Chips({ items, hot, label }: { items: readonly string[]; hot?: string; label: string }) {
  return (
    <div className="flex flex-wrap items-center gap-1 text-xs">
      <span className="mr-1 text-muted-foreground">{label}</span>
      {items.map((x, i) => (
        <span
          key={i}
          className={
            'rounded border px-1.5 py-0.5 font-mono ' +
            (x === hot ? 'border-primary bg-primary/15 font-semibold' : 'bg-muted')
          }
        >
          {x}
        </span>
      ))}
    </div>
  )
}

// ── Figure 1: corpus, normalisation and tokens ───────────────────────────────────────────────────────────────────────

export function TextPipelineSpecimen() {
  const state = useFigureState({
    corpus: choice(SOURCES, 'cat-sat-on-the-mat', { label: 'corpus' }),
    sentences: slider(5, 200, 30, { label: 'sentences', step: 1, when: when('corpus', 'toy corpus') }),
    seed: slider(0, 99, 1, { label: 'seed', step: 1, when: when('corpus', 'toy corpus') }),
    unit: choice(UNITS, 'words', { label: 'tokeniser' }),
    caseFold: setting(true, 'case folding'),
    stripAccents: setting(false, 'strip accents'),
    stopWords: setting(false, 'drop stop words'),
    stem: setting(false, 'Porter stems'),
  })
  const { corpus, sentences, seed, unit, caseFold, stripAccents, stopWords, stem } = state
  const [typed, setTyped] = useState(DEFAULT_TEXT)

  const processed: Processed = useMemo(() => {
    const raw =
      corpus === 'your text'
        ? typed.split('\n').filter((l) => l.trim().length > 0)
        : corpus === 'toy corpus'
          ? [...toyCorpus(stream(seed), { sentences }).documents]
          : [...namedCorpus({ name: corpus }).documents]
    const docs = raw.map((d) => normalise(d, { caseFold, stripAccents, whitespace: false }))
    const tokenised = docs.map((d) =>
      unit === 'characters' ? characterTokenise(d) : tokenise(d, { pattern: unit as TokenPattern }),
    )
    const wordLists = docs.map((d) => tokenise(d).tokens)
    const terms = wordLists.map((ws) => {
      const kept = stopWords ? removeStopWords(ws) : [...ws]
      return stem ? kept.map(porterStem) : kept
    })
    return { raw, docs, tokenised, terms, words: wordLists.flat() }
  }, [corpus, typed, sentences, seed, unit, caseFold, stripAccents, stopWords, stem])

  const tokens = processed.tokenised.reduce((s, t) => s + t.tokens.length, 0)
  const types = new Set(processed.tokenised.flatMap((t) => t.tokens)).size
  const first = processed.tokenised[0]
  const rows = first ? toRows(first.offsets).slice(0, 40) : []

  return (
    <>
      <Figure
        title="Characters to tokens"
        purpose="A tokeniser is a function from text to spans: each token is a substring with offsets, so the text can always be rebuilt from them."
        state={state}
        defaultSize="L"
        hoverReadout={false}
        readouts={
          <>
            <Readout label="documents" value={processed.docs.length} />
            <Readout label="tokens" value={tokens} />
            <Readout label="distinct tokens" value={types} />
          </>
        }
        caption="Pick a corpus or type your own (one document per line). Normalisation (NFKC, case folding, accent stripping) runs before tokenising; hover a token for its [start, end) offsets in UTF-16 code units. The table lists the first document's tokens with their Porter stems."
      >
        <Body>
          {corpus === 'your text' && (
            <Textarea
              aria-label="your text"
              value={typed}
              onChange={(e) => setTyped(e.target.value)}
              className="font-mono text-xs"
              rows={4}
            />
          )}
          <Columns
            panels={[
              {
                title: 'normalised text, tokens tinted',
                body: (
                  <div className="flex flex-col gap-2">
                    {processed.tokenised.slice(0, 12).map((t, d) => (
                      <TokenText key={d} t={t} />
                    ))}
                    {processed.docs.length > 12 && (
                      <span className="text-xs text-muted-foreground">
                        … {processed.docs.length - 12} more documents
                      </span>
                    )}
                  </div>
                ),
              },
              {
                title: 'first document: tokens, offsets, stems',
                body: (
                  <table className="w-full text-xs tabular-nums">
                    <thead className="text-muted-foreground">
                      <tr>
                        <th className="text-left font-normal">token</th>
                        <th className="text-right font-normal">start</th>
                        <th className="text-right font-normal">end</th>
                        <th className="pl-3 text-left font-normal">stem</th>
                      </tr>
                    </thead>
                    <tbody className="font-mono">
                      {rows.map(([s, e], k) => (
                        <tr key={k}>
                          <td>{JSON.stringify(first!.tokens[k])}</td>
                          <td className="text-right">{s}</td>
                          <td className="text-right">{e}</td>
                          <td className="pl-3">{porterStem(first!.tokens[k])}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                ),
              },
            ]}
          />
        </Body>
      </Figure>
      <BpeFigure processed={processed} />
      <WeightingFigure processed={processed} />
      <NgramFigure processed={processed} />
    </>
  )
}

// ── Figure 2: BPE merges, step by step ───────────────────────────────────────────────────────────────────────────────

function BpeFigure({ processed }: { processed: Processed }) {
  const state = useFigureState({
    merges: slider(5, 200, 40, { label: 'merges', step: 1 }),
    minCount: slider(1, 10, 2, { label: 'min pair count', step: 1 }),
  })
  const { merges, minCount } = state
  const run = useMemo(() => {
    if (processed.words.length === 0) return null
    return trace(bpeSteps(processed.words, { merges, minCount }), undefined, merges + 1)
  }, [processed.words, merges, minCount])
  const count = run ? run.steps!.length : 1
  const [step, setStep] = useState(0)
  const at = Math.min(step, count - 1)
  const s: BpeState | null = run ? run.steps![at] : null
  const model = useMemo(() => (run ? bpeModel(run.final) : null), [run])
  const merge = s && at > 0 ? s.merges[at - 1] : null
  const text = processed.docs.slice(0, 8).join('\n')
  const encoded = useMemo(() => (model ? bpeEncode(model, text, { upTo: at }) : null), [model, text, at])
  const curve = useMemo(
    () => (run ? { x: run.steps!.map((st) => st.t), y: run.steps!.map((st) => st.symbols) } : { x: [], y: [] }),
    [run],
  )
  const stepAxis = useAxis({ label: 'merge', integer: true, hold: 'union', key: run })
  const lenAxis = useAxis({ label: 'corpus length (symbols)', hold: 'union', key: run })
  const recent = s ? s.merges.slice(-12).reverse() : []

  return (
    <Figure
      title="Byte-pair encoding, merge by merge"
      purpose="BPE grows a vocabulary bottom up: each step merges the most frequent adjacent pair, and the corpus gets shorter by that pair's count."
      state={state}
      defaultSize="L"
      hoverReadout={false}
      controls={
        <div className="col-span-full">
          <Player label="merge" value={at} onChange={setStep} count={count} defaultSpeed={4} />
        </div>
      }
      readouts={
        s && (
          <>
            <Readout label="merge" value={merge ? `${merge.left} + ${merge.right} → ${merge.merged}` : 'none yet'} />
            <Readout label="pair count" value={merge ? merge.count : '–'} />
            <Readout label="vocabulary" value={s.vocabulary.length} />
            <Readout label="corpus length" value={s.symbols} />
          </>
        )
      }
      caption="Trained on the word tokens of the corpus above (character level, end-of-word marker </w>). Step 0 is every word split into characters. Each step merges the most frequent adjacent pair (ties: the pair met first); the text shows the first documents encoded with the merges so far, the newest merged symbol outlined. Training stops at the merge budget or when the best pair occurs fewer than the minimum count."
    >
      {!s ? (
        <div className="text-sm text-muted-foreground">No words to train on.</div>
      ) : (
        <Columns
          panels={[
            {
              title: 'the text, encoded with the merges so far',
              body: (
                <Body>
                  {encoded && <TokenText t={encoded} highlight={(tok) => merge !== null && tok === merge.merged} />}
                  <Chips label="newest merges" items={recent.map((m) => `${m.merged} (${m.count})`)} />
                  <Chips label="vocabulary" items={s.vocabulary} hot={merge?.merged} />
                </Body>
              ),
            },
            {
              title: 'corpus length in symbols',
              body: (
                <Plot x={stepAxis} y={lenAxis}>
                  <Curve name="Σ count × symbols" x={curve.x} y={curve.y} slot={0} />
                  <Points name="this step" x={[at]} y={[s.symbols]} emphasis live />
                </Plot>
              ),
            },
          ]}
        />
      )}
    </Figure>
  )
}

// ── Figure 3: TF-IDF and BM25 ────────────────────────────────────────────────────────────────────────────────────────

const TF: TfScheme[] = ['raw', 'log', 'augmented', 'binary', 'logAverage']
const IDF: IdfScheme[] = ['smooth', 'standard', 'standardPlusOne', 'probabilistic', 'none']
const NORM: NormScheme[] = ['l2', 'l1', 'none']

function WeightingFigure({ processed }: { processed: Processed }) {
  const state = useFigureState({
    terms: slider(4, 40, 16, { label: 'terms shown', step: 1 }),
    tf: choice(TF, 'raw', { label: 'tf' }),
    idf: choice(IDF, 'smooth', { label: 'idf' }),
    norm: choice(NORM, 'l2', { label: 'norm' }),
    k1: slider(0, 3, 1.2, { label: 'BM25 k₁', step: 0.1 }),
    b: slider(0, 1, 0.75, { label: 'BM25 b', step: 0.05 }),
    delta: slider(0, 2, 0, { label: 'BM25+ δ', step: 0.1 }),
  })
  const { terms, tf, idf, norm, k1, b, delta } = state
  const docs = processed.terms.slice(0, 30)
  const [query, setQuery] = useState('cat mat')

  const data = useMemo(() => {
    if (docs.length === 0 || docs.every((d) => d.length === 0)) return null
    // The full vocabulary weights; the most frequent terms are shown.
    const bag = bagOfWords(docs)
    const counts = toRows(bag.matrix)
    const totals = bag.vocabulary.tokens.map((_, t) => counts.reduce((s, r) => s + r[t], 0))
    const shown = totals
      .map((n, t) => ({ n, t }))
      .sort((p, q) => q.n - p.n || p.t - q.t)
      .slice(0, terms)
      .sort((p, q) => p.t - q.t)
      .map((e) => e.t)
    const pick = (m: number[][]) => m.map((r) => shown.map((t) => r[t]))
    const w = pick(toRows(tfidf(bag.matrix, { tf, idf, norm })))
    const bw = pick(toRows(bm25Weights(bag.matrix, { k1, b, delta })))
    const qTokens = tokenise(normalise(query)).tokens.map((x) =>
      processed.terms.flat().includes(x) ? x : porterStem(x),
    )
    const q = bag.vocabulary.tokens.map((_, t) => 0 + (qTokens.some((x) => tokenId(bag.vocabulary, x) === t) ? 1 : 0))
    const scores = Array.from(toFlat(bm25(bag.matrix, q, { k1, b, delta })))
    return { names: shown.map((t) => bag.vocabulary.tokens[t]), w, bw, scores }
  }, [docs, terms, tf, idf, norm, k1, b, delta, query, processed.terms])

  const docNames = docs.map((_, d) => `d${d + 1}`)
  const termAxis = useAxis({ label: 'term', categories: data?.names ?? [] })
  const docAxis = useAxis({ label: 'document', categories: docNames })
  const docAxis2 = useAxis({ label: 'document', categories: docNames })
  const scoreAxis = useAxis({ label: 'BM25 score', range: [0, undefined] })
  const termAxis2 = useAxis({ label: 'term', categories: data?.names ?? [] })
  const docAxis3 = useAxis({ label: 'document', categories: docNames })
  const docAxis4 = useAxis({ label: 'document', categories: docNames })
  const lengthAxis = useAxis({ label: 'tokens', range: [0, undefined] })
  const docIdx = docs.map((_, d) => d)

  return (
    <Figure
      title="Bag of words, TF-IDF and BM25"
      purpose="Both weightings scale a term's count by how rare it is across documents; BM25 also saturates repeated terms and corrects for document length."
      state={state}
      defaultSize="L"
      controls={
        <label className="col-span-full flex items-center gap-2 text-sm">
          <span className="text-muted-foreground">BM25 query</span>
          <Input
            aria-label="BM25 query"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            className="h-8 max-w-64"
          />
        </label>
      }
      readouts={
        data && (
          <>
            <Readout label="documents" value={docs.length} />
            <Readout label="best match" value={`d${data.scores.indexOf(Math.max(...data.scores)) + 1}`} />
          </>
        )
      }
      caption="Columns are the most frequent terms of the corpus above (after stop-word removal and stemming, if set there); rows are documents (at most 30). Left: TF-IDF with the chosen term-frequency, IDF and normalisation (raw, smooth, l2 is scikit-learn's default). Right: BM25 term weights, IDF log(1 + (N − df + ½)/(df + ½)) times tf(k₁ + 1)/(tf + k₁(1 − b + b L/L̄)), plus δ for BM25+. Bottom: each document's BM25 score for the query."
    >
      {!data ? (
        <div className="text-sm text-muted-foreground">No terms to weigh.</div>
      ) : (
        <Plots rows={2} cols={2} heights={[3, 1.4]}>
          <Plot x={termAxis} y={docAxis} title="TF-IDF">
            <Raster x={data.names.map((_, j) => j)} y={docIdx} z={data.w} valueLabel="tf-idf" />
          </Plot>
          <Plot x={termAxis2} y={docAxis2} title="BM25 weights">
            <Raster x={data.names.map((_, j) => j)} y={docIdx} z={data.bw} valueLabel="BM25 weight" />
          </Plot>
          <Plot x={docAxis3} y={scoreAxis} title="BM25 score of the query">
            <Bars name="score" x={docIdx} y={data.scores} slot={0} width={0.7} />
          </Plot>
          <Plot x={docAxis4} y={lengthAxis} title="document length">
            <Bars name="length" x={docIdx} y={docs.map((d) => d.length)} muted width={0.7} />
          </Plot>
        </Plots>
      )}
    </Figure>
  )
}

// ── Figure 4: n-grams and hashing ────────────────────────────────────────────────────────────────────────────────────

type NgramUnit = 'word' | 'character' | 'character in words'

function NgramFigure({ processed }: { processed: Processed }) {
  const state = useFigureState({
    unit: choice(['word', 'character', 'character in words'] as NgramUnit[], 'word', { label: 'n-grams of' }),
    lo: slider(1, 5, 2, { label: 'n from', step: 1 }),
    hi: slider(1, 5, 2, { label: 'n to', step: 1 }),
    top: slider(5, 30, 15, { label: 'top', step: 1 }),
    features: slider(4, 64, 16, { label: 'hash columns m', step: 1 }),
    signed: setting(true, 'signed hashing'),
  })
  const { unit, lo, hi, top, features, signed } = state
  const data = useMemo(() => {
    const range: [number, number] = [Math.min(lo, hi), Math.max(lo, hi)]
    const grams = processed.docs.map((d, k) =>
      unit === 'word'
        ? wordNgrams(processed.terms[k], range)
        : characterNgrams(d, range, { wordBoundaries: unit === 'character in words' }),
    )
    const counts = new Map<string, number>()
    for (const g of grams.flat()) counts.set(g, (counts.get(g) ?? 0) + 1)
    const best = [...counts.entries()].sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1)).slice(0, top)
    const hashed = toRows(featureHash(grams.slice(0, 30), { features, signed, norm: 'none' }))
    return { grams, distinct: counts.size, best, hashed }
  }, [processed, unit, lo, hi, top, features, signed])
  const names = data.best.map(([g]) => JSON.stringify(g).slice(1, -1))
  const gramAxis = useAxis({ label: 'n-gram', categories: [...names].reverse() })
  const countAxis = useAxis({ label: 'count', range: [0, undefined] })
  const colAxis = useAxis({ label: 'hashed column', categories: Array.from({ length: features }, (_, j) => String(j)) })
  const docAxis = useAxis({ label: 'document', categories: data.hashed.map((_, d) => `d${d + 1}`) })
  return (
    <Figure
      title="n-grams and feature hashing"
      purpose="n-grams add local order to a bag of words at the price of many more features; hashing caps the feature count at m columns without a vocabulary."
      state={state}
      defaultSize="L"
      readouts={
        <>
          <Readout label="n-grams" value={data.grams.reduce((s, g) => s + g.length, 0)} />
          <Readout label="distinct" value={data.distinct} />
          <Readout label="per column (load)" value={formatValue(data.distinct / features)} />
        </>
      }
      caption="Word n-grams use the terms of the corpus above; character n-grams use the normalised text, either across the whole text or inside space-padded words (scikit-learn's char_wb). Right: every document's n-grams hashed into m columns with MurmurHash3, as HashingVectorizer; with signs, collisions can cancel (cells near zero), without them they only add."
    >
      <Plots cols={2} widths={[1, 1.2]}>
        <Plot x={countAxis} y={gramAxis} title="most frequent n-grams">
          <Bars
            name="count"
            x={names.map((_, k) => names.length - 1 - k)}
            y={data.best.map(([, c]) => c)}
            orient="y"
            slot={0}
            width={0.7}
          />
        </Plot>
        <Plot x={colAxis} y={docAxis} title="hashed counts">
          <Raster
            x={Array.from({ length: features }, (_, j) => j)}
            y={data.hashed.map((_, d) => d)}
            z={data.hashed}
            scale={signed ? 'diverging' : 'sequential'}
            valueLabel="hashed count"
          />
        </Plot>
      </Plots>
    </Figure>
  )
}
