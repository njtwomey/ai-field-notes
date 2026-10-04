import { useMemo, useState } from 'react'
import { stream } from 'aifn/foundation/random'
import { toFlat, toRows, transpose, type Tensor } from 'aifn/foundation/tensor'
import { truncatedSvd } from 'aifn/text/cooccurrence'
import { bagOfWords, characterShingles, oneHotTokens, wordNgrams } from 'aifn/text/features'
import { normalise } from 'aifn/text/normalise'
import {
  cosineMap,
  lsa,
  nearestByCosine,
  randomIndexing,
  termDocumentMatrix,
  termTermMatrix,
  type MatrixWeighting,
} from 'aifn/text/representations'
import { tokenise } from 'aifn/text/tokenise'
import { buildVocabulary, type Vocabulary } from 'aifn/text/vocabulary'
import { namedCorpus, topicCorpus, TOPIC_CORPUS_TOPICS } from 'aifn-methods/text/corpora'
import { seriesColor } from '@lab/design/palette'
import { useTheme } from '@lab/design/theme'
import { Columns, Figure } from '@lab/layout'
import { choice, namedPinField, slider, useFigureState, usePinnedName, when } from '@lab/state'
import { Textarea } from '@lab/ui/textarea'
import { Bars, Curve, Plot, Plots, Points, Raster, Readout, Segments, useAxis } from '@lab/viz'
import { formatValue } from '@lab/views'
import { AnalogyFigure } from './analogies'
import { ShingleFigure } from './shingles'

// ── The corpus ───────────────────────────────────────────────────────────────────────────────────────────────────────

type Source = 'topic corpus' | 'pets' | 'your text'

/** A tokenised corpus with what is known of its topics. */
export interface Corpus {
  /** Normalised documents, one sentence each. */
  docs: string[]
  tokens: readonly (readonly string[])[]
  /** Each document's topic (an index into `topics`), when the corpus was generated with topics. */
  labels: number[] | null
  topics: readonly string[]
  /** Each word's topic name, when known. */
  wordTopics: Readonly<Record<string, string>> | null
}

const DEFAULT_TEXT = `the cat chased the mouse in the garden
the dog chased the cat in the garden
the cook baked the bread in the kitchen
the baker baked the cake in the kitchen
the bus drove to the city
the car drove to the town`

/** Word groups: the topics, then function words, then anything else. */
const WORD_GROUPS = [...TOPIC_CORPUS_TOPICS, 'function']

// ── The representations ──────────────────────────────────────────────────────────────────────────────────────────────

type Rep = 'one-hot' | 'bag of words' | 'n-grams' | 'shingles' | 'term–document' | 'co-occurrence' | 'random indexing'
const REPS: Rep[] = [
  'one-hot',
  'bag of words',
  'n-grams',
  'shingles',
  'term–document',
  'co-occurrence',
  'random indexing',
]
const WEIGHTINGS: MatrixWeighting[] = ['raw', 'binary', 'log', 'tfidf', 'ppmi']
type Side = 'both' | 'left' | 'right'
type Distance = 'uniform' | 'hal' | 'harmonic'
type Sort = 'topic' | 'frequency' | 'alphabetical' | 'clustered'

/** What a representation gives: a matrix whose rows are the items embedded, and names for its rows and columns. */
export interface Model {
  rep: Rep
  /** Items × features (float64 [m, n]). */
  matrix: Tensor
  itemKind: 'word' | 'document'
  items: readonly string[]
  /** The group of each item: an index into `groupNames`. */
  groups: number[]
  groupNames: readonly string[]
  features: readonly string[]
  /** The vocabulary of the rows, when they are words (for analogies). */
  vocabulary: Vocabulary | null
  /** Signed entries (random indexing): a diverging colour scale. */
  signed: boolean
  /** One-hot only: the vocabulary × positions matrix of the first documents, and its position labels. */
  positions?: { matrix: number[][]; tokens: string[] }
}

function wordGroups(words: readonly string[], corpus: Corpus): { groups: number[]; groupNames: readonly string[] } {
  if (!corpus.wordTopics) return { groups: words.map(() => 0), groupNames: ['words'] }
  // An 'other' group only when some word has no known topic.
  const known = words.every((w) => WORD_GROUPS.includes(corpus.wordTopics![w] ?? ''))
  const names = known ? WORD_GROUPS : [...WORD_GROUPS, 'other']
  return {
    groups: words.map((w) => {
      const k = names.indexOf(corpus.wordTopics![w] ?? '')
      return k < 0 ? names.length - 1 : k
    }),
    groupNames: names,
  }
}

function documentGroups(corpus: Corpus): { groups: number[]; groupNames: readonly string[] } {
  return corpus.labels
    ? { groups: corpus.labels, groupNames: corpus.topics }
    : { groups: corpus.docs.map(() => 0), groupNames: ['documents'] }
}

/** Rows of a matrix as numbers, keeping only some columns. */
const pickColumns = (m: number[][], cols: readonly number[]) => m.map((r) => cols.map((j) => r[j]))

// ── The page ─────────────────────────────────────────────────────────────────────────────────────────────────────────

export function RepresentationsShowcase() {
  const state = useFigureState({
    corpus: choice(['topic corpus', 'pets', 'your text'] as Source[], 'topic corpus', { label: 'corpus' }),
    sentences: slider(50, 800, 300, { label: 'sentences', step: 10, when: when('corpus', 'topic corpus') }),
    seed: slider(0, 99, 0, { label: 'seed', step: 1, when: when('corpus', 'topic corpus') }),
    rep: choice(REPS, 'co-occurrence', { label: 'representation' }),
    items: choice(['words', 'documents'] as const, 'words', {
      label: 'embed',
      when: (v) => v.rep === 'term–document' || v.rep === 'bag of words',
    }),
    tdWeighting: choice(WEIGHTINGS, 'tfidf', { label: 'weighting', when: when('rep', 'term–document') }),
    ccWeighting: choice(WEIGHTINGS, 'ppmi', { label: 'weighting', when: when('rep', 'co-occurrence') }),
    window: slider(1, 8, 2, {
      label: 'window',
      step: 1,
      when: (v) => v.rep === 'co-occurrence' || v.rep === 'random indexing',
    }),
    side: choice(['both', 'left', 'right'] as Side[], 'both', {
      label: 'sides',
      when: (v) => v.rep === 'co-occurrence' || v.rep === 'random indexing',
    }),
    distance: choice(['uniform', 'hal', 'harmonic'] as Distance[], 'uniform', {
      label: 'by distance',
      when: (v) => v.rep === 'co-occurrence' || v.rep === 'random indexing',
    }),
    n: slider(1, 3, 2, { label: 'n', step: 1, when: when('rep', 'n-grams') }),
    shingle: slider(2, 8, 4, { label: 'shingle k', step: 1, when: when('rep', 'shingles') }),
    dimensions: slider(8, 256, 64, { label: 'index dimensions', step: 8, when: when('rep', 'random indexing') }),
    nonZeros: slider(2, 10, 4, { label: 'non-zeros', step: 2, when: when('rep', 'random indexing') }),
    k: slider(2, 40, 10, { label: 'rank k', step: 1, when: (v) => v.rep !== 'one-hot' }),
    shown: slider(20, 150, 70, { label: 'items shown', step: 5 }),
    pin: namedPinField(),
  })
  const { corpus: source, sentences, seed, rep, items, tdWeighting, ccWeighting, window, side, distance } = state
  const { n, shingle, dimensions, nonZeros, k, shown } = state
  const [typed, setTyped] = useState(DEFAULT_TEXT)
  const { resolved: mode } = useTheme()

  const corpus: Corpus = useMemo(() => {
    if (source === 'topic corpus') {
      const c = topicCorpus(stream(seed), { sentences })
      const docs = [...c.documents]
      return {
        docs,
        tokens: docs.map((d) => tokenise(d).tokens),
        labels: [...c.labels!],
        topics: c.meta.labelNames!,
        wordTopics: c.wordTopics!,
      }
    }
    const raw = source === 'pets' ? [...namedCorpus({ name: 'pets' }).documents] : typed.split('\n')
    const docs = raw.map((d) => normalise(d, { caseFold: true })).filter((d) => d.trim().length > 0)
    return { docs, tokens: docs.map((d) => tokenise(d).tokens), labels: null, topics: [], wordTopics: null }
  }, [source, sentences, seed, typed])

  const model: Model | null = useMemo(() => {
    const docs = corpus.tokens
    if (docs.length < 2 || docs.every((d) => d.length === 0)) return null
    const vocab = buildVocabulary(docs, { specials: [] })
    const words = vocab.tokens
    const windowOptions = {
      window,
      left: side === 'right' ? 0 : window,
      right: side === 'left' ? 0 : window,
      distance,
    }
    const wordModel = (matrix: Tensor, features: readonly string[], signed = false): Model => ({
      rep,
      matrix,
      itemKind: 'word',
      items: words,
      ...wordGroups(words, corpus),
      features,
      vocabulary: vocab,
      signed,
    })
    const docModel = (matrix: Tensor, features: readonly string[]): Model => ({
      rep,
      matrix,
      itemKind: 'document',
      items: corpus.docs,
      ...documentGroups(corpus),
      features,
      vocabulary: null,
      signed: false,
    })
    switch (rep) {
      case 'one-hot': {
        // Each word is a row of the identity; the matrix view shows the first documents' tokens one-hot.
        const first = docs.slice(0, 6).flat()
        return {
          ...wordModel(oneHotTokens(words, vocab), words),
          positions: { matrix: toRows(oneHotTokens(first, vocab)), tokens: first },
        }
      }
      case 'bag of words':
      case 'term–document': {
        const weighting = rep === 'bag of words' ? 'raw' : tdWeighting
        const td = termDocumentMatrix(docs, { terms: vocab, weighting })
        if (items === 'documents') return docModel(transpose(td.matrix) as Tensor, words)
        return wordModel(td.matrix, td.columns)
      }
      case 'n-grams': {
        const grams = docs.map((d) => wordNgrams(d, [n, n]))
        const bag = bagOfWords(grams, { binary: false })
        return docModel(bag.matrix, bag.vocabulary.tokens)
      }
      case 'shingles': {
        const sets = corpus.docs.map((d) => characterShingles(d, shingle))
        const bag = bagOfWords(sets, { binary: true })
        return docModel(bag.matrix, bag.vocabulary.tokens)
      }
      case 'co-occurrence': {
        if (ccWeighting === 'ppmi' && docs.every((d) => d.length < 2)) return null
        const tt = termTermMatrix(docs, { terms: vocab, weighting: ccWeighting, ...windowOptions })
        return wordModel(tt.matrix, tt.columns)
      }
      case 'random indexing': {
        const ri = randomIndexing(docs, {
          words: vocab,
          dimensions,
          nonZeros,
          window,
          left: windowOptions.left,
          right: windowOptions.right,
          weighting: distance,
        })
        return wordModel(
          ri.vectors,
          Array.from({ length: dimensions }, (_, j) => `r${j + 1}`),
          true,
        )
      }
    }
  }, [corpus, rep, items, tdWeighting, ccWeighting, window, side, distance, n, shingle, dimensions, nonZeros])

  // The scree: the leading singular values of the whole matrix (independent of k).
  const scree = useMemo(() => {
    if (!model) return null
    const [m, nf] = model.matrix.shape
    const K = Math.min(40, m, nf)
    if (model.rep === 'one-hot')
      return { values: Array.from({ length: K }, () => 1), energy: Array.from({ length: K }, () => 1 / m) }
    const t = truncatedSvd(model.matrix, K)
    return { values: Array.from(toFlat(t.S)), energy: Array.from(toFlat(t.energy)) }
  }, [model])

  // Rank-k coordinates of every item, and the plane map of the items shown.
  const embedded = useMemo(() => {
    if (!model) return null
    const [m, nf] = model.matrix.shape
    const count = Math.min(shown, m)
    if (model.rep === 'one-hot') {
      // No plane keeps V points equidistant: they are drawn on a circle, and the vectors stay the identity's rows.
      const angle = (i: number) => (2 * Math.PI * i) / count
      return {
        vectors: model.matrix,
        rank: m,
        x: Array.from({ length: count }, (_, i) => Math.cos(angle(i))),
        y: Array.from({ length: count }, (_, i) => Math.sin(angle(i))),
        count,
      }
    }
    const rank = Math.min(k, m, nf)
    const vectors = lsa(model.matrix, rank).rows
    const rows = toRows(vectors).slice(0, count)
    const plane = rank >= 2 ? toRows(cosineMap(rows, 2)) : rows.map((r) => [r[0], 0])
    return { vectors, rank, x: plane.map((p) => p[0]), y: plane.map((p) => p[1]), count }
  }, [model, k, shown])

  // The pin is a word (or document), so it survives a change of representation; its index is in the URL.
  const pins = usePinnedName(state.pin, (i) => state.set('pin', i), model?.items ?? null, 'cat')
  const pinned = pins.name
  const pinnedIndex = pins.pinned ?? -1
  const neighbours = useMemo(
    () => (embedded && pinnedIndex >= 0 ? nearestByCosine(embedded.vectors, pinnedIndex, { count: 10 }) : []),
    [embedded, pinnedIndex],
  )

  const togglePin = (item: string) => pins.toggle(model ? model.items.indexOf(item) : -1)
  const pickNearest = ([px, py]: [number, number]) => {
    if (!embedded || !model) return
    const sx = Math.max(...embedded.x) - Math.min(...embedded.x) || 1
    const sy = Math.max(...embedded.y) - Math.min(...embedded.y) || 1
    let best = -1
    let bestD = Infinity
    for (let i = 0; i < embedded.count; i++) {
      const d = ((embedded.x[i] - px) / sx) ** 2 + ((embedded.y[i] - py) / sy) ** 2
      if (d < bestD) [best, bestD] = [i, d]
    }
    // A click far from every point unpins.
    if (best < 0 || bestD > 0.03 ** 2) pins.clear()
    else togglePin(model.items[best])
  }

  const label = (i: number) => (model?.itemKind === 'document' ? `d${i + 1}` : (model?.items[i] ?? ''))
  const xAxis = useAxis({ label: 'map axis 1', key: [model, k], nice: false })
  const yAxis = useAxis({ label: 'map axis 2', key: [model, k], nice: false, equal: xAxis })
  const shownIdx = embedded ? Array.from({ length: embedded.count }, (_, i) => i) : []
  const strong = pinnedIndex >= 0 && pinnedIndex < (embedded?.count ?? 0) ? [pinnedIndex] : []
  const kept = embedded ? embedded.count : 0
  const energyAtK = scree && embedded ? scree.energy.slice(0, embedded.rank).reduce((s, e) => s + e, 0) : 0

  return (
    <>
      <Figure
        title="Words as vectors"
        purpose="A word is known by the company it keeps: rows of a co-occurrence or term–document matrix, compressed by a truncated SVD, put words of one topic or one syntactic role together; one-hot rows put every word at the same distance from every other."
        state={state}
        defaultSize="L"
        hoverReadout={false}
        controls={
          source === 'your text' ? (
            <Textarea
              aria-label="your text"
              value={typed}
              onChange={(e) => setTyped(e.target.value)}
              className="col-span-full font-mono text-xs"
              rows={4}
            />
          ) : undefined
        }
        readouts={
          model && (
            <>
              <Readout label="documents" value={corpus.docs.length} />
              <Readout label="matrix" value={`${model.matrix.shape[0]} × ${model.matrix.shape[1]}`} />
              {rep === 'one-hot' ? (
                <Readout label="distance between any two words" value="√2 (cosine 0)" />
              ) : (
                <Readout
                  label={`energy in the first ${embedded?.rank ?? 0} components`}
                  value={`${(100 * energyAtK).toFixed(1)}%`}
                />
              )}
              {pinned && <Readout label="pinned" value={pinned} />}
            </>
          )
        }
        caption="The corpus: generated sentences about animals, food, vehicles, colours and places, all built from the same frames ('the cat chases the mouse', 'some buses stop near the market', 'the painter paints the wall red'); or the pets sentences, or your own text (one document per line). Each representation turns the corpus into a matrix whose rows are the items (words, or documents for n-grams and shingles). The map shows the items' rank-k vectors (the truncated SVD, U_k Σ_k) by their cosine geometry: unit-length rows, centred, on their top two principal axes. Colours and shapes are true topics (function words: the, a, some, in, near, to). Labels that would overlap are hidden, the most frequent words kept; hover a point to read its word. Click a word in the map or in the list to pin it and see its nearest neighbours by cosine at rank k; click it again, empty space or press Escape to unpin. Drag rank k and the window: at k = 2 the map is crowded, around k = 10 topics and roles separate, at large k noise returns. With one-hot, every pair of words is at distance √2 with cosine 0, so the neighbour list is a tie and no plane can show the geometry (drawn on a circle)."
      >
        {!model || !embedded ? (
          <div className="text-sm text-muted-foreground">
            Too little text: at least two non-empty documents are needed.
          </div>
        ) : (
          <Columns
            widths={[2.4, 1]}
            panels={[
              {
                title:
                  rep === 'one-hot'
                    ? 'one-hot words: all equidistant (drawn on a circle)'
                    : `the ${model.itemKind === 'word' ? 'words' : 'documents'} at rank ${embedded.rank}, by cosine (first ${kept})`,
                body: (
                  <Plot x={xAxis} y={yAxis} onPlotClick={pickNearest}>
                    {rep === 'one-hot' && strong.length > 0 && (
                      <Segments
                        segments={shownIdx
                          .filter((i) => i !== pinnedIndex)
                          .map((i) => ({
                            from: [embedded.x[pinnedIndex], embedded.y[pinnedIndex]] as const,
                            to: [embedded.x[i], embedded.y[i]] as const,
                          }))}
                      />
                    )}
                    <Points
                      name={model.itemKind}
                      x={embedded.x}
                      y={embedded.y}
                      group={model.groups.slice(0, kept)}
                      groupNames={model.groupNames}
                      thin={kept > 100}
                      labels={shownIdx.map((i) => (i === pinnedIndex ? null : label(i)))}
                    />
                    {strong.length > 0 && (
                      <Points
                        name="pinned"
                        x={[embedded.x[pinnedIndex]]}
                        y={[embedded.y[pinnedIndex]]}
                        labels={[label(pinnedIndex)]}
                        emphasis
                        live
                      />
                    )}
                  </Plot>
                ),
              },
              {
                title: pinned
                  ? `nearest to “${model.itemKind === 'document' ? label(pinnedIndex) : pinned}” by cosine`
                  : 'nearest neighbours',
                body: (
                  <NeighbourList
                    model={model}
                    pinnedIndex={pinnedIndex}
                    neighbours={neighbours}
                    label={label}
                    onPick={(i) => togglePin(model.items[i])}
                    mode={mode}
                  />
                ),
              },
            ]}
          />
        )}
      </Figure>
      {model && embedded && scree && (
        <MatrixFigure
          model={model}
          embedded={embedded}
          scree={scree}
          corpus={corpus}
          pinnedIndex={pinnedIndex}
          onPin={(i) => togglePin(model.items[i])}
        />
      )}
      <ShingleFigure corpus={corpus} />
      {model && embedded && <AnalogyFigure model={model} vectors={embedded.vectors} rank={embedded.rank} />}
    </>
  )
}

// ── The neighbour list ───────────────────────────────────────────────────────────────────────────────────────────────

function NeighbourList({
  model,
  pinnedIndex,
  neighbours,
  label,
  onPick,
  mode,
}: {
  model: Model
  pinnedIndex: number
  neighbours: readonly { index: number; cosine: number }[]
  label: (i: number) => string
  onPick: (i: number) => void
  mode: 'light' | 'dark'
}) {
  if (pinnedIndex < 0)
    return <div className="text-xs text-muted-foreground">Click a point in the map, or a word below, to pin it.</div>
  const colour = (i: number) => seriesColor(mode, Math.min(model.groups[i], 7))
  return (
    <div className="flex h-full flex-col gap-1 overflow-auto pr-1 text-xs">
      {model.itemKind === 'document' && (
        <div className="mb-1 font-mono text-[11px] text-muted-foreground">{model.items[pinnedIndex]}</div>
      )}
      {neighbours.map(({ index, cosine }) => (
        <button
          key={index}
          type="button"
          onClick={() => onPick(index)}
          className="flex items-center gap-2 rounded px-1 py-0.5 text-left hover:bg-muted"
          title={model.itemKind === 'document' ? model.items[index] : `${model.groupNames[model.groups[index]]}`}
        >
          <span aria-hidden className="size-2 shrink-0 rounded-full" style={{ background: colour(index) }} />
          <span className="w-24 shrink-0 truncate font-mono">{label(index)}</span>
          <span className="relative h-2 flex-1 rounded bg-muted">
            <span
              className="absolute inset-y-0 left-0 rounded"
              style={{ width: `${Math.max(0, cosine) * 100}%`, background: colour(index) }}
            />
          </span>
          <span className="w-10 text-right font-mono tabular-nums">{cosine.toFixed(2)}</span>
        </button>
      ))}
    </div>
  )
}

// ── The matrix and the scree ─────────────────────────────────────────────────────────────────────────────────────────

const MAX_COLUMNS = 120

function MatrixFigure({
  model,
  embedded,
  scree,
  corpus,
  pinnedIndex,
  onPin,
}: {
  model: Model
  embedded: { x: number[]; y: number[]; count: number; rank: number }
  scree: { values: number[]; energy: number[] }
  corpus: Corpus
  pinnedIndex: number
  onPin: (i: number) => void
}) {
  const state = useFigureState({
    sort: choice(['topic', 'frequency', 'alphabetical', 'clustered'] as Sort[], 'topic', { label: 'sort rows' }),
    rows: slider(10, 120, 60, { label: 'rows', step: 5 }),
  })
  const { sort, rows } = state

  const view = useMemo(() => {
    const count = Math.min(rows, embedded.count)
    const idx = Array.from({ length: count }, (_, i) => i)
    const angle = (i: number) => Math.atan2(embedded.y[i], embedded.x[i])
    const order =
      sort === 'frequency'
        ? idx
        : sort === 'alphabetical'
          ? [...idx].sort((a, b) => (model.items[a] < model.items[b] ? -1 : model.items[a] > model.items[b] ? 1 : 0))
          : sort === 'topic'
            ? [...idx].sort((a, b) => model.groups[a] - model.groups[b] || a - b)
            : [...idx].sort((a, b) => angle(a) - angle(b))
    const full = toRows(model.matrix)
    let columns: number[]
    let names: string[]
    let z: number[][]
    if (model.positions) {
      // One-hot: the first documents' tokens as columns.
      names = model.positions.tokens
      columns = names.map((_, j) => j)
      z = order.map((i) => model.positions!.matrix[i])
    } else if (model.rep === 'co-occurrence') {
      // The contexts are the same words: order the columns like the rows, so blocks sit on the diagonal.
      columns = order
      names = columns.map((j) => model.features[j])
      z = pickColumns(
        order.map((i) => full[i]),
        columns,
      )
    } else if (model.itemKind === 'word' && model.rep !== 'random indexing' && corpus.labels) {
      // Documents as columns, grouped by topic.
      const docs = corpus.labels.map((_, d) => d).slice(0, MAX_COLUMNS * 2)
      columns = (
        sort === 'topic' || sort === 'clustered'
          ? [...docs].sort((a, b) => corpus.labels![a] - corpus.labels![b] || a - b)
          : docs
      ).slice(0, MAX_COLUMNS * 2)
      names = columns.map((j) => model.features[j])
      z = pickColumns(
        order.map((i) => full[i]),
        columns,
      )
    } else {
      // The heaviest columns.
      const weight = model.features.map((_, j) => full.reduce((s, r) => s + Math.abs(r[j]), 0))
      columns = weight
        .map((w, j) => ({ w, j }))
        .sort((a, b) => b.w - a.w || a.j - b.j)
        .slice(0, model.rep === 'random indexing' ? model.features.length : MAX_COLUMNS)
        .map((e) => e.j)
        .sort((a, b) => a - b)
      names = columns.map((j) => model.features[j])
      z = pickColumns(
        order.map((i) => full[i]),
        columns,
      )
    }
    return { order, names, z }
  }, [model, embedded, sort, rows, corpus.labels])

  const itemName = (i: number) => (model.itemKind === 'document' ? `d${i + 1}` : model.items[i])
  const rowNames = view.order.map(itemName)
  const colAxis = useAxis({
    label: model.positions
      ? 'position in the first documents'
      : model.itemKind === 'word' && model.rep !== 'co-occurrence' && model.rep !== 'random indexing'
        ? 'document'
        : 'feature',
    format: (v) => view.names[Math.round(v)] ?? '',
    nice: false,
    zoom: false,
  })
  const rowAxis = useAxis({ label: model.itemKind, categories: [...rowNames].reverse() })
  const compAxis = useAxis({ label: 'component', integer: true })
  const shareAxis = useAxis({ label: 'share of ‖A‖²_F', range: [0, 1] })
  const at = view.order.indexOf(pinnedIndex)
  const R = view.order.length
  const C = view.names.length
  const ys = view.order.map((_, r) => R - 1 - r)
  const zRev = [...view.z].reverse()
  const cumulative = scree.energy.reduce<number[]>((acc, e) => [...acc, (acc[acc.length - 1] ?? 0) + e], [])
  const comps = scree.energy.map((_, i) => i + 1)

  return (
    <Figure
      title="The matrix and its spectrum"
      purpose="Every representation is a matrix; the singular values say how much of it a rank-k approximation keeps, and the scree's elbow is where structure gives way to noise."
      state={state}
      defaultSize="L"
      readouts={
        <>
          <Readout label="rows shown" value={`${R} of ${model.matrix.shape[0]}`} />
          <Readout label="columns shown" value={`${C} of ${model.matrix.shape[1]}`} />
          <Readout label="σ₁" value={formatValue(scree.values[0] ?? 0)} />
          <Readout label="kept at k" value={`${(100 * (cumulative[embedded.rank - 1] ?? 1)).toFixed(1)}%`} />
        </>
      }
      caption="Left: the matrix itself, rows the most frequent items, columns documents (term–document, grouped by topic), context words (co-occurrence, in the rows' order, so topical blocks sit on the diagonal), the most common n-grams or shingles, random-index dimensions (signed), or, for one-hot, the positions of the first documents' tokens (one 1 per column). Sort the rows by topic, frequency, alphabet, or 'clustered' (by angle in the map above). Click a row to pin its item; the pinned row is outlined. Right: the leading singular values as shares σᵢ²/‖A‖²_F (bars; the first k in colour) and their running total (line). One-hot's identity has every singular value equal to 1: no direction matters more than another."
    >
      <Plots cols={2} widths={[2.4, 1]}>
        <Plot
          x={colAxis}
          y={rowAxis}
          title={`${model.rep}${model.rep === 'term–document' || model.rep === 'co-occurrence' ? '' : ''}`}
          onPlotClick={([, y]) => {
            const r = R - 1 - Math.round(y)
            if (r >= 0 && r < R) onPin(view.order[r])
          }}
        >
          <Raster
            x={view.names.map((_, j) => j)}
            y={ys}
            z={zRev}
            scale={model.signed ? 'diverging' : 'sequential'}
            valueLabel="weight"
          />
          {at >= 0 && (
            <Segments
              emphasis
              width={1.5}
              segments={[
                { from: [-0.5, R - 1 - at - 0.5], to: [C - 0.5, R - 1 - at - 0.5] },
                { from: [-0.5, R - 1 - at + 0.5], to: [C - 0.5, R - 1 - at + 0.5] },
              ]}
            />
          )}
        </Plot>
        <Plot x={compAxis} y={shareAxis} title="scree">
          <Bars
            name="share, first k"
            x={comps.slice(0, embedded.rank)}
            y={scree.energy.slice(0, embedded.rank)}
            slot={0}
            width={0.7}
          />
          <Bars
            name="share, beyond k"
            x={comps.slice(embedded.rank)}
            y={scree.energy.slice(embedded.rank)}
            muted
            width={0.7}
          />
          <Curve name="running total" x={comps} y={cumulative} emphasis />
        </Plot>
      </Plots>
    </Figure>
  )
}
