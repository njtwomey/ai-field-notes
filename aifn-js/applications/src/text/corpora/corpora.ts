/**
 * Toy text corpora for the text pipeline: the small corpora the notes work by hand (the five "cat sat on the mat"
 * documents, the thirteen pet sentences, Sennrich's "low lower newest widest", the Hugging Face WordPiece corpus), and a
 * seeded generator of short sentences about a few topics, with English inflection (plurals, -ed, -ing with doubled
 * consonants) so that stemming and subword merges have something to find.
 */

import type { DatasetInfo, DatasetMeta } from 'aifn/foundation/contracts'
import { child, uniform, type Stream } from 'aifn/foundation/random'
import { definer, type Entry } from 'aifn/foundation/registry'
import { int, oneOf, real, space } from 'aifn/foundation/space'
import { DomainError } from 'aifn/foundation/errors'

/** A corpus: documents of raw text (one sentence or paragraph each) and what they are. */
export interface Corpus {
  readonly kind: 'corpus'
  readonly documents: readonly string[]
  readonly meta: DatasetMeta
}

const repeat = (word: string, n: number) => Array.from({ length: n }, () => word)

/** The named corpora of {@link namedCorpus}. */
export const NAMED_CORPORA = {
  'cat-sat-on-the-mat': {
    description: 'Five short documents about cats and dogs, the worked example of the TF-IDF and BM25 notes.',
    documents: [
      'the cat sat on the mat',
      'the dog sat on the log',
      'the cat chased the dog',
      'a dog and a cat played',
      'the bird sang',
    ],
  },
  pets: {
    description: 'Thirteen sentences about pets, the worked example of the co-occurrence and PMI note.',
    documents: [
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
    ],
  },
  'low-lower-newest-widest': {
    description: 'Sennrich et al. (2016)’s BPE corpus: low ×5, lower ×2, newest ×6, widest ×3.',
    documents: [[...repeat('low', 5), ...repeat('lower', 2), ...repeat('newest', 6), ...repeat('widest', 3)].join(' ')],
  },
  'hug-pug-pun': {
    description: 'The Hugging Face course corpus for WordPiece: hug ×10, pug ×5, pun ×12, bun ×4, hugs ×5.',
    documents: [
      [...repeat('hug', 10), ...repeat('pug', 5), ...repeat('pun', 12), ...repeat('bun', 4), ...repeat('hugs', 5)].join(
        ' ',
      ),
    ],
  },
} as const

/** A named corpus. */
export type CorpusName = keyof typeof NAMED_CORPORA

/** One of the hand-worked corpora of the notes. */
export function namedCorpus(knobs: { name?: CorpusName } = {}): Corpus {
  const name = knobs.name ?? 'cat-sat-on-the-mat'
  const c = NAMED_CORPORA[name]
  if (!c) throw new DomainError('namedCorpus', `namedCorpus: unknown corpus '${String(name)}'`)
  return { kind: 'corpus', documents: [...c.documents], meta: { name, description: c.description, task: 'text' } }
}

// ── A seeded sentence generator ──────────────────────────────────────────────────────────────────────────────────────

type Noun = readonly [singular: string, plural: string]
/** A verb: base form, third person singular, past, -ing form. */
type Verb = readonly [string, string, string, string]

interface Topic {
  readonly nouns: readonly Noun[]
  readonly verbs: readonly Verb[]
  readonly adjectives: readonly string[]
  readonly places: readonly string[]
}

const TOPICS: Readonly<Record<string, Topic>> = {
  pets: {
    nouns: [
      ['cat', 'cats'],
      ['dog', 'dogs'],
      ['mouse', 'mice'],
      ['bird', 'birds'],
      ['puppy', 'puppies'],
      ['fox', 'foxes'],
    ],
    verbs: [
      ['chase', 'chases', 'chased', 'chasing'],
      ['watch', 'watches', 'watched', 'watching'],
      ['hop', 'hops', 'hopped', 'hopping'],
      ['sleep', 'sleeps', 'slept', 'sleeping'],
      ['play', 'plays', 'played', 'playing'],
    ],
    adjectives: ['small', 'lazy', 'quick', 'hungry', 'happy'],
    places: ['garden', 'kitchen', 'mat', 'basket'],
  },
  food: {
    nouns: [
      ['apple', 'apples'],
      ['loaf', 'loaves'],
      ['cheese', 'cheeses'],
      ['berry', 'berries'],
      ['dish', 'dishes'],
      ['cook', 'cooks'],
    ],
    verbs: [
      ['bake', 'bakes', 'baked', 'baking'],
      ['eat', 'eats', 'ate', 'eating'],
      ['cut', 'cuts', 'cut', 'cutting'],
      ['taste', 'tastes', 'tasted', 'tasting'],
      ['serve', 'serves', 'served', 'serving'],
    ],
    adjectives: ['fresh', 'sweet', 'warm', 'ripe', 'salty'],
    places: ['kitchen', 'market', 'table', 'oven'],
  },
  weather: {
    nouns: [
      ['cloud', 'clouds'],
      ['storm', 'storms'],
      ['wind', 'winds'],
      ['river', 'rivers'],
      ['valley', 'valleys'],
      ['tree', 'trees'],
    ],
    verbs: [
      ['cover', 'covers', 'covered', 'covering'],
      ['flood', 'floods', 'flooded', 'flooding'],
      ['shake', 'shakes', 'shook', 'shaking'],
      ['cool', 'cools', 'cooled', 'cooling'],
      ['drift', 'drifts', 'drifted', 'drifting'],
    ],
    adjectives: ['cold', 'dark', 'heavy', 'stormy', 'gentle'],
    places: ['hills', 'valley', 'coast', 'town'],
  },
}

/** The topics of {@link toyCorpus}. */
export const CORPUS_TOPICS = Object.keys(TOPICS)

/** Options of {@link toyCorpus}. */
export interface ToyCorpusOptions {
  /** The number of sentences (default 40). */
  sentences?: number
  /** How many of the topics (pets, food, weather) to mix, from the first (default 3). */
  topics?: number
  /**
   * Zipf exponent of word choice within a topic (default 1): word k of a list is picked with probability ∝ (k + 1)^−s,
   * so a few words dominate as in real text.
   */
  exponent?: number
}

const pick = <T>(xs: readonly T[], u: number, exponent: number): T => {
  const w = xs.map((_, k) => (k + 1) ** -exponent)
  let r = u * w.reduce((a, b) => a + b, 0)
  for (let k = 0; k < xs.length; k++) if ((r -= w[k]) < 0) return xs[k]
  return xs[xs.length - 1]
}

/**
 * A seeded toy corpus: each sentence is drawn from one topic as "the [adjective] noun(s) verb(s|ed|ing) the noun(s) in
 * the place", with determiners, number and tense drawn too. Sentence k depends only on `child(s, k)`, so a longer
 * corpus extends a shorter one.
 */
export function toyCorpus(s: Stream, options: ToyCorpusOptions = {}): Corpus {
  const { sentences = 40, topics = 3, exponent = 1 } = options
  if (!(Number.isInteger(sentences) && sentences >= 1)) throw new DomainError('toyCorpus', 'toyCorpus: sentences ≥ 1')
  const names = CORPUS_TOPICS.slice(0, Math.max(1, Math.min(topics, CORPUS_TOPICS.length)))
  const documents: string[] = []
  for (let k = 0; k < sentences; k++) {
    const u = uniform(child(s, k), 0, 1, { shape: [12] }).data
    const topic = TOPICS[names[Math.min(names.length - 1, Math.floor(u[0] * names.length))]]
    const plural = u[1] < 0.35
    const subject = pick(topic.nouns, u[2], exponent)
    const object = pick(topic.nouns, u[3], exponent)
    const verb = pick(topic.verbs, u[4], exponent)
    const tense = u[5] < 0.45 ? 'past' : u[5] < 0.75 ? 'present' : 'progressive'
    const words: string[] = [u[6] < 0.7 ? 'the' : plural ? 'some' : 'a']
    if (u[7] < 0.5) words.push(pick(topic.adjectives, u[8], exponent))
    words.push(plural ? subject[1] : subject[0])
    if (tense === 'past') words.push(verb[2])
    else if (tense === 'present') words.push(plural ? verb[0] : verb[1])
    else words.push(plural ? 'are' : 'is', verb[3])
    words.push('the', u[9] < 0.5 ? object[0] : object[1])
    if (u[10] < 0.6) words.push(u[11] < 0.5 ? 'in' : 'near', 'the', pick(topic.places, u[11], exponent))
    documents.push(words.join(' '))
  }
  return {
    kind: 'corpus',
    documents,
    meta: {
      name: 'toy corpus',
      description: `${sentences} generated sentences about ${names.join(', ')}, with plurals and verb inflections.`,
      task: 'text',
    },
  }
}

const dataset = definer<DatasetInfo>('dataset', 'text/corpora')

dataset(
  {
    key: 'namedCorpus',
    name: 'Worked-example corpora',
    summary: 'The small corpora the text notes work by hand.',
    task: 'text',
    output: 'corpus',
    knobs: space({ name: oneOf(Object.keys(NAMED_CORPORA) as CorpusName[]) }),
    truth: false,
    notes: [
      'term-frequency-inverse-document-frequency',
      'co-occurrence-matrices-and-pointwise-mutual-information',
      'byte-pair-encoding',
      'wordpiece-and-unigram-tokenisation',
    ],
    cite: ['sennrich2016', 'huggingface2024wordpiece'],
  },
  namedCorpus,
)

dataset(
  {
    key: 'toyCorpus',
    name: 'Toy corpus',
    summary: 'Seeded sentences about pets, food and weather, with plurals and verb inflections.',
    task: 'text',
    output: 'corpus',
    knobs: space({
      sentences: int(1, 2000, { default: 40 }),
      topics: int(1, 3, { default: 3 }),
      exponent: real(0, 3, { default: 1 }),
    }),
    truth: false,
    random: true,
    notes: ['text-representation-pipeline', 'tokenisation', 'bag-of-words'],
  },
  toyCorpus,
)

/** The corpus generators, keyed by name. */
export const corpusDatasets = { namedCorpus, toyCorpus } as unknown as Readonly<
  Record<string, Entry<(...args: never[]) => unknown, DatasetInfo>>
>
