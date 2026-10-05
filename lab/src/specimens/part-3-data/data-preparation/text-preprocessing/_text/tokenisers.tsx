import { useMemo, useState } from 'react'
import { stream } from 'aifn-compute/foundation/random'
import type { Trace } from 'aifn-compute/foundation/trace'
import {
  encodeText,
  tokeniser,
  trainedModel,
  vocabularySize,
  type Encoding,
  type Tokeniser,
  type TrainerState,
} from 'aifn-compute/text/pipeline'
import { tokeniserStatistics } from 'aifn-compute/text/statistics'
import type { BpeState, UnigramLmState, WordPieceState } from 'aifn-compute/text/subword'
import { toyCorpus } from 'aifn-methods/text/corpora'
import {
  SUBWORD_KINDS,
  TOKENISER_CORPUS,
  TOKENISER_KINDS,
  TOKENISER_LABELS,
  tokeniserTrainer,
  untrainedTokeniser,
  type TokeniserKind,
} from 'aifn-methods/text/tokenisers'
import { Button, Player } from 'aifn-render/controls'
import { Columns, ControlRow, Figure } from 'aifn-render/layout'
import {
  call,
  choice,
  packPair,
  pinField,
  setting,
  slider,
  unpackPair,
  useFigureState,
  usePinned,
  when,
  type Task,
} from 'aifn-render/state'
import { Textarea } from 'aifn-render/ui/textarea'
import { formatValue, TrainControls, useTrainedRun } from '@lab/views'
import { Bars, Curve, Plot, Plots, Points, Readout, useAxis } from 'aifn-render/viz'
import { Body, SourceSpan, TokenChips, type Span } from './tokens'
import { visible } from './visible'

const DEFAULT_TEXT = `The newest tokenisers don't split 1234567 the same way!
Unicode: café, naïve, Straße, 東京, привет, 🙂👍🏽 and é (a combining accent).`

type CorpusChoice = 'about tokenisers' | 'toy sentences' | 'your text'
const CORPORA: CorpusChoice[] = ['about tokenisers', 'toy sentences', 'your text']

const f2 = (v: number) => (Number.isFinite(v) ? formatValue(Number(v.toPrecision(3))) : '—')

function corpusOf(choice: CorpusChoice, text: string): string[] {
  if (choice === 'your text') return text.split('\n').filter((l) => l.trim().length > 0)
  if (choice === 'toy sentences') return [...toyCorpus(stream(1), { sentences: 120 }).documents]
  return [...TOKENISER_CORPUS]
}

type SuiteSettings = {
  corpus: CorpusChoice
  /** The training text when the corpus is "your text". */
  typed: string
  vocabularySize: number
  pattern: 'gpt2' | 'cl100k' | 'o200k'
  splitDigits: boolean
  wordMinCount: number
  lowercaseWords: boolean
}

function suiteTask(s: SuiteSettings): Task<Record<TokeniserKind, Tokeniser>> {
  return call('text/tokenisers/tokeniserSuite', corpusOf(s.corpus, s.typed), {
    vocabularySize: s.vocabularySize,
    pattern: s.pattern,
    splitDigits: s.splitDigits,
    wordMinCount: s.wordMinCount,
    lowercaseWords: s.lowercaseWords,
  })
}

const BPE_FAMILY: readonly TokeniserKind[] = ['bpe', 'byteLevelBpe', 'sentencePiece']

// ── Figure 1: one text under every tokeniser ─────────────────────────────────────────────────────────────────────────

export function TokenisersComparedSpecimen() {
  const state = useFigureState({
    corpus: choice(CORPORA, 'about tokenisers', { label: 'train on' }),
    vocabularySize: slider(100, 1200, 400, { label: 'subword vocabulary', step: 50 }),
    pattern: choice(['gpt2', 'cl100k', 'o200k'] as const, 'gpt2', { label: 'byte-level split' }),
    splitDigits: setting(false, 'split digits (LLaMA)'),
    wordMinCount: slider(1, 5, 1, { label: 'word min count', step: 1 }),
    lowercaseWords: setting(false, 'lower-case words'),
    dropout: setting(false, 'BPE-dropout'),
    p: slider(0.05, 0.6, 0.2, { label: 'dropout p', step: 0.05, when: when('dropout', true) }),
    // The pinned span of the text (start and end packed into one), kept in the URL so a link reproduces it.
    pin: pinField(),
  })
  const { corpus, vocabularySize: size, pattern, splitDigits, wordMinCount, lowercaseWords, dropout, p } = state
  const [text, setText] = useState(DEFAULT_TEXT)
  const [seed, setSeed] = useState(0)
  // Hover previews a span; click pins it (again, empty space or Escape: unpins).
  const pins = usePinned(state.pin, (v) => state.set('pin', v))
  const asSpan = (v: number | null): Span | null => (v === null ? null : unpackPair(v))
  const pinned = asSpan(pins.pinned)
  const hot = asSpan(pins.focus)
  const setHover = (span: Span | null) => pins.hover(span ? packPair(span[0], span[1]) : null)
  const pin = (span: Span) => pins.toggle(packPair(span[0], span[1]))
  const unpin = () => pins.clear()
  const settings: SuiteSettings = {
    corpus: corpus as CorpusChoice,
    typed: corpus === 'your text' ? text : '',
    vocabularySize: size,
    pattern,
    splitDigits,
    wordMinCount,
    lowercaseWords,
  }
  const trained = useTrainedRun(settings, suiteTask)
  const suite = trained.run.value

  const rows = useMemo(() => {
    if (!suite) return null
    return TOKENISER_KINDS.map((kind) => {
      const t = suite[kind]
      const sample = dropout && BPE_FAMILY.includes(kind)
      const e = encodeText(t, text, null, sample ? { stream: stream(`dropout-${seed}-${kind}`), dropout: p } : {})
      const stats = tokeniserStatistics(t, [text])
      return { kind, t, e, stats }
    })
  }, [suite, text, dropout, p, seed])

  const kinds = TOKENISER_KINDS.map((k) => TOKENISER_LABELS[k])
  const catAxis = useAxis({ label: 'tokeniser', categories: [...kinds].reverse() })
  const catAxis2 = useAxis({ label: 'tokeniser', categories: [...kinds].reverse() })
  const fertAxis = useAxis({ label: 'tokens per word', range: [0, undefined] })
  const bptAxis = useAxis({ label: 'UTF-8 bytes per token', range: [0, undefined] })
  const idx = TOKENISER_KINDS.map((_, i) => TOKENISER_KINDS.length - 1 - i)

  return (
    <>
      <Figure
        title="One text, eight tokenisers"
        purpose="Tokenisers differ in their units: bytes and characters never meet an unknown symbol but make long sequences; words are short but leave rare words unknown; subword models sit between."
        state={state}
        defaultSize="XL"
        hoverReadout={false}
        controls={
          <>
            <TrainControls
              run={trained as never}
              progress={suite ? 1 : 0}
              progressText={trained.run.running ? 'training' : suite ? 'trained' : 'not trained'}
              actions={
                dropout && (
                  <Button size="sm" variant="outline" onClick={() => setSeed((s) => s + 1)}>
                    Re-sample
                  </Button>
                )
              }
            />
            <div className="col-span-full flex flex-col gap-2" onClick={unpin}>
              <Textarea
                aria-label="text to tokenise"
                value={text}
                onChange={(ev) => setText(ev.target.value)}
                className="font-mono text-xs"
                rows={3}
              />
              <div className="rounded border p-2">
                <div className="mb-1 text-xs text-muted-foreground">
                  offsets: the source span of the hovered or pinned token (hover a word, click to pin)
                </div>
                <SourceSpan text={text} span={hot} onHover={setHover} pinned={pinned} onPin={pin} />
              </div>
            </div>
          </>
        }
        readouts={
          hot && (
            <>
              <Readout label="hovered span" value={`[${hot[0]}, ${hot[1]})`} />
              <Readout label="source" value={JSON.stringify(text.slice(hot[0], hot[1]))} />
            </>
          )
        }
        caption={
          <>
            Press Train to train every tokeniser on the chosen corpus in the worker (well under a second): characters
            (with byte fallback), Penn Treebank words with a word-level vocabulary, and five subword models at the
            vocabulary size set above; byte-level BPE and SentencePiece BPE hold their 256 byte tokens on top of it. The
            byte tokeniser (ByT5) needs no training. Type in the box: each row shows the text cut by one tokeniser.
            Chips alternate blue and purple; dashed chips are special tokens ([CLS], [SEP]), red ones the unknown token,
            green ones <code>&lt;0xNN&gt;</code> byte-fallback tokens; ␣ is a space. Hover a word of the text, or a chip
            of any row (its title gives id and offsets), to preview; click to pin it (pinned chips keep an outline;
            click it again, click empty space or press Escape to unpin; the pin is kept in the link). The span is marked
            in the text and, through the offsets, the chips of every tokeniser that cover it light up while the rest
            dim, so one word can be followed across all eight cuts. The word tokeniser keeps every Treebank token of the
            corpus seen at least min-count times (no size cap; same pre-tokeniser, and lower-casing if set, in training
            and encoding), so its unknowns are exactly the words the corpus never contained: on the default text,
            numbers, accented and non-Latin words and emoji. That gap is why subword tokenisers exist; raise min-count
            to watch coverage fall, or lower-case to watch it rise. With BPE-dropout on, the three BPE models skip each
            merge with probability p; Re-sample draws another segmentation.
          </>
        }
      >
        <Body>
          <div className="flex flex-col gap-3" onClick={unpin}>
            {!rows ? (
              <div className="py-6 text-center text-sm text-muted-foreground">
                Press Train: the eight tokenisers are trained on the corpus, then cut the text above.
              </div>
            ) : (
              rows.map(({ kind, t, e, stats }) => (
                <div key={kind} className="grid grid-cols-[13rem_1fr] items-start gap-3 border-t pt-2">
                  <div className="text-xs">
                    <div className="font-medium">{TOKENISER_LABELS[kind]}</div>
                    <div className="text-muted-foreground tabular-nums">
                      {stats.tokens} tokens · {f2(stats.fertility)} per word · {f2(stats.bytesPerToken)} B/token
                    </div>
                    <div className="text-muted-foreground tabular-nums">vocabulary {vocabularySize(t)}</div>
                    <div
                      className={
                        'tabular-nums ' +
                        (stats.unknownTokens > 0 ? 'font-semibold text-destructive' : 'text-muted-foreground')
                      }
                    >
                      UNK {Math.round(100 * stats.unknownRate)}% of tokens · {Math.round(100 * stats.wordCoverage)}% of
                      words covered
                    </div>
                  </div>
                  <TokenChips
                    e={e}
                    unknownId={t.model.vocabulary.unknown}
                    hot={hot}
                    onHover={setHover}
                    pinned={pinned}
                    onPin={pin}
                  />
                </div>
              ))
            )}
          </div>
        </Body>
      </Figure>
      <Figure
        title="Fertility and compression"
        purpose="Fertility (tokens per word) and compression (bytes per token) are two views of one trade-off: the fewer tokens a text takes, the more each token carries."
        defaultSize="L"
        caption="For the text above, per tokeniser: tokens per run of non-space characters, and UTF-8 bytes per token (excluding special tokens). Bytes give exactly 1 byte per token; words give the fewest tokens until unknown words appear (counted in the first figure). Train above first."
      >
        {!rows ? (
          <div className="py-6 text-center text-sm text-muted-foreground">Press Train in the figure above.</div>
        ) : (
          <Plots cols={2}>
            <Plot x={fertAxis} y={catAxis} title="fertility">
              <Bars
                name="tokens per word"
                x={idx}
                y={rows.map((r) => r.stats.fertility)}
                orient="y"
                slot={0}
                width={0.7}
              />
            </Plot>
            <Plot x={bptAxis} y={catAxis2} title="compression">
              <Bars
                name="bytes per token"
                x={idx}
                y={rows.map((r) => r.stats.bytesPerToken)}
                orient="y"
                slot={1}
                width={0.7}
              />
            </Plot>
          </Plots>
        )}
      </Figure>
      <TrainingFigure text={text} corpus={corpus as CorpusChoice} />
    </>
  )
}

// ── Figure 3: training one subword model, step by step ───────────────────────────────────────────────────────────────

type SubwordKind = (typeof SUBWORD_KINDS)[number]

type StepSettings = { kind: SubwordKind; vocabularySize: number; corpus: CorpusChoice; typed: string }

function stepsTask(s: StepSettings): Task<Trace<TrainerState>> {
  const parts = untrainedTokeniser(s.kind)
  const trainer = tokeniserTrainer(s.kind, s.vocabularySize)
  return call(
    'foundation/trace/trace',
    call('text/pipeline/trainingSteps', parts, corpusOf(s.corpus, s.typed), trainer),
    undefined,
    5000,
  )
}

const isBpe = (k: SubwordKind) => k === 'bpe' || k === 'byteLevelBpe' || k === 'sentencePiece'

/** The size of the vocabulary and the training objective of a state. */
function measures(kind: SubwordKind, s: TrainerState): { size: number; objective: number } {
  if (kind === 'unigram') {
    const u = s as UnigramLmState
    return { size: u.pieces.length, objective: u.logLikelihood }
  }
  if (kind === 'wordPiece') {
    const w = s as WordPieceState
    return { size: w.vocabulary.length, objective: w.logLikelihood }
  }
  const b = s as BpeState
  return { size: b.vocabulary.length, objective: b.symbols }
}

function lastChange(kind: SubwordKind, s: TrainerState): string {
  if (kind === 'unigram') {
    const pruned = (s as UnigramLmState).pruned
    return pruned.length === 0
      ? 'none yet'
      : `pruned ${pruned.length}: ${pruned
          .slice(0, 6)
          .map((x) => visible(x.piece))
          .join(' ')}${pruned.length > 6 ? ' …' : ''}`
  }
  const m = (s as BpeState | WordPieceState).merge
  return m ? `${visible(m.left)} + ${visible(m.right)} → ${visible(m.merged)} (${m.count})` : 'none yet'
}

function TrainingFigure({ text, corpus }: { text: string; corpus: CorpusChoice }) {
  const state = useFigureState({
    kind: choice(
      SUBWORD_KINDS.map((k) => ({ value: k, label: TOKENISER_LABELS[k] })),
      'bpe',
      { label: 'model' },
    ),
    vocabularySize: slider(60, 800, 200, { label: 'vocabulary size', step: 20 }),
  })
  const kind = state.kind as SubwordKind
  const settings: StepSettings = {
    kind,
    vocabularySize: state.vocabularySize,
    corpus,
    typed: corpus === 'your text' ? text : '',
  }
  const trained = useTrainedRun(settings, stepsTask)
  const run = trained.run.value
  const shown = trained.trained ?? settings
  const steps = useMemo(() => run?.steps ?? [], [run])
  const [picked, setPicked] = useState<{ run: unknown; index: number } | null>(null)
  const at = Math.min(picked && picked.run === run ? picked.index : 0, Math.max(0, steps.length - 1))
  const s = steps[at] as TrainerState | undefined
  const parts = useMemo(() => untrainedTokeniser(shown.kind), [shown.kind])
  const trainer = useMemo(() => tokeniserTrainer(shown.kind, shown.vocabularySize), [shown])
  const encoding: Encoding | null = useMemo(() => {
    if (!s) return null
    const model = trainedModel(trainer, s)
    return encodeText(tokeniser({ ...parts, model, specials: trainer.specials ?? [] }), text)
  }, [s, trainer, parts, text])
  const unknownId = useMemo(() => (s ? trainedModel(trainer, s).vocabulary.unknown : -1), [s, trainer])
  const curves = useMemo(() => {
    const m = steps.map((x) => measures(shown.kind, x as TrainerState))
    return { t: steps.map((_, k) => k), size: m.map((x) => x.size), objective: m.map((x) => x.objective) }
  }, [steps, shown.kind])
  const [hot, setHot] = useState<Span | null>(null)
  const stepAxis = useAxis({ label: shown.kind === 'unigram' ? 'pruning round' : 'merge', integer: true, key: run })
  const stepAxis2 = useAxis({ label: shown.kind === 'unigram' ? 'pruning round' : 'merge', integer: true, key: run })
  const sizeAxis = useAxis({ label: 'vocabulary size', hold: 'union', key: run })
  const objAxis = useAxis({
    label: isBpe(shown.kind) ? 'corpus length (symbols)' : 'log-likelihood (nats)',
    hold: 'union',
    key: run,
  })
  const here = s ? measures(shown.kind, s) : null

  return (
    <Figure
      title="Training a subword vocabulary, step by step"
      purpose="BPE and WordPiece grow a vocabulary one merge at a time; the unigram model shrinks a large one round by round. Each step changes how the same text is cut."
      state={state}
      defaultSize="XL"
      hoverReadout={false}
      controls={
        <>
          <TrainControls
            run={trained as never}
            progress={run ? 1 : 0}
            progressText={trained.run.running ? 'training' : run ? `${steps.length - 1} steps` : 'not trained'}
          />
          {steps.length > 0 && (
            <ControlRow label="step">
              <Player
                label={shown.kind === 'unigram' ? 'round' : 'merge'}
                value={at}
                onChange={(i) => setPicked({ run, index: i })}
                count={steps.length}
              />
            </ControlRow>
          )}
        </>
      }
      readouts={
        s && here ? (
          <>
            <Readout label="step" value={at} />
            <Readout label="last change" value={lastChange(shown.kind, s)} />
            <Readout label="vocabulary" value={here.size} />
            <Readout label={isBpe(shown.kind) ? 'corpus symbols' : 'log-likelihood'} value={f2(here.objective)} />
            {encoding && <Readout label="tokens in the text" value={encoding.tokens.length} />}
          </>
        ) : undefined
      }
      caption={
        <>
          Press Train to run the chosen trainer to its vocabulary size in the worker, on the corpus picked in the first
          figure, through the model&apos;s own normaliser and pre-tokeniser (aifn <code>trainingSteps</code>). Step 0 is
          the alphabet (BPE, WordPiece) or the seed vocabulary of frequent substrings (unigram); play or drag the step
          to see the text above re-cut with the vocabulary of that step. The curve counts the alphabet and the learned
          pieces; the model adds its special tokens, and SentencePiece BPE its 256 byte-fallback tokens. BPE&apos;s
          objective is the corpus length in symbols, which falls by the merged pair&apos;s count at each merge;
          WordPiece&apos;s and the unigram model&apos;s is the corpus log-likelihood.
        </>
      }
    >
      {!s ? (
        <div className="py-6 text-center text-sm text-muted-foreground">
          Press Train: the trainer&apos;s steps appear here, from step 0.
        </div>
      ) : (
        <Columns
          panels={[
            {
              title: 'the text, cut with the vocabulary of this step',
              body: (
                <Body>
                  <SourceSpan text={text} span={hot} />
                  {encoding && <TokenChips e={encoding} unknownId={unknownId} hot={hot} onHover={setHot} />}
                </Body>
              ),
            },
            {
              title: 'vocabulary and objective',
              body: (
                <Plots rows={2}>
                  <Plot x={stepAxis} y={sizeAxis} legend={false}>
                    <Curve name="vocabulary size" x={curves.t} y={curves.size} slot={0} />
                    <Points name="this step" x={[at]} y={[here!.size]} emphasis live />
                  </Plot>
                  <Plot x={stepAxis2} y={objAxis} legend={false}>
                    <Curve name="objective" x={curves.t} y={curves.objective} slot={1} />
                    <Points name="this step" x={[at]} y={[here!.objective]} emphasis live />
                  </Plot>
                </Plots>
              ),
            },
          ]}
        />
      )}
    </Figure>
  )
}
