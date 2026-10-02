import { useMemo, useState, type ReactNode } from 'react'
import { topFeatures, firingFeatures, transitionWeights } from 'aifn-applied/inference/sequence-models'
import { stream } from 'aifn/foundation/random'
import { parseTemplates } from 'aifn/text/features'
import { liangHyphenate, hyphenationPatterns, markHyphens } from 'aifn/text/hyphenation'
import { mobyHyphenation } from 'aifn-applied/data/real/hyphenation'
import {
  crfHyphenate,
  gapLabels,
  HYPHENATION_TEMPLATES,
  hyphenationRows,
  hyphenScores,
  thresholdScores,
  type CrfDecision,
  type CrfHyphenationSnapshot,
  type HyphenScores,
  type LiangSnapshot,
  type TaggerSnapshot,
} from 'aifn-applied/text/hyphenation'
import { Button, Select } from '@lab/controls'
import { seriesColor } from '@lab/design/palette'
import { useTheme } from '@lab/design/theme'
import { ControlRow, Figure } from '@lab/layout'
import { cn } from '@lab/lib/utils'
import { choice, int, row, useFigureState } from '@lab/state'
import { Input } from '@lab/ui/input'
import {
  CrfTrainingPlots,
  CrfWeightPlots,
  crfOptimiserField,
  crfTrainingOptions,
  describeOptimiser,
  FiringFeatureList,
  TemplateCells,
  TemplateEditor,
  TrainControls,
  useTrainedRun,
  type ParsedTemplates,
} from '@lab/views'
import { Bars, Plot, Plots, Points, Readout, useAxis } from '@lab/viz'
import { crfTask, type CrfSettings } from './crf-tasks'
import { Marked, ScoreTable } from './showcase'
import { liangTask, MARGINS, SPLIT_SEED, taggerTask, type LiangSettings, type TaggerSettings } from './tasks'

const DECISIONS = [
  { value: 'viterbi' as const, label: 'Viterbi (MAP sequence)' },
  { value: 'posterior' as const, label: 'marginal (max-marginal, posterior)' },
]
const DECISION_NAME: Record<CrfDecision, string> = { viterbi: 'Viterbi', posterior: 'posterior' }
const LIANG_SLOT = 0
const WINDOW_SLOT = 1
const RNN_SLOT = 2
const CRF_SLOT = 3
const LONGEST = 15
const HYPH = 1
const EXAMPLES = ['hyphenation', 'computer', 'information', 'probability', 'beautiful', 'understanding', 'monkey']
const LIANG: LiangSettings = { budget: 1000 }
const TAGGERS: TaggerSettings = { steps: 600, seed: 0 }

const pct = (v: number) => (Number.isFinite(v) ? `${(100 * v).toFixed(1)}%` : '—')

function parse(source: string): ParsedTemplates {
  try {
    return { templates: parseTemplates(source), error: null }
  } catch (e) {
    return { templates: null, error: e as Error }
  }
}

/** A finished run, kept for the comparison of optimisers. */
type RunRecord = {
  key: string
  optimiser: string
  templates: string
  scores: HyphenScores
  active: number
  weights: number
  strings: number
  ms: number
}

export function HyphenationCrfShowcase() {
  const mode = useTheme().resolved
  const [decision, setDecision] = useState<CrfDecision>('viterbi')
  const data = useMemo(() => mobyHyphenation(stream(SPLIT_SEED)), [])
  const testLabels = useMemo(() => gapLabels(data.test.words), [data])

  // ── Settings and the run ──
  const [source, setSource] = useState(HYPHENATION_TEMPLATES.classic)
  const parsed = useMemo(() => parse(source), [source])
  const state = useFigureState({
    data: row('1 · data and features', {
      trainWords: choice(
        [
          { value: 1000, label: '1000' },
          { value: 3000, label: '3000' },
          { value: 0, label: `all ${data.train.words.length}` },
        ],
        0,
        { label: 'training words' },
      ),
      minFrequency: int(1, { label: '-f: keep strings seen at least f times', ge: 1, le: 20 }),
      maxSteps: int(60, { ge: 1, suggestions: [30, 60, 100, 150], label: 'most iterations (epochs for SGD / Adam)' }),
    }),
    optimiser: crfOptimiserField(
      '2 · optimiser and regularisation',
      { C: 1, c1: 0.5, c2: 0.05, batchSize: 64 },
      'words',
    ),
  })
  const options = crfTrainingOptions(state.optimiser)
  const settings: CrfSettings = {
    templates: source,
    trainWords: state.data.trainWords,
    minFrequency: state.data.minFrequency,
    maxSteps: state.data.maxSteps,
    ...options,
  }
  const crf = useTrainedRun(settings, crfTask)
  const snap: CrfHyphenationSnapshot | null = crf.run.value ?? null
  const model = snap?.crf ?? null

  // Finished runs, for L-BFGS against OWL-QN.
  const [records, setRecords] = useState<RunRecord[]>([])
  const doneKey = snap && (snap.done || crf.run.stopped) && !crf.run.running ? JSON.stringify(crf.trained) : null
  if (doneKey && snap?.testScores && !records.some((r) => r.key === doneKey) && crf.trained) {
    const K = snap.crf.labels.length
    const U = snap.crf.index.unigram.length
    let strings = 0
    for (let u = 0; u < U; u++) if (snap.crf.weights.slice(u * K, u * K + K).some((w) => w !== 0)) strings++
    setRecords([
      ...records,
      {
        key: doneKey,
        optimiser: describeOptimiser(crf.trained as never),
        templates: Object.entries(HYPHENATION_TEMPLATES).find(([, v]) => v === crf.trained!.templates)?.[0] ?? 'edited',
        scores: snap.testScores,
        active: snap.history.active.at(-1)!,
        weights: snap.crf.weights.length,
        strings,
        ms: snap.ms,
      },
    ])
  }

  // ── The other three methods ──
  const liang = useTrainedRun(LIANG, liangTask)
  const taggers = useTrainedRun(TAGGERS, taggerTask)
  const liangRun = liang.run.value as LiangSnapshot | null | undefined
  const liangScores = useMemo(() => {
    if (!liangRun) return null
    const patterns = hyphenationPatterns(liangRun.passes.flatMap((p) => p.added))
    return hyphenScores(
      data.test.words,
      data.test.words.map((w) => liangHyphenate(patterns, w.word, MARGINS).hyphens),
    )
  }, [liangRun, data])
  const lastShot = (taggers.run.value as TaggerSnapshot | null | undefined)?.checkpoints.at(-1)
  const windowScores = lastShot ? thresholdScores(testLabels, lastShot.windowTest, 0.5) : null
  const rnnScores = lastShot ? thresholdScores(testLabels, lastShot.rnnTest, 0.5) : null
  const scoreRows = [
    { name: "Liang's patterns", slot: LIANG_SLOT, scores: liangScores, note: '(1000 patterns)' },
    {
      name: 'window MLP',
      slot: WINDOW_SLOT,
      scores: windowScores,
      note: lastShot ? `(step ${lastShot.step}, P ≥ 0.5)` : '',
    },
    { name: 'BiLSTM', slot: RNN_SLOT, scores: rnnScores, note: lastShot ? `(step ${lastShot.step}, P ≥ 0.5)` : '' },
    {
      name: 'linear-chain CRF',
      slot: CRF_SLOT,
      scores: (decision === 'viterbi' ? snap?.testScores : snap?.testScoresPosterior) ?? null,
      note: snap ? `(${DECISION_NAME[decision]}, iteration ${snap.scoredAt})` : '(press Train above)',
    },
  ]
  const decisionSelect = (
    <ControlRow label="decision">
      <Select label="decode by" value={decision} onChange={setDecision} options={DECISIONS} />
    </ControlRow>
  )
  const earlier = records.filter((r) => r.key !== JSON.stringify(crf.trained))
  const recallAxis = useAxis({ label: 'recall', range: [0.5, 1] })
  const precisionAxis = useAxis({ label: 'precision', range: [0.5, 1] })

  return (
    <>
      <Figure
        title="Train a CRF hyphenator"
        purpose="Hyphenation as tagging: each letter gets HYPH (a hyphen follows) or O, and a linear-chain CRF over CRF++ templates learns which letter n-grams around a gap predict a hyphen."
        state={state}
        defaultSize="XL"
        controls={
          <>
            <ControlRow label="3 · templates (CRF++ syntax)">
              <TemplateEditor value={source} onChange={setSource} presets={HYPHENATION_TEMPLATES} parsed={parsed} />
            </ControlRow>
            <TrainControls
              run={crf as never}
              progress={snap ? (snap.done ? 1 : snap.step / snap.maxSteps) : 0}
              progressText={
                snap ? `${snap.step} / ${snap.maxSteps} steps${snap.converged ? ', converged' : ''}` : '0 steps'
              }
            />
          </>
        }
        readouts={{
          'held-out words (Viterbi)': (
            <>
              <Readout label="precision" value={snap?.testScores ? pct(snap.testScores.precision) : '—'} />
              <Readout label="recall" value={snap?.testScores ? pct(snap.testScores.recall) : '—'} />
              <Readout label="F₀.₅" value={snap?.testScores ? pct(snap.testScores.fHalf) : '—'} />
            </>
          ),
          'held-out words (posterior)': (
            <>
              <Readout
                label="precision"
                value={snap?.testScoresPosterior ? pct(snap.testScoresPosterior.precision) : '—'}
              />
              <Readout label="recall" value={snap?.testScoresPosterior ? pct(snap.testScoresPosterior.recall) : '—'} />
            </>
          ),
          model: (
            <>
              <Readout label="unigram strings" value={model ? model.index.unigram.length.toLocaleString() : '—'} />
              <Readout
                label="non-zero weights"
                value={
                  snap
                    ? `${snap.history.active.at(-1)!.toLocaleString()} / ${snap.crf.weights.length.toLocaleString()}`
                    : '—'
                }
              />
              <Readout label="time" value={snap ? `${(snap.ms / 1000).toFixed(1)} s` : '—'} />
            </>
          ),
        }}
        caption={
          <>
            aifn-applied <code>crfHyphenationRun</code> turns each training word of <code>mobyHyphenation</code> into
            rows of [letter, class] (class <code>v</code> for a e i o u y, else <code>c</code>) with labels HYPH or O,
            indexes the template strings (<code>featureIndex</code>, dropping those seen fewer than f times), and trains{' '}
            <code>crfTraining</code> in the worker; every 5 iterations it hyphenates the {data.test.words.length}{' '}
            held-out words by Viterbi and scores every gap. The classic preset reads letters −2 … +3 around the gap
            after the current letter, the bigrams, trigrams and a 4-gram across it, and the vowel/consonant pattern.
            CRF++&apos;s <code>-c C</code> is the L2 strength c₂ = 1/(2C); OWL-QN adds c₁‖λ‖₁ and keeps only the weights
            the data need. SGD and Adam take minibatches of words, one epoch per step (the curves are against epochs, so
            the optimisers compare per pass over the data): a small batch makes many noisy updates per epoch, a large
            one few accurate ones, and the full batch is plain gradient descent. The default run (all words, 60 L-BFGS
            iterations) takes about 15–30 s.
          </>
        }
      >
        <CrfTrainingPlots
          history={snap?.history ?? null}
          maxSteps={snap?.maxSteps ?? settings.maxSteps}
          runKey={crf.trained}
        />
      </Figure>

      <Figure
        title="Against Liang, the MLP and the BiLSTM, and L1 against L2"
        purpose="On the same held-out words, the CRF's Viterbi hyphens are compared with the other three hyphenators; trained with OWL-QN, the CRF keeps a small fraction of its weights at the same precision."
        defaultSize="XL"
        hoverReadout={false}
        controls={
          <>
            {decisionSelect}
            <TrainControls
              label="Liang's patterns (PATGEN, 1000 patterns, about 1 s)"
              run={liang as never}
              progress={liangRun ? liangRun.step / liangRun.steps : 0}
              progressText={`${liangRun?.step ?? 0} / ${liangRun?.steps ?? 20} passes`}
            />
            <TrainControls
              label="window MLP and BiLSTM (600 Adam steps, about 30–60 s)"
              run={taggers as never}
              progress={taggers.run.value ? (taggers.run.value as TaggerSnapshot).step / TAGGERS.steps : 0}
              progressText={`${(taggers.run.value as TaggerSnapshot | null)?.step ?? 0} / ${TAGGERS.steps} steps`}
            />
          </>
        }
        caption={
          <>
            The other three methods are the hyphenation showcase&apos;s, run here with its defaults: Liang&apos;s
            patterns by PATGEN with a budget of 1000 patterns, and the window MLP and BiLSTM after 600 steps with
            hyphens where P ≥ 0.5. Every row counts the same gaps of the same held-out words. The run table collects
            each finished CRF run: train with L-BFGS, then switch the optimiser to OWL-QN and train again to see how
            many weights L1 keeps. A weight is non-zero when it is not exactly 0; a string is active when any of its two
            weights is.
          </>
        }
      >
        <div className="grid gap-4 lg:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
          <div className="flex flex-col gap-3 overflow-x-auto">
            <ScoreTable rows={scoreRows} />
            <table className="w-full text-xs">
              <thead>
                <tr className="text-left text-muted-foreground">
                  <th className="py-1 pr-2 font-medium">CRF run</th>
                  <th className="py-1 pr-2 font-medium">templates</th>
                  <th className="py-1 pr-2 font-medium">precision</th>
                  <th className="py-1 pr-2 font-medium">recall</th>
                  <th className="py-1 pr-2 font-medium">non-zero weights</th>
                  <th className="py-1 pr-2 font-medium">active strings</th>
                  <th className="py-1 font-medium">time</th>
                </tr>
              </thead>
              <tbody>
                {records.length === 0 && (
                  <tr>
                    <td colSpan={7} className="py-2 text-muted-foreground">
                      Finished runs are listed here.
                    </td>
                  </tr>
                )}
                {records.map((r) => (
                  <tr key={r.key} className="border-t border-border">
                    <td className="py-0.5 pr-2">{r.optimiser}</td>
                    <td className="py-0.5 pr-2">{r.templates}</td>
                    <td className="py-0.5 pr-2 font-semibold">{pct(r.scores.precision)}</td>
                    <td className="py-0.5 pr-2">{pct(r.scores.recall)}</td>
                    <td className="py-0.5 pr-2">
                      {r.active.toLocaleString()} / {r.weights.toLocaleString()} ({pct(r.active / r.weights)})
                    </td>
                    <td className="py-0.5 pr-2">{r.strings.toLocaleString()}</td>
                    <td className="py-0.5">{(r.ms / 1000).toFixed(1)} s</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <Plots cols={1} scale={0.6}>
            <Plot x={recallAxis} y={precisionAxis} title="held-out precision and recall">
              {scoreRows.map((r) =>
                r.scores ? (
                  <Points
                    key={r.name}
                    name={r.name}
                    x={[r.scores.recall]}
                    y={[r.scores.precision]}
                    slot={r.slot}
                    size={11}
                  />
                ) : null,
              )}
              {earlier.length > 0 && (
                <Points
                  name="earlier CRF runs"
                  x={earlier.map((r) => r.scores.recall)}
                  y={earlier.map((r) => r.scores.precision)}
                  muted
                  size={7}
                />
              )}
            </Plot>
          </Plots>
        </div>
      </Figure>

      <WordFigure data={data} snap={snap} mode={mode} decision={decision} decisionSelect={decisionSelect} />

      <Figure
        title="The most informative features"
        purpose="The largest weights per label show what the CRF has learned about English hyphenation: which letter pairs a hyphen falls between, and which it never splits."
        defaultSize="XL"
        hoverReadout={false}
        caption={
          <>
            For each label, the unigram strings that favour it most over the other label, ranked by λ_k minus the mean
            of the string&apos;s weights (<code>topFeatures</code> with <code>relative</code>; with two labels this is
            half the log-odds the string adds), with how often each fired in training. A string such as{' '}
            <code>U11:n/t</code> with a large HYPH weight reads &quot;the current letter is n and the next is t&quot;: a
            hyphen between them. The heatmaps show the top strings by max_k |λ| against both labels, and the 2 × 2
            transitions of <code>B</code>: a hyphen is rarely followed by another at the next letter.
          </>
        }
      >
        {model ? (
          <div className="flex flex-col gap-3">
            <div className="grid gap-4 md:grid-cols-2">
              {model.labels.map((y, k) => (
                <div key={y}>
                  <div className="mb-1 text-xs font-medium">label {y}</div>
                  <table className="w-full text-xs">
                    <thead>
                      <tr className="text-left text-muted-foreground">
                        <th className="py-0.5 pr-2 font-medium">string</th>
                        <th className="py-0.5 pr-2 text-right font-medium">λ_k − mean λ</th>
                        <th className="py-0.5 text-right font-medium">count</th>
                      </tr>
                    </thead>
                    <tbody>
                      {topFeatures(model, 14, k, { relative: true }).map((f) => (
                        <tr key={f.string} className="border-t border-border">
                          <td className="py-0.5 pr-2 font-mono">
                            {f.string}/{y}
                          </td>
                          <td
                            className="py-0.5 pr-2 text-right font-mono tabular-nums"
                            style={{ color: seriesColor(mode, 0) }}
                          >
                            {(f.weights[k] - f.weights.reduce((a, w) => a + w, 0) / f.weights.length).toFixed(2)}
                          </td>
                          <td className="py-0.5 text-right tabular-nums">{f.count}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ))}
            </div>
            <CrfWeightPlots crf={model} top={14} scale={0.8} />
            <div className="text-xs text-muted-foreground">
              transitions λ_B:{' '}
              {transitionWeights(model)
                ?.map((r, i) => `${model.labels[i]} → [${r.map((w) => w.toFixed(2)).join(', ')}]`)
                .join('; ') ?? 'none'}
            </div>
          </div>
        ) : (
          <div className="py-10 text-center text-sm text-muted-foreground">Train the CRF in the first figure.</div>
        )}
      </Figure>

      <ErrorsFigure data={data} snap={snap} decision={decision} />
    </>
  )
}

// ── Type a word ──────────────────────────────────────────────────────────────────────────────────────────────────────

function WordFigure({
  data,
  snap,
  mode,
  decision,
  decisionSelect,
}: {
  data: ReturnType<typeof mobyHyphenation>
  snap: CrfHyphenationSnapshot | null
  mode: 'light' | 'dark'
  decision: CrfDecision
  decisionSelect: ReactNode
}) {
  const [typed, setTyped] = useState('hyphenation')
  const word = typed
    .toLowerCase()
    .replace(/[^a-z]/g, '')
    .slice(0, LONGEST)
  const valid = word.length >= 2
  const truth = valid ? data.truth.hyphens(word) : null
  const crf = snap?.crf ?? null
  const result = useMemo(() => (crf && valid ? crfHyphenate(crf, word, decision) : null), [crf, word, valid, decision])
  const otherResult = useMemo(
    () => (crf && valid ? crfHyphenate(crf, word, decision === 'viterbi' ? 'posterior' : 'viterbi') : null),
    [crf, word, valid, decision],
  )
  const viterbi = useMemo(() => {
    if (!result) return null
    const h = new Set(result.hyphens)
    return [...word].map((_, i) => (h.has(i) ? 'HYPH' : 'O'))
  }, [result, word])
  const [pin, setPin] = useState<{ word: string; n: number } | null>(null)
  const [hover, setHover] = useState<number | null>(null)
  const pinned = pin && pin.word === word ? Math.min(pin.n, word.length - 1) : null
  const n = hover ?? pinned ?? (result ? result.probabilities.indexOf(Math.max(...result.probabilities)) : 0)
  const rows = useMemo(() => hyphenationRows(word), [word])
  const features = useMemo(() => (crf && valid ? firingFeatures(crf, rows, Math.max(0, n)) : []), [crf, rows, n, valid])
  const [template, setTemplate] = useState(0)
  const templates = crf?.index.templates ?? null
  const gapLabelsOf = Array.from({ length: Math.max(0, word.length - 1) }, (_, i) => `${word[i]}|${word[i + 1]}`)
  const gapAxis = useAxis({ label: 'gap after letter n', categories: gapLabelsOf })
  const probAxis = useAxis({ label: 'P(HYPH)', range: [0, 1] })
  const previous = n > 0 && viterbi ? (viterbi[n - 1] === 'HYPH' ? HYPH : 0) : null
  const label = viterbi && viterbi[n] === 'HYPH' ? HYPH : 0
  return (
    <Figure
      title="Type any word"
      purpose="For one word: the CRF's per-letter marginal P(hyphen after this letter), its hyphenation by Viterbi or marginal decoding against the dictionary's, and the feature strings that fire at a letter with their weights."
      defaultSize="XL"
      hoverReadout={false}
      controls={
        <>
          {decisionSelect}
          <ControlRow label="word">
            <div className="flex flex-wrap items-center gap-2">
              <Input
                value={typed}
                onChange={(e) => setTyped(e.target.value)}
                className="max-w-48 font-mono"
                aria-label="word"
              />
              {EXAMPLES.map((w) => (
                <Button key={w} size="sm" variant="outline" onClick={() => setTyped(w)}>
                  {w}
                </Button>
              ))}
            </div>
          </ControlRow>
        </>
      }
      readouts={
        valid ? (
          <>
            <Readout label="dictionary" value={truth ? markHyphens(word, truth) : 'not listed'} />
            <Readout
              label={`CRF (${DECISION_NAME[decision]})`}
              value={result ? markHyphens(word, result.hyphens) : '—'}
            />
            <Readout
              label={`CRF (${DECISION_NAME[decision === 'viterbi' ? 'posterior' : 'viterbi']})`}
              value={otherResult ? markHyphens(word, otherResult.hyphens) : '—'}
            />
            <Readout label="labels" value={viterbi ? viterbi.join(' ') : '—'} />
          </>
        ) : undefined
      }
      caption={
        <>
          The bars are the marginals P(y_n = HYPH | word) from forward–backward (<code>templateCrfMarginals</code>), one
          per gap; the dots mark the dictionary&apos;s points. The hyphens shown follow the decision: the Viterbi path
          is the most probable labelling of the whole word (it maximises the joint probability); marginal (posterior)
          decoding puts a hyphen wherever P(HYPH) &gt; P(O), which maximises the expected number of correct labels. With
          two labels marginal decoding is the 0.5 threshold on the bars; the readouts show both. Hover over a letter
          button, or click one to pin it, to list the strings firing at that letter, written string/label with the label
          of the decided path, and the input cells the chosen template reads.
        </>
      }
    >
      {!valid ? (
        <div className="py-6 text-center text-sm text-muted-foreground">Type a word of at least two letters a–z.</div>
      ) : !result ? (
        <div className="py-6 text-center text-sm text-muted-foreground">Train the CRF in the first figure.</div>
      ) : (
        <div className="flex flex-col gap-3">
          <div className="flex flex-wrap items-baseline gap-6 text-sm">
            <span>
              <span className="mr-2 text-xs text-muted-foreground">dictionary</span>
              {truth ? (
                <Marked word={word} hyphens={truth} truth={null} />
              ) : (
                <span className="text-muted-foreground">not listed</span>
              )}
            </span>
            <span>
              <span className="mr-2 text-xs text-muted-foreground">
                <span
                  className="mr-1 inline-block size-2 rounded-full"
                  style={{ background: seriesColor(mode, CRF_SLOT) }}
                />
                CRF
              </span>
              <Marked word={word} hyphens={result.hyphens} truth={truth} />
            </span>
          </div>
          <Plots cols={1} scale={0.4}>
            <Plot x={gapAxis} y={probAxis} title="P(hyphen after the letter)">
              <Bars name="P(HYPH | word)" x={gapLabelsOf.map((_, i) => i)} y={result.probabilities} slot={CRF_SLOT} />
              {truth && <Points name="dictionary point" x={[...truth]} y={truth.map(() => 0.98)} emphasis size={7} />}
            </Plot>
          </Plots>
          <div className="flex flex-wrap items-center gap-1 text-xs" onMouseLeave={() => setHover(null)}>
            <span className="text-muted-foreground">letter n</span>
            {[...word].map((c, i) => (
              <button
                key={i}
                type="button"
                aria-label={`letter ${i} ${c}`}
                onMouseEnter={() => setHover(i)}
                onClick={() => setPin({ word, n: i })}
                className={cn(
                  'rounded border px-1.5 font-mono',
                  i === n ? 'border-foreground bg-muted' : 'border-border',
                  pinned === i && 'outline-2 outline-primary',
                  viterbi?.[i] === 'HYPH' && 'font-bold',
                )}
              >
                {c}
                {viterbi?.[i] === 'HYPH' ? '-' : ''}
              </button>
            ))}
          </div>
          <div className="grid items-start gap-4 lg:grid-cols-[minmax(0,2fr)_minmax(0,3fr)]">
            <TemplateCells
              rows={rows}
              columns={['letter', 'class']}
              n={n}
              templates={templates}
              template={Math.min(template, (templates?.templates.length ?? 1) - 1)}
              onPosition={(i) => setPin({ word, n: i })}
            />
            <FiringFeatureList
              features={features}
              labels={crf!.labels}
              label={label}
              previous={previous}
              trained
              template={template}
              templates={templates}
              onTemplate={setTemplate}
            />
          </div>
        </div>
      )}
    </Figure>
  )
}

// ── Errors ───────────────────────────────────────────────────────────────────────────────────────────────────────────

function ErrorsFigure({
  data,
  snap,
  decision,
}: {
  data: ReturnType<typeof mobyHyphenation>
  snap: CrfHyphenationSnapshot | null
  decision: CrfDecision
}) {
  const errors = useMemo(() => {
    if (!snap) return null
    return data.test.words
      .map((w) => {
        const h = crfHyphenate(snap.crf, w.word, decision).hyphens
        const want = new Set(w.hyphens)
        const got = new Set(h)
        const wrong = h.filter((i) => !want.has(i)).length
        const missed = w.hyphens.filter((i) => !got.has(i)).length
        return { word: w.word, hyphens: h, truth: w.hyphens, wrong, missed }
      })
      .filter((e) => e.wrong + e.missed > 0)
  }, [snap, data, decision])
  const [show, setShow] = useState<'all' | 'wrong' | 'missed'>('all')
  const shown = errors?.filter((e) => (show === 'wrong' ? e.wrong > 0 : show === 'missed' ? e.missed > 0 : true)) ?? []
  return (
    <Figure
      title="Errors against the dictionary"
      purpose="The held-out words the CRF hyphenates differently from Moby, with wrong hyphens in red and missed points as dots; many 'wrong' hyphens are defensible breaks."
      defaultSize="XL"
      hoverReadout={false}
      controls={
        <ControlRow label="show">
          <div className="flex gap-1">
            {(['all', 'wrong', 'missed'] as const).map((k) => (
              <Button key={k} size="sm" variant={show === k ? 'default' : 'outline'} onClick={() => setShow(k)}>
                {k === 'all' ? 'every error' : k === 'wrong' ? 'wrong hyphens' : 'missed points'}
              </Button>
            ))}
          </div>
        </ControlRow>
      }
      readouts={
        errors ? (
          <>
            <Readout label="words with an error" value={`${errors.length} / ${data.test.words.length}`} />
            <Readout label="wrong hyphens" value={errors.reduce((a, e) => a + e.wrong, 0)} />
            <Readout label="missed points" value={errors.reduce((a, e) => a + e.missed, 0)} />
          </>
        ) : undefined
      }
      caption={
        <>
          Each held-out word the CRF (decided as in the word figure) hyphenates differently from the dictionary, in
          frequency order (the first 240 shown). Moby splits by syllable, as in &quot;man-y&quot;, where a typesetter
          would not break, so part of the disagreement is the dictionary&apos;s convention rather than the model&apos;s
          mistake.
        </>
      }
    >
      {!errors ? (
        <div className="py-10 text-center text-sm text-muted-foreground">Train the CRF in the first figure.</div>
      ) : (
        <div className="grid max-h-96 grid-cols-[repeat(auto-fill,minmax(10rem,1fr))] gap-x-4 gap-y-1 overflow-auto">
          {shown.slice(0, 240).map((e) => (
            <Marked key={e.word} word={e.word} hyphens={e.hyphens} truth={e.truth} />
          ))}
        </div>
      )}
    </Figure>
  )
}
