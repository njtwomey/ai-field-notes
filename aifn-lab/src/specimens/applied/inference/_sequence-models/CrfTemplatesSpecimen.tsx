import { useMemo, useState } from 'react'
import {
  firingFeatures,
  templateCrf,
  templateCrfMarginals,
  templateCrfPosterior,
  templateCrfPotentials,
  templateCrfViterbi,
  TOY_POS_TAGS,
  TOY_POS_TEMPLATES,
  toyPosCorpus,
  type CrfSnapshot,
  type LabelledSequence,
  type TemplateCrf,
} from 'aifn-applied/inference/sequence-models'
import { toRows } from 'aifn/foundation/tensor'
import { featureIndex, parseTemplates, type FeatureTemplates } from 'aifn/text/features'
import { Player, Select, Slider } from '@lab/controls'
import { ControlRow, Figure } from '@lab/layout'
import { seriesColor } from '@lab/design/palette'
import { useTheme } from '@lab/design/theme'
import { cn } from '@lab/lib/utils'
import { call, int, row, useFigureState, type Task } from '@lab/state'
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
import { Readout } from '@lab/viz'

/** The first sentences train the tagger; the rest are held out. */
const TRAIN = 36
const COLUMNS = ['word', 'suffix', 'shape'] as const
const MAX_STEPS = 150
const LABELS: readonly string[] = TOY_POS_TAGS

function parse(source: string): ParsedTemplates {
  try {
    return { templates: parseTemplates(source), error: null }
  } catch (e) {
    return { templates: null, error: e as Error }
  }
}

type Settings = {
  source: string
  minFrequency: number
  optimizer: 'lbfgs' | 'owlqn' | 'sgd' | 'adam'
  c1: number
  c2: number
  stepSize?: number
  batchSize?: number
}

function trainTask(s: Settings, train: readonly LabelledSequence[]): Task<CrfSnapshot> {
  return call<CrfSnapshot>(
    'applied/inference/sequence-models/crfTrainingRun',
    call('text/features/parseTemplates', s.source),
    train,
    {
      labels: LABELS,
      minFrequency: s.minFrequency,
      optimizer: s.optimizer,
      c1: s.c1,
      c2: s.c2,
      stepSize: s.stepSize,
      batchSize: s.batchSize,
      maxSteps: s.optimizer === 'sgd' || s.optimizer === 'adam' ? 40 : MAX_STEPS,
      tolerance: 1e-4,
    },
  )
}

const softmaxRows = (rows: number[][]) =>
  rows.map((r) => {
    const m = Math.max(...r)
    const e = r.map((v) => Math.exp(v - m))
    const z = e.reduce((a, b) => a + b, 0)
    return e.map((v) => v / z)
  })

const words = (s: LabelledSequence) => s.rows.map((r) => r[0]).join(' ')

export function CrfTemplatesSpecimen() {
  const corpus = useMemo(() => toyPosCorpus(), [])
  const train = useMemo(() => corpus.slice(0, TRAIN), [corpus])
  const held = useMemo(() => corpus.slice(TRAIN), [corpus])
  const [source, setSource] = useState(TOY_POS_TEMPLATES['unigram window'])
  const parsed = useMemo(() => parse(source), [source])

  const trainState = useFigureState({
    features: row('1 · feature index', {
      minFrequency: int(1, { label: '-f: keep strings seen at least f times', ge: 1, le: 10 }),
    }),
    optimiser: crfOptimiserField('2 · optimiser', { C: 10, c1: 0.2, c2: 0.01 }, 'sentences'),
  })
  const options = crfTrainingOptions(trainState.optimiser)
  const settings: Settings = { source, minFrequency: trainState.features.minFrequency, ...options }
  const training = useTrainedRun(settings, (s) => trainTask(s, train))
  const snap = training.run.value ?? null
  const fresh = snap && training.trained && !training.stale

  // Before training (or after an edit), an untrained CRF over the index of the current templates, so the strings show.
  const preview = useMemo(() => {
    if (!parsed.templates) return null
    const index = featureIndex(
      parsed.templates,
      train.map((s) => s.rows),
      { minFrequency: settings.minFrequency },
    )
    return templateCrf(index, LABELS)
  }, [parsed.templates, train, settings.minFrequency])
  const model: TemplateCrf | null = fresh ? snap.crf : (preview ?? snap?.crf ?? null)
  const trained = Boolean(fresh)

  const heldAccuracy = useMemo(() => {
    if (!snap) return null
    const count = { viterbi: 0, posterior: 0, n: 0 }
    for (const s of held) {
      const v = templateCrfViterbi(snap.crf, s.rows).labels
      const q = templateCrfPosterior(snap.crf, s.rows).labels
      s.labels.forEach((y, i) => {
        count.n++
        if (v[i] === y) count.viterbi++
        if (q[i] === y) count.posterior++
      })
    }
    return { viterbi: count.viterbi / count.n, posterior: count.posterior / count.n }
  }, [snap, held])
  const [decision, setDecision] = useState<Decision>('viterbi')

  // The sentence and the position read; the lattice cell pinned.
  const sentences = useMemo(
    () => [
      ...held.map((s, i) => ({ value: `h${i}`, label: `held-out: ${words(s)}`, s })),
      ...train.map((s, i) => ({ value: `t${i}`, label: `training: ${words(s)}`, s })),
    ],
    [held, train],
  )
  const [picked, setPicked] = useState('h2')
  const sentence = (sentences.find((o) => o.value === picked) ?? sentences[0]).s
  const N = sentence.rows.length
  const [pos, setPos] = useState({ sentence: picked, n: 1 })
  const n = pos.sentence === picked ? Math.min(pos.n, N - 1) : 1
  const setN = (k: number) => setPos({ sentence: picked, n: k })
  const [template, setTemplate] = useState(1)
  const templates: FeatureTemplates | null = model?.index.templates ?? parsed.templates
  const templateIndex = Math.min(template, (templates?.templates.length ?? 1) - 1)

  const inference = useMemo(() => {
    if (!model) return null
    const p = templateCrfPotentials(model, sentence.rows)
    const m = templateCrfMarginals(model, sentence.rows)
    const v = templateCrfViterbi(model, sentence.rows)
    const q = templateCrfPosterior(model, sentence.rows)
    const P = toRows(m.marginals)
    const vPath = Array.from(v.path.data as Int32Array)
    const qPath = Array.from(q.path.data as Int32Array)
    const paths = {
      viterbi: {
        path: vPath,
        labels: v.labels,
        logScore: v.logProbability,
        expected: vPath.reduce((a, k, i) => a + P[i][k], 0),
      },
      posterior: { path: qPath, labels: q.labels, logScore: q.logScore, expected: q.expectedCorrect },
    }
    const chosen = paths[decision]
    const other = paths[decision === 'viterbi' ? 'posterior' : 'viterbi']
    return {
      logPsi: toRows(p.logUnary),
      alpha: softmaxRows(toRows(m.logAlpha)),
      beta: softmaxRows(toRows(m.logBeta)),
      marginals: toRows(m.marginals),
      logZ: m.logZ,
      path: chosen.path,
      other: other.path,
      paths,
      disagree: vPath.flatMap((k, i) => (k !== qPath[i] ? [i] : [])),
    }
  }, [model, sentence, decision])
  const [pin, setPin] = useState<{ sentence: string; n: number; k: number } | null>(null)
  const pinned = pin && pin.sentence === picked && pin.n === n ? pin.k : null
  const label = pinned ?? inference?.path[n] ?? 0
  const previous = n > 0 ? (inference?.path[n - 1] ?? 0) : null
  const features = useMemo(() => (model ? firingFeatures(model, sentence.rows, n) : []), [model, sentence, n])

  const status = !training.trained
    ? 'untrained: the strings fire, but every weight is 0'
    : training.stale
      ? 'settings changed: weights shown are 0 until Retrain'
      : `trained (${describeOptimiser(options)})`

  return (
    <>
      <Figure
        title="CRF++ templates and the features they fire"
        purpose="A template like U02:%x[-1,0] reads one cell of the token table relative to the current row; at every position it expands to a string, and each string conjoined with a label is one feature function with its own weight."
        defaultSize="XL"
        hoverReadout={false}
        controls={
          <>
            <ControlRow label="1 · templates (CRF++ syntax)">
              <TemplateEditor value={source} onChange={setSource} presets={TOY_POS_TEMPLATES} parsed={parsed} />
            </ControlRow>
            <ControlRow label="2 · input and position">
              <Select
                label="sentence"
                value={picked}
                onChange={setPicked}
                options={sentences.map(({ value, label }) => ({ value, label }))}
                className="max-w-xl"
              />
              <Slider label="position n" value={n} onChange={(v) => setN(Math.round(v))} min={0} max={N - 1} step={1} />
            </ControlRow>
          </>
        }
        readouts={{
          index: (
            <>
              <Readout label="unigram strings" value={model ? model.index.unigram.length : '—'} />
              <Readout label="bigram strings" value={model ? model.index.bigram.length : '—'} />
              <Readout
                label="dropped by -f"
                value={model ? model.index.dropped.unigram + model.index.dropped.bigram : '—'}
              />
              <Readout label="weights (U·K + B·K²)" value={model ? model.weights.length : '—'} />
            </>
          ),
          'at n': (
            <>
              <Readout label="token" value={`${n}: ${sentence.rows[n][0]}`} />
              <Readout
                label="label k"
                value={`${LABELS[label]}${pinned === null ? ` (${decision === 'viterbi' ? 'Viterbi' : 'posterior'})` : ' (pinned)'}`}
              />
              <Readout label="weights" value={status} />
            </>
          ),
        }}
        caption={
          <>
            Each token is a row of three columns, as in CRF++&apos;s input: the word, its last two letters and its shape
            (aifn-applied <code>toyPosCorpus</code>, {train.length} training sentences). <code>parseTemplates</code> (in{' '}
            <code>aifn/text/features</code>) reads the template file: a U line is a unigram template, conjoined with the
            label y_n; a B line is a bigram template, conjoined with the pair (y_n−1, y_n); <code>#</code> starts a
            comment, and rows past the ends read <code>_B-1</code>, <code>_B+1</code>, …. The table on the left is the
            input around position n (click a row to move there); the cells the chosen template reads are outlined with
            their macro. On the right, every string firing at n, written string/label as in <code>U01:the/DET</code>,
            with its weight λ once trained: their sum is log ψ_n(k), the log node potential. Click a feature to outline
            its cells. Pick the label k by clicking a lattice cell in the last figure; by default it is the Viterbi
            label.
          </>
        }
      >
        <div className="grid items-start gap-4 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]">
          <div>
            <div className="mb-1 text-xs text-muted-foreground">
              input rows around n = {n}; outlined: the cells of{' '}
              <span className="font-mono">{templates?.templates[templateIndex]?.text ?? '—'}</span>
            </div>
            <TemplateCells
              rows={sentence.rows}
              columns={COLUMNS}
              n={n}
              templates={templates}
              template={templateIndex}
              onPosition={setN}
            />
          </div>
          <FiringFeatureList
            features={features}
            labels={LABELS}
            label={label}
            previous={previous}
            trained={trained}
            template={templateIndex}
            templates={templates}
            onTemplate={setTemplate}
          />
        </div>
      </Figure>

      <Figure
        title="Training by L-BFGS, OWL-QN, SGD or Adam"
        purpose="Training minimises the negative conditional log-likelihood plus a penalty; its gradient is the expected minus the observed feature counts, and an L1 penalty (OWL-QN) sets most weights exactly to zero."
        state={trainState}
        defaultSize="L"
        controls={
          <TrainControls
            run={training as never}
            progress={snap ? (snap.done ? 1 : snap.step / snap.maxSteps) : 0}
            progressText={snap ? `${snap.step} steps${snap.converged ? ', converged' : ''}` : '0 steps'}
          />
        }
        readouts={
          <>
            <Readout label="−log-likelihood" value={snap ? snap.history.nll.at(-1)!.toFixed(3) : '—'} />
            <Readout
              label="non-zero weights"
              value={snap ? `${snap.history.active.at(-1)} / ${snap.crf.weights.length}` : '—'}
            />
            <Readout
              label="held-out tag accuracy (Viterbi / posterior)"
              value={
                heldAccuracy
                  ? `${(100 * heldAccuracy.viterbi).toFixed(1)}% / ${(100 * heldAccuracy.posterior).toFixed(1)}%`
                  : '—'
              }
            />
          </>
        }
        caption={
          <>
            aifn-applied <code>crfTrainingRun</code> builds the feature index of the training sentences (dropping
            strings seen fewer than f times, CRF++&apos;s <code>-f</code>), encodes them, and runs{' '}
            <code>crfTraining</code> in the worker. L-BFGS and OWL-QN come from <code>aifn/optim/second-order</code>:
            one step is one quasi-Newton iteration with its line search. OWL-QN minimises −log-likelihood + c₁‖λ‖₁ +
            c₂‖λ‖² with a pseudo-gradient and steps projected onto an orthant, so weights that cross zero stop there.
            SGD and Adam take minibatches of sentences, one epoch per step, so every optimiser&apos;s curve is against
            passes over the data: a small batch makes many noisy updates per epoch, a large one few accurate ones, and
            the full batch is plain gradient descent (Adam: one adaptive step per epoch). CRF++&apos;s <code>-c C</code>{' '}
            is the L2 strength c₂ = 1/(2C). The gradient norm is that of the pseudo-gradient for OWL-QN.
          </>
        }
      >
        <CrfTrainingPlots
          history={snap?.history ?? null}
          maxSteps={snap?.maxSteps ?? MAX_STEPS}
          stepLabel="step (iteration, or epoch for SGD / Adam)"
          runKey={training.trained}
        />
      </Figure>

      <Figure
        title="The learned weights"
        purpose="Each unigram string has one weight per label and the plain B template one weight per label pair: the strings with the largest weights are the evidence the tagger relies on, and the transitions encode tag grammar such as DET → NOUN."
        defaultSize="XL"
        caption={
          <>
            Left: the {16} unigram strings with the largest max_k |λ_u,k| (<code>topFeatures</code>) against the labels;
            positive weights favour a label, negative ones count against it. Right: the weights of the bigram string{' '}
            <code>B</code> (<code>transitionWeights</code>), previous label i down the side and label k across, the log
            edge potential log Ψ(i, k) when B is the only bigram template. Both use a diverging scale with 0 at the
            midpoint.
          </>
        }
      >
        {snap ? (
          <CrfWeightPlots crf={snap.crf} />
        ) : (
          <div className="py-10 text-center text-sm text-muted-foreground">Press Train in the figure above.</div>
        )}
      </Figure>

      <LatticeFigure
        sentence={sentence}
        inference={inference}
        n={n}
        label={label}
        pinned={pinned}
        trained={trained}
        onPin={(m, k) => {
          setN(m)
          setPin({ sentence: picked, n: m, k })
        }}
        features={features}
        previous={previous}
        templates={templates}
        template={templateIndex}
        onTemplate={setTemplate}
        decision={decision}
        onDecision={setDecision}
      />
    </>
  )
}

// ── The lattice ──────────────────────────────────────────────────────────────────────────────────────────────────────

type Inference = {
  logPsi: number[][]
  alpha: number[][]
  beta: number[][]
  marginals: number[][]
  logZ: number
  /** The chosen decision's path, and the other's. */
  path: number[]
  other: number[]
  paths: Record<Decision, { path: number[]; labels: string[]; logScore: number; expected: number }>
  /** Positions where Viterbi and posterior decoding disagree. */
  disagree: number[]
}

type Decision = 'viterbi' | 'posterior'
const DECISIONS = [
  { value: 'viterbi' as const, label: 'Viterbi (MAP sequence)' },
  { value: 'posterior' as const, label: 'marginal (max-marginal, posterior)' },
]

function LatticeFigure({
  sentence,
  inference,
  n,
  label,
  pinned,
  trained,
  onPin,
  features,
  previous,
  templates,
  template,
  onTemplate,
  decision,
  onDecision,
}: {
  sentence: LabelledSequence
  inference: Inference | null
  n: number
  label: number
  pinned: number | null
  trained: boolean
  onPin: (n: number, k: number) => void
  features: ReturnType<typeof firingFeatures>
  previous: number | null
  templates: FeatureTemplates | null
  template: number
  onTemplate: (t: number) => void
  decision: Decision
  onDecision: (d: Decision) => void
}) {
  const mode = useTheme().resolved
  const N = sentence.rows.length
  const count = 2 * N + 2
  const [at, setAt] = useState<{ key: string; step: number } | null>(null)
  const key = words(sentence)
  const step = at && at.key === key ? Math.min(at.step, count - 1) : 0
  // 0: ψ only; 1 … N: α of positions < step; N + 1 … 2N: β of positions ≥ 2N − step; 2N + 1: marginals and Viterbi.
  const alphaUpTo = Math.min(step, N)
  const betaFrom = step <= N ? N : Math.max(0, 2 * N - step)
  const final = step === count - 1
  const phase =
    step === 0
      ? 'potentials ψ_n'
      : step <= N
        ? `forward: α_${step - 1}`
        : step <= 2 * N
          ? `backward: β_${2 * N - step}`
          : 'marginals and the decided path'
  const psiMax = inference ? Math.max(1e-9, ...inference.logPsi.flat().map(Math.abs)) : 1
  const chosen = inference?.paths[decision] ?? null
  const accuracy = chosen
    ? chosen.labels.filter((y, i) => y === sentence.labels[i]).length / chosen.labels.length
    : null
  const name = (d: Decision) => (d === 'viterbi' ? 'Viterbi' : 'posterior')
  return (
    <Figure
      title="Inference on the lattice: forward–backward and Viterbi"
      purpose="The lattice has one node per position and label; forward messages α and backward messages β sweep along it, their product with ψ gives each label's marginal probability, and a decision rule picks a path: Viterbi the best whole sequence, marginal decoding the best label at each position."
      defaultSize="XL"
      hoverReadout={false}
      controls={
        <>
          <ControlRow label="decision">
            <Select label="decode by" value={decision} onChange={onDecision} options={DECISIONS} />
          </ControlRow>
          <ControlRow label="step through the recursions">
            <Player
              label="step"
              value={step}
              onChange={(s) => setAt({ key, step: s })}
              count={count}
              format={(s) =>
                s === 0 ? 'ψ only' : s <= N ? `α_${s - 1}` : s <= 2 * N ? `β_${2 * N - s}` : 'marginals + decision'
              }
            />
          </ControlRow>
        </>
      }
      readouts={{
        recursion: (
          <>
            <Readout label="phase" value={phase} />
            <Readout label="log Z(x)" value={inference && step > N ? inference.logZ.toFixed(3) : '—'} />
          </>
        ),
        decoding: (
          <>
            <Readout label={name(decision)} value={chosen && final ? chosen.labels.join(' ') : '—'} />
            <Readout label="gold tags" value={sentence.labels.join(' ')} />
            <Readout label="accuracy" value={accuracy !== null && final ? `${(100 * accuracy).toFixed(0)}%` : '—'} />
          </>
        ),
        'Viterbi / posterior': (
          <>
            <Readout
              label="disagree at"
              value={
                inference && final
                  ? inference.disagree.length === 0
                    ? 'nowhere (the same path)'
                    : inference.disagree
                        .map(
                          (i) =>
                            `n = ${i} (${inference.paths.viterbi.labels[i]} / ${inference.paths.posterior.labels[i]})`,
                        )
                        .join(', ')
                  : '—'
              }
            />
            <Readout
              label="log score Σ log ψ + Σ log Ψ"
              value={
                inference && final
                  ? `${inference.paths.viterbi.logScore.toFixed(3)} / ${inference.paths.posterior.logScore.toFixed(3)}`
                  : '—'
              }
            />
            <Readout
              label="expected correct Σ_n P(ŷ_n | x)"
              value={
                inference && final
                  ? `${inference.paths.viterbi.expected.toFixed(3)} / ${inference.paths.posterior.expected.toFixed(3)}`
                  : '—'
              }
            />
          </>
        ),
      }}
      caption={
        <>
          Columns are positions n, rows labels k. Each node shows log ψ_n(k), the sum of the weights of the unigram
          strings firing there (shaded by its size, positive in the first palette colour). The player runs{' '}
          <code>chainForwardBackward</code> (in <code>aifn/inference/exact</code>) in the notation of Twomey et al.
          (2016): forward messages α_n = Ψ_n−1ᵀ γ_n−1 with γ_n = α_n ⊙ ψ_n, then backward messages β_n = Ψ_n δ_n+1 with
          δ_n+1 = β_n+1 ⊙ ψ_n+1, where Ψ is the edge potential of the bigram strings. The two bars in each node are
          α_n(k) and β_n(k), each normalised over k for display. At the last step the fill is the marginal P(y_n = k |
          x) ∝ α_n ⊙ ψ_n ⊙ β_n and the outlined nodes are the decided path: the Viterbi path (<code>chainViterbi</code>
          ), or marginal decoding ŷ_n = argmax_k P(y_n = k | x) (<code>posteriorDecode</code>). Viterbi maximises the
          joint probability of the whole labelling; marginal decoding maximises the expected number of correct labels
          Σ_n P(ŷ_n | x), and can string together labels whose transitions are improbable or even impossible. When the
          two differ, the other path is drawn dashed; the readouts give each path&apos;s log score (Viterbi&apos;s is
          the largest) and expected number of correct labels (the posterior path&apos;s is the largest). Click a node to
          pin it: the list below shows the features that make its potentials.
          {!trained && ' The weights are 0 until the tagger is trained, so every label is equally likely.'}
        </>
      }
    >
      {!inference ? (
        <div className="py-8 text-center text-sm text-muted-foreground">Fix the templates first.</div>
      ) : (
        <div className="flex flex-col gap-3">
          <div className="overflow-x-auto">
            <table className="border-separate border-spacing-1 text-xs">
              <thead>
                <tr>
                  <th />
                  {sentence.rows.map((r, m) => (
                    <th key={m} className={cn('px-1 text-center font-mono font-medium', m === n && 'text-primary')}>
                      <div>{r[0]}</div>
                      <div className="text-[10px] font-normal text-muted-foreground">
                        n = {m} · {sentence.labels[m]}
                      </div>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {LABELS.map((y, k) => (
                  <tr key={y}>
                    <th className="pr-1 text-right font-mono font-medium">{y}</th>
                    {sentence.rows.map((_, m) => {
                      const psi = inference.logPsi[m][k]
                      const onPath = final && inference.path[m] === k
                      const onOther = final && !onPath && inference.other[m] === k
                      const p = inference.marginals[m][k]
                      const isPinned = m === n && k === label && pinned !== null
                      return (
                        <td key={m} className="p-0">
                          <button
                            type="button"
                            aria-label={`node ${m} ${y}`}
                            onClick={() => onPin(m, k)}
                            className={cn(
                              'relative flex h-12 w-16 flex-col justify-between overflow-hidden rounded border border-border p-1 text-left',
                              onPath && 'ring-2 ring-foreground',
                              onOther && 'border-2 border-dashed border-muted-foreground',
                              isPinned && 'outline-2 outline-offset-2 outline-primary',
                            )}
                          >
                            {final && (
                              <span
                                className="absolute inset-0"
                                style={{ background: seriesColor(mode, 2), opacity: 0.85 * p }}
                                aria-hidden
                              />
                            )}
                            <span
                              className="relative font-mono tabular-nums"
                              style={
                                final && p > 0.3
                                  ? undefined
                                  : {
                                      color: psi >= 0 ? seriesColor(mode, 0) : seriesColor(mode, 1),
                                      opacity: 0.4 + (0.6 * Math.abs(psi)) / psiMax,
                                    }
                              }
                            >
                              {psi.toFixed(2)}
                            </span>
                            <span className="relative flex flex-col gap-0.5">
                              <span className="h-1 rounded bg-muted">
                                {m < alphaUpTo && (
                                  <span
                                    className="block h-1 rounded"
                                    style={{
                                      width: `${100 * inference.alpha[m][k]}%`,
                                      background: seriesColor(mode, 3),
                                    }}
                                  />
                                )}
                              </span>
                              <span className="h-1 rounded bg-muted">
                                {m >= betaFrom && (
                                  <span
                                    className="block h-1 rounded"
                                    style={{
                                      width: `${100 * inference.beta[m][k]}%`,
                                      background: seriesColor(mode, 4),
                                    }}
                                  />
                                )}
                              </span>
                            </span>
                            {final && (
                              <span className="relative text-right font-mono text-[10px] tabular-nums">
                                {p.toFixed(2)}
                              </span>
                            )}
                          </button>
                        </td>
                      )
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="flex flex-wrap gap-4 text-xs text-muted-foreground">
            <span>
              <span className="mr-1 inline-block h-1 w-4 rounded" style={{ background: seriesColor(mode, 3) }} />
              α_n(k) (normalised)
            </span>
            <span>
              <span className="mr-1 inline-block h-1 w-4 rounded" style={{ background: seriesColor(mode, 4) }} />
              β_n(k) (normalised)
            </span>
            <span>
              <span className="mr-1 inline-block size-3 rounded" style={{ background: seriesColor(mode, 2) }} />
              P(y_n = k | x)
            </span>
            <span>
              outlined: the {name(decision)} path; dashed: the {name(decision === 'viterbi' ? 'posterior' : 'viterbi')}{' '}
              path where it differs; offset outline: the pinned node
            </span>
          </div>
          <div>
            <div className="mb-1 text-xs text-muted-foreground">
              features of node (n = {n}, k = {LABELS[label]})
              {previous !== null && `, with y_n−1 = ${LABELS[previous]} (${name(decision)}) for the bigram strings`}
            </div>
            <FiringFeatureList
              features={features}
              labels={LABELS}
              label={label}
              previous={previous}
              trained={trained}
              template={template}
              templates={templates}
              onTemplate={onTemplate}
            />
          </div>
        </div>
      )}
    </Figure>
  )
}
