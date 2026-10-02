/**
 * Showcase: topic models compared on the seeded topic corpus of aifn-applied `text/corpora`. Sentences are joined into
 * documents of one topic each (`groupByLabel`); one method of `topicModelRun` (LDA, the HDP, pLSA, NMF, LSA, the
 * correlated topic model, labelled LDA) is fitted in the worker, streaming the topics' top words, the documents' topic proportions, the
 * log-likelihood per token and the NPMI coherence of the topics.
 */
import { useMemo, useState } from 'react'
import { stream } from 'aifn/foundation/random'
import {
  bagOfWordsCorpus,
  groupByLabel,
  TOPIC_METHODS,
  type TopicMethod,
  type TopicSnapshot,
} from 'aifn-applied/inference/topic-models'
import { TOPIC_CORPUS_TOPICS, topicCorpus } from 'aifn-applied/text/corpora'
import { Player } from '@lab/controls'
import { ControlRow, Figure } from '@lab/layout'
import { call, choice, float, int, pinField, row, useFigureState, usePinned, when, type Task } from '@lab/state'
import { formatValue, TrainControls, useTrainedRun } from '@lab/views'
import { Curve, Handle, Plot, Plots, Raster, Readout, Segments, useAxis } from '@lab/viz'

const f3 = (v: number | undefined) =>
  v !== undefined && Number.isFinite(v) ? formatValue(Number(v.toPrecision(3))) : '—'
const STOP = ['the', 'a', 'some', 'in', 'to', 'near']

type Settings = {
  method: TopicMethod
  topics: number
  steps: number
  alpha: number
  beta: number
  gamma: number
  sentences: number
  perDocument: number
  seed: number
}

/** The corpus of a run: seeded sentences as word ids, joined into documents of one corpus topic each. */
function corpusOf(sentences: number, perDocument: number) {
  const c = topicCorpus(stream(3), { sentences })
  const bow = bagOfWordsCorpus(c.documents, { stopWords: STOP })
  const g = groupByLabel(bow.documents, c.labels ?? [], perDocument)
  return { ...g, words: bow.vocabulary }
}

export function TopicModelComparison() {
  const state = useFigureState({
    corpus: row('1 · corpus', {
      sentences: int(600, { ge: 50, le: 3000, suggestions: [300, 600, 1500], label: 'sentences' }),
      perDocument: int(4, { ge: 1, le: 20, suggestions: [1, 4, 8], label: 'sentences per document' }),
    }),
    model: row('2 · model', {
      method: choice(
        TOPIC_METHODS.map((m) => ({ value: m.method, label: m.name })),
        'lda',
        { label: 'method' },
      ),
      topics: int(5, { ge: 2, le: 12, suggestions: [3, 5, 8], label: 'topics K (HDP: to start with)' }),
      alpha: float(0.5, {
        gt: 0,
        scale: 'log10',
        suggestions: [0.1, 0.5, 1],
        label: 'α',
        when: (v) => v.method === 'lda' || v.method === 'hdp',
      }),
      beta: float(0.05, {
        gt: 0,
        scale: 'log10',
        suggestions: [0.01, 0.05, 0.2],
        label: 'β (HDP: η)',
        when: (v) => v.method === 'lda' || v.method === 'hdp',
      }),
      gamma: float(1, {
        gt: 0,
        scale: 'log10',
        suggestions: [0.1, 1, 5],
        label: 'γ (top level)',
        when: when('method', 'hdp'),
      }),
    }),
    run: row('3 · fitting', {
      steps: int(60, { ge: 1, suggestions: [20, 60, 150], label: 'steps (sweeps, EM or NMF updates)' }),
      seed: int(1, { ge: 0, le: 9999, label: 'seed' }),
    }),
    pin: pinField(),
  })
  const settings: Settings = {
    method: state.model.method as TopicMethod,
    topics: state.model.topics,
    steps: state.run.steps,
    alpha: state.model.alpha ?? 0.5,
    beta: state.model.beta ?? 0.05,
    gamma: state.model.gamma ?? 1,
    sentences: state.corpus.sentences,
    perDocument: state.corpus.perDocument,
    seed: state.run.seed,
  }
  const trained = useTrainedRun(settings, (s): Task<TopicSnapshot> => {
    const c = corpusOf(s.sentences, s.perDocument)
    return call<TopicSnapshot>('applied/inference/topic-models/topicModelRun', {
      documents: c.documents,
      vocabulary: c.words.length,
      labels: c.labels,
      method: s.method,
      topics: s.topics,
      steps: s.steps,
      alpha: s.alpha,
      beta: s.beta,
      gamma: s.gamma,
      seed: s.seed,
    })
  })
  const shown = trained.trained ?? settings
  const corpus = useMemo(() => corpusOf(shown.sentences, shown.perDocument), [shown.sentences, shown.perDocument])
  const snap = trained.run.value ?? undefined
  const shots = snap?.checkpoints ?? []
  const [picked, setPicked] = useState<{ run: unknown; index: number } | null>(null)
  const index = Math.min(picked && picked.run === trained.trained ? picked.index : 0, Math.max(0, shots.length - 1))
  const pick = (i: number) => setPicked({ run: trained.trained, index: i })
  const shot = shots[index]
  // The HDP's number of topics changes from checkpoint to checkpoint.
  const K = shot?.topics ?? snap?.topics ?? shown.topics
  const D = corpus.documents.length
  const V = corpus.words.length
  const pins = usePinned(state.pin, (v) => state.set('pin', v), { valid: (v) => v < D })
  const doc = pins.focus

  const field = useMemo(
    () => (shot ? Array.from({ length: D }, (_, d) => Array.from(shot.docTopic.subarray(d * K, (d + 1) * K))) : null),
    [shot, D, K],
  )
  // Lines between the documents of different corpus topics (the documents come grouped by topic).
  const bounds = useMemo(() => {
    const out: { from: [number, number]; to: [number, number] }[] = []
    for (let d = 1; d < D; d++)
      if (corpus.labels[d] !== corpus.labels[d - 1]) out.push({ from: [-0.5, d - 0.5], to: [K - 0.5, d - 0.5] })
    return out
  }, [corpus, D, K])
  const docTop =
    doc !== null && shot
      ? Array.from({ length: K }, (_, k) => k).sort(
          (a, b) => shot.docTopic[doc * K + b] - shot.docTopic[doc * K + a],
        )[0]
      : null

  const tx = useAxis({ label: 'topic', range: [-0.5, K - 0.5], key: K, integer: true })
  const dy = useAxis({ label: 'document (grouped by corpus topic)', range: [-0.5, D - 0.5], key: D, integer: true })
  const sx = useAxis({ label: 'step', range: [0, Math.max(1, snap?.steps ?? shown.steps)], key: snap?.steps })
  const ly = useAxis({ label: 'log-likelihood per token', hold: 'union', key: trained.trained })
  const cy = useAxis({ label: 'mean NPMI coherence', range: [-1, 1] })
  const h = snap?.history
  const hi = h && shot ? h.step.indexOf(shot.step) : -1
  const empty = !trained.trained ? 'press Train to start' : !shot ? 'fitting…' : undefined
  const name = TOPIC_METHODS.find((m) => m.method === shown.method)?.name ?? shown.method
  const marker = shot ? (
    <Handle kind="x" at={shot.step} onDrag={(s) => pick(nearest(shots, s))} label={`step ${shot.step}`} />
  ) : null
  return (
    <Figure
      title="Topic models compared"
      purpose="Each method factors the documents' word counts into topics: LDA and pLSA as mixtures of word distributions (the HDP as LDA with the number of topics inferred), NMF and LSA as matrix factorisations, the CTM with correlated proportions, labelled LDA with the topics fixed by the labels. Coherent topics put words that co-occur together."
      state={state}
      defaultSize="XL"
      controls={
        <>
          <TrainControls
            run={trained as never}
            progress={snap ? snap.step / Math.max(1, snap.steps) : 0}
            progressText={`${snap?.step ?? 0} / ${shown.method === 'lsa' ? 0 : shown.steps} steps`}
          />
          <ControlRow label="checkpoints">
            <Player
              value={index}
              onChange={pick}
              count={Math.max(1, shots.length)}
              label="checkpoint"
              format={(i) => `step ${shots[i]?.step ?? 0}`}
            />
          </ControlRow>
        </>
      }
      readouts={{
        fit: (
          <>
            <Readout label="step" value={shot ? shot.step : '—'} />
            <Readout label="log-likelihood per token" value={hi >= 0 ? f3(h!.logLikelihood[hi]) : '—'} />
            <Readout label="mean NPMI" value={hi >= 0 ? f3(h!.coherence[hi]) : '—'} />
            <Readout label="topics in use" value={shot ? shot.topics : '—'} />
            <Readout label="documents / vocabulary" value={`${D} / ${V}`} />
          </>
        ),
        document: (
          <>
            <Readout
              label="document"
              value={doc === null ? 'click a row' : `${doc}${pins.pinned !== null ? ' (pinned)' : ''}`}
            />
            <Readout label="corpus topic" value={doc === null ? '—' : TOPIC_CORPUS_TOPICS[corpus.labels[doc]]} />
            <Readout label="dominant topic" value={docTop === null ? '—' : `topic ${docTop}`} />
            <Readout
              label="words"
              value={
                doc === null
                  ? '—'
                  : corpus.documents[doc]
                      .slice(0, 16)
                      .map((w) => corpus.words[w])
                      .join(' ')
              }
            />
          </>
        ),
      }}
      caption={
        <>
          aifn-applied <code>topicModelRun</code> fits {name} to {D} documents of {shown.perDocument} sentences each
          from <code>topicCorpus</code> (topics: {TOPIC_CORPUS_TOPICS.join(', ')}; stop words removed). The table lists
          each topic&apos;s top words at the checkpoint with its NPMI coherence (<code>topicCoherence</code>, document
          co-occurrence). Left: every document&apos;s topic proportions (LSA: signed coordinates), documents grouped by
          their corpus topic between the lines; click a row to pin a document, Escape to unpin. Right: the
          log-likelihood per token (not defined for LSA) and the mean coherence against the step; drag the step marker
          or play the checkpoints from step 0. The HDP starts from K topics and opens or closes topics as it samples;
          the number in use is read out beside the fit and is the table&apos;s number of rows.
        </>
      }
    >
      <TopWordsTable shot={shot} words={corpus.words} highlight={docTop} empty={empty} />
      <Plots cols={3} scale={0.8}>
        <Plot
          x={tx}
          y={dy}
          title={empty ?? 'document–topic weights'}
          onPlotClick={([, y]) => pins.toggle(Math.max(0, Math.min(D - 1, Math.round(y))))}
        >
          {field && (
            <Raster
              x={Array.from({ length: K }, (_, k) => k)}
              y={Array.from({ length: D }, (_, d) => d)}
              z={field}
              scale={shown.method === 'lsa' ? 'diverging' : 'sequential'}
              valueLabel={shown.method === 'lsa' ? 'coordinate' : 'θ'}
            />
          )}
          <Segments name="corpus topics" segments={bounds} emphasis width={1} />
          {doc !== null && (
            <Segments
              name={`document ${doc}`}
              segments={[
                { from: [-0.5, doc - 0.5], to: [K - 0.5, doc - 0.5] },
                { from: [-0.5, doc + 0.5], to: [K - 0.5, doc + 0.5] },
              ]}
              emphasis
              width={2}
            />
          )}
        </Plot>
        <Plot x={sx} y={ly} title={empty ?? 'fit'}>
          {h && <Curve name="log-likelihood per token" x={h.step} y={h.logLikelihood} slot={0} />}
          {marker}
        </Plot>
        <Plot x={sx} y={cy} title={empty ?? 'coherence'}>
          {h && <Curve name="mean NPMI" x={h.step} y={h.coherence} slot={1} />}
          {marker}
        </Plot>
      </Plots>
    </Figure>
  )
}

/** Each topic's top words with its coherence, the pinned document's dominant topic highlighted. */
function TopWordsTable({
  shot,
  words,
  highlight,
  empty,
}: {
  shot: TopicSnapshot['checkpoints'][number] | undefined
  words: readonly string[]
  highlight: number | null
  empty: string | undefined
}) {
  if (!shot) return <div className="px-2 py-3 text-sm text-muted-foreground">{empty}</div>
  return (
    <table className="my-2 w-full text-sm">
      <thead>
        <tr className="text-left text-muted-foreground">
          <th className="w-20 py-1 font-normal">topic</th>
          <th className="py-1 font-normal">top words</th>
          <th className="w-24 py-1 text-right font-normal">NPMI</th>
        </tr>
      </thead>
      <tbody>
        {shot.top.map((ws, k) => (
          <tr key={k} className={k === highlight ? 'bg-muted font-medium' : ''}>
            <td className="py-0.5">topic {k}</td>
            <td className="py-0.5">{ws.map((w) => words[w]).join(' · ')}</td>
            <td className="py-0.5 text-right tabular-nums">{f3(shot.coherence[k])}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

function nearest(shots: readonly { step: number }[], step: number): number {
  let best = 0
  shots.forEach((c, i) => {
    if (Math.abs(c.step - step) < Math.abs(shots[best].step - step)) best = i
  })
  return best
}
