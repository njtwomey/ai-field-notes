/**
 * "Topics that drift": aifn-applied `dynamicTopicRun` (the dynamic topic model, MAP variational EM) in the worker, on
 * `driftingTopicCorpus`, whose four themes change their words over time slices. The table shows each fitted topic's
 * top words in every slice; the rasters put one fitted topic's word distribution over time beside the planted theme it
 * matches.
 */
import { useMemo, useState } from 'react'
import { stream } from 'aifn/foundation/random'
import type { DynamicTopicSnapshot } from 'aifn-applied/inference/topic-models'
import { DRIFTING_TOPICS, driftingTopicCorpus } from 'aifn-applied/text/corpora'
import { Player } from '@lab/controls'
import { ControlRow, Figure } from '@lab/layout'
import { call, float, int, row, useFigureState, type Task } from '@lab/state'
import { formatValue, TrainControls, useTrainedRun } from '@lab/views'
import { Curve, Handle, Plot, Plots, Raster, Readout, useAxis } from '@lab/viz'

const THEMES = Object.keys(DRIFTING_TOPICS)
const f3 = (v: number | undefined) =>
  v !== undefined && Number.isFinite(v) ? formatValue(Number(v.toPrecision(4))) : '—'

type Settings = {
  slices: number
  perSlice: number
  width: number
  topics: number
  variance: number
  alpha: number
  steps: number
  seed: number
}

/** The corpus of a run as word ids, with its slices and planted topics. */
function corpusOf(slices: number, perSlice: number, width: number) {
  const c = driftingTopicCorpus(stream('dtm-page'), { slices, documentsPerSlice: perSlice, width })
  const vocabulary = c.topics!.vocabulary
  const index = new Map(vocabulary.map((w, i) => [w, i]))
  return {
    documents: c.documents.map((d) => d.split(' ').map((w) => index.get(w)!)),
    times: c.times!,
    vocabulary,
    truth: c.topics!.topicWord,
  }
}

export function DynamicTopicFigure() {
  const state = useFigureState({
    corpus: row('1 · corpus', {
      slices: int(6, { ge: 2, le: 12, suggestions: [4, 6, 10], label: 'time slices' }),
      perSlice: int(30, { ge: 5, le: 200, suggestions: [15, 30, 60], label: 'documents per slice' }),
      width: float(2.5, { gt: 0, le: 6, suggestions: [1.5, 2.5, 4], label: 'usage window (words)' }),
    }),
    model: row('2 · model', {
      topics: int(4, { ge: 2, le: 8, suggestions: [3, 4, 6], label: 'topics K' }),
      variance: float(0.5, { gt: 0, scale: 'log10', suggestions: [0.01, 0.5, 5], label: 'drift variance σ²' }),
      alpha: float(0.1, { gt: 0, scale: 'log10', suggestions: [0.05, 0.1, 0.5], label: 'α' }),
    }),
    run: row('3 · fitting', {
      steps: int(30, { ge: 1, suggestions: [10, 30, 60], label: 'EM steps' }),
      seed: int(1, { ge: 0, le: 9999, label: 'seed' }),
    }),
    topic: int(0, { ge: 0, le: 7, label: '4 · topic shown' }),
  })
  const settings: Settings = {
    slices: state.corpus.slices,
    perSlice: state.corpus.perSlice,
    width: state.corpus.width,
    topics: state.model.topics,
    variance: state.model.variance,
    alpha: state.model.alpha,
    steps: state.run.steps,
    seed: state.run.seed,
  }
  const trained = useTrainedRun(settings, (s): Task<DynamicTopicSnapshot> => {
    const c = corpusOf(s.slices, s.perSlice, s.width)
    return call<DynamicTopicSnapshot>('applied/inference/topic-models/dynamicTopicRun', {
      documents: c.documents,
      times: c.times,
      vocabulary: c.vocabulary.length,
      topics: s.topics,
      variance: s.variance,
      alpha: s.alpha,
      steps: s.steps,
      seed: s.seed,
    })
  })
  const shown = trained.trained ?? settings
  const corpus = useMemo(
    () => corpusOf(shown.slices, shown.perSlice, shown.width),
    [shown.slices, shown.perSlice, shown.width],
  )
  const snap = trained.run.value ?? undefined
  const shots = snap?.checkpoints ?? []
  const [picked, setPicked] = useState<{ run: unknown; index: number } | null>(null)
  const index = Math.min(picked && picked.run === trained.trained ? picked.index : 0, Math.max(0, shots.length - 1))
  const pick = (i: number) => setPicked({ run: trained.trained, index: i })
  const shot = shots[index]
  const { T, K, V } = snap?.shape ?? { T: shown.slices, K: shown.topics, V: corpus.vocabulary.length }
  const k = Math.min(state.topic, K - 1)
  const phi = (t: number, j: number, w: number) => shot!.topicWord[(t * K + j) * V + w]

  // The planted theme each fitted topic puts most mass on (pooled over slices).
  const match = useMemo(() => {
    if (!shot) return [] as number[]
    return Array.from({ length: K }, (_, j) => {
      const mass = THEMES.map((_, b) => {
        let m = 0
        for (let t = 0; t < T; t++) for (let i = 0; i < 12; i++) m += shot.topicWord[(t * K + j) * V + 12 * b + i]
        return m
      })
      return mass.indexOf(Math.max(...mass))
    })
  }, [shot, T, K, V])
  const theme = match[k] ?? 0
  const themeWords = DRIFTING_TOPICS[THEMES[theme]]
  const fitted = shot ? themeWords.map((_, i) => Array.from({ length: T }, (_, t) => phi(t, k, 12 * theme + i))) : null
  const planted = themeWords.map((_, i) =>
    Array.from({ length: T }, (_, t) => corpus.truth[t]?.[theme]?.[12 * theme + i] ?? 0),
  )
  // Total variation between the fitted topic and its theme, averaged over slices.
  let tv = NaN
  if (shot) {
    tv = 0
    for (let t = 0; t < T; t++)
      for (let w = 0; w < V; w++) tv += Math.abs(phi(t, k, w) - (corpus.truth[t]?.[theme]?.[w] ?? 0)) / 2
    tv /= T
  }

  const tx = useAxis({ label: 'time slice', range: [-0.5, T - 0.5], key: T, integer: true })
  const wy = useAxis({ label: `word`, categories: themeWords })
  const sx = useAxis({ label: 'EM step', range: [0, Math.max(1, snap?.steps ?? shown.steps)], key: snap?.steps })
  const oy = useAxis({ label: 'bound + topic prior', hold: 'union', key: trained.trained })
  const h = snap?.history
  const hi = h && shot ? h.step.indexOf(shot.step) : -1
  const empty = !trained.trained ? 'press Train to start' : !shot ? 'fitting…' : undefined
  const marker = shot ? (
    <Handle kind="x" at={shot.step} onDrag={(s) => pick(nearest(shots, s))} label={`step ${shot.step}`} />
  ) : null
  const slices = Array.from({ length: T }, (_, t) => t)
  return (
    <Figure
      title="Topics that drift"
      purpose="A dynamic topic model lets each topic's word distribution change between time slices: the topic's natural parameters follow a Gaussian random walk, so a theme keeps its identity while its words change, where a static model would split it into an old topic and a new one."
      state={state}
      defaultSize="XL"
      controls={
        <>
          <TrainControls
            run={trained as never}
            progress={snap ? snap.step / Math.max(1, snap.steps) : 0}
            progressText={`${snap?.step ?? 0} / ${shown.steps} steps`}
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
            <Readout label="bound + topic prior" value={hi >= 0 ? f3(h!.objective[hi]) : '—'} />
            <Readout label="log-likelihood per token" value={hi >= 0 ? f3(h!.logLikelihood[hi]) : '—'} />
            <Readout label="documents / vocabulary" value={`${corpus.documents.length} / ${V}`} />
          </>
        ),
        [`topic ${k}`]: (
          <>
            <Readout label="matched theme" value={shot ? THEMES[theme] : '—'} />
            <Readout label="mean total variation from the theme" value={f3(tv)} />
          </>
        ),
      }}
      caption={
        <>
          <code>driftingTopicCorpus</code> draws {shown.perSlice} documents in each of {shown.slices} time slices,
          mixing four themes ({THEMES.join(', ')}) whose words move along a list from old usage to new (a Gaussian
          window of the given width slides along each theme&apos;s twelve words). aifn-applied{' '}
          <code>dynamicTopicRun</code> starts every slice from one static LDA&apos;s topics, then alternates variational
          LDA for each document with the most probable chain of each topic&apos;s word distributions given the random
          walk of variance σ². The table gives each fitted topic&apos;s top three words in every slice. Below, the topic
          shown (choose it with the field) against the planted theme it matches, word by slice: the bright band climbs
          as the theme&apos;s usage moves. A small σ² ties the slices together (the topic cannot follow the drift); a
          large one lets each slice fit on its own. Right: the objective (the evidence lower bound plus the log prior of
          the topic chains) never falls; drag the step marker or play the checkpoints from step 0, where the topics are
          the static LDA&apos;s.
        </>
      }
    >
      {shot ? (
        <table className="my-2 w-full text-sm">
          <thead>
            <tr className="text-left text-muted-foreground">
              <th className="w-28 py-1 font-normal">topic</th>
              {slices.map((t) => (
                <th key={t} className="py-1 font-normal">
                  slice {t}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {Array.from({ length: K }, (_, j) => (
              <tr key={j} className={j === k ? 'bg-muted font-medium' : ''}>
                <td className="py-0.5">
                  {j} · {THEMES[match[j]]}
                </td>
                {slices.map((t) => (
                  <td key={t} className="py-0.5">
                    {Array.from({ length: V }, (_, w) => w)
                      .sort((a, b) => phi(t, j, b) - phi(t, j, a))
                      .slice(0, 3)
                      .map((w) => corpus.vocabulary[w])
                      .join(' · ')}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <div className="px-2 py-3 text-sm text-muted-foreground">{empty}</div>
      )}
      <Plots cols={3} scale={0.8}>
        <Plot x={tx} y={wy} title={empty ?? `fitted topic ${k}`}>
          {fitted && <Raster x={slices} y={themeWords.map((_, i) => i)} z={fitted} valueLabel="φ" />}
        </Plot>
        <Plot x={tx} y={wy} title={`planted theme: ${THEMES[theme]}`}>
          <Raster x={slices} y={themeWords.map((_, i) => i)} z={planted} valueLabel="φ" />
        </Plot>
        <Plot x={sx} y={oy} title={empty ?? 'objective'}>
          {h && <Curve name="bound + topic prior" x={h.step} y={h.objective} slot={0} />}
          {marker}
        </Plot>
      </Plots>
    </Figure>
  )
}

function nearest(shots: readonly { step: number }[], step: number): number {
  let best = 0
  shots.forEach((c, i) => {
    if (Math.abs(c.step - step) < Math.abs(shots[best].step - step)) best = i
  })
  return best
}
