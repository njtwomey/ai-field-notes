import { useMemo, useState } from 'react'
import { stream } from 'aifn-compute/foundation/random'
import { trace } from 'aifn-compute/foundation/trace'
import {
  beamSearch,
  expectedTokensPerCall,
  greedyDecoding,
  samplingDecoding,
  speculativeDecoding,
  type BeamState,
  type DecodingState,
  type LogitsFn,
} from 'aifn-compute/nn/decoding'
import {
  charCorpus,
  decodeChars,
  encodeChars,
  Gpt,
  gptLogits,
  kneserNey,
  type GptTrainingSnapshot,
} from 'aifn-methods/neural/language-models'
import { Figure } from 'aifn-render/layout'
import { Button, Player } from 'aifn-render/controls'
import { Input } from 'aifn-render/ui/input'
import { call, choice, row, setting, slider, useFigureState, useStreamed, when } from 'aifn-render/state'
import { Annotation, Bars, Curve, Plot, Plots, Points, Readout, Segments, formatNumber, useAxis } from 'aifn-render/viz'
import { cn } from 'aifn-render/lib/utils'

const fmt = (v: number) => formatNumber(v)
const range = (n: number) => Array.from({ length: n }, (_, i) => i)
const CORPUS = charCorpus()
const V = CORPUS.vocabulary.tokens.length
const show = (c: string) => (c === ' ' ? '␣' : c === '\n' ? '⏎' : c)
const TOP = 20

/** The generated text with the prompt muted and the token of the current step highlighted. */
function TextStrip({
  tokens,
  promptLength,
  current,
}: {
  tokens: readonly number[]
  promptLength: number
  current: number
}) {
  return (
    <div className="mb-2 rounded border border-border bg-muted/30 px-2 py-1.5 font-mono text-sm leading-relaxed break-words whitespace-pre-wrap">
      {tokens.map((t, i) => (
        <span
          key={i}
          className={cn(
            i < promptLength && 'text-muted-foreground',
            i === current && 'rounded-sm bg-foreground text-background',
          )}
        >
          {CORPUS.vocabulary.tokens[t]}
        </span>
      ))}
    </div>
  )
}

// ---------------------------------------------------------------------------------------------------------------------
// 1. Decoding strategies over an n-gram model or a tiny GPT trained in the browser.

const MODELS = [
  { value: 'ngram', label: 'Kneser–Ney n-gram' },
  { value: 'gpt', label: 'tiny GPT (trained here)' },
] as const
const STRATEGIES = [
  { value: 'greedy', label: 'greedy' },
  { value: 'sampling', label: 'sampling' },
  { value: 'beam', label: 'beam search' },
] as const

export function DecodingSpecimen() {
  const [prompt, setPrompt] = useState('mary had a ')
  const state = useFigureState({
    model: row('1 · model', {
      model: choice(MODELS, 'ngram', { label: 'model' }),
      order: slider(1, 7, 4, { label: 'n-gram order', step: 1, when: when('model', 'ngram') }),
      trainSteps: slider(50, 600, 300, { label: 'training steps', step: 50, when: when('model', 'gpt') }),
    }),
    strategy: row('2 · strategy', {
      strategy: choice(STRATEGIES, 'sampling', { label: 'strategy' }),
      temperature: slider(0.1, 2, 1, { label: 'temperature', step: 0.05, when: when('strategy', 'sampling') }),
      topK: slider(0, 20, 0, { label: 'top-k (0: off)', step: 1, when: when('strategy', 'sampling') }),
      topP: slider(0.05, 1, 0.9, { label: 'top-p', step: 0.01, when: when('strategy', 'sampling') }),
      penalty: slider(1, 2, 1, { label: 'repetition penalty', step: 0.05, when: when('strategy', 'sampling') }),
      beams: slider(1, 8, 3, { label: 'beams', step: 1, when: when('strategy', 'beam') }),
      lengthPenalty: slider(0, 2, 0, { label: 'length penalty α', step: 0.1, when: when('strategy', 'beam') }),
    }),
    run: row('3 · run', {
      tokens: slider(5, 60, 30, { label: 'tokens to generate', step: 1 }),
      seed: slider(0, 50, 0, { label: 'seed', step: 1, when: when('strategy', 'sampling') }),
    }),
  })
  const { model: which, order, trainSteps } = state.model
  const { strategy, temperature, topK, topP, penalty, beams, lengthPenalty } = state.strategy
  const { tokens: maxTokens, seed } = state.run

  // The GPT trains in a worker; the page decodes from its latest parameters as they stream in.
  const [trainRequest, setTrainRequest] = useState(0)
  const task = useMemo(
    () =>
      which === 'gpt' && trainRequest > 0
        ? call<GptTrainingSnapshot>('neural/language-models/gptTrainingRun', {
            steps: trainSteps,
            every: 25,
            seed: trainRequest,
          })
        : null,
    [which, trainSteps, trainRequest],
  )
  const training = useStreamed(task)
  const snapshot = training.value

  const ngram = useMemo(() => kneserNey({ order }).fit(CORPUS), [order])
  const logits: LogitsFn | null = useMemo(() => {
    if (which === 'ngram') return ngram.logits
    if (!snapshot) return null
    return gptLogits(Gpt(snapshot.config), snapshot.params)
  }, [which, ngram, snapshot])
  const promptIds = useMemo(() => encodeChars(CORPUS, prompt.toLowerCase()), [prompt])

  const decoded = useMemo(() => {
    if (!logits) return null
    const base = { prompt: promptIds, maxTokens }
    if (strategy === 'beam') {
      const tr = trace(beamSearch(logits, { ...base, beams, lengthPenalty }), undefined, maxTokens, { keep: 'all' })
      return { kind: 'beam' as const, steps: tr.steps as readonly BeamState[] }
    }
    const alg =
      strategy === 'greedy'
        ? greedyDecoding(logits, base)
        : samplingDecoding(logits, {
            ...base,
            temperature,
            topK: topK > 0 ? topK : undefined,
            topP: topP < 1 ? topP : undefined,
            repetitionPenalty: penalty !== 1 ? penalty : undefined,
          })
    const tr = trace(alg, undefined, maxTokens, { keep: 'all', stream: stream(`decode-${seed}`) })
    return { kind: 'tokens' as const, steps: tr.steps as readonly DecodingState[] }
  }, [logits, promptIds, maxTokens, strategy, beams, lengthPenalty, temperature, topK, topP, penalty, seed])

  const [position, setPosition] = useState(1)
  const count = decoded?.steps.length ?? 1
  const pos = Math.max(0, Math.min(position, count - 1))

  // The distribution panel (token decoders): the step's raw and processed distributions over the top tokens.
  const dist = useMemo(() => {
    if (!decoded || decoded.kind !== 'tokens' || pos === 0) return null
    const st = decoded.steps[pos]
    const order = range(V)
      .sort((a, b) => st.probs[b] - st.probs[a])
      .slice(0, TOP)
    return {
      order,
      labels: order.map((t) => show(CORPUS.vocabulary.tokens[t])),
      raw: order.map((t) => st.probs[t]),
      filtered: order.map((t) => (st.kept[t] ? st.filtered[t] : NaN)),
      cut: order.map((t) => (st.kept[t] ? NaN : st.probs[t])),
      chosen: order.indexOf(st.token),
      keptMass: st.probs.reduce((a, p, t) => a + (st.kept[t] ? p : 0), 0),
      keptCount: st.kept.filter(Boolean).length,
      token: st.token,
    }
  }, [decoded, pos])

  // The beam tree up to the current step: x = step, y = rank of the node among its step's expansions.
  const tree = useMemo(() => {
    if (!decoded || decoded.kind !== 'beam') return null
    const st = decoded.steps[pos]
    const byStep = new Map<number, typeof st.tree>()
    for (const n of st.tree) byStep.set(n.step, [...(byStep.get(n.step) ?? []), n])
    const y = new Map<number, number>()
    for (const nodes of byStep.values())
      [...nodes].sort((a, b) => b.logProb - a.logProb).forEach((n, r) => y.set(n.id, -r))
    const kept = st.tree.filter((n) => n.kept)
    const pruned = st.tree.filter((n) => !n.kept)
    const best = new Set<number>()
    let node = st.best.node
    while (node >= 0) {
      best.add(node)
      node = st.tree[node].parent
    }
    const edges = (keep: boolean) =>
      st.tree
        .filter((n) => n.parent >= 0 && n.kept === keep)
        .map((n) => ({
          from: [n.step - 1, y.get(n.parent)!] as [number, number],
          to: [n.step, y.get(n.id)!] as [number, number],
        }))
    return {
      st,
      y,
      kept,
      pruned,
      best,
      keptEdges: edges(true),
      prunedEdges: edges(false),
      bestPath: st.tree.filter((n) => best.has(n.id)).sort((a, b) => a.step - b.step),
    }
  }, [decoded, pos])

  const current =
    decoded?.kind === 'tokens'
      ? decoded.steps[pos].tokens
      : decoded?.kind === 'beam'
        ? decoded.steps[pos].best.tokens
        : promptIds

  const tx = useAxis({ label: 'next token (most probable first)', categories: dist?.labels ?? [] })
  const ty = useAxis({ label: 'probability', range: [0, 1] })
  const bx = useAxis({ label: 'step', range: [-0.5, maxTokens + 0.5] })
  const by = useAxis({ label: 'rank by log-probability', hold: 'union', key: `${beams}${maxTokens}` })
  const lx = useAxis({ label: 'training step' })
  const ly = useAxis({ label: 'loss (nats per character)' })

  return (
    <Figure
      title="Decoding strategies"
      purpose="A language model gives a distribution over the next token; greedy decoding takes its mode, sampling draws from it after temperature, top-k, top-p and a repetition penalty reshape it, and beam search keeps the few most probable continuations at once."
      defaultSize="L"
      state={state}
      controls={
        <div className="flex flex-col gap-2">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-xs text-muted-foreground">prompt</span>
            <Input
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              className="max-w-xs font-mono"
              aria-label="prompt"
            />
            {which === 'gpt' && (
              <Button variant="outline" size="sm" onClick={() => setTrainRequest((r) => r + 1)}>
                {training.running ? 'training…' : snapshot ? 'train again' : 'train the GPT'}
              </Button>
            )}
            {which === 'gpt' && training.running && (
              <Button variant="ghost" size="sm" onClick={training.stop}>
                stop
              </Button>
            )}
          </div>
          {decoded && (
            <Player
              value={pos}
              onChange={setPosition}
              count={count}
              label="4 · token"
              format={(k) => `${k} of ${count - 1}`}
              startReason="step 0 is the prompt alone; the first generated token has a distribution to show"
            />
          )}
        </div>
      }
      readouts={{
        'this step': dist ? (
          <>
            <Readout label="chosen" value={`“${show(CORPUS.vocabulary.tokens[dist.token])}”`} />
            <Readout label="tokens kept" value={`${dist.keptCount} of ${V}`} />
            <Readout label="raw mass kept" value={fmt(dist.keptMass)} />
            <Readout label="log p of the text" value={fmt((decoded!.steps[pos] as DecodingState).logProb)} />
          </>
        ) : tree ? (
          <>
            <Readout label="live beams" value={String(tree.st.beams.length)} />
            <Readout label="finished" value={String(tree.st.finished.length)} />
            <Readout label="best log p" value={fmt(tree.st.best.logProb)} />
            <Readout label="best score" value={fmt(tree.st.best.score)} />
          </>
        ) : (
          <Readout label="step" value="the prompt" />
        ),
        model: (
          <>
            <Readout
              label="model"
              value={
                which === 'ngram'
                  ? `Kneser–Ney, order ${order}`
                  : snapshot
                    ? `GPT, step ${snapshot.step} of ${snapshot.steps}`
                    : 'GPT not trained yet'
              }
            />
            {which === 'ngram' && (
              <Readout label="perplexity on the corpus" value={fmt(ngram.perplexity(CORPUS.ids.slice(0, 300)))} />
            )}
            {which === 'gpt' && snapshot && <Readout label="training loss" value={fmt(snapshot.losses.at(-1)!)} />}
          </>
        ),
      }}
      caption="aifn-compute/nn/decoding greedyDecoding, samplingDecoding and beamSearch over aifn-methods/neural/language-models: a Kneser–Ney n-gram model or a tiny GPT trained on nursery rhymes in a worker (press train; decoding follows the latest parameters). The strip shows the text with the prompt muted and the current token highlighted. Token decoders: grey bars are the model's probabilities, coloured bars the processed distribution actually sampled; grey alone means cut off by top-k or top-p. Beam search: every expansion by step and rank, kept ones coloured, the best hypothesis in ink."
    >
      <TextStrip tokens={current} promptLength={promptIds.length} current={pos > 0 ? current.length - 1 : -1} />
      {which === 'gpt' && !snapshot ? (
        <div className="py-8 text-center text-sm text-muted-foreground">Train the GPT to decode from it.</div>
      ) : tree ? (
        <Plot x={bx} y={by}>
          <Segments segments={tree.prunedEdges} width={1} />
          <Segments segments={tree.keptEdges} slot={0} width={1.5} />
          <Points
            name="pruned"
            x={tree.pruned.map((n) => n.step)}
            y={tree.pruned.map((n) => tree.y.get(n.id)!)}
            muted
            thin
          />
          <Points name="kept" x={tree.kept.map((n) => n.step)} y={tree.kept.map((n) => tree.y.get(n.id)!)} slot={0} />
          <Curve
            name="best"
            x={tree.bestPath.map((n) => n.step)}
            y={tree.bestPath.map((n) => tree.y.get(n.id)!)}
            emphasis
          />
          {tree.kept
            .filter((n) => n.step > 0)
            .map((n) => (
              <Annotation key={n.id} at={[n.step, tree.y.get(n.id)!]} text={show(CORPUS.vocabulary.tokens[n.token])} />
            ))}
        </Plot>
      ) : dist ? (
        <Plots cols={which === 'gpt' ? 2 : 1} widths={which === 'gpt' ? [2, 1] : [1]}>
          <Plot x={tx} y={ty}>
            <Bars name="model probability" x={range(dist.order.length)} y={dist.raw} muted />
            <Bars name="sampled from" x={range(dist.order.length)} y={dist.filtered} slot={0} width={0.5} />
            {dist.chosen >= 0 && <Annotation x={dist.chosen} text="chosen" dashed />}
          </Plot>
          {which === 'gpt' && snapshot ? (
            <Plot x={lx} y={ly}>
              <Curve name="loss" x={range(snapshot.losses.length)} y={snapshot.losses} slot={1} />
            </Plot>
          ) : null}
        </Plots>
      ) : (
        <div className="py-8 text-center text-sm text-muted-foreground">
          Step the player to see each token’s distribution.
        </div>
      )}
    </Figure>
  )
}

// ---------------------------------------------------------------------------------------------------------------------
// 2. Speculative decoding: a cheap n-gram drafts, a larger one verifies.

export function SpeculativeSpecimen() {
  const state = useFigureState({
    models: row('1 · models', {
      target: slider(3, 7, 6, { label: 'target n-gram order', step: 1 }),
      draft: slider(1, 4, 2, { label: 'draft n-gram order', step: 1 }),
    }),
    run: row('2 · speculation', {
      lookahead: slider(1, 8, 4, { label: 'lookahead γ', step: 1 }),
      temperature: slider(0.3, 1.5, 1, { label: 'temperature', step: 0.05 }),
      seed: slider(0, 50, 0, { label: 'seed', step: 1 }),
      greedyline: setting(true, 'one token per call (no speculation)'),
    }),
  })
  const { target, draft } = state.models
  const { lookahead, temperature, seed, greedyline } = state.run
  const models = useMemo(
    () => ({ target: kneserNey({ order: target }).fit(CORPUS), draft: kneserNey({ order: draft }).fit(CORPUS) }),
    [target, draft],
  )
  const prompt = useMemo(() => encodeChars(CORPUS, 'the '), [])
  const tr = useMemo(
    () =>
      trace(
        speculativeDecoding(models.target.logits, models.draft.logits, {
          prompt,
          maxTokens: 80,
          lookahead,
          temperature,
        }),
        undefined,
        80,
        { keep: 'all', stream: stream(`spec-${seed}`) },
      ),
    [models, prompt, lookahead, temperature, seed],
  )
  const [position, setPosition] = useState(1)
  const pos = Math.max(0, Math.min(position, tr.steps.length - 1))
  const at = tr.steps[pos]
  const round = at.round
  const alpha = at.proposed > 0 ? at.acceptedTotal / at.proposed : NaN
  const calls = useMemo(() => tr.steps.map((s) => s.targetCalls), [tr])
  const generated = useMemo(() => tr.steps.map((s) => s.generated), [tr])
  const ax = useAxis({
    label: 'draft token',
    categories: round ? round.drafted.map((t, i) => `${i + 1}: ${show(CORPUS.vocabulary.tokens[t])}`) : [],
  })
  const ay = useAxis({ label: 'acceptance min(1, p/q)', range: [0, 1] })
  const cx = useAxis({ label: 'target calls' })
  const cy = useAxis({ label: 'tokens generated' })
  const acceptedBars = round ? round.acceptance.map((a, i) => (i < round.accepted ? a : NaN)) : []
  const rejectedBars = round ? round.acceptance.map((a, i) => (i >= round.accepted ? a : NaN)) : []
  return (
    <Figure
      title="Speculative decoding"
      purpose="A small model drafts γ tokens and the large one verifies them in one pass, keeping each with probability min(1, p/q) and resampling the first rejection from the residual, so the text is distributed exactly as the large model's while each of its calls yields several tokens."
      defaultSize="L"
      state={state}
      controls={
        <Player
          value={pos}
          onChange={setPosition}
          count={tr.steps.length}
          label="3 · round"
          format={(k) => `${k} of ${tr.steps.length - 1}`}
          startReason="round 0 has drafted nothing yet"
        />
      }
      readouts={{
        'this round': round ? (
          <>
            <Readout label="drafted" value={String(round.drafted.length)} />
            <Readout label="accepted" value={String(round.accepted)} />
            <Readout
              label="then"
              value={`${round.rejected ? 'resampled' : 'bonus'} “${show(CORPUS.vocabulary.tokens[round.correction])}”`}
            />
          </>
        ) : (
          <Readout label="round" value="none yet" />
        ),
        'so far': (
          <>
            <Readout label="acceptance rate α̂" value={fmt(alpha)} />
            <Readout label="tokens per target call" value={fmt(at.generated / Math.max(1, at.targetCalls))} />
            <Readout
              label="expected per call at α̂"
              value={Number.isFinite(alpha) ? fmt(expectedTokensPerCall(alpha, lookahead)) : '—'}
            />
          </>
        ),
      }}
      caption="aifn-compute/nn/decoding speculativeDecoding with Kneser–Ney models of two orders (aifn-methods/neural/language-models) as target and draft, from the prompt “the ”. Left: each draft token of the round with its acceptance probability, accepted ones coloured, the first rejection and the rest grey. Right: tokens generated against target calls; the dashed line is one token per call."
    >
      <TextStrip tokens={at.tokens} promptLength={prompt.length} current={-1} />
      <div className="mb-2 text-xs text-muted-foreground">{decodeChars(CORPUS, at.tokens).length} characters</div>
      <Plots cols={2}>
        <Plot x={ax} y={ay}>
          <Bars name="accepted" x={range(acceptedBars.length)} y={acceptedBars} slot={0} />
          <Bars name="rejected or not reached" x={range(rejectedBars.length)} y={rejectedBars} muted />
        </Plot>
        <Plot x={cx} y={cy}>
          <Curve name="speculative" x={calls} y={generated} slot={0} showPoints />
          {greedyline && <Curve name="one per call" x={calls} y={calls} dashed muted />}
          <Annotation x={at.targetCalls} text="now" dashed />
        </Plot>
      </Plots>
    </Figure>
  )
}
