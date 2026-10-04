import { useMemo, useState } from 'react'
import { stream } from 'aifn/foundation/random'
import { trace } from 'aifn/foundation/trace'
import { expectedTokensPerCall, speculativeDecoding } from 'aifn/nn/decoding'
import { charCorpus, decodeChars, encodeChars, kneserNey } from 'aifn-applied/neural/language-models'
import {
  Annotation,
  Bars,
  ControlRow,
  Curve,
  Figure,
  NumberSelector,
  Player,
  Plot,
  Plots,
  Readout,
  formatNumber,
  useAxis,
} from 'aifn-render'

const fmt = (v: number) => formatNumber(v)
const range = (n: number) => Array.from({ length: n }, (_, i) => i)
const CORPUS = charCorpus()
const show = (c: string) => (c === ' ' ? '␣' : c === '\n' ? '⏎' : c)

export function SpeculativeDecodingExplorer() {
  const [targetOrder, setTargetOrder] = useState(6)
  const [draftOrder, setDraftOrder] = useState(2)
  const [lookahead, setLookahead] = useState(4)
  const [temperature, setTemperature] = useState(1.0)
  const [position, setPosition] = useState(1)

  const seed = 0

  const models = useMemo(
    () => ({
      target: kneserNey({ order: targetOrder }).fit(CORPUS),
      draft: kneserNey({ order: draftOrder }).fit(CORPUS),
    }),
    [targetOrder, draftOrder],
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
  const cx = useAxis({ label: 'target model calls' })
  const cy = useAxis({ label: 'tokens generated' })

  const acceptedBars = round ? round.acceptance.map((a, i) => (i < round.accepted ? a : NaN)) : []
  const rejectedBars = round ? round.acceptance.map((a, i) => (i >= round.accepted ? a : NaN)) : []

  return (
    <Figure
      title="Speculative decoding in action"
      purpose="A smaller draft model proposes γ tokens in parallel, and the large target model evaluates them in a single forward pass, accepting each with probability min(1, p/q). One target forward pass produces multiple tokens with zero distribution shift."
      controls={
        <>
          <ControlRow label="Model orders">
            <NumberSelector
              label="Target n-gram order"
              value={targetOrder}
              onChange={setTargetOrder}
              min={3}
              max={7}
              step={1}
              suggestions={[4, 5, 6]}
            />
            <NumberSelector
              label="Draft n-gram order"
              value={draftOrder}
              onChange={setDraftOrder}
              min={1}
              max={4}
              step={1}
              suggestions={[1, 2, 3]}
            />
          </ControlRow>
          <ControlRow label="Speculation parameters">
            <NumberSelector
              label="Lookahead window γ"
              value={lookahead}
              onChange={setLookahead}
              min={1}
              max={8}
              step={1}
              suggestions={[2, 4, 6]}
            />
            <NumberSelector
              label="Sampling temperature"
              value={temperature}
              onChange={setTemperature}
              min={0.3}
              max={1.5}
              step={0.1}
              suggestions={[0.7, 1.0, 1.2]}
            />
            <Player
              value={pos}
              onChange={setPosition}
              count={tr.steps.length}
              label="Speculation round"
              format={(k) => `${k} of ${tr.steps.length - 1}`}
            />
          </ControlRow>
        </>
      }
      readouts={{
        'current round': round ? (
          <>
            <Readout label="Drafted tokens" value={String(round.drafted.length)} />
            <Readout label="Accepted tokens" value={String(round.accepted)} />
            <Readout
              label="Correction/bonus"
              value={`${round.rejected ? 'resampled' : 'bonus'} “${show(CORPUS.vocabulary.tokens[round.correction])}”`}
            />
          </>
        ) : (
          <Readout label="Round" value="prompt" />
        ),
        cumulative: (
          <>
            <Readout label="Acceptance rate α̂" value={fmt(alpha)} />
            <Readout label="Tokens per target call" value={fmt(at.generated / Math.max(1, at.targetCalls))} />
            <Readout
              label="Theoretical speedup"
              value={Number.isFinite(alpha) ? fmt(expectedTokensPerCall(alpha, lookahead)) : '—'}
            />
          </>
        ),
      }}
      caption="Draft tokens from a fast 2-gram model verified by a 6-gram model from prompt 'the '. Left: acceptance probabilities min(1, p/q) per draft token; green bars were accepted, grey rejected or skipped. Right: cumulative generated tokens vs target calls; above the dashed line demonstrates acceleration over standard autoregressive decoding."
    >
      <div className="mb-3 rounded border border-border bg-muted/30 px-3 py-2 font-mono text-sm leading-relaxed break-words whitespace-pre-wrap">
        <span className="text-muted-foreground font-semibold">Prompt: </span>
        <span className="text-muted-foreground">{decodeChars(CORPUS, prompt)}</span>
        <span>{decodeChars(CORPUS, at.tokens.slice(prompt.length))}</span>
      </div>
      <Plots cols={2}>
        <Plot x={ax} y={ay} title="Draft acceptance probabilities">
          <Bars name="accepted" x={range(acceptedBars.length)} y={acceptedBars} slot={0} />
          <Bars name="rejected or skipped" x={range(rejectedBars.length)} y={rejectedBars} muted />
        </Plot>
        <Plot x={cx} y={cy} title="Tokens per target call">
          <Curve name="speculative decoding" x={calls} y={generated} slot={0} showPoints />
          <Curve name="standard autoregressive (1:1)" x={calls} y={calls} dashed muted />
          <Annotation x={at.targetCalls} text="current step" dashed />
        </Plot>
      </Plots>
    </Figure>
  )
}
