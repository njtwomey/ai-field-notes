import { useMemo, useState } from 'react'
import { stream } from 'aifn-compute/foundation/random'
import { trace } from 'aifn-compute/foundation/trace'
import {
  beamSearch,
  greedyDecoding,
  samplingDecoding,
  type BeamState,
  type DecodingState,
} from 'aifn-compute/nn/decoding'
import { charCorpus, decodeChars, encodeChars, kneserNey } from 'aifn-methods/neural/language-models'
import {
  Annotation,
  Bars,
  ControlRow,
  Curve,
  Figure,
  NumberSelector,
  Player,
  Plot,
  Points,
  Readout,
  Segments,
  Select,
  formatNumber,
  useAxis,
} from 'aifn-render'

const fmt = (v: number) => formatNumber(v)
const range = (n: number) => Array.from({ length: n }, (_, i) => i)
const CORPUS = charCorpus()
const V = CORPUS.vocabulary.tokens.length
const show = (c: string) => (c === ' ' ? '␣' : c === '\n' ? '⏎' : c)
const TOP = 16

const STRATEGIES = [
  { value: 'sampling', label: 'Sampling (temperature / top-p / top-k)' },
  { value: 'greedy', label: 'Greedy decoding (argmax)' },
  { value: 'beam', label: 'Beam search' },
]

export function DecodingStrategiesExplorer() {
  const [strategy, setStrategy] = useState<'sampling' | 'greedy' | 'beam'>('sampling')
  const [temperature, setTemperature] = useState(0.8)
  const [topP, setTopP] = useState(0.9)
  const [topK, setTopK] = useState(0)
  const penalty = 1.0
  const [beams, setBeams] = useState(3)
  const [position, setPosition] = useState(1)

  const promptText = 'mary had a '
  const maxTokens = 35
  const seed = 0

  const ngram = useMemo(() => kneserNey({ order: 4 }).fit(CORPUS), [])
  const promptIds = useMemo(() => encodeChars(CORPUS, promptText), [])

  const decoded = useMemo(() => {
    const base = { prompt: promptIds, maxTokens }
    if (strategy === 'beam') {
      const alg = beamSearch(ngram.logits, { ...base, beams, lengthPenalty: 0 })
      const tr = trace(alg, undefined, maxTokens, { keep: 'all' })
      return { kind: 'beam' as const, steps: tr.steps as readonly BeamState[] }
    }
    const alg =
      strategy === 'greedy'
        ? greedyDecoding(ngram.logits, base)
        : samplingDecoding(ngram.logits, {
            ...base,
            temperature,
            topK: topK > 0 ? topK : undefined,
            topP: topP < 1 ? topP : undefined,
            repetitionPenalty: penalty !== 1 ? penalty : undefined,
          })
    const tr = trace(alg, undefined, maxTokens, { keep: 'all', stream: stream(`decode-${seed}`) })
    return { kind: 'tokens' as const, steps: tr.steps as readonly DecodingState[] }
  }, [ngram, promptIds, maxTokens, strategy, beams, temperature, topK, topP, penalty, seed])

  const count = decoded?.steps.length ?? 1
  const pos = Math.max(0, Math.min(position, count - 1))

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
      chosen: order.indexOf(st.token),
      keptMass: st.probs.reduce((a, p, t) => a + (st.kept[t] ? p : 0), 0),
      keptCount: st.kept.filter(Boolean).length,
      token: st.token,
    }
  }, [decoded, pos])

  const tree = useMemo(() => {
    if (!decoded || decoded.kind !== 'beam') return null
    const st = decoded.steps[pos]
    const byStep = new Map<number, typeof st.tree>()
    for (const n of st.tree) byStep.set(n.step, [...(byStep.get(n.step) ?? []), n])
    const y = new Map<number, number>()
    for (const nodes of byStep.values()) {
      ;[...nodes].sort((a, b) => b.logProb - a.logProb).forEach((n, r) => y.set(n.id, -r))
    }
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

  const tx = useAxis({ label: 'candidate token (ranked by model probability)', categories: dist?.labels ?? [] })
  const ty = useAxis({ label: 'probability', range: [0, 1] })
  const bx = useAxis({ label: 'step' })
  const by = useAxis({ label: 'hypothesis rank' })

  return (
    <Figure
      title="Decoding strategies compared step by step"
      purpose="A language model outputs logits over the vocabulary; greedy decoding selects the argmax mode, sampling draws from the tail-truncated distribution (temperature, top-k, top-p), and beam search tracks multiple high-likelihood trajectories simultaneously."
      controls={
        <>
          <ControlRow label="Decoding algorithm">
            <Select
              label="Strategy"
              value={strategy}
              onChange={(v) => setStrategy(v as 'sampling' | 'greedy' | 'beam')}
              options={STRATEGIES}
            />
            {strategy === 'sampling' && (
              <>
                <NumberSelector
                  label="Temperature T"
                  value={temperature}
                  onChange={setTemperature}
                  min={0.2}
                  max={1.8}
                  step={0.1}
                  suggestions={[0.5, 0.8, 1.0, 1.2]}
                />
                <NumberSelector
                  label="Nucleus top-p"
                  value={topP}
                  onChange={setTopP}
                  min={0.1}
                  max={1.0}
                  step={0.05}
                  suggestions={[0.5, 0.8, 0.9, 1.0]}
                />
                <NumberSelector
                  label="Top-k (0 = off)"
                  value={topK}
                  onChange={setTopK}
                  min={0}
                  max={20}
                  step={2}
                  suggestions={[0, 4, 10]}
                />
              </>
            )}
            {strategy === 'beam' && (
              <NumberSelector
                label="Beam width"
                value={beams}
                onChange={setBeams}
                min={2}
                max={6}
                step={1}
                suggestions={[2, 3, 5]}
              />
            )}
          </ControlRow>
          <ControlRow label="Generation step">
            <Player
              value={pos}
              onChange={setPosition}
              count={count}
              label="Token"
              format={(k) => `${k} of ${count - 1}`}
            />
          </ControlRow>
        </>
      }
      readouts={{
        'current token': dist ? (
          <>
            <Readout label="Chosen token" value={`“${show(CORPUS.vocabulary.tokens[dist.token])}”`} />
            <Readout label="Tokens kept" value={`${dist.keptCount} of ${V}`} />
            <Readout label="Raw mass kept" value={fmt(dist.keptMass)} />
            <Readout label="Sequence log-probability" value={fmt((decoded!.steps[pos] as DecodingState).logProb)} />
          </>
        ) : tree ? (
          <>
            <Readout label="Active beams" value={String(tree.st.beams.length)} />
            <Readout label="Finished beams" value={String(tree.st.finished.length)} />
            <Readout label="Best log-probability" value={fmt(tree.st.best.logProb)} />
            <Readout label="Best score" value={fmt(tree.st.best.score)} />
          </>
        ) : (
          <Readout label="State" value="prompt prefix" />
        ),
      }}
      caption="Interactive generation from prompt 'mary had a ' over a 4-gram Kneser–Ney language model. For sampling: grey bars represent raw model probabilities; coloured bars show the truncated, temperature-scaled sampling distribution. For beam search: tracked paths with pruned candidates."
    >
      <div className="mb-3 rounded border border-border bg-muted/30 px-3 py-2 font-mono text-sm leading-relaxed break-words whitespace-pre-wrap">
        <span className="font-semibold text-muted-foreground">Generated: </span>
        <span className="text-muted-foreground">{decodeChars(CORPUS, promptIds)}</span>
        <span>{decodeChars(CORPUS, current.slice(promptIds.length))}</span>
      </div>
      {tree ? (
        <Plot x={bx} y={by} title="Beam search expansion tree">
          <Segments segments={tree.prunedEdges} width={1} />
          <Segments segments={tree.keptEdges} slot={0} width={1.5} />
          <Points
            name="pruned"
            x={tree.pruned.map((n) => n.step)}
            y={tree.pruned.map((n) => tree.y.get(n.id)!)}
            muted
          />
          <Points name="kept" x={tree.kept.map((n) => n.step)} y={tree.kept.map((n) => tree.y.get(n.id)!)} slot={0} />
          <Curve
            name="best hypothesis"
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
        <Plot x={tx} y={ty} title="Next-token distribution">
          <Bars name="raw model probability" x={range(dist.order.length)} y={dist.raw} muted />
          <Bars name="sampled distribution" x={range(dist.order.length)} y={dist.filtered} slot={0} width={0.5} />
          {dist.chosen >= 0 && <Annotation x={dist.chosen} text="sampled" dashed />}
        </Plot>
      ) : (
        <div className="py-8 text-center text-sm text-muted-foreground">
          Step the player forward to view token distributions.
        </div>
      )}
    </Figure>
  )
}
