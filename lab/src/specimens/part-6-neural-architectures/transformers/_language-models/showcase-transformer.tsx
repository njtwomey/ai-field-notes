import { useMemo, useState, type ReactNode } from 'react'
import { stream } from 'aifn-compute/foundation/random'
import { fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { run } from 'aifn-compute/foundation/trace'
import {
  beamSearch,
  greedyDecoding,
  samplingDecoding,
  type BeamState,
  type DecodingState,
} from 'aifn-compute/nn/decoding'
import { recordActivations } from 'aifn-compute/nn/training'
import {
  SEQUENCE_VOCABULARY,
  encodeSequence,
  decodeSequence,
  sequenceTasks,
  type SequenceTaskData,
  type SequenceTaskName,
} from 'aifn-methods/data/synthetic'
import { Gpt, gptLogits, type GptParams, type TaskSnapshot } from 'aifn-methods/neural/language-models'
import { Diagram, type DiagramEdge, type DiagramGroup, type DiagramNode, type DiagramSpec } from 'aifn-render/diagram'
import { Button, Player } from 'aifn-render/controls'
import { ControlRow, Figure } from 'aifn-render/layout'
import { cn } from 'aifn-render/lib/utils'
import { call, choice, float, int, row, slider, useFigureState, when, type Task } from 'aifn-render/state'
import { Input } from 'aifn-render/ui/input'
import { AttentionPanel, attentionPattern, formatValue, TrainControls, useTrainedRun } from '@lab/views'
import { Bars, Curve, Handle, Plot, Plots, Raster, Readout, useAxis } from 'aifn-render/viz'

const f3 = (v: number) => (Number.isFinite(v) ? formatValue(Number(v.toPrecision(3))) : '—')
const pct = (v: number) => `${Math.round(100 * v)}%`
const VOCAB = SEQUENCE_VOCABULARY
const END = VOCAB.indexOf('.')
const SAMPLES = 8

const TASKS = [
  { value: 'reverse', label: 'reverse' },
  { value: 'copy', label: 'copy' },
  { value: 'sort', label: 'sort' },
  { value: 'induction', label: 'associative recall (induction)' },
  { value: 'dyck2', label: 'close brackets (Dyck-2)' },
  { value: 'addition', label: 'addition' },
] as const
const POSITIONS = [
  { value: 'learned', label: 'learned' },
  { value: 'sinusoidal', label: 'sinusoidal' },
  { value: 'rope', label: 'rotary (RoPE)' },
  { value: 'alibi', label: 'ALiBi' },
  { value: 'none', label: 'none (NoPE)' },
] as const
const STRATEGIES = [
  { value: 'greedy', label: 'greedy' },
  { value: 'sampling', label: 'sampling' },
  { value: 'beam', label: 'beam search' },
] as const

/** A prompt to start from, per task (the box accepts any prompt the task's truth accepts). */
const EXAMPLE: Record<SequenceTaskName, string> = {
  copy: 'bfcad',
  reverse: 'bfcad',
  sort: 'dbfca',
  dyck1: '(()(',
  dyck2: '([(',
  addition: '47+85',
  induction: 'cfadbd',
}

type Settings = {
  task: SequenceTaskName
  maxLength: number
  testLength: number
  layers: number
  heads: number
  width: number
  position: string
  norm: 'layer' | 'rms'
  steps: number
  stepSize: number
  seed: number
}

const dataKnobs = (s: Settings) => ({
  task: s.task,
  n: 2000,
  testN: 200,
  minLength: s.task === 'induction' ? 3 : s.task === 'addition' ? 1 : 2,
  maxLength: s.maxLength,
  testLength: s.testLength,
})
const dataSeed = (s: Settings) => `sequence-tasks-${s.seed}`

/** The worker task: draw the data, then train on it, streaming a snapshot per checkpoint. */
function transformerTask(s: Settings): Task<TaskSnapshot> {
  return call<TaskSnapshot>(
    'neural/language-models/taskTrainingRun',
    call('data/synthetic/sequenceTasks', call('foundation/random/stream', dataSeed(s)), dataKnobs(s)),
    {
      layers: s.layers,
      heads: s.heads,
      width: s.width,
      position: s.position,
      norm: s.norm,
      steps: s.steps,
      every: Math.max(5, Math.round(s.steps / 16)),
      evaluate: 200,
      stepSize: s.stepSize,
      seed: s.seed,
    },
  )
}

// ── The architecture diagram ─────────────────────────────────────────────────────────────────────────────────────────

/** What clicking a module shows: the activation it taps (see `Gpt.apply`), or a special panel. */
type ModuleId = string

const BLOCK_WIDTH = 9.6

/** The diagram: the residual stream left to right, each block's two branches above it, clickable modules. */
function architecture(layers: number, heads: number, position: string, norm: 'layer' | 'rms', selected: ModuleId) {
  const normName = norm === 'rms' ? 'RMSNorm' : 'LayerNorm'
  const nodes: DiagramNode[] = []
  const edges: DiagramEdge[] = []
  const groups: DiagramGroup[] = []
  const absolute = position === 'learned' || position === 'sinusoidal'
  const box = (id: string, x: number, y: number, label: string, tone: number | 'neutral', w = 2.2): DiagramNode => ({
    id,
    x,
    y,
    w,
    h: 0.9,
    label,
    tone,
    shape: 'box',
    small: true,
    selected: id === selected,
  })
  const plus = (id: string, x: number): DiagramNode => ({
    id,
    x,
    y: 3,
    shape: 'op',
    label: '$+$',
    selected: id === selected,
    ariaLabel: id === 'embed' ? 'embedding sum' : `${id}: residual stream`,
  })
  nodes.push({ id: 'tokens', x: 0, y: 3, w: 1.6, h: 0.9, shape: 'pill', label: 'tokens', small: true, tone: 'neutral' })
  nodes.push(box('tok', 2.2, 3, 'token emb. $W_E$', 0, 2))
  nodes.push({
    ...box(
      'pos',
      2.2,
      1.4,
      position === 'none' ? 'no positions' : absolute ? `${position} pos.` : `${position} (in attention)`,
      0,
      2.2,
    ),
    dashed: !absolute,
  })
  nodes.push(plus('embed', 4))
  edges.push({ from: 'tokens', to: 'tok' }, { from: 'tok', to: 'embed' })
  if (absolute) edges.push({ from: 'pos:e', to: 'embed:n', via: [[4, 1.4]] })
  let x = 4
  for (let i = 0; i < layers; i++) {
    const b = `b${i}`
    const x0 = x
    nodes.push(box(`${b}.norm1`, x0 + 1.7, 1.4, normName, 'neutral', 2))
    nodes.push(box(`${b}.attn`, x0 + 4, 1.4, `attention ×${heads}`, 1, 2.2))
    nodes.push(plus(`${b}.add1`, x0 + 5.6))
    nodes.push(box(`${b}.norm2`, x0 + 7, 4.6, normName, 'neutral', 2))
    nodes.push(box(`${b}.mlp`, x0 + 8.8 - 0.2, 4.6, 'MLP', 2, 1))
    nodes.push(plus(`${b}.add2`, x0 + BLOCK_WIDTH))
    const prev = i === 0 ? 'embed' : `b${i - 1}.add2`
    edges.push(
      { from: prev, to: `${b}.add1` },
      { from: `${b}.add1`, to: `${b}.add2` },
      {
        from: `${prev}:e`,
        to: `${b}.norm1:w`,
        via: [
          [x0 + 0.45, 3],
          [x0 + 0.45, 1.4],
        ],
        arrow: 'end',
      },
      { from: `${b}.norm1`, to: `${b}.attn` },
      { from: `${b}.attn:e`, to: `${b}.add1:n`, via: [[x0 + 5.6, 1.4]] },
      {
        from: `${b}.add1:e`,
        to: `${b}.norm2:w`,
        via: [
          [x0 + 5.95, 3],
          [x0 + 5.95, 4.6],
        ],
      },
      { from: `${b}.norm2`, to: `${b}.mlp` },
      { from: `${b}.mlp:e`, to: `${b}.add2:s`, via: [[x0 + BLOCK_WIDTH, 4.6]] },
    )
    groups.push({
      id: `${b}.group`,
      label: `block ${i + 1}`,
      rect: { x: x0 + 0.25, y: 0.6, w: BLOCK_WIDTH - 0.1, h: 4.8 },
      dashed: true,
      labelAt: 'top-left',
    })
    x += BLOCK_WIDTH
  }
  const last = layers > 0 ? `b${layers - 1}.add2` : 'embed'
  nodes.push(box('final', x + 1.6, 3, `final ${normName}`, 'neutral', 2.2))
  nodes.push(box('unembed', x + 4.1, 3, 'unembed $W_E^{\\top}$', 0, 2.2))
  nodes.push(box('softmax', x + 6.3, 3, 'softmax', 3, 1.5))
  edges.push({ from: last, to: 'final' }, { from: 'final', to: 'unembed' }, { from: 'unembed', to: 'softmax' })
  return { nodes, edges, groups, unit: 40, fitLabels: true } satisfies DiagramSpec
}

/** The activation paths a module shows, with a title each. */
function tensorsOf(id: ModuleId): { path: string; title: string; scale: 'diverging' | 'sequential' }[] {
  const block = /^b(\d+)\.(\w+)$/.exec(id)
  if (block) {
    const p = `blocks.${block[1]}`
    switch (block[2]) {
      case 'norm1':
        return [{ path: `${p}.attentionNorm`, title: 'normalised stream into attention', scale: 'diverging' }]
      case 'attn':
        return [{ path: `${p}.attention`, title: 'attention output (added to the stream)', scale: 'diverging' }]
      case 'add1':
        return [{ path: `${p}.residual`, title: 'residual stream after attention', scale: 'diverging' }]
      case 'norm2':
        return [{ path: `${p}.feedForwardNorm`, title: 'normalised stream into the MLP', scale: 'diverging' }]
      case 'mlp':
        return [
          { path: `${p}.feedForward.hidden`, title: 'MLP hidden units (after GELU)', scale: 'diverging' },
          { path: `${p}.feedForward`, title: 'MLP output (added to the stream)', scale: 'diverging' },
        ]
      case 'add2':
        return [{ path: p, title: `residual stream after block ${Number(block[1]) + 1}`, scale: 'diverging' }]
    }
  }
  switch (id) {
    case 'tok':
      return [{ path: 'embedding.tokens', title: 'token embeddings', scale: 'diverging' }]
    case 'pos':
      return [{ path: 'embedding.positions', title: 'position vectors', scale: 'diverging' }]
    case 'embed':
      return [{ path: 'embedding', title: 'residual stream into block 1', scale: 'diverging' }]
    case 'final':
      return [{ path: 'final', title: 'final normalisation', scale: 'diverging' }]
    case 'unembed':
      return [{ path: 'logits', title: 'logits', scale: 'diverging' }]
  }
  return []
}

// ── Small views ──────────────────────────────────────────────────────────────────────────────────────────────────────

const glyph = (t: string) => (t === '_' ? '·' : t)

/** Unique labels for a token axis (a repeated token gets its position). */
function tokenLabels(tokens: readonly string[]): string[] {
  return tokens.map((t, i) => `${i} ${glyph(t)}`)
}

/** A tokens × features heatmap of one activation [T, k], tokens down (first at the top). */
function TokenHeatmap({
  tensor,
  tokens,
  title,
  scale,
  feature,
}: {
  tensor: Tensor
  tokens: readonly string[]
  title: string
  scale: 'diverging' | 'sequential'
  feature: string
}) {
  const [T, k] = tensor.shape.length === 2 ? tensor.shape : [1, tensor.shape[0]]
  const z = useMemo(() => {
    const v = toFlat(tensor)
    return Array.from({ length: T }, (_, i) => Array.from({ length: k }, (_, j) => v[i * k + j])).reverse()
  }, [tensor, T, k])
  const labels = useMemo(() => [...tokenLabels(tokens)].reverse(), [tokens])
  const xs = useMemo(() => Array.from({ length: k }, (_, j) => j), [k])
  const ys = useMemo(() => Array.from({ length: T }, (_, i) => i), [T])
  const xa = useAxis({ label: feature, range: [-0.5, k - 0.5], nice: false, integer: true })
  const ya = useAxis({ label: 'token', categories: labels })
  return (
    <Plot x={xa} y={ya} title={title}>
      <Raster x={xs} y={ys} z={z} scale={scale} valueLabel={title} />
    </Plot>
  )
}

/** The answer of a sample with each character marked right or wrong against the target. */
function Marked({ output, target }: { output: string; target: string }) {
  return (
    <span className="font-mono">
      {[...output].map((c, i) => (
        <span key={i} className={c === target[i] ? 'text-success' : 'rounded-sm bg-destructive/15 text-destructive'}>
          {c}
        </span>
      ))}
      {output.length < target.length && (
        <span className="text-destructive">{'_'.repeat(target.length - output.length)}</span>
      )}
    </span>
  )
}

// ── The page ─────────────────────────────────────────────────────────────────────────────────────────────────────────

export function TransformerShowcase() {
  const state = useFigureState({
    task: row('1 · task', {
      task: choice(TASKS, 'reverse', { label: 'task' }),
      maxLength: slider(3, 7, 6, { label: 'longest training prompt', step: 1 }),
      testLength: slider(3, 9, 6, { label: 'longest test prompt', step: 1 }),
    }),
    model: row('2 · model', {
      layers: choice([1, 2], 2, { label: 'layers' }),
      heads: choice([1, 2, 4], 2, { label: 'heads' }),
      width: choice([16, 32, 48], 32, { label: 'width d' }),
      position: choice(POSITIONS, 'learned', { label: 'positions' }),
      norm: choice(
        [
          { value: 'layer', label: 'LayerNorm' },
          { value: 'rms', label: 'RMSNorm' },
        ],
        'layer',
        { label: 'normalisation' },
      ),
    }),
    run: row('3 · training run', {
      steps: int(300, { ge: 1, suggestions: [150, 300, 450, 600], label: 'Adam steps (batch 32)' }),
      stepSize: float(0.003, { label: 'step size', gt: 0, le: 0.1, scale: 'log10', suggestions: [0.001, 0.003, 0.01] }),
      seed: int(0, { label: 'seed', ge: 0, le: 9999 }),
    }),
    decode: row('4 · decoding', {
      strategy: choice(STRATEGIES, 'greedy', { label: 'strategy' }),
      temperature: slider(0.1, 2, 1, { label: 'temperature', step: 0.05, when: when('strategy', 'sampling') }),
      topK: slider(0, 10, 0, { label: 'top-k (0: off)', step: 1, when: when('strategy', 'sampling') }),
      topP: slider(0.05, 1, 1, { label: 'top-p', step: 0.01, when: when('strategy', 'sampling') }),
      beams: slider(2, 6, 3, { label: 'beams', step: 1, when: when('strategy', 'beam') }),
    }),
  })
  const settings: Settings = {
    task: state.task.task as SequenceTaskName,
    maxLength: state.task.maxLength,
    testLength: Math.max(state.task.testLength, state.task.maxLength),
    layers: Number(state.model.layers),
    heads: Number(state.model.heads),
    width: Number(state.model.width),
    position: state.model.position,
    norm: state.model.norm as 'layer' | 'rms',
    steps: Number(state.run.steps),
    stepSize: state.run.stepSize,
    seed: state.run.seed,
  }
  const trained = useTrainedRun(settings, transformerTask)
  const result = trained.run.value
  const shown = trained.trained ?? settings
  // The data are rebuilt only when what draws them changes (a plain-data key).
  const dataKey = JSON.stringify({ ...dataKnobs(shown), seed: dataSeed(shown) })
  const data = useMemo(() => {
    const { seed, ...knobs } = JSON.parse(dataKey) as ReturnType<typeof dataKnobs> & { seed: string }
    return sequenceTasks(stream(seed), knobs)
  }, [dataKey])
  const shots = useMemo(() => result?.checkpoints ?? [], [result])

  // The checkpoint shown belongs to the run it was picked on; a new run opens at step 0.
  const [picked, setPicked] = useState<{ run: Settings | null; index: number } | null>(null)
  const index = Math.min(picked && picked.run === trained.trained ? picked.index : 0, Math.max(0, shots.length - 1))
  const shot = shots[index]
  const pick = (i: number) => setPicked({ run: trained.trained, index: i })
  const pickStep = (step: number) => {
    let best = 0
    shots.forEach((c, i) => {
      if (Math.abs(c.step - step) < Math.abs(shots[best].step - step)) best = i
    })
    pick(best)
  }
  const config = result?.config
  const model = useMemo(() => (config ? Gpt(config) : null), [config])
  const logits = useMemo(() => (model && shot ? gptLogits(model, shot.params as GptParams) : null), [model, shot])

  // Sample generations on the first test prompts, with the chosen decoder.
  const { strategy, temperature, topK, topP, beams } = state.decode
  const samples = useMemo(() => {
    if (!logits || !model) return null
    return data.test.prompts.slice(0, SAMPLES).map((prompt, i) => {
      const ids = encodeSequence(`^${prompt}=`)
      const base = { prompt: ids, maxTokens: Math.max(1, model.config.context - ids.length), stop: [END] }
      let tokens: readonly number[]
      if (strategy === 'beam') {
        const best = run(beamSearch(logits, { ...base, beams }), undefined, base.maxTokens) as BeamState
        tokens = best.best.tokens
      } else {
        const alg =
          strategy === 'greedy'
            ? greedyDecoding(logits, base)
            : samplingDecoding(logits, {
                ...base,
                temperature,
                topK: topK > 0 ? topK : undefined,
                topP: topP < 1 ? topP : undefined,
              })
        tokens = (run(alg, undefined, base.maxTokens, { stream: stream(`sample-${i}`) }) as DecodingState).tokens
      }
      const output = decodeSequence(tokens.slice(ids.length))
      return { prompt, target: `${data.test.answers[i]}.`, output, score: data.truth.score(prompt, output) }
    })
  }, [logits, model, data, strategy, temperature, topK, topP, beams])

  // Curves: the loss of every step (thinned), and accuracies at the checkpoints.
  const curves = useMemo(() => {
    if (!result) return null
    const n = result.losses.length
    const stride = Math.max(1, Math.floor(n / 400))
    const x: number[] = []
    const y: number[] = []
    for (let i = 0; i < n; i += stride) {
      x.push(i)
      y.push(result.losses[i])
    }
    const cs = result.checkpoints
    return {
      x,
      y,
      cx: cs.map((c) => c.step),
      trainExact: cs.map((c) => c.train.exact),
      testExact: cs.map((c) => c.test.exact),
      testToken: cs.map((c) => c.test.token),
      testLoss: cs.map((c) => c.test.loss),
    }
  }, [result])

  const stepAxis = useAxis({ label: 'step', range: [0, result?.steps ?? shown.steps], key: result?.steps })
  const lossAxis = useAxis({
    label: 'loss (nats per answer token)',
    hold: 'union',
    key: trained.trained,
    range: [0, undefined],
  })
  const accAxis = useAxis({ label: 'accuracy', range: [0, 1] })
  const marker = shot ? <Handle kind="x" at={shot.step} onDrag={pickStep} label={`step ${shot.step}`} /> : null
  const done = result?.step ?? 0
  const player =
    shots.length > 0 ? (
      <Player
        label="checkpoint"
        value={index}
        onChange={pick}
        count={shots.length}
        format={(k) => `step ${shots[k]?.step ?? 0}`}
      />
    ) : null

  const trainingFigure = (
    <Figure
      title="Training a tiny transformer on a toy task"
      purpose="A two-layer decoder-only transformer learns an algorithmic task from examples in under a minute: the loss falls, exact-match accuracy jumps, and the generations go from noise to the answer."
      state={state}
      defaultSize="XL"
      controls={
        <>
          <TrainControls
            run={trained as never}
            progress={done / (trained.trained?.steps ?? settings.steps)}
            progressText={`${done} / ${trained.trained?.steps ?? settings.steps} steps`}
          />
          {player && <ControlRow label="5 · checkpoint">{player}</ControlRow>}
        </>
      }
      readouts={{
        checkpoint: (
          <>
            <Readout label="step" value={shot ? shot.step : '—'} />
            <Readout label="train exact" value={shot ? pct(shot.train.exact) : '—'} />
            <Readout label="test exact" value={shot ? pct(shot.test.exact) : '—'} />
            <Readout label="test token accuracy" value={shot ? pct(shot.test.token) : '—'} />
            <Readout label="test loss" value={shot ? f3(shot.test.loss) : '—'} />
          </>
        ),
        data: (
          <>
            <Readout label="task" value={shown.task} />
            <Readout label="train / test" value={`${data.train.prompts.length} / ${data.test.prompts.length}`} />
            <Readout label="row length (context)" value={data.width} />
            <Readout
              label="parameters"
              value={
                shot
                  ? Object.values(flatParams(shot.params as GptParams))
                      .reduce((a, t) => a + t, 0)
                      .toLocaleString()
                  : '—'
              }
            />
          </>
        ),
      }}
      caption={
        <>
          aifn-methods <code>sequenceTasks</code> draws 2000 training and 200 test rows <code>^ prompt = answer .</code>{' '}
          over a fixed 27-token vocabulary; <code>taskTrainingRun</code> trains a <code>Gpt</code> (pre-norm blocks of
          <code> aifn/nn/attention</code>, tied embedding) in the worker by AdamW on the cross-entropy of the answer
          tokens only, batch 32. Accuracy is teacher-forced on 200 rows: exact means every answer token is the most
          probable one, which is when greedy decoding reproduces the answer. A longest test prompt above the longest
          training prompt tests length generalisation; learned positions do not extend past the training lengths. Drag
          the step marker on either chart, or play the checkpoints; the generations below and the forward pass in the
          next figure follow the checkpoint. On the default settings, reverse reaches about 95% test exact match in 300
          steps (about 45 s in the browser).
        </>
      }
    >
      <Plots cols={2} scale={0.55}>
        <Plot x={stepAxis} y={lossAxis} title="loss">
          {curves && <Curve name="minibatch loss" x={curves.x} y={curves.y} muted thin />}
          {curves && <Curve name="test loss" x={curves.cx} y={curves.testLoss} slot={1} showPoints />}
          {marker}
        </Plot>
        <Plot x={stepAxis} y={accAxis} title="accuracy">
          {curves && <Curve name="train exact" x={curves.cx} y={curves.trainExact} slot={0} showPoints />}
          {curves && <Curve name="test exact" x={curves.cx} y={curves.testExact} slot={1} showPoints />}
          {curves && <Curve name="test per token" x={curves.cx} y={curves.testToken} slot={1} dashed />}
          {marker}
        </Plot>
      </Plots>
      <div className="mt-2 overflow-x-auto">
        {samples ? (
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-xs text-muted-foreground">
                <th className="py-1 pr-3 font-medium">test prompt</th>
                <th className="py-1 pr-3 font-medium">answer</th>
                <th className="py-1 pr-3 font-medium">
                  {strategy === 'beam' ? `beam search (${beams})` : strategy} at step {shot?.step}
                </th>
                <th className="py-1 font-medium">score</th>
              </tr>
            </thead>
            <tbody>
              {samples.map((s, i) => (
                <tr key={i} className="border-t border-border">
                  <td className="py-0.5 pr-3 font-mono">{s.prompt}</td>
                  <td className="py-0.5 pr-3 font-mono text-muted-foreground">{s.target}</td>
                  <td className="py-0.5 pr-3">
                    <Marked output={s.output} target={s.target} />
                  </td>
                  <td className={cn('py-0.5 text-xs', s.score.exact ? 'text-success' : 'text-muted-foreground')}>
                    {s.score.exact ? 'exact' : `${pct(s.score.tokenAccuracy)} of tokens`}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <div className="py-4 text-center text-sm text-muted-foreground">
            Press Train: generations on test prompts appear here at each checkpoint.
          </div>
        )}
      </div>
    </Figure>
  )

  return (
    <>
      {trainingFigure}
      <ForwardPass
        task={shown.task}
        data={data}
        model={model}
        params={(shot?.params as GptParams | undefined) ?? null}
        step={shot?.step ?? null}
        player={player}
        layers={result?.config.layers ?? shown.layers}
        heads={result?.config.heads ?? shown.heads}
        position={result?.config.position ?? shown.position}
        norm={result?.config.norm ?? shown.norm}
      />
    </>
  )
}

/** Parameter counts by leaf. */
function flatParams(p: GptParams): Record<string, number> {
  const out: Record<string, number> = {}
  const walk = (x: unknown, path: string) => {
    if (x && typeof x === 'object' && 'shape' in x && 'data' in x) {
      out[path] = (x as Tensor).shape.reduce((a, b) => a * b, 1)
      return
    }
    if (Array.isArray(x)) x.forEach((v, i) => walk(v, `${path}[${i}]`))
    else if (x && typeof x === 'object') for (const [k, v] of Object.entries(x)) walk(v, path ? `${path}.${k}` : k)
  }
  walk(p, '')
  return out
}

/** Clean a typed prompt: characters of the vocabulary that may appear in a prompt. */
const cleanPrompt = (text: string) =>
  [...text.toLowerCase()].filter((c) => VOCAB.includes(c) && !'^=._'.includes(c)).join('')

function ForwardPass({
  task,
  data,
  model,
  params,
  step,
  player,
  layers,
  heads,
  position,
  norm,
}: {
  task: SequenceTaskName
  data: SequenceTaskData
  model: Gpt | null
  params: GptParams | null
  step: number | null
  player: ReactNode
  layers: number
  heads: number
  position: string
  norm: 'layer' | 'rms'
}) {
  const [typed, setTyped] = useState<string | null>(null)
  const [selected, setSelected] = useState<ModuleId>('b0.attn')
  const [at, setAt] = useState<number | null>(null)
  const text = typed ?? EXAMPLE[task]
  // A prompt the task cannot answer (e.g. unbalanced brackets) falls back to the example.
  const prompt = useMemo(() => {
    const p = cleanPrompt(text).slice(0, data.width - 3)
    try {
      data.truth.answer(p)
      return { text: p, valid: p.length > 0 }
    } catch {
      return { text: EXAMPLE[task], valid: false }
    }
  }, [text, data, task])

  // Greedy completion from the prompt, then one forward pass over prompt and completion, recording every module.
  const pass = useMemo(() => {
    if (!model || !params) return null
    const ids = encodeSequence(`^${prompt.text}=`).slice(0, model.config.context - 1)
    const logitsFn = gptLogits(model, params)
    const maxTokens = Math.max(1, model.config.context - ids.length)
    const generated = (
      run(greedyDecoding(logitsFn, { prompt: ids, maxTokens, stop: [END] }), undefined, maxTokens) as DecodingState
    ).tokens
    // Feed everything but the last generated token: the last position then predicts it.
    const fed = generated.length > ids.length ? generated.slice(0, -1) : generated
    const recorded = recordActivations((ctx) => model.apply(params, fed, ctx))
    const tokens = fed.map((t) => VOCAB[t])
    const logits = recorded.output as Tensor
    const V = VOCAB.length
    const lv = toFlat(logits)
    const probs = Array.from({ length: fed.length }, (_, i) => {
      const row = Array.from(lv.slice(i * V, (i + 1) * V))
      const m = Math.max(...row)
      const e = row.map((v) => Math.exp(v - m))
      const s = e.reduce((a, b) => a + b, 0)
      return e.map((v) => v / s)
    })
    return {
      tokens,
      ids: fed,
      promptLength: ids.length,
      output: decodeSequence(generated.slice(ids.length)),
      activations: recorded.activations as Record<string, Tensor>,
      probs,
    }
  }, [model, params, prompt.text])

  const T = pass?.tokens.length ?? 0
  // The position whose next-token distribution is shown: the separator (it predicts the first answer token) by default.
  const position0 = pass ? pass.promptLength - 1 : 0
  const where = Math.min(at ?? position0, Math.max(0, T - 1))
  const spec = useMemo(
    () => architecture(layers, heads, position, norm, selected),
    [layers, heads, position, norm, selected],
  )
  const answer = (() => {
    try {
      return data.truth.answer(prompt.text)
    } catch {
      return ''
    }
  })()

  const px = useAxis({ label: 'next token', categories: VOCAB.map(glyph) })
  const py = useAxis({ label: 'probability', range: [0, 1] })
  const block = /^b(\d+)\.attn$/.exec(selected)
  const views = tensorsOf(selected)
  const pattern =
    pass && block
      ? attentionPattern({
          keys: pass.tokens,
          weights: pass.activations[`blocks.${block[1]}.attention.weights`],
          mask: null,
        })
      : null
  const missing = selected === 'pos' && pass && !pass.activations['embedding.positions']

  let body: ReactNode
  if (!pass) {
    body = (
      <div className="py-8 text-center text-sm text-muted-foreground">
        Train the model above: the forward pass of the checkpoint shown appears here.
      </div>
    )
  } else if (selected === 'tokens') {
    body = (
      <div className="flex flex-wrap gap-1 py-4 font-mono text-sm">
        {pass.tokens.map((t, i) => (
          <span key={i} className="rounded border border-border px-1.5 py-0.5">
            {glyph(t)} <span className="text-xs text-muted-foreground">#{pass.ids[i]}</span>
          </span>
        ))}
      </div>
    )
  } else if (selected === 'softmax') {
    body = (
      <Plots cols={2} scale={0.62}>
        <TokenHeatmap
          tensor={probsTensor(pass.probs)}
          tokens={pass.tokens}
          title="next-token probabilities"
          scale="sequential"
          feature="vocabulary index"
        />
        <Plot x={px} y={py} title={`after “${glyph(pass.tokens[where])}” (position ${where})`}>
          <Bars name="probability" x={VOCAB.map((_, k) => k)} y={pass.probs[where]} slot={3} />
        </Plot>
      </Plots>
    )
  } else if (missing) {
    body = (
      <div className="py-8 text-center text-sm text-muted-foreground">
        With {position} positions nothing is added to the embeddings: position enters inside attention (RoPE rotates
        queries and keys; ALiBi biases the scores by distance), or nowhere (NoPE: only the causal mask tells positions
        apart).
      </div>
    )
  } else {
    body = (
      <>
        {pattern && <AttentionPanel pattern={pattern} head="all" focus={where} />}
        <Plots cols={views.length} scale={pattern ? 0.45 : 0.62}>
          {views.map((v) =>
            pass.activations[v.path] ? (
              <TokenHeatmap
                key={v.path}
                tensor={pass.activations[v.path]}
                tokens={pass.tokens}
                title={v.title}
                scale={v.scale}
                feature={
                  v.path === 'logits' ? 'vocabulary index' : v.path.endsWith('hidden') ? 'hidden unit' : 'dimension'
                }
              />
            ) : null,
          )}
        </Plots>
      </>
    )
  }

  return (
    <Figure
      title="One forward pass, module by module"
      purpose="Every module of the transformer on one input you type: click a module in the diagram to see the tensor it produces, tokens down and features across, at the checkpoint chosen above."
      defaultSize="XL"
      hoverReadout={false}
      controls={
        <>
          <ControlRow label="input">
            <div className="flex flex-wrap items-center gap-2">
              <Input
                value={text}
                onChange={(e) => setTyped(e.target.value)}
                className="max-w-48 font-mono"
                aria-label="prompt"
              />
              <span className="text-xs text-muted-foreground">
                {prompt.valid
                  ? `reads ^${prompt.text}= and completes greedily; answer ${answer}.`
                  : `not a ${task} prompt: showing ${EXAMPLE[task]}`}
              </span>
              <Button size="sm" variant="outline" onClick={() => setTyped(data.test.prompts[0])}>
                a test prompt
              </Button>
            </div>
          </ControlRow>
          {pass && (
            <ControlRow label="position (next-token distribution)">
              <div className="flex flex-wrap gap-1">
                {pass.tokens.map((t, i) => (
                  <button
                    key={i}
                    type="button"
                    aria-label={`position ${i}`}
                    onClick={() => setAt(i)}
                    className={cn(
                      'rounded border px-1.5 py-0.5 font-mono text-xs',
                      i === where ? 'border-foreground bg-muted' : 'border-border',
                      i >= pass.promptLength && 'text-muted-foreground',
                    )}
                  >
                    {glyph(t)}
                  </button>
                ))}
              </div>
            </ControlRow>
          )}
          {player && <ControlRow label="checkpoint">{player}</ControlRow>}
        </>
      }
      readouts={
        pass ? (
          <>
            <Readout label="step" value={step ?? '—'} />
            <Readout label="completion" value={`${pass.output}`} />
            <Readout label="right" value={data.truth.score(prompt.text, pass.output).exact ? 'yes' : 'no'} />
            <Readout
              label={`most probable after position ${where}`}
              value={(() => {
                const p = pass.probs[where]
                const k = p.indexOf(Math.max(...p))
                return `“${glyph(VOCAB[k])}” (${f3(p[k])})`
              })()}
            />
            <Readout label="module" value={selected} />
          </>
        ) : undefined
      }
      caption={
        <>
          The diagram is the model trained above: token embedding (and absolute positions), then per block a{' '}
          {norm === 'rms' ? 'RMSNorm' : 'LayerNorm'} and multi-head causal attention added to the residual stream, then
          a normalisation and a GELU MLP added to it, then the final normalisation, the unembedding (the embedding
          matrix transposed, tied) and the softmax. Click a module: its tensor for this input is recorded by aifn{' '}
          <code>recordActivations</code> from the taps of <code>Gpt.apply</code> and <code>transformerBlock</code>.
          Heatmaps put tokens down (the first at the top) and features across, signed values on a diverging scale.
          Attention shows each head&apos;s weights (queries down, keys across; hover a query in the strip). Pick a
          position to see its next-token distribution under softmax. The input is the prompt, the separator and the
          model&apos;s own greedy answer, so a trained reverse model attends from each answer token back to its mirror
          image in the prompt; play the checkpoints to watch that pattern form.
        </>
      }
    >
      <div className="flex flex-col gap-2">
        <Diagram
          spec={spec}
          ariaLabel="The transformer's modules; click one to see its tensor"
          height={210}
          onNodeClick={(id) => setSelected(id)}
        />
        {body}
      </div>
    </Figure>
  )
}

/** Rows of probabilities as a tensor [T, V] for the heatmap. */
function probsTensor(rows: number[][]): Tensor {
  return fromData(Float64Array.from(rows.flat()), [rows.length, rows[0]?.length ?? 0])
}
