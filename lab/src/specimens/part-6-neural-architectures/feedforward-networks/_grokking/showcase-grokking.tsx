import { useMemo, useState } from 'react'
import { stream } from 'aifn-compute/foundation/random'
import { toFlat, unwrap, type Tensor } from 'aifn-compute/foundation/tensor'
import { recordActivations } from 'aifn-compute/nn/training'
import { modularArithmetic, type ModularOperation } from 'aifn-methods/data/synthetic'
import { ModularMlp, pairsOf, type GrokkingSnapshot, type ModularMlpParams } from 'aifn-methods/neural/grokking'
import { Button, Player, Slider } from 'aifn-render/controls'
import { ControlRow, Figure } from 'aifn-render/layout'
import { call, choice, float, int, row, slider, useFigureState, type Task } from 'aifn-render/state'
import { formatValue, optimiserField, TrainControls, useTrainedRun } from '@lab/views'
import { Bars, Curve, Handle, Plot, Plots, Points, Raster, Readout, useAxis } from 'aifn-render/viz'

const f3 = (v: number) => (Number.isFinite(v) ? formatValue(Number(v.toPrecision(3))) : '—')
const pct = (v: number) => `${Math.round(100 * v)}%`

const OPERATIONS = [
  { value: '+', label: 'a + b' },
  { value: '-', label: 'a − b' },
  { value: '*', label: 'a × b' },
  { value: '/', label: 'a ÷ b' },
] as const

type Settings = {
  p: number
  op: ModularOperation
  fraction: number
  weightDecay: number
  width: number
  steps: number
  stepSize: number
  /** AdamW (default) or full-batch L-BFGS on the cross-entropy plus (λ/2)‖θ‖². */
  method: 'adamw' | 'lbfgs'
  memory: number
  seed: number
}

const knobsOf = (s: Settings) => ({ p: s.p, op: s.op, fraction: s.fraction })
const seedOf = (s: Settings) => `grokking-data-${s.seed}`

function grokkingTask(s: Settings): Task<GrokkingSnapshot> {
  return call<GrokkingSnapshot>(
    'neural/grokking/grokkingRun',
    call('data/synthetic/modularArithmetic', call('foundation/random/stream', seedOf(s)), knobsOf(s)),
    {
      width: s.width,
      embed: 32,
      steps: s.steps,
      stepSize: s.stepSize,
      weightDecay: s.weightDecay,
      method: s.method,
      memory: s.memory,
      recordEvery: 10,
      checkpointEvery: Math.max(10, Math.round(s.steps / 30 / 10) * 10),
      seed: s.seed,
    },
  )
}

/** Coordinates of every residue's embedding in the plane of one frequency's cosine and sine components. */
function circleOf(embedding: Tensor, basis: Tensor, p: number, k: number): { x: number[]; y: number[] } {
  const E = toFlat(embedding)
  const B = toFlat(basis)
  const d = embedding.shape[1]
  // c = Σ_a B[a, cos k] E[a], s likewise: the embedding's two coefficient vectors at frequency k.
  const coef = (col: number) => {
    const v = new Float64Array(d)
    for (let a = 0; a < p; a++) for (let j = 0; j < d; j++) v[j] += B[a * p + col] * E[a * d + j]
    return v
  }
  const c = coef(2 * k - 1)
  const s = 2 * k < p ? coef(2 * k) : new Float64Array(d)
  // An orthonormal basis of span(c, s) by Gram–Schmidt, then each residue's projection onto it.
  const dot = (u: Float64Array, v: Float64Array) => u.reduce((acc, x, j) => acc + x * v[j], 0)
  const nc = Math.sqrt(dot(c, c)) || 1
  const u1 = c.map((x) => x / nc)
  const r = s.map((x, j) => x - dot(s, u1) * u1[j])
  const nr = Math.sqrt(dot(r, r)) || 1
  const u2 = r.map((x) => x / nr)
  const x: number[] = []
  const y: number[] = []
  for (let a = 0; a < p; a++) {
    const row = Float64Array.from({ length: d }, (_, j) => E[a * d + j])
    x.push(dot(row, u1))
    y.push(dot(row, u2))
  }
  return { x, y }
}

export function GrokkingShowcase() {
  const state = useFigureState({
    task: row('1 · task', {
      op: choice(OPERATIONS, '+', { label: 'operation' }),
      p: choice([23, 31, 37, 43], 31, { label: 'modulus p' }),
      fraction: slider(0.3, 0.8, 0.5, { label: 'training fraction', step: 0.05 }),
    }),
    model: row('2 · model and optimiser', {
      width: int(128, { ge: 8, le: 512, suggestions: [64, 128, 256], label: 'hidden units' }),
      weightDecay: float(2, { ge: 0, step: 0.5, suggestions: [0, 0.5, 1, 2, 4], label: 'weight decay λ' }),
    }),
    optimiser: optimiserField({
      label: '3 · optimiser (full batch)',
      adamLabel: 'AdamW (decoupled decay)',
      stepSize: 0.01,
      suggestions: [0.003, 0.01, 0.02],
    }),
    run: row('4 · training run', {
      steps: int(1500, { ge: 1, suggestions: [1000, 1500, 2000, 3000], label: 'full-batch steps' }),
      seed: int(0, { label: 'seed', ge: 0, le: 9999 }),
    }),
  })
  const settings: Settings = {
    p: Number(state.task.p),
    op: state.task.op as ModularOperation,
    fraction: state.task.fraction,
    weightDecay: Number(state.model.weightDecay),
    width: Number(state.model.width),
    steps: Number(state.run.steps),
    stepSize: Number(state.optimiser.values.stepSize ?? 0.01),
    method: state.optimiser.key === 'lbfgs' ? 'lbfgs' : 'adamw',
    memory: Number(state.optimiser.values.memory ?? 10),
    seed: state.run.seed,
  }
  const trained = useTrainedRun(settings, grokkingTask)
  const result = trained.run.value
  const shown = trained.trained ?? settings
  const { p: shownP, op: shownOp, fraction: shownFraction, seed: shownSeed } = shown
  const data = useMemo(
    () =>
      modularArithmetic(stream(seedOf({ seed: shownSeed } as Settings)), {
        p: shownP,
        op: shownOp,
        fraction: shownFraction,
      }),
    [shownP, shownOp, shownFraction, shownSeed],
  )
  const p = data.p
  const shots = useMemo(() => result?.checkpoints ?? [], [result])

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

  // The embedding's Fourier spectrum at every checkpoint (core rfft, through the truth's `spectrum`).
  const spectra = useMemo(
    () => shots.map((c) => Array.from(data.truth.spectrum((c.params as ModularMlpParams).embedding))),
    [shots, data],
  )
  const spectrum = spectra[index]
  const half = Math.floor(p / 2)
  const frequencies = useMemo(() => Array.from({ length: half }, (_, k) => k + 1), [half])
  // The heatmap: frequency (rows, 1 … ⌊p/2⌋; the constant is left out) by checkpoint.
  const field = useMemo(() => frequencies.map((k) => spectra.map((s) => s[k])), [frequencies, spectra])
  const top = useMemo(() => {
    if (!spectrum) return [] as number[]
    return frequencies
      .slice()
      .sort((a, b) => spectrum[b] - spectrum[a])
      .slice(0, 5)
  }, [spectrum, frequencies])
  const topShare = spectrum ? top.reduce((a, k) => a + spectrum[k], 0) : 0

  // The residues in the plane of the strongest frequency at this checkpoint.
  const circle = useMemo(
    () =>
      shot && top.length
        ? circleOf((shot.params as ModularMlpParams).embedding, data.truth.fourierBasis, p, top[0])
        : null,
    [shot, top, data, p],
  )

  // One hidden unit over the whole table.
  const config = result?.config
  const model = useMemo(() => (config ? ModularMlp(config) : null), [config])
  const tablePairs = useMemo(() => pairsOf(data.table), [data])
  const hidden = useMemo(() => {
    if (!model || !shot) return null
    const rec = recordActivations((ctx) => model.apply(shot.params as ModularMlpParams, tablePairs, ctx))
    return rec.activations.hidden as Tensor
  }, [model, shot, tablePairs])
  const [neuronPick, setNeuron] = useState<number | null>(null)
  // By default the unit with the largest outgoing weights at this checkpoint.
  const strongest = useMemo(() => {
    if (!shot) return 0
    const out = (shot.params as ModularMlpParams).out
    const W = toFlat(unwrap(out) as Tensor)
    const [h, q] = out.shape
    let best = 0
    let bestNorm = -1
    for (let j = 0; j < h; j++) {
      let n = 0
      for (let k = 0; k < q; k++) n += W[j * q + k] ** 2
      if (n > bestNorm) [best, bestNorm] = [j, n]
    }
    return best
  }, [shot])
  const width = result?.config.width ?? shown.width
  const neuron = Math.min(neuronPick ?? strongest, width - 1)
  const neuronGrid = useMemo(() => {
    if (!hidden) return null
    const v = toFlat(hidden)
    const h = hidden.shape[1]
    const x = toFlat(data.table.x)
    const grid = Array.from({ length: p }, () => new Array<number>(p).fill(NaN))
    for (let i = 0; i < v.length / h; i++) grid[x[2 * i + 1]][x[2 * i]] = v[i * h + neuron]
    return grid
  }, [hidden, neuron, data, p])

  const c = result?.curves
  const stepAxis = useAxis({ label: 'step', range: [0, result?.steps ?? shown.steps], key: result?.steps })
  const accAxis = useAxis({ label: 'accuracy', range: [0, 1] })
  const lossAxis = useAxis({ label: 'cross-entropy (nats)', log: true, hold: 'union', key: trained.trained })
  const normAxis = useAxis({ label: 'weight norm ‖θ‖₂', hold: 'union', key: trained.trained })
  const freqAxis = useAxis({ label: 'frequency k', range: [0.5, half + 0.5], nice: false, integer: true })
  const shareAxis = useAxis({ label: 'share of embedding power', range: [0, 1] })
  const ckAxis = useAxis({
    label: 'step',
    range: [-0.5 * (shots[1]?.step ?? 1), (shots.at(-1)?.step ?? 1) + 0.5 * (shots[1]?.step ?? 1)],
    nice: false,
    key: shots.length,
  })
  const cx = useAxis({ label: `cos component (k = ${top[0] ?? '—'})` })
  const cy = useAxis({ label: 'sin component', equal: cx })
  const ax = useAxis({ label: 'a', range: [-0.5, p - 0.5], nice: false, integer: true })
  const bx = useAxis({ label: 'b', range: [-0.5, p - 0.5], nice: false, integer: true, equal: ax })
  const residues = useMemo(() => Array.from({ length: p }, (_, a) => a), [p])
  const marker = shot ? <Handle kind="x" at={shot.step} onDrag={pickStep} label={`step ${shot.step}`} /> : null
  const recordIndex = c
    ? Math.max(
        0,
        c.steps.findIndex((s) => s >= (shot?.step ?? 0)),
      )
    : 0
  const done = result?.step ?? 0
  const total = trained.trained?.steps ?? settings.steps
  const player =
    shots.length > 0 ? (
      <ControlRow label="4 · checkpoint">
        <Player
          label="checkpoint"
          value={index}
          onChange={pick}
          count={shots.length}
          format={(k) => `step ${shots[k]?.step ?? 0}`}
        />
      </ControlRow>
    ) : null
  const waiting = (
    <div className="py-6 text-center text-sm text-muted-foreground">
      Press Train: this panel follows the checkpoint.
    </div>
  )

  return (
    <>
      <Figure
        title="Grokking modular arithmetic"
        purpose="A network fits its training pairs of a ∘ b mod p within a few hundred steps, but only generalises to the held-out pairs much later: weight decay keeps pushing it from memorising towards a solution built from a few Fourier frequencies."
        state={state}
        defaultSize="XL"
        controls={
          <>
            <TrainControls run={trained as never} progress={done / total} progressText={`${done} / ${total} steps`} />
            {player}
          </>
        }
        readouts={
          <>
            <Readout label="step" value={shot ? shot.step : '—'} />
            <Readout label="train accuracy" value={c ? pct(c.trainAccuracy[recordIndex]) : '—'} />
            <Readout label="test accuracy" value={c ? pct(c.testAccuracy[recordIndex]) : '—'} />
            <Readout label="weight norm" value={c ? f3(c.weightNorm[recordIndex]) : '—'} />
            <Readout label="train / test pairs" value={`${data.train.x.shape[0]} / ${data.test.x.shape[0]}`} />
          </>
        }
        caption={
          <>
            aifn-methods <code>modularArithmetic</code> splits the table of a {shown.op} b mod {p} into a training share
            of {shown.fraction} and a test rest; <code>grokkingRun</code> trains an MLP (a shared 32-dimensional
            embedding of the residues, [E_a, E_b] → {width} ReLU units → {p} logits) in the worker by full-batch AdamW
            (β₂ = 0.98) with decoupled weight decay λ, or by full-batch L-BFGS on the cross-entropy plus (λ/2)‖θ‖²,
            which heads straight for the regularised minimum instead of drifting to it. The published runs (Power et
            al., 2022; Nanda et al., 2023) use a 1-layer transformer, p = 97 or 113 and 10⁴–10⁵ steps; this MLP on p =
            31 with step size 0.01 and λ = 2 is the fastest setting found to grok reliably in a browser: train accuracy
            reaches 100% by about step 50, test accuracy stays near 0 until about step 300 and then climbs past 80% by
            step 1500 (about 45 s; 2000 or 3000 steps finish the climb). With λ = 0 the test accuracy stays low for the
            whole run; a smaller training fraction delays it or prevents it. Drag the step marker or play the
            checkpoints; the figure below follows.
          </>
        }
      >
        <Plots cols={3} scale={0.6}>
          <Plot x={stepAxis} y={accAxis} title="accuracy">
            {c && <Curve name="train" x={c.steps} y={c.trainAccuracy} slot={0} />}
            {c && <Curve name="test" x={c.steps} y={c.testAccuracy} slot={1} />}
            {marker}
          </Plot>
          <Plot x={stepAxis} y={lossAxis} title="loss">
            {c && <Curve name="train" x={c.steps} y={c.trainLoss} slot={0} />}
            {c && <Curve name="test" x={c.steps} y={c.testLoss} slot={1} />}
            {marker}
          </Plot>
          <Plot x={stepAxis} y={normAxis} title="weight norm" legend={false}>
            {c && <Curve name="‖θ‖₂" x={c.steps} y={c.weightNorm} slot={2} />}
            {marker}
          </Plot>
        </Plots>
      </Figure>
      <Figure
        title="Inside the network: Fourier features and neurons"
        purpose="Generalisation arrives with structure: the embedding's power concentrates on a few frequencies of ℤ_p, the residues line up on circles in those frequencies' planes, and hidden units become periodic in a and b."
        defaultSize="XL"
        hoverReadout={false}
        controls={
          <>
            {player}
            <ControlRow label="5 · hidden unit">
              <div className="flex flex-wrap items-center gap-2">
                <Slider label="unit" value={neuron} onChange={setNeuron} min={0} max={width - 1} step={1} />
                {neuronPick !== null && (
                  <Button size="sm" variant="outline" onClick={() => setNeuron(null)}>
                    strongest unit
                  </Button>
                )}
              </div>
            </ControlRow>
          </>
        }
        readouts={
          <>
            <Readout label="step" value={shot ? shot.step : '—'} />
            <Readout label="top frequencies" value={spectrum ? top.slice(0, 3).join(', ') : '—'} />
            <Readout label="power in the top 5" value={spectrum ? pct(topShare) : '—'} />
            <Readout label="hidden unit" value={neuronPick === null ? `${neuron} (strongest)` : neuron} />
          </>
        }
        caption={
          <>
            Top left: the share of the embedding&apos;s power at each frequency k = 1 … {half} (the constant left out),
            by checkpoint, from the truth&apos;s <code>spectrum</code>: |DFT|² of each embedding column along the
            residues (aifn <code>rfft</code>), folded over ±k and summed. Top middle: that spectrum at the checkpoint.
            Top right: every residue&apos;s embedding projected on the cosine and sine components of the strongest
            frequency k, joined in the order 0, 1, …, p − 1; when the frequency is used the points lie on an ellipse at
            angles 2πka/p. Bottom: one hidden unit&apos;s activation over every pair (a, b), recorded by{' '}
            <code>recordActivations</code> (by default the unit with the largest outgoing weights at this checkpoint).
            For addition a generalising unit depends on a + b through periodic stripes.
          </>
        }
      >
        <Plots cols={3} scale={0.55}>
          <Plot x={ckAxis} y={freqAxis} title="embedding spectrum over training">
            {spectra.length > 0 && (
              <Raster
                x={shots.map((s) => s.step)}
                y={frequencies}
                z={field}
                scale="sequential"
                range={[0, Math.max(0.05, ...field.flat())]}
                valueLabel="share"
              />
            )}
            {marker}
          </Plot>
          <Plot x={freqAxis} y={shareAxis} title={`spectrum at step ${shot?.step ?? 0}`} legend={false}>
            {spectrum && <Bars name="share" x={frequencies} y={frequencies.map((k) => spectrum[k])} slot={0} />}
          </Plot>
          <Plot x={cx} y={cy} title="residues in the top frequency's plane" legend={false}>
            {circle && (
              <Curve name="0 → p − 1" x={[...circle.x, circle.x[0]]} y={[...circle.y, circle.y[0]]} muted thin />
            )}
            {circle && <Points name="residue" x={circle.x} y={circle.y} slot={0} />}
          </Plot>
        </Plots>
        {neuronGrid ? (
          <Plot x={ax} y={bx} title={`hidden unit ${neuron} over the table`} scale={0.6}>
            <Raster x={residues} y={residues} z={neuronGrid} scale="sequential" valueLabel="activation" />
          </Plot>
        ) : (
          waiting
        )}
      </Figure>
    </>
  )
}
