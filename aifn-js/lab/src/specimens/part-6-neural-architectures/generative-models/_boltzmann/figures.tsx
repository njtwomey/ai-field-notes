/**
 * Energy-based networks of binary units (`aifn-applied/generative/boltzmann`): an RBM learning bars and stripes by
 * contrastive divergence, with its exact log-likelihood, receptive fields and Gibbs samples; a deep belief network
 * stacked greedily from RBMs on noisy digits, sampled top-down and fine-tuned from a few labels; and Hopfield networks
 * recalling 5 × 7 digits, classical (Hebbian weights, asynchronous sweeps) against modern (one attention step), with
 * their capacity curves.
 */
import { useMemo, useState } from 'react'
import { digitGlyphs } from 'aifn-applied/data/synthetic'
import {
  capacityCurve,
  corruptPattern,
  hebbianWeights,
  hopfieldRecall,
  modernHopfieldEnergy,
  modernHopfieldUpdate,
  overlaps,
  type DbnRun,
  type DbnRunOptions,
  type RbmRun,
  type RbmRunOptions,
} from 'aifn-applied/generative/boltzmann'
import { child, stream } from 'aifn/foundation/random'
import { fromData, toFlat } from 'aifn/foundation/tensor'
import { Player, usePlayhead } from '@lab/controls'
import { ControlRow, Figure } from '@lab/layout'
import { call, choice, float, int, row, setting, slider, useComputed, useFigureState, type Task } from '@lab/state'
import { TrainControls, useTrainedRun } from '@lab/views'
import { Bars, Curve, formatNumber, Plot, Plots, Raster, Readout, useAxis } from '@lab/viz'

const fmt = (v: number, digits = 3) => (Number.isFinite(v) ? formatNumber(Number(v.toPrecision(digits))) : '—')

/** Images of h × w pixels (rows of h·w) laid out in `cols` columns with a 1-pixel gap, as raster rows (top first). */
function tiles(images: ArrayLike<number>, count: number, h: number, w: number, cols: number, map = (v: number) => v) {
  const r = Math.ceil(count / cols)
  const H = r * (h + 1) - 1
  const W = cols * (w + 1) - 1
  const out = Array.from({ length: H }, () => new Array<number>(W).fill(NaN))
  for (let k = 0; k < count; k++) {
    const R = Math.floor(k / cols)
    const C = k % cols
    for (let i = 0; i < h; i++)
      for (let j = 0; j < w; j++) out[H - 1 - (R * (h + 1) + i)][C * (w + 1) + j] = map(images[k * h * w + i * w + j])
  }
  return { z: out, x: Array.from({ length: W }, (_, j) => j), y: Array.from({ length: H }, (_, i) => i) }
}

// ── 1 · RBM ──────────────────────────────────────────────────────────────────────────────────────────────────────────

type RbmSettings = { size: number; options: RbmRunOptions }
const rbmTask = ({ size, options }: RbmSettings): Task<RbmRun> =>
  call<RbmRun>(
    'applied/generative/boltzmann/rbmRun',
    { x: call('applied/data/synthetic/barsAndStripes', { size }) },
    options,
  )

export function RbmSpecimen() {
  const state = useFigureState({
    setup: row('1 · data and machine', {
      size: choice([3, 4], 3, { label: 'bars and stripes size' }),
      hidden: int(8, { ge: 1, le: 16, suggestions: [4, 8, 12, 16], label: 'hidden units' }),
      k: int(1, { ge: 1, le: 50, suggestions: [1, 5, 10], label: 'Gibbs steps k' }),
      persistent: setting(false, 'persistent chains (PCD)'),
    }),
    run: row('2 · training run', {
      epochs: int(1000, { ge: 1, suggestions: [200, 500, 1000, 2000], label: 'epochs' }),
      rate: float(0.1, { gt: 0, scale: 'log10', suggestions: [0.01, 0.05, 0.1, 0.3], label: 'learning rate' }),
      seed: int(1, { ge: 0, le: 9999, label: 'seed' }),
    }),
  })
  const { size, hidden, k, persistent } = state.setup
  const settings: RbmSettings = {
    size: Number(size),
    options: {
      hidden,
      k,
      persistent,
      epochs: state.run.epochs,
      learningRate: state.run.rate,
      seed: state.run.seed,
      batchSize: 4,
    },
  }
  const trained = useTrainedRun(settings, rbmTask)
  const run = trained.run.value
  const shots = run?.checkpoints ?? []
  const [picked, setPicked] = useState(0)
  const index = Math.min(picked, Math.max(0, shots.length - 1))
  const shot = shots[index]
  const s = trained.trained?.size ?? Number(size)
  const D = s * s
  const H = run?.hidden ?? hidden
  const epochAxis = useAxis({
    label: 'epoch',
    range: [0, trained.trained?.options.epochs ?? state.run.epochs],
    key: JSON.stringify(trained.trained),
  })
  const llAxis = useAxis({ label: 'mean log-likelihood (nats)', hold: 'union', key: JSON.stringify(trained.trained) })
  const fields = useMemo(() => {
    if (!shot) return null
    // Hidden unit j's receptive field: column j of W, as an s × s image.
    const img = new Float64Array(H * D)
    for (let j = 0; j < H; j++) for (let i = 0; i < D; i++) img[j * D + i] = shot.weights[i * H + j]
    return tiles(img, H, s, s, Math.min(4, H))
  }, [shot, H, D, s])
  const samples = useMemo(() => (shot ? tiles(shot.samples, shot.samples.length / D, s, s, 4) : null), [shot, D, s])
  const wMax = useMemo(() => (shot ? Math.max(...Array.from(shot.weights, Math.abs)) || 1 : 1), [shot])
  const fx = useAxis({ label: '', range: [-0.5, (fields?.x.length ?? 1) - 0.5], key: fields?.x.length })
  const fy = useAxis({ label: '', range: [-0.5, (fields?.y.length ?? 1) - 0.5], key: fields?.y.length, equal: fx })
  const sx = useAxis({ label: '', range: [-0.5, (samples?.x.length ?? 1) - 0.5], key: samples?.x.length })
  const sy = useAxis({ label: '', range: [-0.5, (samples?.y.length ?? 1) - 0.5], key: samples?.y.length, equal: sx })
  const ll = run?.logLikelihood ?? []
  const epochs = ll.map((_, e) => e)
  return (
    <Figure
      title="A restricted Boltzmann machine learns bars and stripes"
      purpose="An RBM is trained by the gap between correlations in the data and in its own samples; contrastive divergence approximates the second by k Gibbs steps from the data, and with few hidden units the exact log-likelihood shows how close it gets to the uniform distribution over the patterns."
      state={state}
      defaultSize="L"
      controls={
        <>
          <TrainControls
            run={trained as never}
            progress={run ? (ll.length - 1) / (trained.trained?.options.epochs ?? 1) : 0}
            progressText={run ? `${ll.length - 1} epochs` : ''}
          />
          <ControlRow label="3 · checkpoints">
            <Player
              className="col-span-full"
              value={index}
              onChange={setPicked}
              count={Math.max(1, shots.length)}
              label="checkpoint"
              format={(i) => `epoch ${shots[i]?.epoch ?? 0}`}
            />
          </ControlRow>
        </>
      }
      readouts={
        <>
          <Readout label="epoch" value={shot ? shot.epoch : '—'} />
          <Readout label="log-likelihood" value={shot ? fmt(ll[shot.epoch]) : '—'} />
          <Readout label="best possible" value={run ? fmt(run.bestLogLikelihood) : '—'} />
        </>
      }
      caption="aifn barsAndStripes (every pattern of whole rows or whole columns on, 14 for 3 × 3) and rbmRun (CD-k or PCD-k, minibatches of 4, weights N(0, 0.1²)). Left: the exact mean log-likelihood per epoch (log Z summed over the 2ᴴ hidden states), against −log(number of patterns), the best any model can do. Middle: each hidden unit's weights to the pixels (diverging scale). Right: samples after 200 Gibbs steps from random images. Press Train; play the checkpoints."
    >
      <Plots cols={3}>
        <Plot x={epochAxis} y={llAxis} title={!trained.trained ? 'press Train to start' : 'exact log-likelihood'}>
          {run && (
            <Curve
              name="best (uniform on the patterns)"
              x={[0, epochs.length]}
              y={[run.bestLogLikelihood, run.bestLogLikelihood]}
              emphasis
              dashed
            />
          )}
          {run && <Curve name="RBM" slot={0} x={epochs} y={ll} />}
        </Plot>
        <Plot x={fx} y={fy} title="receptive fields">
          {fields && (
            <Raster
              x={fields.x}
              y={fields.y}
              z={fields.z}
              scale="diverging"
              range={[-wMax, wMax]}
              valueLabel="weight"
            />
          )}
        </Plot>
        <Plot x={sx} y={sy} title="Gibbs samples">
          {samples && (
            <Raster x={samples.x} y={samples.y} z={samples.z} scale="sequential" range={[0, 1]} valueLabel="pixel" />
          )}
        </Plot>
      </Plots>
    </Figure>
  )
}

// ── 2 · Deep belief network ──────────────────────────────────────────────────────────────────────────────────────────

const DBN_SHAPES = {
  '32 → 16': [32, 16],
  '32 → 12': [32, 12],
  '64 → 16': [64, 16],
  '32': [32],
  '32 → 16 → 12': [32, 16, 12],
}
type DbnSettings = { flip: number; perClass: number; dataSeed: number; options: DbnRunOptions }
const dbnTask = ({ flip, perClass, dataSeed, options }: DbnSettings): Task<DbnRun> =>
  call<DbnRun>(
    'applied/generative/boltzmann/dbnRun',
    call('applied/data/synthetic/digits', call('foundation/random/stream', dataSeed), { perClass, flip, noise: 0 }),
    options,
  )

export function DbnSpecimen() {
  const state = useFigureState({
    data: row('1 · noisy digits', {
      flip: slider(0, 0.3, 0.1, { step: 0.01, label: 'pixel flip probability' }),
      perClass: int(30, { ge: 2, le: 200, suggestions: [10, 30, 100], label: 'images per digit' }),
    }),
    stack: row('2 · stack', {
      shape: choice(Object.keys(DBN_SHAPES) as (keyof typeof DBN_SHAPES)[], '32 → 16', { label: 'hidden layers' }),
      epochs: int(150, { ge: 1, suggestions: [50, 150, 400], label: 'CD epochs per layer' }),
      rate: float(0.1, { gt: 0, scale: 'log10', suggestions: [0.03, 0.1, 0.3], label: 'learning rate' }),
      k: int(1, { ge: 1, le: 50, suggestions: [1, 5], label: 'Gibbs steps k' }),
    }),
    tune: row('3 · fine-tune', {
      labels: int(1, { ge: 1, le: 20, suggestions: [1, 2, 5], label: 'labels per digit' }),
      fineTune: int(200, { ge: 0, suggestions: [0, 100, 200, 500], label: 'Adam epochs' }),
      seed: int(1, { ge: 0, le: 9999, label: 'seed' }),
    }),
  })
  const settings: DbnSettings = {
    flip: state.data.flip,
    perClass: state.data.perClass,
    dataSeed: state.tune.seed,
    options: {
      layers: DBN_SHAPES[state.stack.shape],
      epochs: state.stack.epochs,
      learningRate: state.stack.rate,
      k: state.stack.k,
      labelsPerClass: state.tune.labels,
      fineTuneEpochs: state.tune.fineTune,
      seed: state.tune.seed,
    },
  }
  const trained = useTrainedRun(settings, dbnTask)
  const run = trained.run.value
  const shots = run?.checkpoints ?? []
  const [picked, setPicked] = useState(0)
  const index = Math.min(picked, Math.max(0, shots.length - 1))
  const shot = shots[index]
  const sizes = run?.sizes ?? DBN_SHAPES[state.stack.shape]
  const H1 = sizes[0]
  const epochs = trained.trained?.options.epochs ?? state.stack.epochs
  const key = JSON.stringify(trained.trained)
  const fields = useMemo(() => {
    if (!shot) return null
    const img = new Float64Array(H1 * 35)
    for (let j = 0; j < H1; j++) for (let i = 0; i < 35; i++) img[j * 35 + i] = shot.weights[i * H1 + j]
    return tiles(img, H1, 7, 5, 8)
  }, [shot, H1])
  const wMax = useMemo(() => (shot ? Math.max(...Array.from(shot.weights, Math.abs)) || 1 : 1), [shot])
  const samples = useMemo(() => (shot ? tiles(shot.samples, shot.samples.length / 35, 7, 5, 8) : null), [shot])
  const errAxis = useAxis({ label: 'epoch of the layer', range: [0, epochs], key })
  const eAxis = useAxis({ label: 'reconstruction error (squared, per image)', hold: 'union', key })
  const fx = useAxis({ label: '', range: [-0.5, (fields?.x.length ?? 1) - 0.5], key: fields?.x.length })
  const fy = useAxis({ label: '', range: [-0.5, (fields?.y.length ?? 1) - 0.5], key: fields?.y.length, equal: fx })
  const sx = useAxis({ label: '', range: [-0.5, (samples?.x.length ?? 1) - 0.5], key: samples?.x.length })
  const sy = useAxis({ label: '', range: [-0.5, (samples?.y.length ?? 1) - 0.5], key: samples?.y.length, equal: sx })
  const tuneEpochs = trained.trained?.options.fineTuneEpochs ?? state.tune.fineTune
  const tAxis = useAxis({ label: 'fine-tune epoch', range: [0, Math.max(1, tuneEpochs)], key })
  const accAxis = useAxis({ label: 'test accuracy', range: [0, 1] })
  const ft = run?.fineTune ?? null
  const totalEpochs = sizes.length * epochs + (tuneEpochs || 0)
  const doneEpochs = run
    ? run.reconstructionError.reduce((t, e) => t + Math.max(0, e.length - 1), 0) + (ft ? ft.pretrained.length - 1 : 0)
    : 0
  return (
    <Figure
      title="A deep belief network, stacked one RBM at a time"
      purpose="Greedy layer-wise training fits an RBM to the data, then another RBM to the first one's hidden probabilities, and so on; the stack generates by Gibbs sampling in the top RBM and one directed pass down, and its weights are a starting point that lets a classifier learn digits from one label each."
      state={state}
      defaultSize="L"
      controls={
        <>
          <TrainControls
            run={trained as never}
            progress={run ? doneEpochs / Math.max(1, totalEpochs) : 0}
            progressText={run ? (run.phase === 'pretraining' ? `layer ${shot ? shot.layer + 1 : 1}` : run.phase) : ''}
          />
          <ControlRow label="4 · checkpoints">
            <Player
              className="col-span-full"
              value={index}
              onChange={setPicked}
              count={Math.max(1, shots.length)}
              label="checkpoint"
              format={(i) => (shots[i] ? `layer ${shots[i].layer + 1}, epoch ${shots[i].epoch}` : 'layer 1, epoch 0')}
            />
          </ControlRow>
        </>
      }
      readouts={
        <>
          <Readout
            label="checkpoint"
            value={shot ? `layer ${shot.layer + 1} of ${sizes.length}, epoch ${shot.epoch}` : '—'}
          />
          <Readout
            label="test accuracy (DBN init · random init)"
            value={ft ? `${fmt(ft.pretrained.at(-1) ?? NaN)} · ${fmt(ft.random.at(-1) ?? NaN)}` : '—'}
          />
          <Readout label="labelled · test images" value={ft ? `${ft.labelled} · ${ft.test}` : '—'} />
        </>
      }
      caption="aifn digits (5 × 7 glyphs, each pixel flipped with the chosen probability) and dbnRun: half the images, without labels, train each layer by CD-k for the chosen epochs on the hidden probabilities of the layer below (minibatches of 10). Top left: each layer's reconstruction error over its own epochs (slot = layer). Top right: layer 1's receptive fields. Bottom left: samples from the stack trained so far (200 Gibbs steps in the top RBM, then one pass down; pixel probabilities). Bottom right: the fine-tune, a sigmoid MLP with a softmax head trained by Adam on the chosen number of labels per digit, scored on the other half, from the DBN's weights (slot 0) and from random weights of the same scale (slot 1). Press Train; play the checkpoints."
    >
      <Plots cols={2}>
        <Plot x={errAxis} y={eAxis} title={!trained.trained ? 'press Train to start' : 'reconstruction error'}>
          {run?.reconstructionError.map((e, l) => (
            <Curve key={l} name={`layer ${l + 1}`} slot={l} x={e.map((_, i) => i).slice(1)} y={e.slice(1)} />
          ))}
        </Plot>
        <Plot x={fx} y={fy} title="layer 1 receptive fields">
          {fields && (
            <Raster
              x={fields.x}
              y={fields.y}
              z={fields.z}
              scale="diverging"
              range={[-wMax, wMax]}
              valueLabel="weight"
            />
          )}
        </Plot>
        <Plot x={sx} y={sy} title="samples (top-down)">
          {samples && (
            <Raster x={samples.x} y={samples.y} z={samples.z} scale="sequential" range={[0, 1]} valueLabel="pixel" />
          )}
        </Plot>
        <Plot x={tAxis} y={accAxis} title="fine-tune from few labels">
          {ft && <Curve name="DBN weights" slot={0} x={ft.pretrained.map((_, i) => i)} y={ft.pretrained} />}
          {ft && <Curve name="random weights" slot={1} x={ft.random.map((_, i) => i)} y={ft.random} />}
        </Plot>
      </Plots>
    </Figure>
  )
}

// ── 3 · Hopfield ─────────────────────────────────────────────────────────────────────────────────────────────────────

export function HopfieldSpecimen() {
  const state = useFigureState({
    memory: row('1 · memory', {
      stored: int(4, { ge: 1, le: 10, suggestions: [2, 4, 7, 10], label: 'digits stored (0 …)' }),
      cue: int(1, { ge: 0, le: 9, label: 'cued digit' }),
      flip: slider(0, 0.5, 0.15, { step: 0.01, label: 'cue noise (flip probability)' }),
      seed: int(1, { ge: 0, le: 9999, label: 'seed' }),
    }),
    modern: row('2 · modern network', {
      beta: float(0.5, { gt: 0, scale: 'log10', suggestions: [0.05, 0.1, 0.5, 2], label: 'inverse temperature β' }),
    }),
  })
  const { stored, cue: cueDigit, flip, seed } = state.memory
  const { beta } = state.modern
  const glyphs = useMemo(() => Float64Array.from(toFlat(digitGlyphs()), (v) => 2 * v - 1), [])
  const n = Math.max(stored, Math.min(cueDigit + 1, 10))
  const X = useMemo(() => fromData(glyphs.slice(0, n * 35), [n, 35]), [glyphs, n])
  const recall = useMemo(() => {
    const W = hebbianWeights(X)
    const cue = corruptPattern(glyphs.subarray(cueDigit * 35, (cueDigit + 1) * 35), stream(`cue/${seed}`), flip)
    const classical = hopfieldRecall(W, cue, stream(`recall/${seed}`))
    const modern = modernHopfieldUpdate(X, cue, beta)
    return {
      cue,
      classical,
      modern,
      modernEnergy: [modernHopfieldEnergy(X, cue, beta), modernHopfieldEnergy(X, modern.state, beta)],
    }
  }, [X, glyphs, cueDigit, flip, seed, beta])
  const [sweep, setSweep] = usePlayhead(recall.classical.states.length)
  const state0 = recall.classical.states[Math.min(sweep, recall.classical.states.length - 1)]
  const ov = overlaps(X, state0)
  const curve = useComputed(
    () => capacityCurve(child(stream('capacity'), seed), { units: 35 * 2, maxPatterns: 20, trials: 6, flip, beta }),
    [seed, flip, beta],
    { mode: 'release' },
  )
  const pictures = tiles(
    Float64Array.from([...recall.cue, ...state0, ...recall.modern.state]),
    3,
    7,
    5,
    3,
    (v) => (v + 1) / 2,
  )
  const px = useAxis({ label: 'cue · classical · modern', range: [-0.5, pictures.x.length - 0.5] })
  const py = useAxis({ label: '', range: [-0.5, pictures.y.length - 0.5], equal: px })
  const digitAxis = useAxis({ label: 'stored digit', range: [-0.5, n - 0.5] })
  const wAxis = useAxis({ label: 'overlap / softmax weight', range: [-1, 1] })
  const loadAxis = useAxis({ label: 'random patterns stored (70 units)', range: [1, 20] })
  const qAxis = useAxis({ label: 'final overlap with the cued pattern', range: [0, 1] })
  return (
    <Figure
      title="Hopfield networks: classical and modern"
      purpose="A classical Hopfield network stores patterns in Hebbian weights and recalls by sign updates that never raise its energy, but only up to about 0.14 patterns per unit; the modern network's update, one softmax-weighted average of the stored patterns, is attention, and recalls far past that."
      state={state}
      defaultSize="L"
      controls={
        <ControlRow label="3 · classical sweeps">
          <Player
            className="col-span-full"
            value={sweep}
            onChange={setSweep}
            count={recall.classical.states.length}
            label="sweep"
          />
        </ControlRow>
      }
      readouts={{
        classical: (
          <>
            <Readout
              label="energy"
              value={fmt(recall.classical.energies[Math.min(sweep, recall.classical.energies.length - 1)])}
            />
            <Readout label="overlap with the cued digit" value={fmt(ov[cueDigit] ?? NaN)} />
            <Readout label="converged" value={recall.classical.converged ? 'yes' : 'no'} />
          </>
        ),
        modern: (
          <>
            <Readout label="weight on the cued digit" value={fmt(recall.modern.weights[cueDigit] ?? NaN)} />
            <Readout
              label="energy before → after"
              value={`${fmt(recall.modernEnergy[0])} → ${fmt(recall.modernEnergy[1])}`}
            />
          </>
        ),
      }}
      caption="aifn digitGlyphs as ±1 patterns of 35 pixels; the first digits are stored (the cued digit is always among them). Left: the noisy cue (corruptPattern), the classical network's state at the chosen sweep (hebbianWeights, hopfieldRecall: asynchronous updates in random order) and the modern network's one-step result (modernHopfieldUpdate). Middle: overlaps of the classical state with each stored digit (slot 0) and the modern softmax weights (slot 1). Right: capacityCurve on random patterns of 70 units: classical recall breaks down near 0.14 × 70 ≈ 10 patterns, the modern update holds. Digit glyphs overlap strongly (they are not random), so the classical network fails much earlier on them."
    >
      <Plots cols={3}>
        <Plot x={px} y={py} title="cue, classical recall, modern recall">
          <Raster x={pictures.x} y={pictures.y} z={pictures.z} scale="sequential" range={[0, 1]} valueLabel="pixel" />
        </Plot>
        <Plot x={digitAxis} y={wAxis} title="which memory">
          <Bars
            name="classical overlap"
            slot={0}
            x={Array.from({ length: n }, (_, i) => i - 0.2)}
            y={Array.from(ov)}
            width={0.35}
          />
          <Bars
            name="modern weight"
            slot={1}
            x={Array.from({ length: n }, (_, i) => i + 0.2)}
            y={Array.from(recall.modern.weights)}
            width={0.35}
          />
        </Plot>
        <Plot x={loadAxis} y={qAxis} title="capacity">
          <Curve
            name="classical"
            slot={0}
            x={curve.value.patterns}
            y={Array.from(curve.value.classical, (v) => Math.max(0, v))}
            showPoints
          />
          <Curve
            name="modern"
            slot={1}
            x={curve.value.patterns}
            y={Array.from(curve.value.modern, (v) => Math.max(0, v))}
            showPoints
          />
          <Curve name="0.14 N" x={[0.138 * 70, 0.138 * 70]} y={[0, 1]} emphasis dashed thin />
        </Plot>
      </Plots>
    </Figure>
  )
}
