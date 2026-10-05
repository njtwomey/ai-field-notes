/**
 * "Quantisation: bits against error": core `aifn/nn/quantise` on a weight matrix (the levels over the histogram, the
 * error against bits for each scheme), a small MLP trained in the worker and quantised by round-to-nearest and GPTQ
 * (aifn-methods `quantisationStudy`), and the serving memory and decoding speed that fewer bits buy
 * (`servingMemory`, `decodeThroughput`).
 */
import { useMemo } from 'react'
import {
  QUANTISATION_METHODS,
  decodeThroughput,
  servingMemory,
  type QuantisationMethod,
  type QuantisationSnapshot,
  type QuantisationStudyOptions,
} from 'aifn-methods/neural/quantisation'
import { quantisationError, quantisationParams, type QuantisationParams } from 'aifn-compute/nn/quantise'
import { child, normal, stream, uniform } from 'aifn-compute/foundation/random'
import { fromData, toFlat, type Tensor } from 'aifn-compute/foundation/tensor'
import { Figure } from 'aifn-render/layout'
import { call, choice, float, int, row, slider, useFigureState, type Task } from 'aifn-render/state'
import { CLASSIFICATION_CASES, TrainControls, datasetChoice, useTrainedRun, type DatasetValue } from '@lab/views'
import { Annotation, Bars, Curve, Plot, Plots, Points, Readout, formatNumber, useAxis } from 'aifn-render/viz'

const fmt = (v: number) => (Number.isFinite(v) ? formatNumber(v) : '—')
const BITS = [2, 3, 4, 5, 6, 7, 8]

/** A weight matrix of 16 output rows × 64 inputs: Gaussian or Student-t entries, rows scaled by log-uniform factors. */
function weightMatrix(kind: string, seed: number): Tensor {
  const s = stream(`quantise-weights-${seed}`)
  const z = toFlat(normal(child(s, 'z'), 0, 1, { shape: [16, 64] }) as Tensor)
  const chi = toFlat(normal(child(s, 'chi'), 0, 1, { shape: [16, 64, 3] }) as Tensor)
  const scale = toFlat(uniform(child(s, 'rows'), -1.2, 1.2, { shape: [16] }) as Tensor).map((v) => 10 ** v)
  const out = Float64Array.from(z, (v, i) => {
    // Student t with 3 degrees of freedom: z / √(χ²₃/3).
    const t = kind === 'heavy' ? v / Math.sqrt((chi[3 * i] ** 2 + chi[3 * i + 1] ** 2 + chi[3 * i + 2] ** 2) / 3) : v
    const r = Math.floor(i / 64)
    return 0.1 * t * (kind === 'rows' ? scale[r] : 1)
  })
  return fromData(out, [16, 64])
}

function histogram(values: ArrayLike<number>, lo: number, hi: number, bins: number) {
  const counts = new Array<number>(bins).fill(0)
  const w = (hi - lo) / bins
  for (let i = 0; i < values.length; i++) {
    const k = Math.floor((values[i] - lo) / w)
    if (k >= 0 && k < bins) counts[k]++
  }
  return { x: counts.map((_, k) => lo + (k + 0.5) * w), y: counts, width: w }
}

const VARIANTS = [
  { key: 'tensor', label: 'per tensor, nearest', axis: undefined, rounding: 'nearest' },
  { key: 'channel', label: 'per channel, nearest', axis: 0, rounding: 'nearest' },
  { key: 'stochastic', label: 'per tensor, stochastic', axis: undefined, rounding: 'stochastic' },
] as const

export function BitsErrorSpecimen() {
  const state = useFigureState({
    weights: row('1 · weights', {
      kind: choice(
        [
          { value: 'gaussian', label: 'Gaussian' },
          { value: 'heavy', label: 'heavy-tailed (Student t₃)' },
          { value: 'rows', label: 'rows on different scales' },
        ],
        'rows',
        { label: 'weights W (16 × 64)' },
      ),
      seed: int(0, { ge: 0, le: 9999, label: 'seed' }),
    }),
    quantiser: row('2 · quantiser', {
      bits: slider(1, 8, 4, { label: 'bits b', step: 1 }),
      scheme: choice(['symmetric', 'affine'], 'symmetric', { label: 'scheme' }),
      granularity: choice(
        [
          { value: 'tensor', label: 'per tensor' },
          { value: 'channel', label: 'per output row' },
        ],
        'tensor',
        { label: 'scale per' },
      ),
    }),
  })
  const { kind, seed } = state.weights
  const { bits, scheme, granularity } = state.quantiser
  const W = useMemo(() => weightMatrix(kind, seed), [kind, seed])
  const w = useMemo(() => toFlat(W), [W])
  const params: QuantisationParams = useMemo(
    () =>
      quantisationParams(W, {
        bits,
        scheme: scheme as 'symmetric' | 'affine',
        ...(granularity === 'channel' ? { axis: 0 } : {}),
      }),
    [W, bits, scheme, granularity],
  )
  const err = useMemo(() => quantisationError(W, params), [W, params])
  const curves = useMemo(
    () =>
      VARIANTS.map((v) =>
        BITS.map((b) => {
          const p = quantisationParams(W, {
            bits: b,
            scheme: scheme as 'symmetric' | 'affine',
            ...(v.axis === undefined ? {} : { axis: v.axis }),
          })
          return quantisationError(
            W,
            p,
            v.rounding === 'stochastic' ? { rounding: 'stochastic', stream: stream('quantise-sr') } : {},
          ).sqnr
        }),
      ),
    [W, scheme],
  )
  const span = Math.max(...w.map(Math.abs)) * 1.05
  const hist = useMemo(() => histogram(w, -span, span, 80), [w, span])
  // The levels s(q − z) of the first row's quantiser (all rows' when per tensor).
  const levels = useMemo(() => {
    const s = params.scale[0]
    const z = params.zeroPoint[0]
    const n = params.qmax - params.qmin + 1
    if (n > 64) return []
    return Array.from({ length: n }, (_, k) => s * (params.qmin + k - z))
  }, [params])
  const row0 = useMemo(() => histogram(w.slice(0, 64), -span, span, 80), [w, span])
  const xv = useAxis({ label: 'weight value', range: [-span, span], key: `${kind}${seed}` })
  const yc = useAxis({ label: 'count', hold: 'union', key: `${kind}${seed}` })
  const xb = useAxis({ label: 'bits b', integer: true, range: [1.5, 8.5] })
  const ys = useAxis({ label: 'SQNR (dB)', hold: 'union', key: `${kind}${seed}${scheme}` })
  return (
    <Figure
      title="Quantisation: bits against error"
      purpose="A b-bit quantiser rounds each weight to one of 2ᵇ evenly spaced levels set by a scale and zero point. Every extra bit halves the step and adds about 6 dB of signal-to-noise; a scale per output row fits rows of different size far better than one scale for the tensor."
      defaultSize="XL"
      state={state}
      readouts={{
        quantiser: (
          <>
            <Readout label="levels 2ᵇ" value={2 ** bits} />
            <Readout label="scale s (row 0)" value={fmt(params.scale[0])} />
            <Readout label="zero point z (row 0)" value={params.zeroPoint[0]} />
            <Readout label="MSE" value={fmt(err.mse)} />
            <Readout label="SQNR" value={`${fmt(err.sqnr)} dB`} />
            <Readout label="clipped" value={`${fmt(100 * err.clipped)}%`} />
          </>
        ),
      }}
      caption="Left: the histogram of all weights (first colour) and of the first output row (second colour), with the quantiser's levels for that row as vertical lines (drawn up to 6 bits). With one scale for the tensor, a row of small weights falls between a few levels near zero; with a scale per row each row uses the whole grid. Affine quantisers shift the grid by a zero point so it covers [min, max]; symmetric ones centre it on zero. Right: the signal-to-quantisation-noise ratio against bits for round-to-nearest per tensor and per row and for stochastic rounding (unbiased, but noisier), the chosen b marked. Change the weight distribution: heavy tails stretch the range so most weights get few levels."
    >
      <Plots cols={2}>
        <Plot x={xv} y={yc} title="weights and quantisation levels">
          <Bars name="all weights" x={hist.x} y={hist.y} width={hist.width} slot={0} opacity={0.6} />
          <Bars name="row 0" x={row0.x} y={row0.y} width={row0.width} slot={1} opacity={0.8} />
          {levels.map((l, k) => (
            <Annotation key={k} x={l} dashed />
          ))}
        </Plot>
        <Plot x={xb} y={ys} title="error against bits">
          {VARIANTS.map((v, j) => (
            <Curve key={v.key} name={v.label} x={BITS} y={curves[j]} slot={j} showPoints />
          ))}
          <Annotation x={bits} />
        </Plot>
      </Plots>
    </Figure>
  )
}

// ── A trained MLP ────────────────────────────────────────────────────────────────────────────────────────────────────

const DATA = datasetChoice(CLASSIFICATION_CASES)
const LABELS: Record<QuantisationMethod, string> = {
  'per-tensor': 'round to nearest, per tensor',
  'per-channel': 'round to nearest, per channel',
  gptq: 'GPTQ (per channel)',
  awq: 'AWQ (per channel)',
}
type Settings = { data: DatasetValue; width: number; depth: number; steps: number; stepSize: number; seed: number }
const dataSeed = (d: DatasetValue) => `quantise-study-data-${d.key}`

function studyTask(s: Settings): Task<QuantisationSnapshot> {
  const options: QuantisationStudyOptions = {
    width: s.width,
    depth: s.depth,
    steps: s.steps,
    stepSize: s.stepSize,
    seed: s.seed,
    bits: BITS,
  }
  return call<QuantisationSnapshot>(
    'applied/neural/quantisation/quantisationStudy',
    DATA.task(s.data, dataSeed(s.data)),
    options,
  )
}

export function TrainedMlpSpecimen() {
  const state = useFigureState({
    data: DATA.field({ label: '1 · data', initial: 'spirals' }),
    net: row('2 · network and training', {
      width: int(24, { ge: 2, le: 64, suggestions: [8, 16, 24, 32], label: 'hidden units' }),
      depth: int(2, { ge: 1, le: 4, suggestions: [1, 2, 3], label: 'hidden layers' }),
      steps: int(1500, { ge: 1, suggestions: [500, 1500, 3000], label: 'Adam steps' }),
      stepSize: float(0.01, { gt: 0, scale: 'log10', suggestions: [0.003, 0.01, 0.03], label: 'Adam η' }),
      seed: int(0, { ge: 0, le: 9999, label: 'seed' }),
    }),
  })
  const settings: Settings = {
    data: { key: state.data.key, values: { ...state.data.values } },
    width: Number(state.net.width),
    depth: Number(state.net.depth),
    steps: Number(state.net.steps),
    stepSize: Number(state.net.stepSize),
    seed: Number(state.net.seed),
  }
  const trained = useTrainedRun(settings, studyTask)
  const r = trained.run.value
  const runKey = trained.trained
  const stepAxis = useAxis({ label: 'Adam step', key: runKey, hold: 'union' })
  const lossAxis = useAxis({ label: 'training loss', log: true, key: runKey, hold: 'union' })
  const bitAxis = useAxis({ label: 'bits b', integer: true, range: [1.5, 8.5] })
  const accAxis = useAxis({ label: 'training accuracy', range: [0.4, 1.02] })
  const results = r?.results ?? []
  const done = r?.done ?? 0
  const total = r?.total ?? settings.steps + BITS.length
  return (
    <Figure
      title="Quantising a trained network"
      purpose="The same trained MLP with its weights rounded to b bits: one scale for each weight matrix loses accuracy first, a scale per output unit holds out longer; GPTQ feeds each column's rounding error into the columns not yet rounded, and AWQ scales up the input channels that meet large activations before rounding."
      defaultSize="XL"
      state={state}
      controls={
        <TrainControls run={trained as never} progress={total ? done / total : 0} progressText={`${done} / ${total}`} />
      }
      readouts={{
        network: (
          <>
            <Readout
              label="float accuracy"
              value={r && Number.isFinite(r.accuracy) ? `${Math.round(100 * r.accuracy)}%` : '—'}
            />
            <Readout label="weights" value={r ? r.weights.length : '—'} />
            <Readout label="phase" value={r?.phase ?? '—'} />
          </>
        ),
        ...Object.fromEntries(
          results
            .filter((x) => x.bits === 4 || x.bits === 3)
            .map((x) => [
              `at ${x.bits} bits`,
              <>
                {QUANTISATION_METHODS.map((m) => (
                  <Readout
                    key={m}
                    label={LABELS[m]}
                    value={`${Math.round(100 * x.accuracy[m])}% (${fmt(x.sqnr[m])} dB)`}
                  />
                ))}
              </>,
            ]),
        ),
      }}
      caption="Press Train: aifn-methods quantisationStudy trains a ReLU MLP on the chosen two-class set by Adam in the worker (the loss streams in), then quantises every weight matrix (biases stay in floating point) at 2 to 8 bits with symmetric quantisers and reports the training accuracy of the quantised network. GPTQ quantises each layer column by column using the Hessian 2XᵀX/n of that layer's own inputs. AWQ multiplies input channel j by s_j = (mean |X_j|)^α before rounding and divides it back out after, with α searched on a grid in [0, 1] for the smallest layer output error (α = 0 is round-to-nearest per channel). The dashed line is the floating-point accuracy. Harder sets (two spirals) need more of the network's precision; few bits break them first."
    >
      {r ? (
        <Plots cols={2}>
          <Plot x={stepAxis} y={lossAxis} title="training" legend={false}>
            <Curve name="loss" x={r.step} y={r.loss} slot={0} />
          </Plot>
          <Plot x={bitAxis} y={accAxis} title="accuracy after quantisation">
            {Number.isFinite(r.accuracy) && <Annotation y={r.accuracy} dashed />}
            {QUANTISATION_METHODS.map((m, j) => (
              <Curve
                key={m}
                name={LABELS[m]}
                x={results.map((x) => x.bits)}
                y={results.map((x) => x.accuracy[m])}
                slot={j}
                showPoints
              />
            ))}
          </Plot>
        </Plots>
      ) : (
        <div className="py-6 text-center text-sm text-muted-foreground">
          Press Train: the loss streams in, then the accuracy at each bit width.
        </div>
      )}
    </Figure>
  )
}

// ── Serving cost ─────────────────────────────────────────────────────────────────────────────────────────────────────

const SERVE_BITS = [16, 8, 6, 4, 3, 2]

export function ServingCostSpecimen() {
  const state = useFigureState({
    model: row('1 · model', {
      billions: float(7, { gt: 0, suggestions: [1, 7, 13, 70], label: 'parameters (billions)' }),
      layers: int(32, { ge: 1, suggestions: [24, 32, 80], label: 'layers' }),
      kvHeads: int(8, { ge: 1, suggestions: [8, 32], label: 'KV heads' }),
      headDim: int(128, { ge: 1, suggestions: [64, 128], label: 'head size' }),
    }),
    serve: row('2 · serving', {
      context: int(4096, { ge: 1, suggestions: [2048, 8192, 32768], label: 'context (tokens)' }),
      batch: int(1, { ge: 1, suggestions: [1, 8, 64], label: 'batch' }),
      kvBits: choice([16, 8, 4], 16, { label: 'KV-cache bits' }),
    }),
    device: row('3 · device', {
      memoryGb: float(24, { gt: 0, suggestions: [16, 24, 80], label: 'memory (GB)' }),
      bandwidthGbs: float(1000, { gt: 0, suggestions: [400, 1000, 3350], label: 'bandwidth (GB/s)' }),
      tflops: float(150, { gt: 0, suggestions: [80, 150, 990], label: 'TFLOP/s' }),
    }),
  })
  const shape = {
    parameters: Number(state.model.billions) * 1e9,
    layers: Number(state.model.layers),
    kvHeads: Number(state.model.kvHeads),
    headDim: Number(state.model.headDim),
  }
  const device = {
    memoryGb: Number(state.device.memoryGb),
    bandwidthGbs: Number(state.device.bandwidthGbs),
    tflops: Number(state.device.tflops),
  }
  const rows = SERVE_BITS.map((b) => {
    const setup = {
      weightBits: b,
      kvBits: Number(state.serve.kvBits),
      batch: Number(state.serve.batch),
      context: Number(state.serve.context),
    }
    return { bits: b, mem: servingMemory(shape, setup, device), speed: decodeThroughput(shape, setup, device) }
  })
  const xb = useAxis({ label: 'weight bits', categories: SERVE_BITS.map(String) })
  const ym = useAxis({ label: 'memory (GB)', range: [0, undefined], hold: 'union' })
  const yt = useAxis({ label: 'tokens / s (roofline)', range: [0, undefined], hold: 'union' })
  return (
    <Figure
      title="What fewer bits buy at serving time"
      purpose="Generating a token reads every weight once, so at small batch a decoder is limited by memory bandwidth: halving the bits per weight halves the bytes read and roughly doubles the tokens per second, until the arithmetic becomes the limit."
      defaultSize="L"
      state={state}
      readouts={{
        'at 4 bits': (
          <>
            <Readout label="weights" value={`${fmt(rows[3].mem.weights / 1e9)} GB`} />
            <Readout label="KV cache" value={`${fmt(rows[3].mem.kvCache / 1e9)} GB`} />
            <Readout label="fits" value={rows[3].mem.fits ? 'yes' : 'no'} />
            <Readout label="bound" value={rows[3].speed.bound} />
            <Readout label="ridge batch" value={fmt(rows[3].speed.ridgeBatch)} />
          </>
        ),
      }}
      caption="Left: the memory of the weights and of the key–value cache (2 · layers · KV heads · head size · context · batch values) at each weight precision; the dashed line is the device's memory. Right: the roofline estimate of decoding speed, the slower of reading every byte once at the device's bandwidth and doing 2N FLOPs per token. Raise the batch: the weights are read once per step for the whole batch, so throughput grows until the step becomes compute-bound (the ridge batch). Estimates, not measurements."
    >
      <Plots cols={2}>
        <Plot x={xb} y={ym} title="memory">
          <Bars
            name="weights"
            x={SERVE_BITS.map((_, k) => k - 0.2)}
            y={rows.map((r) => r.mem.weights / 1e9)}
            width={0.38}
            slot={0}
          />
          <Bars
            name="KV cache"
            x={SERVE_BITS.map((_, k) => k + 0.2)}
            y={rows.map((r) => r.mem.kvCache / 1e9)}
            width={0.38}
            slot={1}
          />
          <Annotation y={device.memoryGb} dashed />
        </Plot>
        <Plot x={xb} y={yt} title="decoding speed" legend={false}>
          <Bars
            name="tokens / s"
            x={SERVE_BITS.map((_, k) => k)}
            y={rows.map((r) => r.speed.tokensPerSecond)}
            slot={2}
          />
          <Points
            name="compute-bound"
            x={rows.flatMap((r, k) => (r.speed.bound === 'compute' ? [k] : []))}
            y={rows.flatMap((r) => (r.speed.bound === 'compute' ? [r.speed.tokensPerSecond] : []))}
            emphasis
          />
        </Plot>
      </Plots>
    </Figure>
  )
}
