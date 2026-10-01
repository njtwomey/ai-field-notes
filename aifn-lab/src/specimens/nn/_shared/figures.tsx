import { moons, shapesImage, spirals, xor } from 'aifn-applied/data/synthetic'
import { type Dataset } from 'aifn-applied/data'
import { accuracy } from 'aifn/learning/metrics'
import { trainingLoop } from 'aifn/nn/training'
import { adamRule } from 'aifn/optim/first-order'
import { conv2d, convOutputSize, maxPool2d, relu } from 'aifn/nn/functional'
import { Mlp, MultiHeadAttention, scaledDotProductAttention } from 'aifn/nn/layers'
import { xavierUniform } from 'aifn/nn/init'
import { type Params } from 'aifn/foundation/pytree'
import { binaryCrossEntropyWithLogits } from 'aifn/learning/losses'
import { stream } from 'aifn/foundation/random'
import { sigmoid } from 'aifn/numerics/special'
import { fromData, fromRows, tensor, toFlat, toRows, unwrap, type Tensor } from 'aifn/foundation/tensor'
import { trace } from 'aifn/foundation/trace'
import { useMemo, useState } from 'react'
import { Figure } from '@lab/layout'
import { Player } from '@lab/controls'
import { choice, row, setting, slider, toggle, useComputed, useFigureState } from '@lab/state'
import {
  Annotation,
  Bars,
  Curve,
  Handle,
  Plot,
  Plots,
  Points,
  Raster,
  Readout,
  Vectors,
  formatNumber,
  useAxis,
} from '@lab/viz'
import { ParamsPanel } from '@lab/views'

const fmt = (v: number) => formatNumber(v)
const raw = (v: unknown) => unwrap(v as Tensor) as Tensor

// ---------------------------------------------------------------------------------------------------------------------
// 1. An MLP learning a decision surface.

type DataId = 'spirals' | 'xor' | 'moons'
const DATASETS: { value: DataId; label: string }[] = [
  { value: 'spirals', label: 'two spirals' },
  { value: 'xor', label: 'XOR' },
  { value: 'moons', label: 'moons' },
]

function datasetOf(id: DataId): Dataset {
  const s = stream(`nn-data-${id}`)
  if (id === 'spirals') return spirals(s, { n: 200, turns: 1.25, noise: 0.04 })
  if (id === 'moons') return moons(s, { n: 200, noise: 0.12 })
  return xor(s, { n: 200 })
}

const STEPS = 600
const GRID = 45

/** Evenly spaced grid coordinates spanning the data with a margin. */
function gridAxes(x: Tensor) {
  const rows = toRows(x)
  const span = (a: number) => {
    const v = rows.map((r) => r[a])
    const lo = Math.min(...v)
    const hi = Math.max(...v)
    const pad = 0.15 * (hi - lo)
    return Array.from({ length: GRID }, (_, i) => lo - pad + ((hi - lo + 2 * pad) * i) / (GRID - 1))
  }
  return { gx: span(0), gy: span(1) }
}

export function MlpTrainingSpecimen() {
  const state = useFigureState({
    setup: row('1 · data and network', {
      dataId: choice(DATASETS, 'spirals', { label: 'data' }),
      width: slider(2, 32, 16, { label: 'hidden units per layer', step: 1 }),
      lr: slider(0.005, 0.1, 0.03, { label: 'Adam learning rate', step: 0.005 }),
    }),
  })
  const { dataId, width, lr } = state.setup
  const data = useMemo(() => datasetOf(dataId), [dataId])
  const labels = useMemo(() => toFlat(data.y!), [data])
  const model = useMemo(() => Mlp([2, width, width, 1], { activation: 'tanh', init: xavierUniform() }), [width])
  // 600 Adam steps: run once the slider is released, not on every pointer move.
  const run = useComputed(
    () => {
      const y = fromData(Float64Array.from(labels), [labels.length, 1])
      const alg = trainingLoop({
        loss: (p: Params[], b: { x: Tensor; y: Tensor }) => binaryCrossEntropyWithLogits(model.apply(p, b.x), b.y),
        data: { x: data.x, y },
        optimizer: adamRule({ stepSize: lr }),
      })
      const tr = trace(alg, { params: model.init(stream('nn-init')) }, STEPS, {
        every: 10,
        record: { loss: (s) => s.loss, 'gradient norm': (s) => s.gradNorm },
      })
      return { tr, model, loss: toFlat(tr.series.loss) }
    },
    [model, data, labels, lr],
    { mode: 'release' },
  )
  const { tr, loss } = run.value
  const net = run.value.model
  const [position, setPosition] = useState(0)
  const pos = Math.min(position, tr.steps.length - 1)
  const at = tr.steps[pos]
  const { gx, gy } = useMemo(() => gridAxes(data.x), [data])
  const gridPoints = useMemo(() => fromRows(gy.flatMap((y) => gx.map((x) => [x, y]))), [gx, gy])
  const surface = useMemo(() => {
    const p = toFlat(raw(sigmoid(net.apply(at.params, gridPoints))))
    return gy.map((_, i) => p.slice(i * GRID, (i + 1) * GRID))
  }, [net, at, gridPoints, gy])
  const trainAccuracy = useMemo(() => {
    const p = toFlat(raw(net.apply(at.params, data.x)))
    return accuracy(
      labels,
      p.map((z) => (z > 0 ? 1 : 0)),
    )
  }, [net, at, data, labels])
  const points = useMemo(() => {
    const rows = toRows(data.x)
    return { x: rows.map((r) => r[0]), y: rows.map((r) => r[1]) }
  }, [data])
  const steps = tr.index
  const x1 = useAxis({ label: 'x₁' })
  const x2 = useAxis({ label: 'x₂', equal: x1 })
  const stepAxis = useAxis({ label: 'step', range: [0, STEPS] })
  const lossAxis = useAxis({ label: 'loss', log: true, hold: 'union', key: `${dataId}${width}` })
  return (
    <>
      <Figure
        title="An MLP learning a decision surface"
        purpose="Gradient descent bends a tanh network's decision surface around the data: early steps give a nearly linear boundary, later ones wrap it around each class."
        defaultSize="L"
        state={state}
        controls={
          <Player
            value={pos}
            onChange={setPosition}
            count={tr.steps.length}
            format={(k) => String(tr.index[k])}
            label="2 · training step"
          />
        }
        readouts={{
          'at this step': (
            <>
              <Readout label="step" value={String(tr.index[pos])} />
              <Readout label="loss" value={fmt(at.loss)} />
              <Readout label="training accuracy" value={fmt(trainAccuracy)} />
              <Readout label="gradient norm" value={fmt(at.gradNorm)} />
            </>
          ),
        }}
        caption={`Mlp(2 → ${width} → ${width} → 1) with tanh units, trained by full-batch Adam on binary cross-entropy (aifn/nn training, aifn/learning/losses). Colour: P(class 1) over the plane at the chosen step (pale at 0.5, the boundary). Right: the loss against the step; play from the initial network or drag the vertical line. Markers: circles class 0, squares class 1.`}
      >
        <Plots cols={2} widths={[1.2, 1]}>
          <Plot x={x1} y={x2}>
            <Raster
              x={gx}
              y={gy}
              z={surface}
              scale="diverging"
              range={[0, 1]}
              valueLabel="P(class 1)"
              stale={run.stale}
            />
            <Points name="data" x={points.x} y={points.y} group={labels} groupNames={['class 0', 'class 1']} />
          </Plot>
          <Plot x={stepAxis} y={lossAxis}>
            <Curve name="training loss" x={steps} y={loss} stale={run.stale} />
            <Handle
              kind="x"
              at={steps[pos]}
              label="step"
              onDrag={(x) => setPosition(Math.max(0, Math.min(tr.steps.length - 1, Math.round(x / 10))))}
            />
          </Plot>
        </Plots>
      </Figure>
      <Figure
        title="The network's parameters at the chosen step"
        purpose="Every parameter tensor of the MLP with its shape, norm and gradient norm at the step chosen above; pick one to see its values."
      >
        <ParamsPanel params={at.params} grads={at.grads} />
      </Figure>
    </>
  )
}

// ---------------------------------------------------------------------------------------------------------------------
// 2. Convolution: kernels, feature maps and the receptive field.

const SIZE = 24
const KERNELS: { name: string; k: number[][] }[] = [
  {
    name: 'vertical edges',
    k: [
      [-1, 0, 1],
      [-2, 0, 2],
      [-1, 0, 1],
    ],
  },
  {
    name: 'horizontal edges',
    k: [
      [-1, -2, -1],
      [0, 0, 0],
      [1, 2, 1],
    ],
  },
  {
    name: 'blur',
    k: [
      [1, 2, 1],
      [2, 4, 2],
      [1, 2, 1],
    ].map((r) => r.map((v) => v / 16)),
  },
  {
    name: 'sharpen',
    k: [
      [0, -1, 0],
      [-1, 5, -1],
      [0, -1, 0],
    ],
  },
]

/** Rows of an [H, W] tensor flipped so row 0 (the top of the image) is drawn at the top of a heatmap. */
const upright = (m: Tensor) => toRows(m).reverse()
const axis = (n: number) => Array.from({ length: n }, (_, i) => i)

const OFFSETS = ['−1', '0', '+1']
const OFFSETS_UP = ['+1', '0', '−1']
const KERNEL_CHOICES = KERNELS.map((kk, i) => ({ value: i, label: kk.name }))

export function ConvolutionSpecimen() {
  const state = useFigureState({
    kernel: row('1 · kernel', { k: choice(KERNEL_CHOICES, 0, { label: 'kernel' }) }),
    geometry: row('2 · geometry', {
      stride: slider(1, 3, 1, { label: 'stride', step: 1 }),
      padding: slider(0, 3, 1, { label: 'padding', step: 1 }),
      dilation: slider(1, 3, 1, { label: 'dilation', step: 1 }),
    }),
    reveal: row('3 · reveal', { pool: toggle(false, 'ReLU and 2 × 2 max pooling') }),
    // The chosen output position (column, row from the bottom of the drawn map): moved by its handle.
    pi: slider(0, SIZE + 6, 3, { onChart: true, step: 1 }),
    pj: slider(0, SIZE + 6, 8, { onChart: true, step: 1 }),
  })
  const { stride, padding, dilation } = state.geometry
  const { pool } = state.reveal
  const k = state.kernel.k
  const image = useMemo(() => shapesImage({ size: SIZE }), [])
  const imageRows = useMemo(() => upright(image), [image])
  const kernels = useMemo(() => fromData(Float64Array.from(KERNELS.flatMap((kk) => kk.k.flat())), [4, 1, 3, 3]), [])
  const maps = useMemo(() => {
    const y = conv2d(fromData(Float64Array.from(toFlat(image)), [1, SIZE, SIZE]), kernels, {
      stride,
      padding,
      dilation,
    })
    return raw(pool ? maxPool2d(relu(y), 2) : y)
  }, [image, kernels, stride, padding, dilation, pool])
  const [, H, W] = maps.shape
  const map = useMemo(
    () => fromData(Float64Array.from(toFlat(maps).slice(k * H * W, (k + 1) * H * W)), [H, W]),
    [maps, k, H, W],
  )
  const mapRows = useMemo(() => upright(map), [map])
  const kernelRows = useMemo(() => [...KERNELS[k].k].reverse(), [k])
  const pi = Math.min(Math.max(0, Math.round(state.pi)), W - 1)
  const pj = Math.min(Math.max(0, Math.round(state.pj)), H - 1)
  // The receptive field of output (row, col) in input pixels (before pooling), as a rectangle.
  const outRow = H - 1 - pj
  const scale = pool ? 2 : 1
  const r0 = outRow * scale * stride - padding
  const c0 = pi * scale * stride - padding
  const extent = (scale - 1) * stride + dilation * 2 + 1
  const box = [
    [c0 - 0.5, r0 - 0.5],
    [c0 + extent - 0.5, r0 - 0.5],
    [c0 + extent - 0.5, r0 + extent - 0.5],
    [c0 - 0.5, r0 + extent - 0.5],
    [c0 - 0.5, r0 - 0.5],
  ].map(([c, r]) => [c, SIZE - 1 - r])
  const outSize = convOutputSize(SIZE, 3, stride, padding, dilation)
  const ix = useAxis({ label: 'column' })
  const iy = useAxis({ label: 'row', equal: ix })
  const kx = useAxis({ label: 'kernel column', categories: OFFSETS })
  const ky = useAxis({ label: 'kernel row', categories: OFFSETS_UP, equal: kx })
  const ox = useAxis({ label: 'output column' })
  const oy = useAxis({ label: 'output row', equal: ox })
  return (
    <Figure
      title="Convolution on a small image"
      purpose="Each output of a convolution is a weighted sum over a small window of the input; stride, padding and dilation set where the windows sit, and so the output size and the receptive field."
      defaultSize="L"
      state={state}
      readouts={{
        sizes: (
          <>
            <Readout label="input" value={`${SIZE} × ${SIZE}`} />
            <Readout label="conv output" value={`${outSize} × ${outSize}`} />
            <Readout label="shown map" value={`${H} × ${W}`} />
          </>
        ),
        'chosen output': (
          <>
            <Readout label={`output at (${outRow}, ${pi})`} value={fmt(mapRows[pj]?.[pi] ?? NaN)} />
            <Readout label="receptive field" value={`${extent} × ${extent} pixels`} />
          </>
        ),
      }}
      caption="aifn/nn conv2d (cross-correlation, as in every deep-learning library) of the shapes test image with four fixed 3 × 3 kernels. Left: the input with the receptive field of the chosen output; middle: the kernel; right: its feature map (diverging colour: sign, pale at zero). Drag the point on the feature map to move the output; the box follows on the input. Raise the dilation: the box grows while the kernel keeps 9 weights."
    >
      <Plots cols={3} widths={[1.3, 0.7, 1.3]}>
        <Plot x={ix} y={iy}>
          <Raster x={axis(SIZE)} y={axis(SIZE)} z={imageRows} valueLabel="pixel" colorBar={false} />
          <Curve name="receptive field" x={box.map((q) => q[0])} y={box.map((q) => q[1])} slot={1} live />
        </Plot>
        <Plot x={kx} y={ky}>
          <Raster x={axis(3)} y={axis(3)} z={kernelRows} scale="diverging" valueLabel="weight" colorBar={false} />
        </Plot>
        <Plot x={ox} y={oy}>
          <Raster x={axis(W)} y={axis(H)} z={mapRows} scale="diverging" valueLabel="output" />
          <Handle
            kind="point"
            at={[pi, pj]}
            label="output"
            onDrag={([x, y]) => {
              state.set('pi', x)
              state.set('pj', y)
            }}
          />
        </Plot>
      </Plots>
    </Figure>
  )
}

// ---------------------------------------------------------------------------------------------------------------------
// 3. Attention weights with a draggable query.

const KEYS = [
  [1.2, 0.3],
  [0.8, 1.1],
  [-0.4, 1.3],
  [-1.3, 0.2],
  [-0.6, -1.1],
  [0.9, -0.9],
]
const TOKENS = ['the', 'cat', 'sat', 'on', 'a', 'mat']

const KEY_X = KEYS.map((k) => k[0])
const KEY_Y = KEYS.map((k) => k[1])
const POSITIONS = axis(TOKENS.length)

export function AttentionQuerySpecimen() {
  const state = useFigureState({
    sharpness: row('1 · sharpness', {
      temperature: slider(0.1, 3, 1, { label: 'temperature (× √d)', step: 0.05 }),
    }),
    qx: slider(-2, 2, 1.4, { onChart: true }),
    qy: slider(-2, 2, 0.6, { onChart: true }),
  })
  const { temperature } = state.sharpness
  const q = useMemo((): [number, number] => [state.qx, state.qy], [state.qx, state.qy])
  const keys = useMemo(() => tensor(KEYS), [])
  const w = useMemo(() => {
    const { weights } = scaledDotProductAttention(tensor([q]), keys, keys, { scale: 1 / (Math.SQRT2 * temperature) })
    return toFlat(raw(weights))
  }, [q, keys, temperature])
  const top = w.indexOf(Math.max(...w))
  const vectors = useMemo(() => [{ from: [0, 0] as [number, number], to: q, slot: 1, label: 'q' }], [q])
  const d1 = useAxis({ label: 'dimension 1', range: [-2, 2] })
  const d2 = useAxis({ label: 'dimension 2', range: [-2, 2], equal: d1 })
  const tok = useAxis({ label: 'key (token)', categories: TOKENS })
  const wa = useAxis({ label: 'attention weight', range: [0, 1] })
  return (
    <Figure
      title="Attention weights of one query"
      purpose="A query attends to each key in proportion to exp(q·k/√d): keys pointing the same way as the query get most of the weight, and a lower temperature sharpens the choice."
      state={state}
      readouts={{
        query: (
          <>
            <Readout label="q" value={`(${fmt(q[0])}, ${fmt(q[1])})`} />
            <Readout label="largest weight" value={`${TOKENS[top]}: ${fmt(w[top])}`} />
          </>
        ),
      }}
      caption={`aifn/nn scaledDotProductAttention with d = 2 and scale 1/(T√d). Left: six keys (tokens ${TOKENS.join(', ')}) and the query arrow; drag its tip. Right: the softmax weights, which sum to one.`}
    >
      <Plots cols={2}>
        <Plot x={d1} y={d2}>
          <Points name="keys" x={KEY_X} y={KEY_Y} slot={0} />
          {KEYS.map((k, i) => (
            <Annotation key={TOKENS[i]} at={[k[0], k[1]]} text={TOKENS[i]} slot={0} />
          ))}
          <Vectors vectors={vectors} live />
          <Handle {...state.handle(['qx', 'qy'], { label: 'q' })} />
        </Plot>
        <Plot x={tok} y={wa}>
          <Bars name="weight" x={POSITIONS} y={w} slot={0} />
        </Plot>
      </Plots>
    </Figure>
  )
}

const HEADS = [
  { value: 0, label: 'head 1' },
  { value: 1, label: 'head 2' },
]
const TOKENS_UP = [...TOKENS].reverse()

export function MultiHeadSpecimen() {
  const state = useFigureState({
    layer: row('1 · layer', { head: choice(HEADS, 0, { label: 'head' }), causal: setting(true, 'causal mask') }),
  })
  const { head: h, causal } = state.layer
  const layer = useMemo(() => MultiHeadAttention(8, { heads: 2, causal }), [causal])
  const params = useMemo(() => layer.init(stream('mha')), [layer])
  const x = useMemo(
    () =>
      tensor(
        TOKENS.map((_, t) => Array.from({ length: 8 }, (_, j) => Math.sin(0.9 * t * (j + 1)) + 0.3 * Math.cos(j - t))),
      ),
    [],
  )
  const weights = useMemo(() => {
    let captured: Tensor | null = null
    layer.apply(params, x, { tap: (path, v) => ((captured = path === 'weights' ? raw(v) : captured), v) })
    return captured as unknown as Tensor
  }, [layer, params, x])
  const T = TOKENS.length
  // Row 0 drawn at the top: the first query token heads the matrix.
  const z = useMemo(
    () => toRows(fromData(Float64Array.from(toFlat(weights).slice(h * T * T, (h + 1) * T * T)), [T, T])).reverse(),
    [weights, h, T],
  )
  const kx = useAxis({ label: 'key', categories: TOKENS })
  const qy = useAxis({ label: 'query', categories: TOKENS_UP, equal: kx })
  return (
    <Figure
      title="Multi-head self-attention weights"
      purpose="Each head of a multi-head attention layer has its own weight matrix over the sequence; a causal mask zeroes the upper triangle, so each token sees only itself and earlier tokens."
      state={state}
      caption="aifn/nn MultiHeadAttention(8, 2 heads) with random initial weights on a fixed six-token input; the weights are read through the layer's tap. Rows are queries (top: the first token), columns keys; each row sums to one."
    >
      <Plot x={kx} y={qy}>
        <Raster x={POSITIONS} y={POSITIONS} z={z} range={[0, 1]} valueLabel="weight" />
      </Plot>
    </Figure>
  )
}
