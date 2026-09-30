import { moons, shapesImage, spirals, xor, type Dataset } from 'aifn/datasets'
import { accuracy } from 'aifn/metrics'
import {
  adam,
  conv2d,
  convOutputSize,
  maxPool2d,
  Mlp,
  MultiHeadAttention,
  relu,
  scaledDotProductAttention,
  training,
  xavierUniform,
  type Params,
} from 'aifn/nn'
import { binaryCrossEntropyWithLogits } from 'aifn/losses'
import { stream } from 'aifn/random'
import { sigmoid } from 'aifn/special'
import { fromData, fromRows, tensor, toFlat, toRows, unwrap, type Tensor } from 'aifn/tensor'
import { trace } from 'aifn/trace'
import { useMemo, useState } from 'react'
import { ControlRow, Figure } from '@lab/layout'
import { Player, Select, Slider, Switch } from '@lab/controls'
import { Heatmap, Panel, Readout, Subplots, XYChart, formatNumber, type Handle, type XYSeries } from '@lab/viz'
import { ParamsView } from '@lab/views'

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
  const [dataId, setDataId] = useState<DataId>('spirals')
  const [width, setWidth] = useState(16)
  const [lr, setLr] = useState(0.03)
  const data = useMemo(() => datasetOf(dataId), [dataId])
  const labels = useMemo(() => toFlat(data.y!), [data])
  const model = useMemo(() => Mlp([2, width, width, 1], { activation: 'tanh', init: xavierUniform() }), [width])
  const tr = useMemo(() => {
    const y = fromData(Float64Array.from(labels), [labels.length, 1])
    const alg = training({
      loss: (p: Params[], b: { x: Tensor; y: Tensor }) => binaryCrossEntropyWithLogits(model.apply(p, b.x), b.y),
      data: { x: data.x, y },
      optimizer: adam({ lr }),
    })
    return trace(alg, { params: model.init(stream('nn-init')) }, STEPS, {
      every: 10,
      record: { loss: (s) => s.loss, 'gradient norm': (s) => s.gradNorm },
    })
  }, [model, data, labels, lr])
  const [position, setPosition] = useState<number | null>(null)
  const pos = Math.min(position ?? tr.steps.length - 1, tr.steps.length - 1)
  const state = tr.steps[pos]
  const { gx, gy } = useMemo(() => gridAxes(data.x), [data])
  const gridPoints = useMemo(() => fromRows(gy.flatMap((y) => gx.map((x) => [x, y]))), [gx, gy])
  const surface = useMemo(() => {
    const p = toFlat(raw(sigmoid(model.apply(state.params, gridPoints))))
    return gy.map((_, i) => p.slice(i * GRID, (i + 1) * GRID))
  }, [model, state, gridPoints, gy])
  const trainAccuracy = useMemo(() => {
    const p = toFlat(raw(model.apply(state.params, data.x)))
    return accuracy(
      labels,
      p.map((z) => (z > 0 ? 1 : 0)),
    )
  }, [model, state, data, labels])
  const points = useMemo(() => {
    const rows = toRows(data.x)
    return { name: 'data', type: 'scatter' as const, x: rows.map((r) => r[0]), y: rows.map((r) => r[1]), group: labels }
  }, [data, labels])
  const steps = tr.index
  const loss = toFlat(tr.series.loss)
  const lossSeries: XYSeries[] = [{ name: 'training loss', type: 'line', x: steps, y: loss }]
  const cursor: XYSeries[] = [
    {
      name: 'step',
      type: 'line',
      x: [steps[pos], steps[pos]],
      y: [Math.min(...loss), Math.max(...loss)],
      emphasis: true,
    },
  ]
  return (
    <>
      <Figure
        title="An MLP learning a decision surface"
        description="Gradient descent bends a tanh network's decision surface around the data: early steps give a nearly linear boundary, later ones wrap it around each class."
        defaultSize="L"
        controls={
          <>
            <ControlRow label="1 · data and network">
              <Select label="data" value={dataId} onChange={setDataId} options={DATASETS} />
              <Slider label="hidden units per layer" value={width} min={2} max={32} step={1} onChange={setWidth} />
              <Slider label="Adam learning rate" value={lr} min={0.005} max={0.1} step={0.005} onChange={setLr} />
            </ControlRow>
            <ControlRow label="2 · training step">
              <Player
                value={pos}
                onChange={setPosition}
                count={tr.steps.length}
                format={(k) => String(tr.index[k])}
                label="step"
              />
            </ControlRow>
          </>
        }
        readouts={
          <>
            <Readout label="step" value={String(tr.index[pos])} />
            <Readout label="loss" value={fmt(state.loss)} />
            <Readout label="training accuracy" value={fmt(trainAccuracy)} />
            <Readout label="gradient norm" value={fmt(state.gradNorm)} />
          </>
        }
        caption={`Mlp(2 → ${width} → ${width} → 1) with tanh units, trained by full-batch Adam on binary cross-entropy (aifn/nn training, aifn/losses). Colour: P(class 1) over the plane at the chosen step. Right: the loss against the step; drag the vertical line or play the steps.`}
      >
        <Subplots cols={2} widthRatios={[1.2, 1]}>
          <Panel>
            <Heatmap
              x={gx}
              y={gy}
              z={surface}
              scale="diverging"
              range={[0, 1]}
              valueLabel="P(class 1)"
              overlay={[points]}
              xLabel="x₁"
              yLabel="x₂"
            />
          </Panel>
          <Panel>
            <XYChart
              series={lossSeries}
              live={cursor}
              xLabel="step"
              yLabel="loss"
              yLog
              handles={[
                {
                  kind: 'x',
                  at: steps[pos],
                  label: 'step',
                  onDrag: (x) => setPosition(Math.max(0, Math.min(tr.steps.length - 1, Math.round(x / 10)))),
                },
              ]}
            />
          </Panel>
        </Subplots>
      </Figure>
      <ParamsView
        title="The network's parameters at the chosen step"
        params={state.params}
        grads={state.grads}
        description="Every parameter tensor of the MLP with its shape, norm and gradient norm; pick one to see its values."
      />
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

export function ConvolutionSpecimen() {
  const [stride, setStride] = useState(1)
  const [padding, setPadding] = useState(1)
  const [dilation, setDilation] = useState(1)
  const [pool, setPool] = useState(false)
  const [which, setWhich] = useState('0')
  const image = useMemo(() => shapesImage({ size: SIZE }), [])
  const kernels = useMemo(() => fromData(Float64Array.from(KERNELS.flatMap((k) => k.k.flat())), [4, 1, 3, 3]), [])
  const maps = useMemo(() => {
    const y = conv2d(fromData(Float64Array.from(toFlat(image)), [1, SIZE, SIZE]), kernels, {
      stride,
      padding,
      dilation,
    })
    return raw(pool ? maxPool2d(relu(y), 2) : y)
  }, [image, kernels, stride, padding, dilation, pool])
  const [, H, W] = maps.shape
  const k = Number(which)
  const map = useMemo(
    () => fromData(Float64Array.from(toFlat(maps).slice(k * H * W, (k + 1) * H * W)), [H, W]),
    [maps, k, H, W],
  )
  const [probe, setProbe] = useState<[number, number]>([5, 5])
  const pi = Math.min(Math.max(0, Math.round(probe[0])), W - 1)
  const pj = Math.min(Math.max(0, Math.round(probe[1])), H - 1)
  // The receptive field of output (row, col) in input pixels (before pooling), as a rectangle.
  const row = H - 1 - pj
  const scale = pool ? 2 : 1
  const r0 = row * scale * stride - padding
  const c0 = pi * scale * stride - padding
  const extent = (scale - 1) * stride + dilation * 2 + 1
  const box = [
    [c0 - 0.5, r0 - 0.5],
    [c0 + extent - 0.5, r0 - 0.5],
    [c0 + extent - 0.5, r0 + extent - 0.5],
    [c0 - 0.5, r0 + extent - 0.5],
    [c0 - 0.5, r0 - 0.5],
  ].map(([c, r]) => [c, SIZE - 1 - r])
  const handles: Handle[] = [{ kind: 'point', at: [pi, pj], label: 'output', onDrag: ([x, y]) => setProbe([x, y]) }]
  const outSize = convOutputSize(SIZE, 3, stride, padding, dilation)
  return (
    <Figure
      title="Convolution on a small image"
      description="Each output of a convolution is a weighted sum over a small window of the input; stride, padding and dilation set where the windows sit, and so the output size and the receptive field."
      defaultSize="L"
      controls={
        <>
          <ControlRow label="1 · kernel">
            <Select
              label="kernel"
              value={which}
              onChange={setWhich}
              options={KERNELS.map((kk, i) => ({ value: String(i), label: kk.name }))}
            />
          </ControlRow>
          <ControlRow label="2 · geometry">
            <Slider label="stride" value={stride} min={1} max={3} step={1} onChange={setStride} />
            <Slider label="padding" value={padding} min={0} max={3} step={1} onChange={setPadding} />
            <Slider label="dilation" value={dilation} min={1} max={3} step={1} onChange={setDilation} />
          </ControlRow>
          <ControlRow label="3 · reveal">
            <Switch label="ReLU and 2 × 2 max pooling" checked={pool} onChange={setPool} />
          </ControlRow>
        </>
      }
      readouts={
        <>
          <Readout label="input" value={`${SIZE} × ${SIZE}`} />
          <Readout label="conv output" value={`${outSize} × ${outSize}`} />
          <Readout label="shown map" value={`${H} × ${W}`} />
          <Readout label={`output at (${row}, ${pi})`} value={fmt(toRows(map)[row]?.[pi] ?? NaN)} />
          <Readout label="receptive field" value={`${extent} × ${extent} pixels`} />
        </>
      }
      caption="aifn/nn conv2d (cross-correlation, as in every deep-learning library) of the shapes test image with four fixed 3 × 3 kernels. Left: the input with the receptive field of the chosen output; middle: the kernel; right: its feature map (diverging colour: sign). Drag the point on the feature map to move the output; the box follows on the input."
    >
      <Subplots cols={3} widthRatios={[1.3, 0.7, 1.3]}>
        <Panel>
          <Heatmap
            x={axis(SIZE)}
            y={axis(SIZE)}
            z={upright(image)}
            valueLabel="pixel"
            equalAspect
            overlay={[
              { name: 'receptive field', type: 'line', x: box.map((p) => p[0]), y: box.map((p) => p[1]), slot: 1 },
            ]}
          />
        </Panel>
        <Panel>
          <Heatmap
            x={axis(3)}
            y={axis(3)}
            z={[...KERNELS[k].k].reverse()}
            scale="diverging"
            valueLabel="weight"
            equalAspect
          />
        </Panel>
        <Panel>
          <Heatmap
            x={axis(W)}
            y={axis(H)}
            z={upright(map)}
            scale="diverging"
            valueLabel="output"
            equalAspect
            handles={handles}
          />
        </Panel>
      </Subplots>
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

export function AttentionQuerySpecimen() {
  const [q, setQ] = useState<[number, number]>([1.4, 0.6])
  const [temperature, setTemperature] = useState(1)
  const keys = useMemo(() => tensor(KEYS), [])
  const { weights } = useMemo(
    () => scaledDotProductAttention(tensor([q]), keys, keys, { scale: 1 / (Math.SQRT2 * temperature) }),
    [q, keys, temperature],
  )
  const w = toFlat(raw(weights))
  const keySeries: XYSeries[] = [
    { name: 'keys', type: 'scatter', x: KEYS.map((k) => k[0]), y: KEYS.map((k) => k[1]), slot: 0 },
  ]
  const bars: XYSeries[] = [{ name: 'weight', type: 'bar', x: axis(KEYS.length).map((i) => i + 1), y: w, slot: 0 }]
  const top = w.indexOf(Math.max(...w))
  return (
    <Figure
      title="Attention weights of one query"
      description="A query attends to each key in proportion to exp(q·k/√d): keys pointing the same way as the query get most of the weight, and a lower temperature sharpens the choice."
      controls={
        <ControlRow label="1 · sharpness">
          <Slider
            label="temperature (× √d)"
            value={temperature}
            min={0.1}
            max={3}
            step={0.05}
            onChange={setTemperature}
          />
        </ControlRow>
      }
      readouts={
        <>
          <Readout label="q" value={`(${fmt(q[0])}, ${fmt(q[1])})`} />
          <Readout label="largest weight" value={`${TOKENS[top]}: ${fmt(w[top])}`} />
        </>
      }
      caption={`aifn/nn scaledDotProductAttention with d = 2 and scale 1/(T√d). Left: six keys (tokens ${TOKENS.join(', ')}) and the query arrow; drag its tip. Right: the softmax weights, which sum to one.`}
    >
      <Subplots cols={2}>
        <Panel>
          <XYChart
            series={keySeries}
            aspect="equal"
            xRange={[-2, 2]}
            yRange={[-2, 2]}
            vectors={[{ from: [0, 0], to: q, slot: 1, label: 'q' }]}
            handles={[{ kind: 'point', at: q, label: 'q', onDrag: (p) => setQ([p[0], p[1]]) }]}
            xLabel="dimension 1"
            yLabel="dimension 2"
          />
        </Panel>
        <Panel>
          <XYChart series={bars} integerX xLabel="key (token)" yLabel="attention weight" yRange={[0, 1]} />
        </Panel>
      </Subplots>
    </Figure>
  )
}

export function MultiHeadSpecimen() {
  const [causal, setCausal] = useState(true)
  const [head, setHead] = useState('0')
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
  const h = Number(head)
  const T = TOKENS.length
  const z = toRows(fromData(Float64Array.from(toFlat(weights).slice(h * T * T, (h + 1) * T * T)), [T, T])).reverse()
  return (
    <Figure
      title="Multi-head self-attention weights"
      description="Each head of a multi-head attention layer has its own weight matrix over the sequence; a causal mask zeroes the upper triangle, so each token sees only itself and earlier tokens."
      controls={
        <ControlRow label="1 · layer">
          <Select
            label="head"
            value={head}
            onChange={setHead}
            options={[
              { value: '0', label: 'head 1' },
              { value: '1', label: 'head 2' },
            ]}
          />
          <Switch label="causal mask" checked={causal} onChange={setCausal} />
        </ControlRow>
      }
      caption="aifn/nn MultiHeadAttention(8, 2 heads) with random initial weights on a fixed six-token input; the weights are read through the layer's tap. Rows are queries (top: the first token), columns keys; each row sums to one."
    >
      <Heatmap
        x={axis(T)}
        y={axis(T).map((i) => i - (T - 1))}
        z={z}
        range={[0, 1]}
        valueLabel="weight"
        xLabel="key position"
        yLabel="−query position"
        equalAspect
      />
    </Figure>
  )
}
