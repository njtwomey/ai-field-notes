import { useMemo } from 'react'
import {
  Bars,
  choice,
  Figure,
  float,
  formatNumber,
  int,
  Plot,
  Readout,
  row,
  slider,
  useAxis,
  useFigureState,
} from 'aifn-render'
import { stream } from 'aifn-compute/foundation/random'
import { fromData, get, reshape, toFlat, toRows, type Tensor } from 'aifn-compute/foundation/tensor'
import {
  classActivationMap,
  explanationSimilarity,
  featureMaps,
  inputGradient,
  normaliseMap,
} from 'aifn-compute/learning/explain'
import { activations, gradCam } from 'aifn-compute/nn/training'
import { upsampleBilinear } from 'aifn-compute/signal/image'
import { renderShapes, type ShapeItem, type ShapeName } from 'aifn-methods/data/synthetic'
import { FixedHeight, ImageTile, useTrainedCnn } from '../_shared/figure-helpers'
import {
  asInput,
  LAYER,
  SHAPE_CLASSES,
  shapesSetup,
  SIZE,
  smallCnn,
  softmax,
  type CnnParams,
} from '../_shared/small-cnn'

type Method = 'cam' | 'grad-cam' | 'grad-cam++' | 'grad-cam-conv1' | 'gradient'
const METHODS: { value: Method; label: string }[] = [
  { value: 'cam', label: 'CAM (classifier weights)' },
  { value: 'grad-cam', label: 'Grad-CAM at conv2' },
  { value: 'grad-cam++', label: 'Grad-CAM++ at conv2' },
  { value: 'grad-cam-conv1', label: 'Grad-CAM at conv1' },
  { value: 'gradient', label: 'input gradient |∂y/∂x|' },
]

type Randomise = 'none' | 'linear' | 'conv2' | 'all'
const RANDOMISE: { value: Randomise; label: string; layers: number[] }[] = [
  { value: 'none', label: 'none (trained)', layers: [] },
  { value: 'linear', label: 'linear layer', layers: [LAYER.linear] },
  { value: 'conv2', label: 'linear + conv2', layers: [LAYER.linear, LAYER.conv2] },
  { value: 'all', label: 'linear + conv2 + conv1', layers: [LAYER.linear, LAYER.conv2, LAYER.conv1] },
]

const RADIUS = { a: 3.5, b: 4 }
const net = smallCnn(SHAPE_CLASSES.length)
const fresh = net.init(stream(99)) as CnnParams

/** The chosen explanation of class c as a non-negative 16 x 16 map scaled to a largest value of 1. */
function explain(params: CnnParams, x: Tensor, c: number, method: Method): Float64Array {
  if (method === 'gradient') {
    const f = (v: Tensor) => get(net.apply(params, reshape(v, [1, 1, SIZE, SIZE])), 0, c)
    return normaliseMap(inputGradient(f, toFlat(x)).gradient.map(Math.abs))
  }
  if (method === 'grad-cam-conv1') return normaliseMap(gradCam(net, params, x, c, String(LAYER.relu1)).map)
  let coarse: Float64Array
  if (method === 'cam') {
    const A = featureMaps(activations(net, params, x).activations[String(LAYER.relu2)] as Tensor)
    const w = toRows((params[LAYER.linear] as { weight: Tensor }).weight) as number[][]
    coarse = classActivationMap(
      A,
      w.map((r) => r[c]),
    ).map((v) => Math.max(0, v))
  } else {
    coarse = gradCam(net, params, x, c, String(LAYER.relu2), {
      weighting: method === 'grad-cam++' ? 'grad-cam++' : 'grad-cam',
    }).map
  }
  return normaliseMap(toFlat(upsampleBilinear(fromData(coarse, [SIZE / 2, SIZE / 2]), [SIZE, SIZE])))
}

/**
 * Class activation maps of a small network trained in the page: two shapes on one input, the map for a chosen class
 * by CAM, Grad-CAM, Grad-CAM++ or the plain input gradient, and the model-randomisation sanity check.
 */
export function ClassActivationExplorer() {
  const state = useFigureState({
    training: row('1 · train the network', {
      steps: int(120, { ge: 10, le: 600, suggestions: [60, 120, 240], label: 'Adam steps' }),
      rate: float(0.02, { gt: 0, le: 0.5, suggestions: [0.005, 0.01, 0.02, 0.05], label: 'learning rate' }),
    }),
    scene: row('2 · input', {
      first: choice<ShapeName>(SHAPE_CLASSES, 'disc', { label: 'first object' }),
      second: choice<ShapeName | 'none'>(['none', ...SHAPE_CLASSES], 'cross', { label: 'second object' }),
    }),
    map: row('3 · map', {
      target: choice<number>(
        SHAPE_CLASSES.map((s, c) => ({ value: c, label: s })),
        2,
        { label: 'class c' },
      ),
      method: choice<Method>(METHODS, 'cam', { label: 'method' }),
      randomise: choice<Randomise>(RANDOMISE, 'none', { label: 'layers re-initialised (sanity check)' }),
    }),
    ax: slider(RADIUS.a, SIZE - RADIUS.a, 4.5, { onChart: true }),
    ay: slider(RADIUS.a, SIZE - RADIUS.a, 5, { onChart: true }),
    bx: slider(RADIUS.b, SIZE - RADIUS.b, 11.5, { onChart: true }),
    by: slider(RADIUS.b, SIZE - RADIUS.b, 11, { onChart: true }),
  })
  const { steps, rate } = state.training
  const { first, second } = state.scene
  const { target, method, randomise } = state.map
  const setup = useMemo(() => shapesSetup(steps, rate), [steps, rate])
  const progress = useTrainedCnn(setup)
  const trained = progress.params

  const params = useMemo(() => {
    if (!trained) return null
    const layers = RANDOMISE.find((r) => r.value === randomise)!.layers
    return trained.map((p, i) => (layers.includes(i) ? fresh[i] : p))
  }, [trained, randomise])

  const { ax, ay, bx, by } = state
  const image = useMemo(() => {
    const items: ShapeItem[] = [{ shape: first, texture: 'solid', x: ax, y: ay, radius: RADIUS.a }]
    if (second !== 'none') items.push({ shape: second, texture: 'solid', x: bx, y: by, radius: RADIUS.b })
    return renderShapes(SIZE, items, { noise: 0.05, stream: stream(5) })
  }, [first, second, ax, ay, bx, by])
  const x = useMemo(() => asInput(image), [image])

  const result = useMemo(() => {
    if (!params || !trained) return null
    const z = toFlat(net.apply(params, x) as Tensor)
    const map = explain(params, x, target, method)
    let check: { mean: number; bias: number } | null = null
    if (method === 'cam') {
      const A = featureMaps(activations(net, params, x).activations[String(LAYER.relu2)] as Tensor)
      const w = toRows((params[LAYER.linear] as { weight: Tensor }).weight) as number[][]
      const cam = classActivationMap(
        A,
        w.map((r) => r[target]),
      )
      check = {
        mean: cam.reduce((a, b) => a + b, 0) / cam.length,
        bias: toFlat((params[LAYER.linear] as { bias: Tensor }).bias)[target],
      }
    }
    const similarity =
      randomise === 'none' ? null : explanationSimilarity(explain(trained, x, target, method), map).spearman
    return { z: Array.from(z), p: softmax(Array.from(z)), map, check, similarity }
  }, [params, trained, x, target, method, randomise])

  const classAxis = useAxis({ categories: [...SHAPE_CLASSES] })
  const probAxis = useAxis({ label: 'p(class | x)', range: [0, 1] })
  const handles = [
    {
      at: [state.ax - 0.5, state.ay - 0.5] as [number, number],
      onDrag: ([px, py]: [number, number]) => {
        state.set('ax', Math.min(SIZE - RADIUS.a, Math.max(RADIUS.a, px + 0.5)))
        state.set('ay', Math.min(SIZE - RADIUS.a, Math.max(RADIUS.a, py + 0.5)))
      },
      label: 'first object',
    },
  ]
  if (second !== 'none')
    handles.push({
      at: [state.bx - 0.5, state.by - 0.5],
      onDrag: ([px, py]: [number, number]) => {
        state.set('bx', Math.min(SIZE - RADIUS.b, Math.max(RADIUS.b, px + 0.5)))
        state.set('by', Math.min(SIZE - RADIUS.b, Math.max(RADIUS.b, py + 0.5)))
      },
      label: 'second object',
    })
  const methodLabel = METHODS.find((m) => m.value === method)!.label
  return (
    <Figure
      title="Where the network finds each class"
      state={state}
      caption={
        'A network of two 3 × 3 convolutions, global average pooling and a linear layer is trained in the page on 16 × 16 images of single discs, squares and crosses. Left: an input with two objects; drag either object by its centre. The ink line is the half-maximum contour of the map. Middle: the map for class c, scaled to a largest value of 1; CAM and Grad-CAM at conv2 are computed on the 8 × 8 map and resampled bilinearly to 16 × 16. Right: the class probabilities. CAM and Grad-CAM at conv2 give the same map here, since the head is pooling plus a linear layer. Re-initialise layers from the top to run the model-randomisation test: a faithful map changes when the weights it describes change.'
      }
      readouts={
        <>
          <Readout
            label="training"
            value={progress.done ? `${steps} steps done` : `step ${progress.losses.length} of ${steps}`}
          />
          {result &&
            SHAPE_CLASSES.map((s, c) => <Readout key={s} label={`p(${s})`} value={formatNumber(result.p[c])} />)}
          {result?.check && (
            <Readout
              label={`mean of CAM + bias = logit of ${SHAPE_CLASSES[target]}`}
              value={`${formatNumber(result.check.mean)} + ${formatNumber(result.check.bias)} = ${formatNumber(result.z[target])}`}
            />
          )}
          {result && result.similarity !== null && (
            <Readout label="rank correlation with the trained network's map" value={formatNumber(result.similarity)} />
          )}
        </>
      }
    >
      <div className="grid grid-cols-1 gap-4 md:grid-cols-[1fr_1fr_1fr]">
        <ImageTile
          values={image}
          width={SIZE}
          height={SIZE}
          range={[0, 1]}
          title="input"
          contour={result ? { field: result.map, levels: [0.5] } : undefined}
          handles={handles}
          ariaLabel="Input image with draggable objects"
        />
        <ImageTile
          values={result ? result.map : new Float64Array(SIZE * SIZE)}
          width={SIZE}
          height={SIZE}
          range={[0, 1]}
          title={`${methodLabel}, class ${SHAPE_CLASSES[target]}`}
          ariaLabel="Class activation map"
        />
        <FixedHeight height={220}>
          <Plot x={classAxis} y={probAxis} ariaLabel="Class probabilities">
            <Bars name="p(class | x)" x={[0, 1, 2]} y={result ? result.p : [0, 0, 0]} />
          </Plot>
        </FixedHeight>
      </div>
    </Figure>
  )
}
