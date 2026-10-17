/**
 * The small convolutional network the figures of this chapter train and then interpret: two 3 x 3 convolutions with
 * ReLU, a 2 x 2 max pool between them, global average pooling and one linear layer, on 16 x 16 greyscale images of
 * shapes. It is the architecture class activation maps need (pooled maps fed straight to a linear classifier). The
 * model and the training are engine layers and the engine's training loop; this file only fixes the recipe.
 */
import { child, stream } from 'aifn-compute/foundation/random'
import type { Params } from 'aifn-compute/foundation/pytree'
import { fromData, tensor, toFlat, toRows, type Tensor } from 'aifn-compute/foundation/tensor'
import { live } from 'aifn-compute/foundation/trace'
import { softmaxCrossEntropy } from 'aifn-compute/learning/losses'
import {
  ActivationLayer,
  AvgPool2d,
  Conv2d,
  Flatten,
  Linear,
  MaxPool2d,
  Sequential,
  type Context,
} from 'aifn-compute/nn/layers'
import { trainingLoop } from 'aifn-compute/nn/training'
import { adamRule } from 'aifn-compute/optim/first-order'
import { shapeTextureDataset, type ShapeName, type TextureName } from 'aifn-methods/data/synthetic'

/** Image side in pixels. */
export const SIZE = 16
/** Channels of the two convolutions. */
export const C1 = 6
export const C2 = 8

/** The layers of `smallCnn` by index, which is also each layer's tap path. */
export const LAYER = { conv1: 0, relu1: 1, pool1: 2, conv2: 3, relu2: 4, gap: 5, flatten: 6, linear: 7 } as const

/** The layers of `smallCnn`, in order. */
function cnnLayers(classes: number) {
  return [
    Conv2d(1, C1, 3, { padding: 1 }),
    ActivationLayer('relu'),
    MaxPool2d(2),
    Conv2d(C1, C2, 3, { padding: 1 }),
    ActivationLayer('relu'),
    AvgPool2d(SIZE / 2),
    Flatten(1),
    Linear(C2, classes),
  ]
}

/** Conv(1 -> 6, 3 x 3, pad 1), ReLU, max pool 2, Conv(6 -> 8, 3 x 3, pad 1), ReLU, global average pool, Linear(8 -> k). */
export function smallCnn(classes: number) {
  return Sequential(...cnnLayers(classes))
}

/**
 * The first `depth` layers of `smallCnn` (apply it with the first `depth` entries of the parameters): depth 4 ends at
 * the second convolution, before its ReLU, with output [n, 8, 8, 8].
 */
export function smallCnnTrunk(depth: number) {
  return Sequential(...cnnLayers(1).slice(0, depth))
}

/**
 * The input rows (or columns) a unit at row (or column) i of the second convolution's 8 x 8 map reads: conv1 adds one
 * pixel on each side, the pool doubles, conv2 adds one pooled cell (two pixels) on each side, so rows 2i - 3 to 2i + 4.
 */
export const conv2Field = (i: number): [number, number] => [2 * i - 3, 2 * i + 4]

export type SmallCnn = ReturnType<typeof smallCnn>
/** The parameter list of `smallCnn`: one entry per layer. */
export type CnnParams = Params[]

/** The classes of the plain-shapes task (solid fill), used by the feature-visualisation and CAM figures. */
export const SHAPE_CLASSES: readonly ShapeName[] = ['disc', 'square', 'cross']

/** The plain-shapes recipe: 40 solid shapes per class, minibatches of 30, a given number of Adam steps and rate. */
export const shapesSetup = (steps: number, learningRate: number): TrainSetup => ({
  shapes: SHAPE_CLASSES,
  textures: ['solid', 'solid', 'solid'],
  restyle: 0,
  perClass: 40,
  steps,
  learningRate,
  batchSize: 30,
  seed: 1,
})

/** What a training run is given. */
export type TrainSetup = {
  shapes: readonly ShapeName[]
  textures: readonly TextureName[]
  /** Probability that a training image's texture is redrawn at random (0: texture always matches shape). */
  restyle: number
  perClass: number
  steps: number
  learningRate: number
  batchSize: number
  seed: number
}

/** A checkpoint of a run: the step, the minibatch loss and, on the last one, the trained parameters. */
export type TrainCheckpoint = { step: number; loss: number; params: CnnParams | null }

/** The training set of a setup: `shapeTextureDataset` with matched pairing and the setup's restyle probability. */
export function trainingSet(setup: TrainSetup) {
  return shapeTextureDataset(child(stream(setup.seed), 'train'), {
    size: SIZE,
    perClass: setup.perClass,
    shapes: setup.shapes,
    textures: setup.textures,
    restyle: setup.restyle,
  })
}

/**
 * Adam on softmax cross-entropy over minibatches, as a generator that yields one checkpoint per step and the trained
 * parameters with the last, so a figure can run it in slices between frames.
 */
export function* trainSmallCnn(setup: TrainSetup): Generator<TrainCheckpoint> {
  const net = smallCnn(setup.shapes.length)
  const data = trainingSet(setup)
  const y = tensor(Array.from(data.shape))
  const loss = (p: CnnParams, b: { x: Tensor; y: Tensor }, ctx: Context) =>
    softmaxCrossEntropy(net.apply(p, b.x, ctx), b.y)
  const alg = trainingLoop({
    loss,
    data: { x: data.x, y },
    batchSize: setup.batchSize,
    optimizer: adamRule({ stepSize: setup.learningRate }),
  })
  const start = { params: net.init(child(stream(setup.seed), 'init')) as CnnParams }
  for (const { step, state } of live(alg, start, { stream: child(stream(setup.seed), 'run') })) {
    const last = step >= setup.steps || state.diverged
    yield { step, loss: state.loss, params: last ? state.params : null }
    if (last) return
  }
}

/** Logits of a batch of images [n, 1, SIZE, SIZE], one row per image. */
export function logits(net: SmallCnn, params: CnnParams, x: Tensor): number[][] {
  return toRows(net.apply(params, x) as Tensor) as number[][]
}

/** One image (SIZE * SIZE values) as the network's input [1, 1, SIZE, SIZE]. */
export function asInput(image: ArrayLike<number>): Tensor {
  return fromData(Float64Array.from(image), [1, 1, SIZE, SIZE])
}

/** Softmax of a row of logits. */
export function softmax(z: readonly number[]): number[] {
  const m = Math.max(...z)
  const e = z.map((v) => Math.exp(v - m))
  const s = e.reduce((a, b) => a + b, 0)
  return e.map((v) => v / s)
}

/** The index of the largest entry. */
export const argmax = (z: readonly number[]) => z.indexOf(Math.max(...z))

/** The image of example i of a dataset's x, [n, 1, SIZE, SIZE], as SIZE * SIZE values. */
export function imageAt(x: Tensor, i: number): Float64Array {
  return Float64Array.from(toFlat(x).slice(i * SIZE * SIZE, (i + 1) * SIZE * SIZE))
}
