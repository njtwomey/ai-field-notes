/**
 * A coordinate network for a ray-cast world: an MLP from a camera pose (x, y, θ) to the distances of R rays across the
 * field of view. The training set is the ray caster itself, run at poses drawn uniformly over the free floor and
 * headings. Inputs pass through a fixed positional encoding first (sines and cosines at octave frequencies, as in
 * NeRF), because a plain MLP fits the low frequencies of a function first and the walls' edges are high frequencies.
 */

import { Mlp, type Layer } from 'aifn-compute/nn/layers'
import { trainingLoop, type TrainingState } from 'aifn-compute/nn/training'
import { adamRule } from 'aifn-compute/optim/first-order'
import { meanSquaredErrorLoss } from 'aifn-compute/learning/losses'
import { fromData, toFlat, unwrap, type Tensor } from 'aifn-compute/foundation/tensor'
import { child, stream, uniform, type Stream } from 'aifn-compute/foundation/random'
import type { Params } from 'aifn-compute/foundation/pytree'
import type { Algorithm } from 'aifn-compute/foundation/trace'
import { castFan, castRay, freeCells, isFree, MAX_DISTANCE, type Pose, type World } from '../_shared/world'

export type Encoding = 'raw' | 'fourier'

export type Form = 'fan' | 'ray'

/** What the network outputs: the distance divided by the map size, or its logarithm. */
export type Target = 'distance' | 'log'

export type ModelOptions = {
  form: Form
  target: Target
  rays: number
  fov: number
  encoding: Encoding
  /** Octaves of the positional encoding: frequencies 2^0 … 2^(L−1). */
  octaves: number
  width: number
  depth: number
  learningRate: number
  batchSize: number
}

/** Distances are divided by this before fitting, so targets lie roughly in [0, 1]. */
export const distanceScale = (w: World) => Math.max(w.width, w.height)

/** Number of input features of the encoding. */
export const featureCount = (encoding: Encoding, octaves: number) => (encoding === 'raw' ? 4 : 6 * octaves)

/**
 * Write the features of a pose into `out` at `offset`. Positions are rescaled to u = x / width and v = y / height in
 * [0, 1]. Raw: (2u − 1, 2v − 1, cos θ, sin θ). Fourier: sin and cos of 2^k π u, 2^k π v and 2^k θ for k < L; integer
 * multiples of θ keep every feature periodic in the heading.
 */
export function encode(w: World, p: Pose, encoding: Encoding, octaves: number, out: Float64Array, offset: number) {
  const u = p.x / w.width
  const v = p.y / w.height
  if (encoding === 'raw') {
    out[offset] = 2 * u - 1
    out[offset + 1] = 2 * v - 1
    out[offset + 2] = Math.cos(p.theta)
    out[offset + 3] = Math.sin(p.theta)
    return
  }
  let j = offset
  for (let k = 0; k < octaves; k++) {
    const f = 2 ** k
    out[j++] = Math.sin(f * Math.PI * u)
    out[j++] = Math.cos(f * Math.PI * u)
    out[j++] = Math.sin(f * Math.PI * v)
    out[j++] = Math.cos(f * Math.PI * v)
    out[j++] = Math.sin(f * p.theta)
    out[j++] = Math.cos(f * p.theta)
  }
}

/** n poses drawn uniformly over the free floor (a free cell, then a point in it clear of walls) and headings. */
export function samplePoses(w: World, n: number, s: Stream): Pose[] {
  const cells = freeCells(w)
  const poses: Pose[] = []
  // Points too close to a wall are skipped, so draw in rounds until there are enough.
  for (let round = 0; poses.length < n && round < 20; round++) {
    const d = toFlat(uniform(child(s, round), 0, 1, { shape: [4 * n] }) as Tensor)
    for (let i = 0; i < n && poses.length < n; i++) {
      const c = cells[Math.min(cells.length - 1, Math.floor(d[4 * i] * cells.length))]
      const x = c.x - 0.5 + d[4 * i + 1]
      const y = c.y - 0.5 + d[4 * i + 2]
      if (isFree(w, x, y, 0.1)) poses.push({ x, y, theta: Math.PI * (2 * d[4 * i + 3] - 1) })
    }
  }
  return poses
}

export type Dataset = { x: Tensor; y: Tensor }

const toTarget = (d: number, scale: number, t: Target) => (t === 'log' ? Math.log(d) : d / scale)
const fromTarget = (v: number, scale: number, t: Target) => (t === 'log' ? Math.exp(v) : v * scale)

/**
 * Features and scaled distances. Fan form: one row per pose, its R ray distances as the targets. Ray form: one row per
 * ray, the pose's heading replaced by the ray's absolute angle and its one distance as the target; a pose's fan is
 * then R rows.
 */
export function makeDataset(
  w: World,
  poses: Pose[],
  offsets: Float64Array,
  o: ModelOptions,
  perRay: Float64Array | null,
): Dataset {
  const F = featureCount(o.encoding, o.octaves)
  const scale = distanceScale(w)
  if (o.form === 'fan') {
    const R = offsets.length
    const xs = new Float64Array(poses.length * F)
    const ys = new Float64Array(poses.length * R)
    poses.forEach((p, i) => {
      encode(w, p, o.encoding, o.octaves, xs, i * F)
      castFan(w, p, offsets).forEach((h, r) => (ys[i * R + r] = toTarget(h.distance, scale, o.target)))
    })
    return { x: fromData(xs, [poses.length, F]), y: fromData(ys, [poses.length, R]) }
  }
  const angles = perRay ?? Float64Array.of(0)
  const n = poses.length * angles.length
  const xs = new Float64Array(n * F)
  const ys = new Float64Array(n)
  let i = 0
  for (const p of poses)
    for (const a of angles) {
      const q = { ...p, theta: p.theta + a }
      encode(w, q, o.encoding, o.octaves, xs, i * F)
      ys[i++] = toTarget(castRay(w, q.x, q.y, q.theta).distance, scale, o.target)
    }
  return { x: fromData(xs, [n, F]), y: fromData(ys, [n, 1]) }
}

/** The network for these options: an MLP from the encoded input to R distances (fan form) or one (ray form). */
export function network(o: ModelOptions): Layer<Params[]> {
  const F = featureCount(o.encoding, o.octaves)
  return Mlp([F, ...Array.from({ length: o.depth }, () => o.width), o.form === 'fan' ? o.rays : 1])
}

const rootStream = (w: World, seed: number) => stream(`neural-ray-casting/${w.id}/${seed}`)

/** The initial weights, the same wherever they are drawn (the page and the training worker). */
export const initialParams = (w: World, o: ModelOptions, seed = 1): Params[] =>
  network(o).init(child(rootStream(w, seed), 'init'))

export type Model = {
  net: Layer<Params[]>
  alg: Algorithm<{ params: Params[] }, TrainingState<Params[]>>
  start: { params: Params[] }
  train: Dataset
  /** 200 training poses, each with its whole fan, for the training error. */
  trainCheck: Dataset
  /** 200 held-out poses, each with its whole fan (R rows in ray form). */
  test: Dataset
}

export function buildModel(w: World, offsets: Float64Array, o: ModelOptions, seed = 1): Model {
  const root = rootStream(w, seed)
  // The same number of distances in both forms: 4000 fans of R rays, or 4000 R single rays.
  const poses = samplePoses(w, o.form === 'fan' ? 4000 : 4000 * offsets.length, child(root, 'train'))
  const train = makeDataset(w, poses, offsets, o, null)
  const trainCheck = makeDataset(w, poses.slice(0, 200), offsets, o, offsets)
  const test = makeDataset(w, samplePoses(w, 200, child(root, 'test')), offsets, o, offsets)
  const net = network(o)
  const alg = trainingLoop({
    loss: (p: Params[], b: { x: Tensor; y: Tensor }) => meanSquaredErrorLoss(net.apply(p, b.x), b.y),
    data: { x: train.x, y: train.y },
    batchSize: o.batchSize,
    optimizer: adamRule({ stepSize: o.learningRate }),
  })
  return { net, alg, start: { params: initialParams(w, o, seed) }, train, trainCheck, test }
}

/** The network's ray distances (unscaled) for the fan at one pose. */
export function predict(
  net: Layer<Params[]>,
  w: World,
  params: Params[],
  p: Pose,
  offsets: Float64Array,
  o: ModelOptions,
): Float64Array {
  const d = makeDataset(w, [p], offsets, o, offsets)
  const out = toFlat(unwrap(net.apply(params, d.x)) as Tensor)
  const scale = distanceScale(w)
  return Float64Array.from(out, (v) => Math.min(MAX_DISTANCE, Math.max(0.05, fromTarget(v, scale, o.target))))
}

/** Errors of the network's distances on a dataset: root mean square in cells, and the median relative error. */
export function errors(net: Layer<Params[]>, w: World, params: Params[], d: Dataset, o: ModelOptions) {
  const out = toFlat(unwrap(net.apply(params, d.x)) as Tensor)
  const y = toFlat(d.y)
  const scale = distanceScale(w)
  let s = 0
  const rel = new Float64Array(y.length)
  for (let i = 0; i < y.length; i++) {
    const truth = fromTarget(y[i], scale, o.target)
    const e = fromTarget(out[i], scale, o.target) - truth
    s += e * e
    rel[i] = Math.abs(e) / truth
  }
  rel.sort()
  const within = rel.findIndex((r) => r > 0.1)
  return {
    rmse: Math.sqrt(s / y.length),
    relative: rel[Math.floor(rel.length / 2)],
    /** Fraction of rays whose distance is within 10 % of the truth, so their wall slice is about the right height. */
    within10: (within < 0 ? rel.length : within) / rel.length,
  }
}
