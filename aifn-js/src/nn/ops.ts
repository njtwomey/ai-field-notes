/**
 * The primitives `nn` adds to `aifn/tensor`, each with its vjp written with primitives so that it can be differentiated
 * again:
 *
 * - `gather` / `scatterAdd` (flat indices): each is the other's adjoint. Embedding lookups (`takeRows`) and max
 *   pooling are gathers.
 * - 2-D convolution with stride, padding and dilation, and its two adjoints (with respect to the input and to the
 *   kernel). The three form a closed family: each one's vjp is made of the other two, so convolutions have derivatives
 *   of every order. 1-D convolution is the 2-D one on a height-1 image.
 * - Average pooling and its adjoint.
 *
 * Convolution is cross-correlation, as in every deep-learning library: y[n, o, i, j] = Σ_{c, a, b} x[n, c, i·s_h − p_h
 * + a·d_h, j·s_w − p_w + b·d_w] · w[o, c, a, b], with zeros outside the image (Dumoulin & Visin, 2016, "A guide to
 * convolution arithmetic for deep learning", §2–5, for the output sizes).
 */

import {
  defineOp,
  fromData,
  gather,
  take,
  isContiguous,
  reshape,
  shapeOfValue,
  toFlat,
  unwrap,
  type Op,
  type Tensor,
  type Value,
} from 'aifn/tensor'

/** Row-major float64 data of a raw value (no copy when already contiguous float64 at offset 0). */
export function f64(x: number | Tensor): Float64Array {
  if (typeof x === 'number') return new Float64Array([x])
  if (x.dtype === 'float64' && isContiguous(x) && x.offset === 0 && x.data.length === sizeOf(x.shape)) {
    return x.data as Float64Array
  }
  return Float64Array.from(toFlat(x))
}

export function sizeOf(shape: readonly number[]): number {
  return shape.reduce((a, b) => a * b, 1)
}

function tensorOf(x: number | Tensor, where: string): Tensor {
  if (typeof x === 'number') throw new Error(`${where}: expected a tensor, got a number`)
  return x
}

// ── Gather and take (moved to aifn/tensor) ──────────────────────────────────────────────────────────────────────────

// TODO(consolidation WP5): remove; import `gather` and `take` from aifn/tensor.
export { gather }

/**
 * The rows `indices` of x along its first axis (an embedding lookup). Deprecated: `take` from `aifn/tensor`.
 * TODO(consolidation WP5): remove.
 */
export const takeRows: typeof take = take

// ── Convolution ──────────────────────────────────────────────────────────────────────────────────────────────────────

/** Stride, zero padding and dilation along (height, width). */
export type ConvGeometry = {
  stride: readonly [number, number]
  padding: readonly [number, number]
  dilation: readonly [number, number]
}

type ConvParams = ConvGeometry & { x: readonly number[]; w: readonly number[] }

/** The output length of a convolution along one axis: ⌊(n + 2p − d(k − 1) − 1)/s⌋ + 1. */
export function convOutputSize(n: number, k: number, stride: number, padding: number, dilation: number): number {
  return Math.floor((n + 2 * padding - dilation * (k - 1) - 1) / stride) + 1
}

/** The output shape [N, O, H′, W′] of a convolution. */
function convOutputShape({ x, w, stride, padding, dilation }: ConvParams): number[] {
  return [
    x[0],
    w[0],
    convOutputSize(x[2], w[2], stride[0], padding[0], dilation[0]),
    convOutputSize(x[3], w[3], stride[1], padding[1], dilation[1]),
  ]
}

/**
 * Visit every (input, kernel, output) triple of flat offsets that a convolution multiplies together. The three
 * primitives differ only in which two of the three arrays they read and which one they accumulate into.
 */
function convLoop(p: ConvParams, visit: (xi: number, wi: number, yi: number) => void): void {
  const [N, C, H, W] = p.x
  const [O, , KH, KW] = p.w
  const [, , HO, WO] = convOutputShape(p)
  const [sh, sw] = p.stride
  const [ph, pw] = p.padding
  const [dh, dw] = p.dilation
  for (let n = 0; n < N; n++)
    for (let o = 0; o < O; o++)
      for (let i = 0; i < HO; i++)
        for (let j = 0; j < WO; j++) {
          const yi = ((n * O + o) * HO + i) * WO + j
          for (let c = 0; c < C; c++)
            for (let a = 0; a < KH; a++) {
              const r = i * sh - ph + a * dh
              if (r < 0 || r >= H) continue
              for (let b = 0; b < KW; b++) {
                const q = j * sw - pw + b * dw
                if (q < 0 || q >= W) continue
                visit(((n * C + c) * H + r) * W + q, ((o * C + c) * KH + a) * KW + b, yi)
              }
            }
        }
}

const conv2dOp: Op<ConvGeometry> = defineOp<ConvGeometry>(
  'conv2d',
  ([x, w], g) => {
    const p = { ...g, x: tensorOf(x, 'conv2d').shape, w: tensorOf(w, 'conv2d').shape }
    const xd = f64(x)
    const wd = f64(w)
    const shape = convOutputShape(p)
    const y = new Float64Array(sizeOf(shape))
    convLoop(p, (xi, wi, yi) => (y[yi] += xd[xi] * wd[wi]))
    return fromData(y, shape)
  },
  (g, [x, w], _y, geometry) => [
    conv2dInputGradOp([g, w], { ...geometry, x: shapeOfValue(x) }),
    conv2dWeightGradOp([x, g], { ...geometry, w: shapeOfValue(w) }),
  ],
)

// The adjoint in the input: gx = Jᵀg for J = conv(·, w). Its vjp: conv(u, w) for g, weightGrad(u, g) for w.
const conv2dInputGradOp: Op<ConvGeometry & { x: readonly number[] }> = defineOp<
  ConvGeometry & { x: readonly number[] }
>(
  'conv2dInputGrad',
  ([g, w], p) => {
    const params = { ...p, w: tensorOf(w, 'conv2dInputGrad').shape }
    const gd = f64(g)
    const wd = f64(w)
    const gx = new Float64Array(sizeOf(p.x))
    convLoop(params, (xi, wi, yi) => (gx[xi] += gd[yi] * wd[wi]))
    return fromData(gx, p.x)
  },
  (u, [g, w], _y, p) => [
    conv2dOp([u, w], geometryOf(p)),
    conv2dWeightGradOp([u, g], { ...geometryOf(p), w: shapeOfValue(w) }),
  ],
)

// The adjoint in the kernel: ⟨gw, v⟩ = ⟨g, conv(x, v)⟩. Its vjp: inputGrad(g, u) for x, conv(x, u) for g.
const conv2dWeightGradOp: Op<ConvGeometry & { w: readonly number[] }> = defineOp<
  ConvGeometry & { w: readonly number[] }
>(
  'conv2dWeightGrad',
  ([x, g], p) => {
    const params = { ...p, x: tensorOf(x, 'conv2dWeightGrad').shape }
    const xd = f64(x)
    const gd = f64(g)
    const gw = new Float64Array(sizeOf(p.w))
    convLoop(params, (xi, wi, yi) => (gw[wi] += gd[yi] * xd[xi]))
    return fromData(gw, p.w)
  },
  (u, [x, g], _y, p) => [
    conv2dInputGradOp([g, u], { ...geometryOf(p), x: shapeOfValue(x) }),
    conv2dOp([x, u], geometryOf(p)),
  ],
)

function geometryOf(p: ConvGeometry): ConvGeometry {
  return { stride: p.stride, padding: p.padding, dilation: p.dilation }
}

/** A per-axis option: one number for both axes or a pair. */
export type Pair = number | readonly [number, number]

const pair = (v: Pair): [number, number] => (typeof v === 'number' ? [v, v] : [v[0], v[1]])

/** Options of `conv2d` and `conv1d`. */
export type ConvOptions<P = Pair> = {
  /** Step between output positions. Default 1. */
  stride?: P
  /** Zeros added on each side of the input. Default 0. */
  padding?: P
  /** Spacing between kernel taps (1 = contiguous). Default 1. */
  dilation?: P
}

/**
 * 2-D convolution (cross-correlation) of x, shape [N, C, H, W] (or [C, H, W]), with kernels w, shape [O, C, KH, KW],
 * giving [N, O, H′, W′] (or [O, H′, W′]) with H′ = ⌊(H + 2p − d(KH − 1) − 1)/s⌋ + 1. A primitive differentiable in x and
 * w to any order. Matches `torch.nn.functional.conv2d` (without bias; add it by broadcasting).
 */
export function conv2d(x: Value, w: Value, options: ConvOptions = {}): Value {
  const geometry: ConvGeometry = {
    stride: pair(options.stride ?? 1),
    padding: pair(options.padding ?? 0),
    dilation: pair(options.dilation ?? 1),
  }
  const xs = shapeOfValue(x)
  const ws = shapeOfValue(w)
  if (ws.length !== 4) throw new Error(`conv2d: kernels need shape [O, C, KH, KW], got [${ws.join(', ')}]`)
  const batched = xs.length === 4
  if (!batched && xs.length !== 3) throw new Error(`conv2d: input needs shape [N, C, H, W] or [C, H, W]`)
  const x4 = batched ? x : reshape(x, [1, ...xs])
  if (shapeOfValue(x4)[1] !== ws[1])
    throw new Error(`conv2d: input has ${shapeOfValue(x4)[1]} channels, kernels ${ws[1]}`)
  const y = conv2dOp([x4, w], geometry)
  return batched ? y : reshape(y, shapeOfValue(y).slice(1))
}

/**
 * 1-D convolution of x, shape [N, C, L] (or [C, L]), with kernels w, shape [O, C, K], giving [N, O, L′]. Computed as
 * `conv2d` on a height-1 image, so it shares its derivatives.
 */
export function conv1d(x: Value, w: Value, options: ConvOptions<number> = {}): Value {
  const xs = shapeOfValue(x)
  const ws = shapeOfValue(w)
  if (ws.length !== 3) throw new Error(`conv1d: kernels need shape [O, C, K]`)
  const batched = xs.length === 3
  const x4 = reshape(x, batched ? [xs[0], xs[1], 1, xs[2]] : [1, xs[0], 1, xs[1]])
  const w4 = reshape(w, [ws[0], ws[1], 1, ws[2]])
  const y = conv2d(x4, w4, {
    stride: [1, options.stride ?? 1],
    padding: [0, options.padding ?? 0],
    dilation: [1, options.dilation ?? 1],
  })
  const [n, o, , l] = shapeOfValue(y)
  return reshape(y, batched ? [n, o, l] : [o, l])
}

// ── Pooling ──────────────────────────────────────────────────────────────────────────────────────────────────────────

type PoolParams = { kernel: [number, number]; stride: [number, number]; padding: [number, number]; x: number[] }

/** Visit each (input offset, output offset) pair of a pooling window over [N, C, H, W]; padded taps are skipped. */
function poolLoop(p: PoolParams, visit: (xi: number, yi: number) => void): void {
  const [N, C, H, W] = p.x
  const [, , HO, WO] = poolShape(p)
  for (let nc = 0; nc < N * C; nc++)
    for (let i = 0; i < HO; i++)
      for (let j = 0; j < WO; j++) {
        const yi = (nc * HO + i) * WO + j
        for (let a = 0; a < p.kernel[0]; a++) {
          const r = i * p.stride[0] - p.padding[0] + a
          if (r < 0 || r >= H) continue
          for (let b = 0; b < p.kernel[1]; b++) {
            const q = j * p.stride[1] - p.padding[1] + b
            if (q < 0 || q >= W) continue
            visit((nc * H + r) * W + q, yi)
          }
        }
      }
}

function poolShape(p: PoolParams): number[] {
  const [N, C, H, W] = p.x
  return [
    N,
    C,
    convOutputSize(H, p.kernel[0], p.stride[0], p.padding[0], 1),
    convOutputSize(W, p.kernel[1], p.stride[1], p.padding[1], 1),
  ]
}

const avgPoolOp: Op<PoolParams> = defineOp<PoolParams>(
  'avgPool2d',
  ([x], p) => {
    const xd = f64(x)
    const shape = poolShape(p)
    const y = new Float64Array(sizeOf(shape))
    const area = p.kernel[0] * p.kernel[1]
    poolLoop(p, (xi, yi) => (y[yi] += xd[xi] / area))
    return fromData(y, shape)
  },
  (g, _inputs, _y, p) => [avgPoolAdjointOp([g], p)],
)

const avgPoolAdjointOp: Op<PoolParams> = defineOp<PoolParams>(
  'avgPool2dAdjoint',
  ([g], p) => {
    const gd = f64(g)
    const gx = new Float64Array(sizeOf(p.x))
    const area = p.kernel[0] * p.kernel[1]
    poolLoop(p, (xi, yi) => (gx[xi] += gd[yi] / area))
    return fromData(gx, p.x)
  },
  (u, _inputs, _y, p) => [avgPoolOp([u], p)],
)

/** Options of the pooling functions. */
export type PoolOptions<P = Pair> = {
  /** Step between windows. Default: the kernel size (non-overlapping windows). */
  stride?: P
  /** Padding on each side: zeros for average pooling (counted in the mean), −∞ for max pooling. Default 0. */
  padding?: P
}

function pool4(x: Value, kernel: Pair, options: PoolOptions, what: string) {
  const xs = shapeOfValue(x)
  const batched = xs.length === 4
  if (!batched && xs.length !== 3) throw new Error(`${what}: input needs shape [N, C, H, W] or [C, H, W]`)
  const x4 = batched ? x : reshape(x, [1, ...xs])
  const k = pair(kernel)
  const p: PoolParams = {
    kernel: k,
    stride: pair(options.stride ?? k),
    padding: pair(options.padding ?? 0),
    x: shapeOfValue(x4),
  }
  const unbatch = (y: Value) => (batched ? y : reshape(y, shapeOfValue(y).slice(1)))
  return { x4, p, unbatch }
}

/**
 * 2-D average pooling of [N, C, H, W] (or [C, H, W]) over kernel windows; padded zeros count in the mean (PyTorch's
 * default `count_include_pad`). Differentiable to any order.
 */
export function avgPool2d(x: Value, kernel: Pair, options: PoolOptions = {}): Value {
  const { x4, p, unbatch } = pool4(x, kernel, options, 'avgPool2d')
  return unbatch(avgPoolOp([x4], p))
}

/**
 * 2-D max pooling of [N, C, H, W] (or [C, H, W]) over kernel windows. The maximum's position is found on the values
 * and the output gathers it, so the gradient flows to the first maximum of each window (ties go to the earliest tap).
 */
export function maxPool2d(x: Value, kernel: Pair, options: PoolOptions = {}): Value {
  const { x4, p, unbatch } = pool4(x, kernel, options, 'maxPool2d')
  const xd = f64(unwrap(x4))
  const shape = poolShape(p)
  const best = new Float64Array(sizeOf(shape)).fill(-Infinity)
  const at = new Int32Array(sizeOf(shape)).fill(-1)
  poolLoop(p, (xi, yi) => {
    if (xd[xi] > best[yi] || at[yi] < 0) {
      best[yi] = xd[xi]
      at[yi] = xi
    }
  })
  return unbatch(gather(x4, at, shape))
}

/** 1-D average pooling of [N, C, L] (or [C, L]); see `avgPool2d`. */
export function avgPool1d(x: Value, kernel: number, options: PoolOptions<number> = {}): Value {
  return pool1d(x, kernel, options, avgPool2d)
}

/** 1-D max pooling of [N, C, L] (or [C, L]); see `maxPool2d`. */
export function maxPool1d(x: Value, kernel: number, options: PoolOptions<number> = {}): Value {
  return pool1d(x, kernel, options, maxPool2d)
}

function pool1d(
  x: Value,
  kernel: number,
  options: PoolOptions<number>,
  pool2: (x: Value, k: Pair, o: PoolOptions) => Value,
): Value {
  const xs = shapeOfValue(x)
  const batched = xs.length === 3
  const x4 = reshape(x, batched ? [xs[0], xs[1], 1, xs[2]] : [1, xs[0], 1, xs[1]])
  const y = pool2(x4, [1, kernel], {
    stride: [1, options.stride ?? kernel],
    padding: [0, options.padding ?? 0],
  })
  const [n, c, , l] = shapeOfValue(y)
  return reshape(y, batched ? [n, c, l] : [c, l])
}
