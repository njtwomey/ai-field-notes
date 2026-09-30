/**
 * The shared layer of `aifn/signal`: the constructors and readers of the signal-processing objects defined in
 * `aifn/foundation/contracts` (design S §2.13): `Signal` (samples with a sample rate), `Spectrum` and
 * `TimeFrequency`. Every function of the family takes a `SignalInput` (a `Signal`, or a bare tensor or array, which
 * means fs = 1 unless an `fs` option says otherwise) and returns these objects, so axes and units carry through a
 * chain. Frequencies follow the owner's decision 14: Hz with `fs` for sampled data, always tagged by `axis`.
 * Conventions follow scipy.signal (Virtanen et al., 2020, "SciPy 1.0", Nature Methods 17).
 */

import { dense, fromData, isTensor, type Tensor } from 'aifn/foundation/tensor'
import type { Scalar, Size, Signal, Spectrum, TimeFrequency, VectorLike } from 'aifn/foundation/contracts'
import { ShapeError } from 'aifn/foundation/errors'
import { complex } from 'aifn/systems'

export type { Signal, Spectrum, TimeFrequency } from 'aifn/foundation/contracts'

/** What a signal-processing function accepts: a `Signal`, or bare samples (a rank-1 tensor or an array). */
export type SignalInput = Signal | VectorLike

/** Options of `signal`. */
export type SignalOptions = {
  /** Samples per second. Default 1 (time in samples). */
  fs?: Scalar
  /** Time of sample 0, in seconds. Default 0. */
  t0?: Scalar
  /** Unit of the values, e.g. `V`, `Pa`, `a.u.`. */
  unit?: string
  /** Channel names, for a [channels, n] signal. */
  channels?: readonly string[]
}

/** True for a `Signal` object (`kind: 'signal'`). */
export function isSignal(x: unknown): x is Signal {
  return typeof x === 'object' && x !== null && (x as { kind?: unknown }).kind === 'signal'
}

/**
 * A `Signal` from samples: `data` is [n] or [channels, n] (a copy is taken of an array; a tensor is kept). A `Signal`
 * passed in keeps its data and has the given options replace its own.
 *
 * @example signal([0, 1, 0, -1], { fs: 4 }) // one cycle of a 1 Hz sine sampled at 4 Hz
 */
export function signal(data: SignalInput | Tensor, options: SignalOptions = {}): Signal {
  const base = isSignal(data) ? data : undefined
  const values = base ? base.data : isTensor(data) ? data : dense.vec(dense.toF64(data as VectorLike, 'signal'))
  if (values.shape.length < 1 || values.shape.length > 2)
    throw new ShapeError('signal', `signal: data must be [n] or [channels, n], got [${values.shape.join(', ')}]`)
  const fs = options.fs ?? base?.fs ?? 1
  if (!(fs > 0)) throw new RangeError('signal: fs must be positive')
  const unit = options.unit ?? base?.unit
  const channels = options.channels ?? base?.channels
  return {
    kind: 'signal',
    data: values,
    fs,
    t0: options.t0 ?? base?.t0 ?? 0,
    ...(unit !== undefined ? { unit } : {}),
    ...(channels !== undefined ? { channels } : {}),
  }
}

/** The samples of a rank-1 signal input, with its sample rate and start time. */
export type Samples = { values: dense.F64; fs: Scalar; t0: Scalar; unit?: string }

/**
 * The samples of a single-channel `SignalInput` as a fresh float64 array (never shared with the input), with fs (the
 * `fs` option if given, else the signal's, else 1) and t0.
 */
export function readSamples(x: SignalInput, where: string, fs?: Scalar): Samples {
  if (isSignal(x)) {
    if (x.data.shape.length !== 1)
      throw new ShapeError(where, `${where}: expected a single-channel signal, got [${x.data.shape.join(', ')}]`)
    return { values: Float64Array.from(dense.toF64(x.data, where)), fs: fs ?? x.fs, t0: x.t0, unit: x.unit }
  }
  return { values: Float64Array.from(dense.toF64(x, where)), fs: fs ?? 1, t0: 0 }
}

/** The sample times t0 + k/fs of a signal, as a rank-1 tensor. */
export function sampleTimes(s: Signal): Tensor {
  const n = s.data.shape[s.data.shape.length - 1]
  return fromData(
    Float64Array.from({ length: n }, (_, k) => s.t0 + k / s.fs),
    [n],
  )
}

/** The fields of a `Spectrum` other than `kind`. */
export type SpectrumFields = Omit<Spectrum, 'kind'>

/** A `Spectrum` from its fields (the `kind` brand is added). */
export function spectrum(fields: SpectrumFields): Spectrum {
  return { kind: 'spectrum', ...fields }
}

/** The fields of a `TimeFrequency` other than `kind`. */
export type TimeFrequencyFields = Omit<TimeFrequency, 'kind'>

/** A `TimeFrequency` raster from its fields (the `kind` brand is added). */
export function timeFrequency(fields: TimeFrequencyFields): TimeFrequency {
  return { kind: 'time-frequency', ...fields }
}

/** Unit of a density or power spectrum of a signal with value unit `unit`: `u²/Hz` or `u²`. */
export function powerUnit(unit: string | undefined, density: boolean): string | undefined {
  if (unit === undefined) return undefined
  return density ? `${unit}²/Hz` : `${unit}²`
}

/**
 * Complex values in the interim layout (phase 3 brings `complex128`): a [...shape, 2] float64 tensor of (re, im)
 * pairs, the layout of `aifn/systems`' `complex` helpers.
 */
export function complexValues(re: ArrayLike<number>, im: ArrayLike<number>, shape: readonly Size[]): Tensor {
  return fromData(dense.data(complex.pairsOf(re, im)), [...shape, 2])
}
