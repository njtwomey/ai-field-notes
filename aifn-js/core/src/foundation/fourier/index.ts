/**
 * `aifn/foundation/fourier`: discrete Fourier transforms with numpy's conventions. Complex arrays are `{ re, im }`
 * pairs of float64 tensors (`ComplexTensor`); real signals are rank-1 tensors or arrays (`Signal`).
 *
 * - Transforms: `fft`, `ifft` (any length: radix-2 or Bluestein), `rfft`, `irfft`, `fft2`, `ifft2`, `dft`, and the
 *   frequency grids `fftfreq`, `rfftfreq`, `fftshift`, `ifftshift`; `nextPowerOfTwo`, `isPowerOfTwo`.
 * - Complex helpers (until `complex128` lands in tensor): `magnitude`, `phase`, `power`, `decibels`.
 *
 * Fourier analysis (windows, spectra, time-frequency) is `aifn/signal`.
 */

export { decibels, magnitude, phase, power, type ComplexTensor, type Signal } from './complex'
export {
  dft,
  fft,
  fft2,
  fftfreq,
  fftshift,
  ifft,
  ifft2,
  ifftshift,
  irfft,
  isPowerOfTwo,
  nextPowerOfTwo,
  rfft,
  rfftfreq,
} from './fft'

// Raw-array helpers for the transforms in `aifn/signal` (spectral estimators, Hilbert transform, wavelets).
export { complexOf, readSignal, readValues } from './complex'
export { transformInPlace } from './fft'
export { dct } from './dct'
