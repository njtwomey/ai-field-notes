/**
 * `aifn/dsp`: signal processing with numpy's and scipy.signal's conventions. Complex arrays are `{ re, im }` pairs of
 * float64 tensors (`ComplexTensor`); real signals are rank-1 tensors or arrays.
 *
 * - Fourier: `fft`, `ifft` (any length: radix-2 or Bluestein), `rfft`, `irfft`, `fft2`, `ifft2`, `dft`, `fftfreq`,
 *   `rfftfreq`, `fftshift`, `ifftshift`, `nextPowerOfTwo`, `isPowerOfTwo`; `magnitude`, `phase`, `power`, `decibels`.
 * - Windows: `getWindow` (hann, hamming, blackman, kaiser, gaussian, tukey, …), `besselI0`.
 * - Spectral estimation: `periodogram`, `welch`, `multitaper`, `dpss`; time–frequency: `stft`, `spectrogram`.
 * - Filters: `firwin`, `kaiserBeta`, `kaiserOrder`, `kaiserAttenuation`; `iirfilter`, `butter`, `cheby1`, `cheby2`;
 *   `lfilter`, `lfilterZi`, `filtfilt`; `freqz`, `groupDelay`, `unwrap`.
 * - Convolution: `convolve`, `fftConvolve`, `correlate`, `correlationLags`.
 * - Analytic signal: `hilbert`, `instantaneous`, `envelope`, `hilbertSpectrum`.
 * - Wavelets: `waveletFilters`, `dwt`, `idwt`, `wavedec`, `waverec`, `wavefun` (cascade), `morlet`, `cwt`.
 * - Empirical mode decomposition: `emd`, `eemd`, `siftSteps` (traceable), `siftImf`, `extrema`.
 * - Audio: `hzToMel`, `melToHz`, `melFilterbank`, `dct`, `mfcc`.
 * - Images: `correlate2d`, `convolve2d`, `separableFilter`, `gaussianKernel`, `gaussianBlur`, `sobel`,
 *   `laplacianOfGaussian`.
 * - Test signals: `chirp`, `tones`, `sampleTimes`.
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
export { besselI0, getWindow, type WindowInput, type WindowName, type WindowSpec } from './windows'
export {
  dpss,
  multitaper,
  periodogram,
  spectrogram,
  stft,
  welch,
  type Detrend,
  type Dpss,
  type SegmentOptions,
  type Spectrogram,
  type Spectrum,
  type Stft,
} from './spectral'
export {
  butter,
  cheby1,
  cheby2,
  filtfilt,
  firwin,
  freqz,
  groupDelay,
  iirfilter,
  kaiserAttenuation,
  kaiserBeta,
  kaiserOrder,
  lfilter,
  lfilterZi,
  unwrap,
  type FirwinOptions,
  type IirFilter,
  type IirOptions,
  type ResponseOptions,
  type Zpk,
} from './filters'
export {
  convolve,
  correlate,
  correlationLags,
  fftConvolve,
  type ConvolutionMode,
  type ConvolveOptions,
} from './convolution'
export { envelope, hilbert, hilbertSpectrum, instantaneous, type HilbertSpectrum, type Instantaneous } from './hilbert'
export {
  cwt,
  dwt,
  idwt,
  morlet,
  wavedec,
  waveletFilters,
  wavefun,
  waverec,
  type Cwt,
  type WaveletDecomposition,
  type WaveletFilters,
  type WaveletName,
} from './wavelets'
export {
  eemd,
  emd,
  extrema,
  RILLING_RULE,
  siftImf,
  siftSteps,
  type EmdOptions,
  type EmdResult,
  type SiftOptions,
  type SiftState,
  type StopRule,
} from './emd'
export {
  dct,
  hzToMel,
  melFilterbank,
  melToHz,
  mfcc,
  type MelFilterbank,
  type MelScale,
  type Mfcc,
  type MfccOptions,
} from './audio'
export {
  convolve2d,
  correlate2d,
  gaussianBlur,
  gaussianKernel,
  laplacianOfGaussian,
  separableFilter,
  sobel,
  type Border,
  type Gradients,
  type ImageInput,
} from './image'
export { chirp, sampleTimes, tones } from './signals'
