/**
 * `aifn/signal/spectral`: spectral estimation (periodogram, Welch, multitaper with DPSS tapers), returning `Spectrum`s,
 * and the short-time Fourier transform and spectrogram, returning `TimeFrequency` rasters; the inverse STFT (`istft`)
 * with the COLA/NOLA window tests.
 */

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
} from './spectral'
export { checkCola, checkNola, istft, type IstftOptions } from './inverse'
