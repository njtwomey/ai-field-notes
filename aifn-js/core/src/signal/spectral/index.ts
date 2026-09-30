/**
 * `aifn/signal/spectral`: spectral estimation (periodogram, Welch, multitaper with DPSS tapers), returning `Spectrum`s,
 * and the short-time Fourier transform and spectrogram, returning `TimeFrequency` rasters.
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
