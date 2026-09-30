/**
 * `aifn/signal`: signal processing, as scipy.signal, pywt and PyEMD: windows, filters, spectral estimation,
 * time–frequency analysis, wavelets, statistical signal processing and decompositions. The shared layer holds the
 * signal objects: `signal` (samples with a sample rate), `spectrum`, `timeFrequency`. Children: windows, filters,
 * spectral, time-frequency, wavelets, statistical, decompositions; gap: multirate.
 */

export {
  isSignal,
  sampleTimes,
  signal,
  spectrum,
  timeFrequency,
  type Signal,
  type SignalInput,
  type SignalOptions,
  type Spectrum,
  type TimeFrequency,
} from './signal'
export { getWindow } from './windows'
export { firwin, butter, lfilter, freqz } from './filters'
export { periodogram, welch } from './spectral'
export { hilbert } from './time-frequency'
export { dwt, idwt } from './wavelets'
export { yuleWalker, burg } from './statistical'
export { emd } from './decompositions'
