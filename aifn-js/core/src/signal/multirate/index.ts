/**
 * `aifn/signal/multirate`: sample-rate change and filter banks, as scipy.signal: `resamplePoly` (rational resampling
 * by a polyphase FIR), `decimateSignal` (scipy's `decimate`: anti-aliasing filter, then every q-th sample), `upfirdn`
 * (upsample, FIR filter, downsample; `aifn/foundation/convolution`'s), `polyphase` (the polyphase components of a
 * filter) and `dftFilterBank` (the uniform DFT analysis filter bank, by its polyphase implementation).
 */

export { upfirdn } from 'aifn/foundation/convolution'
export {
  decimateSignal,
  dftFilterBank,
  polyphase,
  resamplePoly,
  type DecimateOptions,
  type ResamplePolyOptions,
} from './multirate'
export { multirateFunctions } from './registry'
