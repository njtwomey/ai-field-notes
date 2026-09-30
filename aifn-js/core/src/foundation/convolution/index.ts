/**
 * `aifn/foundation/convolution`: linear convolution and correlation of 1-D signals with numpy's and scipy.signal's
 * conventions: `convolve`, `correlate` and `fftConvolve` in `full`, `same` or `valid` mode, and `correlationLags`.
 */

export {
  convolve,
  correlate,
  correlationLags,
  fftConvolve,
  type ConvolutionMode,
  type ConvolveOptions,
} from './convolution'
export { correlate2d, convolve2d, separableFilter, type Border, type ImageInput } from './image'

export { readImage } from './image'
