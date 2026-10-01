/** The functions of `aifn-applied/vision/filters`. */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn/foundation/registry'
import * as image from './image'

const fn = definer<FunctionInfo>('function', 'vision/filters')

fn(
  {
    key: 'gaussianKernel',
    name: 'Gaussian kernel',
    role: 'construction',
    notes: ['image-filtering'],
    cite: ['lindeberg1994'],
  },
  image.gaussianKernel,
)
fn(
  {
    key: 'gaussianBlur',
    name: 'Gaussian blur',
    role: 'transform',
    notes: ['image-filtering'],
    cite: ['lindeberg1994'],
  },
  image.gaussianBlur,
)
fn(
  {
    key: 'sobel',
    name: 'Sobel gradient',
    role: 'transform',
    notes: ['edge-detection', 'image-filtering'],
    cite: ['gonzalez1985'],
  },
  image.sobel,
)
fn(
  {
    key: 'laplacianOfGaussian',
    name: 'Laplacian of Gaussian',
    role: 'construction',
    notes: ['edge-detection'],
    cite: ['marr1980'],
  },
  image.laplacianOfGaussian,
)

/** The functions of the module, keyed by name. */
export const visionFilterFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', image) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
