/** The functions of `aifn-applied/learning/preprocessing` besides its registered transformers. */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn/foundation/registry'
import * as encoding from './encoding'
import * as transformer from './transformer'

const fn = definer<FunctionInfo>('function', 'learning/preprocessing')

fn(
  { key: 'fitTransform', name: 'Fit and transform', role: 'transform', notes: ['feature-scaling', 'data-leakage'] },
  transformer.fitTransform,
)
fn({ key: 'checkColumns', name: 'Check columns', role: 'property' }, transformer.checkColumns)
fn(
  {
    key: 'targetEncodeCrossFit',
    name: 'Cross-fitted target encoding',
    summary: 'Target means per category computed out of fold, so a row never sees its own label.',
    role: 'transform',
    notes: ['categorical-encoding', 'data-leakage'],
    cite: ['micci2001'],
  },
  encoding.targetEncodeCrossFit,
)

/** The functions of the module, keyed by name. */
export const preprocessingFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', transformer, encoding) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
