/** The functions of `aifn-applied/unsupervised/embedding/manifold`. */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn/foundation/registry'
import * as manifold from './manifold'

definer<FunctionInfo>('function', 'unsupervised/embedding/manifold')(
  {
    key: 'neighbourGraph',
    name: 'Neighbourhood graph',
    role: 'construction',
    notes: ['isomap', 'laplacian-eigenmaps', 'locally-linear-embedding'],
  },
  manifold.neighbourGraph,
)

/** The functions of the module, keyed by name. */
export const manifoldFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', manifold) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
