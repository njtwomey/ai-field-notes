/** The functions of `aifn/graph/propagation`, registered with the notes they serve. */

import { definer, entries, type Entry, type FunctionInfo } from 'aifn/foundation/registry'
import * as propagation from './propagation'

const fn = definer<FunctionInfo>('function', 'graph/propagation')

fn(
  {
    key: 'propagate',
    name: 'Message passing',
    summary:
      'Gather source features along edges, apply an edge function, aggregate at destinations (sum, mean or max).',
    role: 'transform',
    notes: ['message-passing-neural-network', 'graph-convolutional-network', 'graph-attention-network'],
  },
  propagation.propagate,
)
fn(
  { key: 'messageEdges', name: 'Message edges', role: 'construction', notes: ['message-passing-neural-network'] },
  propagation.messageEdges,
)

/** The functions of the module, keyed by name. */
export const propagationFunctions: Readonly<Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>> =
  entries<FunctionInfo>('function', propagation) as Readonly<
    Record<string, Entry<(...args: never[]) => unknown, FunctionInfo>>
  >
