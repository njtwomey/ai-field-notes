import type { Specimen } from '../specimen'
import {
  BroadcastSpecimen,
  EinsumSpecimen,
  LogSumExpSpecimen,
  PrimitiveSpecimen,
  ViewsSpecimen,
} from './_tensor/figures'

export const specimens: Specimen[] = [
  {
    module: 'tensor',
    title: 'Broadcasting',
    description:
      'A column of shape [m, 1] and a row of shape [1, n] combine elementwise into an m×n grid (NumPy rules).',
    tags: ['broadcasting', 'elementwise', 'heatmap'],
    render: () => <BroadcastSpecimen />,
  },
  {
    module: 'tensor',
    title: 'Views: slices, transposes and strides',
    description:
      'Slicing, reversing and permuting return views over the same data: only shape, strides and offset change.',
    tags: ['views', 'strides', 'slice', 'transpose'],
    render: () => <ViewsSpecimen />,
  },
  {
    module: 'tensor',
    title: 'logsumexp without overflow',
    description: 'logsumexp stays finite for values near ±800, where the naive log Σ exp gives ∞ or −∞.',
    tags: ['reductions', 'stable forms', 'softmax'],
    render: () => <LogSumExpSpecimen />,
  },
  {
    module: 'tensor',
    title: 'einsum and batched matmul',
    description:
      "einsum('bij,jk->bik') equals matmul with a broadcast batch axis; the rank-3 result is browsed by slider.",
    tags: ['einsum', 'matmul', 'batched'],
    render: () => <EinsumSpecimen />,
  },
  {
    module: 'tensor',
    title: 'A primitive defined once',
    description: 'defineUnary makes one function (here softplus) that maps numbers to numbers and tensors to tensors.',
    tags: ['defineUnary', 'primitives', 'autodiff'],
    render: () => <PrimitiveSpecimen />,
  },
]
