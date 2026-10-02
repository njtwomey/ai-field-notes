import type { Specimen } from '@lab/specimen'
import {
  BroadcastSpecimen,
  EinsumSpecimen,
  LogSumExpSpecimen,
  PrimitiveSpecimen,
  ViewsSpecimen,
} from './_tensor/figures'

export const specimens: Specimen[] = [
  {
    module: 'part-1-mathematical-foundations/mathematics/linear-algebra',
    title: 'Broadcasting',
    description:
      'A column of shape [m, 1] and a row of shape [1, n] combine elementwise into an m×n grid (NumPy rules).',
    tags: ['broadcasting', 'elementwise', 'heatmap'],
    render: () => <BroadcastSpecimen />,
  },
  {
    module: 'part-1-mathematical-foundations/mathematics/linear-algebra',
    title: 'Views: slices, transposes and strides',
    description:
      'Slicing, reversing and permuting return views over the same data: only shape, strides and offset change.',
    tags: ['views', 'strides', 'slice', 'transpose'],
    render: () => <ViewsSpecimen />,
  },
  {
    module: 'part-1-mathematical-foundations/mathematics/linear-algebra',
    title: 'logsumexp without overflow',
    description: 'logsumexp stays finite for values near ±800, where the naive log Σ exp gives ∞ or −∞.',
    tags: ['reductions', 'stable forms', 'softmax'],
    render: () => <LogSumExpSpecimen />,
  },
  {
    module: 'part-1-mathematical-foundations/mathematics/linear-algebra',
    title: 'einsum and batched matmul',
    description:
      "einsum('bij,jk->bik') equals matmul with a broadcast batch axis; the rank-3 result is browsed by slider.",
    tags: ['einsum', 'matmul', 'batched'],
    render: () => <EinsumSpecimen />,
  },
  {
    module: 'part-1-mathematical-foundations/mathematics/linear-algebra',
    title: 'A primitive defined once',
    description: 'defineUnary makes one function (here softplus) that maps numbers to numbers and tensors to tensors.',
    tags: ['defineUnary', 'primitives', 'autodiff'],
    render: () => <PrimitiveSpecimen />,
  },
]
