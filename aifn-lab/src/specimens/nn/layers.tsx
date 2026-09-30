import type { Specimen } from '../../specimen'
import { AttentionQuerySpecimen, MultiHeadSpecimen } from './_shared/figures'

export const specimens: Specimen[] = [
  {
    module: 'nn/layers',
    title: 'Attention weights',
    description:
      'Scaled dot-product attention of a draggable query over six keys, and the per-head weight matrices of a causal multi-head self-attention layer.',
    tags: ['attention', 'transformer', 'softmax', 'causal mask', 'multi-head'],
    render: () => (
      <>
        <AttentionQuerySpecimen />
        <MultiHeadSpecimen />
      </>
    ),
  },
]
