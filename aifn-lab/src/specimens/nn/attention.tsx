import type { Specimen } from '../../specimen'
import {
  AttentionPositionsSpecimen,
  BiasSpecimen,
  FlashAttentionSpecimen,
  KvCacheSpecimen,
  RopeSpecimen,
} from './_attention/figures'
import { AttentionQuerySpecimen, MultiHeadSpecimen } from './_shared/figures'

export const specimens: Specimen[] = [
  {
    module: 'nn/attention',
    title: 'Attention, masks and positions',
    description:
      'Self-attention over an editable sequence with every positional scheme (none, sinusoidal, learned, RoPE, ALiBi, T5) and mask (causal, sliding window, padding); RoPE rotating queries and keys; ALiBi and T5 biases; key–value cache memory; the tiled online softmax of FlashAttention.',
    tags: ['attention', 'transformer', 'positional encoding', 'RoPE', 'ALiBi', 'mask', 'KV cache', 'FlashAttention'],
    render: () => (
      <>
        <AttentionQuerySpecimen />
        <AttentionPositionsSpecimen />
        <RopeSpecimen />
        <BiasSpecimen />
        <KvCacheSpecimen />
        <MultiHeadSpecimen />
        <FlashAttentionSpecimen />
      </>
    ),
  },
]
