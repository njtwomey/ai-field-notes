import type { Specimen } from '@lab/specimen'
import { TransformerShowcase } from './_language-models/showcase-transformer'

export const specimens: Specimen[] = [
  {
    module: 'part-6-neural-architectures/transformers',
    title: 'Showcase: a transformer, module by module',
    description:
      'A two-layer decoder-only transformer trained in the browser on reversing, copying, sorting, bracket completion, addition or associative recall, with every module of one forward pass on an input you type: embeddings, attention per head, the residual stream, normalisation, MLP activations, logits and next-token probabilities.',
    tags: ['showcase', 'transformer', 'GPT', 'attention', 'residual stream', 'training', 'induction', 'decoding'],
    render: () => <TransformerShowcase />,
  },
]
