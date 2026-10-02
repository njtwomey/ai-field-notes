import type { Specimen } from '../../../specimen'
import { GrokkingShowcase } from './_grokking/showcase-grokking'

export const specimens: Specimen[] = [
  {
    module: 'applied/neural/grokking',
    title: 'Showcase: grokking',
    description:
      'An MLP trained in the browser on part of the table of a ∘ b mod p fits its training pairs early and generalises much later under weight decay; the embedding’s Fourier spectrum, the residues’ circles and periodic hidden units emerge over the checkpoints.',
    tags: ['showcase', 'grokking', 'modular arithmetic', 'weight decay', 'Fourier', 'generalisation', 'training'],
    render: () => <GrokkingShowcase />,
  },
]
