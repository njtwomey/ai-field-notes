import type { Specimen } from '../../../specimen'
import { ClipAblation, ClipShowcase } from './_contrastive/showcase'

export const specimens: Specimen[] = [
  {
    module: 'applied/neural/contrastive',
    title: 'Showcase: contrastive alignment of two views (a tiny CLIP)',
    description:
      'Shapes seen as an image and as a caption, two MLP encoders trained by symmetric InfoNCE with a learnable temperature, the embeddings aligning on the unit circle, zero-shot classification by class captions, retrieval, and an ablation over batch size and temperature.',
    tags: ['showcase', 'CLIP', 'InfoNCE', 'contrastive', 'zero-shot', 'retrieval', 'alignment', 'uniformity'],
    render: () => (
      <>
        <ClipShowcase />
        <ClipAblation />
      </>
    ),
  },
]
