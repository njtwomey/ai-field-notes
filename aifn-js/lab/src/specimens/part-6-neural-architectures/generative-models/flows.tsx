import type { Specimen } from '@lab/specimen'
import { RealNvpShowcase } from './_flows/showcase-realnvp'

export const specimens: Specimen[] = [
  {
    module: 'part-6-neural-architectures/generative-models',
    title: 'Showcase: RealNVP, a coupling flow',
    description:
      'A RealNVP normalising flow trained in the worker on moons, rings, pinwheels and spirals: the model density against the truth, samples, the data pushed through each coupling layer to a Gaussian, and the training likelihood.',
    tags: ['showcase', 'normalising flow', 'RealNVP', 'affine coupling', 'change of variables', 'training'],
    render: () => <RealNvpShowcase />,
  },
]
