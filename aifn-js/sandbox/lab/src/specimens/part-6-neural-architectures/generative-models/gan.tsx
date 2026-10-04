import type { Specimen } from '@lab/specimen'
import { GanShowcase } from './_gan/showcase-gan'

export const specimens: Specimen[] = [
  {
    module: 'part-6-neural-architectures/generative-models',
    title: 'Showcase: a GAN on 2-d data',
    description:
      'An MLP generator and discriminator trained in the worker on rings, grids, pinwheels and moons under the minimax, non-saturating, WGAN-GP and hinge games: generated points against real ones over training, the discriminator field with the push on the generator, the optimal discriminator from the known density, both losses and mode coverage.',
    tags: ['showcase', 'GAN', 'mode collapse', 'WGAN-GP', 'discriminator', 'training'],
    render: () => <GanShowcase />,
  },
]
