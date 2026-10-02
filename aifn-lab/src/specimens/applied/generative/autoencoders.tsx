import type { Specimen } from '../../../specimen'
import { AutoencoderShowcase } from './_autoencoders/showcase-autoencoders'

export const specimens: Specimen[] = [
  {
    module: 'applied/generative/autoencoders',
    title: 'Showcase: autoencoders and VAEs',
    description:
      'An autoencoder, a (β-)VAE, a conditional VAE and a VQ-VAE trained in the worker on 2-d point clouds or 5 × 7 digits: reconstructions, prior samples and the decoded code grid, the codes of the data, and the loss parts over training.',
    tags: ['showcase', 'autoencoder', 'VAE', 'β-VAE', 'conditional VAE', 'VQ-VAE', 'latent space', 'training'],
    render: () => <AutoencoderShowcase />,
  },
]
