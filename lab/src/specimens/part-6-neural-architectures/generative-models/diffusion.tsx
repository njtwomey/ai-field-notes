import type { Specimen } from '@lab/specimen'
import { ForwardNoisingSpecimen, LearnedDenoiserSpecimen, ReverseSamplingSpecimen } from './_diffusion/figures'
import { DiffusionForward, DiffusionReverse } from './_diffusion/showcase-diffusion'

export const specimens: Specimen[] = [
  {
    module: 'part-6-neural-architectures/generative-models',
    title: 'Forward noising',
    description: 'A 2-D Gaussian mixture noised by the DDPM forward process under the linear and cosine schedules.',
    tags: ['DDPM', 'noise schedule', 'forward process', 'cosine schedule'],
    render: () => <ForwardNoisingSpecimen />,
  },
  {
    module: 'part-6-neural-architectures/generative-models',
    title: 'DDPM, DDIM and the probability-flow ODE',
    description:
      'Reverse sampling paths of three samplers from the same noise, all driven by the exact score of a Gaussian mixture, so no training is needed.',
    tags: ['DDPM', 'DDIM', 'probability flow', 'score', 'sampling', 'trace'],
    render: () => <ReverseSamplingSpecimen />,
  },
  {
    module: 'part-6-neural-architectures/generative-models',
    title: 'A learned noise predictor',
    description: 'An MLP denoiser trained on a 2-D Gaussian mixture with the simple DDPM loss, then sampled with DDIM.',
    tags: ['denoiser', 'training', 'MLP', 'DDIM', 'mixture'],
    render: () => <LearnedDenoiserSpecimen />,
  },
  {
    module: 'part-6-neural-architectures/generative-models',
    title: 'Showcase: diffusion in two dimensions',
    description:
      'A seven-component mixture noised over t with its exact score field, then sampled back by DDPM, DDIM and the probability-flow ODE from the same noise, with the particle paths played over t.',
    tags: ['showcase', 'DDPM', 'DDIM', 'probability flow', 'score', 'VP SDE'],
    render: () => (
      <>
        <DiffusionForward />
        <DiffusionReverse />
      </>
    ),
  },
]
