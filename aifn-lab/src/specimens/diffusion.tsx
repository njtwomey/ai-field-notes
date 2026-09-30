import type { Specimen } from '../specimen'
import { ForwardNoisingSpecimen, LearnedDenoiserSpecimen, ReverseSamplingSpecimen } from './_diffusion/figures'

export const specimens: Specimen[] = [
  {
    module: 'diffusion',
    title: 'Forward noising',
    description: 'A 2-D Gaussian mixture noised by the DDPM forward process under the linear and cosine schedules.',
    tags: ['DDPM', 'noise schedule', 'forward process', 'cosine schedule'],
    render: () => <ForwardNoisingSpecimen />,
  },
  {
    module: 'diffusion',
    title: 'DDPM, DDIM and the probability-flow ODE',
    description:
      'Reverse sampling paths of three samplers from the same noise, all driven by the exact score of a Gaussian mixture, so no training is needed.',
    tags: ['DDPM', 'DDIM', 'probability flow', 'score', 'sampling', 'trace'],
    render: () => <ReverseSamplingSpecimen />,
  },
  {
    module: 'diffusion',
    title: 'A learned noise predictor',
    description: 'An MLP denoiser trained on a 2-D Gaussian mixture with the simple DDPM loss, then sampled with DDIM.',
    tags: ['denoiser', 'training', 'MLP', 'DDIM', 'mixture'],
    render: () => <LearnedDenoiserSpecimen />,
  },
]
