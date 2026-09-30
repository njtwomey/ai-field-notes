import type { Specimen } from '../../specimen'
import { CmaEsSpecimen, NelderMeadSpecimen } from './_shared/figures'

export const specimens: Specimen[] = [
  {
    module: 'optim/derivative-free',
    title: 'Nelder–Mead simplex',
    description: "The Nelder–Mead simplex on Himmelblau's function, with the operation of each step.",
    tags: ['derivative free', 'simplex'],
    render: () => <NelderMeadSpecimen />,
  },
  {
    module: 'optim/derivative-free',
    title: 'CMA-ES population',
    description: 'CMA-ES on the Rastrigin function: each generation’s samples, the mean path and the step size σ.',
    tags: ['evolution strategy', 'stochastic', 'stream', 'global optimisation'],
    render: () => <CmaEsSpecimen />,
  },
]
