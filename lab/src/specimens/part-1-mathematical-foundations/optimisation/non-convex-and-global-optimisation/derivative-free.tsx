import type { Specimen } from '@lab/specimen'
import { CmaEsSpecimen, NelderMeadSpecimen } from '../_shared/figures'

export const specimens: Specimen[] = [
  {
    module: 'part-1-mathematical-foundations/optimisation/non-convex-and-global-optimisation',
    title: 'Nelder–Mead simplex',
    description: "The Nelder–Mead simplex on Himmelblau's function, with the operation of each step.",
    tags: ['derivative free', 'simplex'],
    render: () => <NelderMeadSpecimen />,
  },
  {
    module: 'part-1-mathematical-foundations/optimisation/non-convex-and-global-optimisation',
    title: 'CMA-ES population',
    description: 'CMA-ES on the Rastrigin function: each generation’s samples, the mean path and the step size σ.',
    tags: ['evolution strategy', 'stochastic', 'stream', 'global optimisation'],
    render: () => <CmaEsSpecimen />,
  },
]
