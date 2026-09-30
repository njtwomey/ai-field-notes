import type { Specimen } from '../../specimen'
import { LineSearchSpecimen } from './_shared/figures'

export const specimens: Specimen[] = [
  {
    module: 'optim/line-search',
    title: 'Line-search trials',
    description:
      'Gradient descent with a backtracking (Armijo) or strong Wolfe line search: the trial points of each step on the surface and on φ(α), with the sufficient-decrease line.',
    tags: ['line search', 'Armijo', 'Wolfe', 'backtracking'],
    render: () => <LineSearchSpecimen />,
  },
]
