import type { Specimen } from '../../specimen'
import { DatasetFromCode } from './_interpreter/figures'

export const specimens: Specimen[] = [
  {
    module: 'applied/interpreter',
    title: 'Interpreter: a dataset from code',
    description:
      'A JavaScript program, run by aifn/interpreter with seeded draws, makes a dataset; linear or logistic regression is fitted to it.',
    tags: ['interpreter', 'code editor', 'linear regression', 'logistic regression', 'seeds'],
    render: () => <DatasetFromCode />,
  },
]
