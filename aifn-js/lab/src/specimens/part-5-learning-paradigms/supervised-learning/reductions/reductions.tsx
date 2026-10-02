import type { Specimen } from '@lab/specimen'
import { OutputCodeSpecimen } from '../_shared/figures'

export const specimens: Specimen[] = [
  {
    module: 'part-5-learning-paradigms/supervised-learning/reductions',
    title: 'Output codes',
    description: 'Code matrices for one-versus-rest, one-versus-one, exhaustive and random codes, with codeDistance.',
    tags: ['outputCode', 'exhaustiveCode', 'randomCode', 'codeDistance', 'ECOC'],
    render: () => <OutputCodeSpecimen />,
  },
]
