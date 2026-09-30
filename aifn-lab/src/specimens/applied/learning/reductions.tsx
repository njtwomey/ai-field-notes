import type { Specimen } from '../../../specimen'
import { OutputCodeSpecimen } from './_shared/figures'

export const specimens: Specimen[] = [
  {
    module: 'applied/learning/reductions',
    title: 'Output codes',
    description: 'Code matrices for one-versus-rest, one-versus-one, exhaustive and random codes, with codeDistance.',
    tags: ['outputCode', 'exhaustiveCode', 'randomCode', 'codeDistance', 'ECOC'],
    render: () => <OutputCodeSpecimen />,
  },
]
