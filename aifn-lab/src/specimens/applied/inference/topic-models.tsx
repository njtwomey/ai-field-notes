import type { Specimen } from '../../../specimen'
import { LdaSpecimen } from './_topic-models/LdaSpecimen'

export const specimens: Specimen[] = [
  {
    module: 'applied/inference/topic-models',
    title: 'LDA by collapsed Gibbs sampling',
    description: 'The registered LDA engine, picked by infer() from the model description, learning the bars topics.',
    tags: ['LDA', 'topic model', 'collapsed Gibbs', 'custom engine'],
    render: () => <LdaSpecimen />,
  },
]
