import type { Specimen } from '../../../specimen'
import { LdaSpecimen } from './_topic-models/LdaSpecimen'
import { TopicModelComparison } from './_topic-models/compare'
import { DynamicTopicFigure } from './_topic-models/dynamic'

export const specimens: Specimen[] = [
  {
    module: 'applied/inference/topic-models',
    title: 'LDA by collapsed Gibbs sampling',
    description: 'The registered LDA engine, picked by infer() from the model description, learning the bars topics.',
    tags: ['LDA', 'topic model', 'collapsed Gibbs', 'custom engine'],
    render: () => <LdaSpecimen />,
  },
  {
    module: 'applied/inference/topic-models',
    title: 'Showcase: topic models compared',
    description:
      'LDA by collapsed Gibbs sampling, the HDP (topics inferred), pLSA by EM, NMF, LSA, a correlated topic model and labelled LDA fitted in the browser to a seeded corpus about animals, food, vehicles, colours and places: each topic’s top words with its NPMI coherence, the documents’ topic proportions, and the fit and coherence step by step; then a dynamic topic model following four themes whose words change over time.',
    tags: [
      'showcase',
      'topic models',
      'LDA',
      'pLSA',
      'NMF',
      'LSA',
      'correlated topic model',
      'labelled LDA',
      'HDP',
      'hierarchical Dirichlet process',
      'dynamic topic model',
      'coherence',
      'NPMI',
      'topicModelRun',
    ],
    render: () => (
      <>
        <TopicModelComparison />
        <DynamicTopicFigure />
      </>
    ),
  },
]
