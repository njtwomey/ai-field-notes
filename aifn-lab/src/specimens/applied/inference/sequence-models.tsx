import type { Specimen } from '../../../specimen'
import { CrfTemplatesSpecimen } from './_sequence-models/CrfTemplatesSpecimen'

export const specimens: Specimen[] = [
  {
    module: 'applied/inference/sequence-models',
    title: 'Conditional random fields: feature templates and inference',
    description:
      'A linear-chain CRF over CRF++ feature templates on a toy part-of-speech task: edit the templates (U unigram, B bigram, %x[row,column] macros), see which cells a template reads at a position and the feature strings that fire there with their weights, train by L-BFGS, OWL-QN, SGD or Adam, read the weights and transitions as heatmaps, and step forward–backward and Viterbi along the lattice.',
    tags: [
      'CRF',
      'CRF++',
      'feature templates',
      'sequence labelling',
      'forward–backward',
      'Viterbi',
      'OWL-QN',
      'L-BFGS',
    ],
    render: () => <CrfTemplatesSpecimen />,
  },
]
