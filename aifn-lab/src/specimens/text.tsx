import type { Specimen } from '../specimen'
import { TextPipelineSpecimen } from './_text/figures'

export const specimens: Specimen[] = [
  {
    module: 'text',
    title: 'Text: from characters to features',
    description:
      'One corpus through the text pipeline: normalisation and tokens with offsets, byte-pair encoding stepped merge by merge, the bag of words weighted by TF-IDF and BM25, and word and character n-grams hashed into a fixed number of columns.',
    tags: ['tokenisation', 'BPE', 'TF-IDF', 'BM25', 'n-grams', 'feature hashing', 'Porter stemmer', 'normalisation'],
    render: () => <TextPipelineSpecimen />,
  },
]
