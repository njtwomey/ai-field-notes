import type { Specimen } from '../specimen'
import { TextPipelineSpecimen } from './_text/figures'
import { TokenisersComparedSpecimen } from './_text/tokenisers'

export const specimens: Specimen[] = [
  {
    module: 'text',
    title: 'Text: from characters to features',
    description:
      'One corpus through the text pipeline: normalisation and tokens with offsets, byte-pair encoding stepped merge by merge, the bag of words weighted by TF-IDF and BM25, and word and character n-grams hashed into a fixed number of columns.',
    tags: ['tokenisation', 'BPE', 'TF-IDF', 'BM25', 'n-grams', 'feature hashing', 'Porter stemmer', 'normalisation'],
    render: () => <TextPipelineSpecimen />,
  },
  {
    module: 'text',
    title: 'Text: tokenisers compared',
    description:
      'One text under eight tokenisers trained on one corpus (characters, bytes, Treebank words, BPE, byte-level BPE, WordPiece, unigram, SentencePiece BPE with byte fallback): coloured token chips with ids and offsets, fertility and compression, BPE-dropout, and a subword vocabulary trained step by step.',
    tags: [
      'tokenisation',
      'BPE',
      'byte-level BPE',
      'WordPiece',
      'unigram',
      'SentencePiece',
      'byte fallback',
      'BPE-dropout',
      'offsets',
      'fertility',
    ],
    render: () => <TokenisersComparedSpecimen />,
  },
]
