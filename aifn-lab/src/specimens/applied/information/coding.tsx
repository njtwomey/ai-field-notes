import type { Specimen } from '../../../specimen'
import { HuffmanSpecimen } from './_coding/figures'

export const specimens: Specimen[] = [
  {
    module: 'applied/information/coding',
    title: 'Huffman tree against the entropy',
    description:
      'huffmanSteps on a Zipf distribution, drawn as a tree merge by merge with codeword paths, then codeword lengths against −log₂ p and expected lengths of the Huffman, Shannon–Fano and Shannon codes against H.',
    tags: ['huffmanCode', 'huffmanSteps', 'Tree', 'shannonFanoCode', 'shannonCode', 'kraftSum', 'coding'],
    render: () => <HuffmanSpecimen />,
  },
]
