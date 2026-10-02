import type { Specimen } from '@lab/specimen'
import { HuffmanShowcase } from './_coding/figures'

export const specimens: Specimen[] = [
  {
    module: 'part-2-signals-systems-and-information/information-theory/channel-capacity-and-coding',
    title: 'Showcase: Huffman coding, merge by merge',
    description:
      'huffmanSteps played from the queue of leaves to the finished tree, binary or D-ary, with a tie rule and its effect on length variance; click a leaf to pin its codeword, encode and decode a message; then codeword lengths against −log p (Zipf, uniform, dyadic, skewed, English) and block coding.',
    tags: [
      'showcase',
      'huffmanSteps',
      'huffmanCode',
      'canonicalCode',
      'sourceExtension',
      'prefixEncode',
      'prefixDecode',
      'kraftSum',
      'Diagram',
      'coding',
    ],
    render: () => <HuffmanShowcase />,
  },
]
