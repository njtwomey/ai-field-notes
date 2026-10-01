import type { Specimen } from '../../specimen'
import { DecodingSpecimen, SpeculativeSpecimen } from './_decoding/figures'

export const specimens: Specimen[] = [
  {
    module: 'nn/decoding',
    title: 'Decoding strategies',
    description:
      'Greedy decoding, sampling with temperature, top-k, top-p and a repetition penalty, and beam search over a Kneser–Ney n-gram model or a tiny GPT trained in the browser, one token per step; speculative decoding with an n-gram draft and an acceptance trace.',
    tags: ['decoding', 'language model', 'sampling', 'top-p', 'beam search', 'speculative decoding', 'GPT', 'n-gram'],
    render: () => (
      <>
        <DecodingSpecimen />
        <SpeculativeSpecimen />
      </>
    ),
  },
]
