import type { Specimen } from '../../../specimen'
import { HyphenationCrfShowcase } from './_hyphenation/crf'
import { HyphenationShowcase } from './_hyphenation/showcase'

export const specimens: Specimen[] = [
  {
    module: 'applied/text/hyphenation',
    title: "Showcase: hyphenation, from TeX's patterns to a small LSTM",
    description:
      "Four hyphenators learned in the browser from 6800 common English words of the public-domain Moby Hyphenator list: Liang's patterns (TeX's algorithm) learned by PATGEN level by level, a NETtalk-style 7-letter window MLP, a bidirectional LSTM tagger and a linear-chain CRF over CRF++ templates. Type any word to compare them, see which patterns fire at each gap, and score hyphen points on held-out words with precision first.",
    tags: ['showcase', 'hyphenation', 'Liang', 'PATGEN', 'TeX', 'NETtalk', 'LSTM', 'sequence tagging', 'precision'],
    render: () => <HyphenationShowcase />,
  },
  {
    module: 'applied/text/hyphenation',
    title: 'Showcase: hyphenation with a CRF',
    description:
      'A linear-chain CRF over CRF++ feature templates learns to hyphenate 6800 common English words: edit the templates, the training size, -c and -f, pick L-BFGS, OWL-QN (L1 + L2), SGD or Adam, compare held-out precision and recall with Liang, the window MLP and the BiLSTM, watch L1 keep a fraction of the weights, type a word to see its marginals, Viterbi path and firing features, and read the most informative features and the errors.',
    tags: ['showcase', 'hyphenation', 'CRF', 'CRF++', 'feature templates', 'OWL-QN', 'L1', 'sequence tagging'],
    render: () => <HyphenationCrfShowcase />,
  },
]
