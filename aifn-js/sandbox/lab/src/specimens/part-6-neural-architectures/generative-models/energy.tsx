import type { Specimen } from '@lab/specimen'
import { JemShowcase } from './_energy/showcase-jem'

export const specimens: Specimen[] = [
  {
    module: 'part-6-neural-architectures/generative-models',
    title: 'Showcase: your classifier is secretly an energy-based model',
    description:
      'A softmax classifier trained by cross-entropy beside the same network trained by JEM on 2-d classes: matching decision regions, but only JEM shapes the energy −logsumexp f(x) like the data, draws Langevin samples on the data, separates far points by energy and stays calibrated; a logit shift c(x) leaves p(y | x) fixed and moves the energy.',
    tags: ['showcase', 'JEM', 'energy-based model', 'Langevin', 'calibration', 'out-of-distribution'],
    render: () => <JemShowcase />,
  },
]
