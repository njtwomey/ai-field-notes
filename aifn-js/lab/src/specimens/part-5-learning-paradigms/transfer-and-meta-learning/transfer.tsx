import type { Specimen } from '@lab/specimen'
import { LabelShiftSpecimen, PrototypicalSpecimen } from './_transfer/few-shot'
import { ContinualSpecimen, DomainAdaptationSpecimen, MamlSpecimen } from './_transfer/figures'

export const specimens: Specimen[] = [
  {
    module: 'part-5-learning-paradigms/transfer-and-meta-learning',
    title: 'Domain adaptation and continual learning',
    description:
      'Source-only training against MMD, CORAL and DANN on shifted moons, with the decision boundary and the aligned features; forgetting across a task sequence with naive fine-tuning, EWC and replay; MAML on sinusoids against a pretrained network; prototypical networks learning a metric for unseen few-shot classes; and BBSE against EM estimating target class priors under label shift.',
    tags: [
      'transfer learning',
      'domain adaptation',
      'DANN',
      'MMD',
      'CORAL',
      'continual learning',
      'EWC',
      'replay',
      'MAML',
      'meta-learning',
      'prototypical networks',
      'few-shot learning',
      'label shift',
      'BBSE',
    ],
    render: () => (
      <>
        <DomainAdaptationSpecimen />
        <ContinualSpecimen />
        <MamlSpecimen />
        <PrototypicalSpecimen />
        <LabelShiftSpecimen />
      </>
    ),
  },
]
