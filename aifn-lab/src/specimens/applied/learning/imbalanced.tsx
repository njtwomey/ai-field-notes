import type { Specimen } from '../../../specimen'
import { ResamplingSpecimen } from './_imbalanced/figures'

export const specimens: Specimen[] = [
  {
    module: 'applied/learning/preprocessing',
    title: 'Imbalanced learning: resampling',
    description:
      'Random over- and under-sampling, SMOTE, borderline-SMOTE, ADASYN and Tomek-link cleaning on two overlapping Gaussian classes with a rare class: where synthetic points go, how the logistic-regression boundary moves, and what happens to recall, precision, AUROC and average precision.',
    tags: ['class imbalance', 'SMOTE', 'borderline-SMOTE', 'ADASYN', 'Tomek links', 'over-sampling', 'under-sampling'],
    render: () => <ResamplingSpecimen />,
  },
]
