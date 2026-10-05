import type { Specimen } from '@lab/specimen'
import { AnomalyComparison } from './_anomaly/compare'

export const specimens: Specimen[] = [
  {
    module: 'part-5-learning-paradigms/anomaly-detection',
    title: 'Showcase: anomaly detection compared',
    description:
      'Isolation forest, local outlier factor, k-NN distance, one-class SVM, SVDD, classical and robust (MCD) Mahalanobis, PCA reconstruction and an ensemble on 2-D data with planted anomalies: the score field with a draggable threshold, the score histograms, precision against recall, and every detector’s AUROC and average precision.',
    tags: [
      'showcase',
      'anomaly detection',
      'isolation forest',
      'local outlier factor',
      'one-class SVM',
      'SVDD',
      'Mahalanobis',
      'minimum covariance determinant',
      'anomalyScores',
      'compareDetectors',
    ],
    render: () => <AnomalyComparison />,
  },
]
