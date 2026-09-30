import type { Specimen } from '../../specimen'
import {
  ClusteringSpecimen,
  CurveGallerySpecimen,
  RankingSpecimen,
  ReliabilitySpecimen,
  TextSpecimen,
} from './_metrics/figures'
import { ClassifierDashboardSpecimen } from './_metrics/classifier-dashboard'

export const specimens: Specimen[] = [
  {
    module: 'learning/metrics',
    title: 'A threshold, its confusion matrix, ROC and PR',
    description:
      'A classifier-evaluation dashboard: a 2-D dataset from aifn/datasets and one of thirteen classifiers (logistic, polynomial and spline logistic, k-NN, LDA, QDA, naive Bayes, RBF SVM, tree, forest, boosting, GP, MLP), scored on a held-out set. One threshold moves the decision boundary, the class histograms, the contingency table with its margins (confusionMatrix, confusionMargins) and the operating point on rocCurve and precisionRecallCurve.',
    tags: [
      'confusion matrix',
      'contingency table',
      'threshold',
      'decision boundary',
      'ROC',
      'precision-recall',
      'AUROC',
      'average precision',
      'dashboard',
    ],
    render: () => <ClassifierDashboardSpecimen />,
  },
  {
    module: 'learning/metrics',
    title: 'Curve gallery: ROC, PR, DET, gain, cost, PRG',
    description:
      'Two scoring classifiers drawn by CurveView as any of the typed curves, with DeLong and bootstrap intervals for AUROC.',
    tags: ['CurveView', 'ROC', 'DET', 'cost curve', 'lift', 'precision-recall-gain', 'DeLong', 'bootstrap'],
    render: () => <CurveGallerySpecimen />,
  },
  {
    module: 'learning/metrics',
    title: 'Reliability diagram and calibration errors',
    description:
      'reliabilityDiagram of a temperature-mis-scaled model, with ECE, MCE, RMS and debiased calibration errors and the Murphy decomposition of the Brier score.',
    tags: ['calibration', 'ECE', 'reliability diagram', 'Brier', 'CurveView'],
    render: () => <ReliabilitySpecimen />,
  },
  {
    module: 'learning/metrics',
    title: 'nDCG of a noisy ranking',
    description:
      'Discounted gains of a ranking by noisy scores against the ideal order, with nDCG, DCG, precision at k, AP, reciprocal rank and ERR.',
    tags: ['ranking', 'nDCG', 'DCG', 'ERR', 'MAP', 'MRR'],
    render: () => <RankingSpecimen />,
  },
  {
    module: 'learning/metrics',
    title: 'Clustering scores for a degrading clustering',
    description:
      'ARI, AMI, NMI, Rand, Fowlkes–Mallows and the silhouette as a growing fraction of labels is reassigned at random.',
    tags: ['clustering', 'ARI', 'AMI', 'NMI', 'silhouette'],
    render: () => <ClusteringSpecimen />,
  },
  {
    module: 'learning/metrics',
    title: 'Text metrics of a sentence pair',
    description:
      'BLEU (plain and smoothed), chrF, ROUGE-1/2/L, WER, CER and TER of an editable reference and candidate.',
    tags: ['BLEU', 'chrF', 'ROUGE', 'WER', 'TER'],
    render: () => <TextSpecimen />,
  },
]
