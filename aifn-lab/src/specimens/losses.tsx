import type { Specimen } from '../specimen'
import { MarginLossesSpecimen, RankingLossesSpecimen, RegressionLossesSpecimen } from './_losses/figures'

export const specimens: Specimen[] = [
  {
    module: 'losses',
    title: 'Classification losses and the margin',
    description:
      'The 0–1 loss and its surrogates (hinge, squared hinge, logistic, exponential, modified Huber) as functions of the margin y·f(x), with their derivatives.',
    tags: ['surrogate loss', 'hinge', 'logistic', 'margin', 'classification'],
    render: () => <MarginLossesSpecimen />,
  },
  {
    module: 'losses',
    title: 'Regression losses',
    description: 'Squared, absolute, Huber, log-cosh and pinball losses of the residual, with δ and τ adjustable.',
    tags: ['regression', 'Huber', 'quantile', 'pinball', 'robust'],
    render: () => <RegressionLossesSpecimen />,
  },
  {
    module: 'losses',
    title: 'Ranking losses on a toy list',
    description:
      'Pointwise, pairwise (RankNet, hinge, LambdaRank) and listwise (softmax, ListNet, ListMLE, ApproxNDCG) losses of six draggable scores, with the push each loss gives every item.',
    tags: ['ranking', 'learning to rank', 'LambdaRank', 'ListMLE', 'recommendation'],
    render: () => <RankingLossesSpecimen />,
  },
]
