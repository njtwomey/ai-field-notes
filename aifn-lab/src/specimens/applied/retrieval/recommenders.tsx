import type { Specimen } from '../../../specimen'
import { FeedbackLoopShowcase } from './_recommenders/feedback'
import { MatchboxSpecimen } from './_recommenders/matchbox'
import { RecommenderShowcase } from './_recommenders/showcase'

export const specimens: Specimen[] = [
  {
    module: 'applied/retrieval/recommenders',
    title: 'Showcase: recommenders, from popularity to two towers',
    description:
      'Popularity, user- and item-kNN, implicit ALS, logistic MF, BPR, factorisation machines, field-aware FM, Wide & Deep, DeepFM, NCF, a two-tower model and SASRec, trained in the browser on implicit feedback with a known low-rank taste and popularity-biased exposure: every user’s scores, held-out recall and NDCG by epoch, the learned embedding map and a pinned user’s top k; and the feedback loop, where policies retrained on their own clicks concentrate exposure.',
    tags: [
      'showcase',
      'recommender systems',
      'collaborative filtering',
      'matrix factorisation',
      'BPR',
      'two-tower',
      'SASRec',
      'feedback loop',
      'popularity bias',
      'recommenderRun',
      'feedbackLoop',
    ],
    render: () => (
      <>
        <RecommenderShowcase />
        <FeedbackLoopShowcase />
      </>
    ),
  },
  {
    module: 'applied/retrieval/recommenders',
    title: 'Matchbox: Bayesian recommendation with feature traits',
    description:
      'Matchbox (Stern, Herbrich and Graepel, 2009) trained online by assumed-density filtering on synthetic ordinal ratings whose traits depend on user and item features: held-out error for known and cold-start users, the recovered item trait map and a predicted rating distribution.',
    tags: [
      'Matchbox',
      'recommender systems',
      'Bayesian',
      'assumed-density filtering',
      'expectation propagation',
      'cold start',
      'ordinal regression',
      'matchboxRun',
    ],
    render: () => <MatchboxSpecimen />,
  },
]
