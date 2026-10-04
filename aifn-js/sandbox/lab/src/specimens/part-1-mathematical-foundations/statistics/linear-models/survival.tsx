import type { Specimen } from '@lab/specimen'
import { PartialLikelihoodSpecimen, SurvivalCurvesSpecimen } from './_survival/figures'

export const specimens: Specimen[] = [
  {
    module: 'part-1-mathematical-foundations/statistics/linear-models',
    title: 'Survival analysis: Kaplan–Meier to Cox',
    description:
      'Censored data with a known Weibull truth: Kaplan–Meier curves by arm with censor ticks and the log-rank test, then Cox and accelerated-failure-time fits against the true survival; the Cox partial likelihood under Efron and Breslow ties, with Newton’s steps.',
    tags: [
      'survival analysis',
      'Kaplan–Meier',
      'log-rank test',
      'Cox proportional hazards',
      'AFT',
      'Weibull',
      'censoring',
    ],
    render: () => (
      <>
        <SurvivalCurvesSpecimen />
        <PartialLikelihoodSpecimen />
      </>
    ),
  },
]
