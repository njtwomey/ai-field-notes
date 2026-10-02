import type { Specimen } from '@lab/specimen'
import { HypothesisTestSpecimen } from './_tests/hypothesis'
import { PeekingPlayerSpecimen, PeekingSimulationSpecimen } from './_tests/peeking'

export const specimens: Specimen[] = [
  {
    module: 'part-1-mathematical-foundations/statistics/hypothesis-testing',
    title: 'Hypothesis tests: statistic, null, p-value',
    description:
      'Sixteen tests on one protocol: generate or drag the data, pick the alternative and move α; the null law with the rejection region, the p-value area, the interval and the effect size.',
    tags: ['t-test', 'p-value', 'rejection region', 'binomial', 'Fisher', 'Mann–Whitney', 'Kolmogorov–Smirnov', 'χ²'],
    render: () => <HypothesisTestSpecimen />,
  },
  {
    module: 'part-1-mathematical-foundations/statistics/hypothesis-testing',
    title: 'Peeking and sequential tests',
    description:
      'Repeated t-tests inflate the type-I error (a simulation) while the mixture SPRT and the confidence sequence hold it; one experiment played observation by observation, with Wald’s SPRT.',
    tags: ['peeking', 'optional stopping', 'mSPRT', 'confidence sequence', 'SPRT', 'e-value'],
    render: () => (
      <>
        <PeekingSimulationSpecimen />
        <PeekingPlayerSpecimen />
      </>
    ),
  },
]
