import type { Specimen } from '../../specimen'
import { PipelineStepsSpecimen, TransformedTargetSpecimen } from './_compose/figures'

export const specimens: Specimen[] = [
  {
    module: 'learning/compose',
    title: "A pipeline's fitted state, step by step",
    description:
      'pipeline(standardScaler(), polynomialFeatures(), logisticRegression()): the data as each step sees it, each step’s fitted state, and P(y = 1 | x) of the whole pipeline.',
    tags: ['pipeline', 'standardScaler', 'polynomialFeatures', 'StateTree'],
    render: () => <PipelineStepsSpecimen />,
  },
  {
    module: 'learning/compose',
    title: "A transformed target's predictive",
    description:
      'transformTarget(linearRegression(), logTarget()) returns a log-normal predictive: quantile band, median and mean against x, and the density at one x.',
    tags: ['transformTarget', 'log-normal', 'pushforward', 'Box–Cox'],
    render: () => <TransformedTargetSpecimen />,
  },
]
