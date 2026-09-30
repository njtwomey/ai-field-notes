import type { Specimen } from '../specimen'
import { DecisionRuleSpecimen, IrlsTraceSpecimen, LinearFitSpecimen } from './_estimators/figures'

export const specimens: Specimen[] = [
  {
    module: 'estimators',
    title: 'Linear regression and its predictive',
    description:
      'linearRegression (least squares or ridge) in FitView: the data, decide(x), E[y | x] and a 90% band of the Gaussian predictive, with the fitted state as a tree.',
    tags: ['FitView', 'linearRegression', 'ridge', 'predictive'],
    render: () => <LinearFitSpecimen />,
  },
  {
    module: 'estimators',
    title: 'Decision rules on a probabilistic classifier',
    description:
      'withDecision with a cost matrix turns P(y = 1 | x) from logisticRegression into Bayes decisions; the threshold moves with the cost of a missed positive.',
    tags: ['withDecision', 'cost matrix', 'logisticRegression', 'FitView'],
    render: () => <DecisionRuleSpecimen />,
  },
  {
    module: 'estimators',
    title: 'The training trace of a fit',
    description: "logisticRegression's Newton/IRLS loop as a trace: loss and Newton decrement per step.",
    tags: ['TraceView', 'IRLS', 'Newton', 'training'],
    render: () => <IrlsTraceSpecimen />,
  },
]
